# -*- coding: utf-8 -*-
# =====================================================================
#  تحميل البيانات مع التقسيم المختوم — framework/common/data.py
# ---------------------------------------------------------------------
#  ثلاثة مسارات مستقلّة (لا تُدمج مصفوفات ميزاتها):
#     uavids    — تدفّقات شبكة NS-3 ، 18 ميزة
#     isot      — حركة WiFi من DJI Tello ، 60 ميزة
#     telemetry — قياسات طيران PX4 (نوافذ المرحلة الثالثة ، بروتوكول LOAO)
#
#  كلّ محمّل يقبل partition ∈ {dev, sealed, all}:
#     dev    = بيانات التطوير (المراحل 1–4 وكلّ ضبط وبحث)
#     sealed = مجموعة الاختبار المختومة — تُفتح في المرحلة الخامسة فقط
#     all    = كلّ البيانات (لمطابقة نتائج المراحل القديمة فقط)
#  وحدة الختم ليست العيّنة بل الوحدة الطبيعية للبيانات (انظري sealing.py).
#
#  المنطق منسوخ من step4/data_core.py مع فرق وحيد: القناع الجزئي.
# =====================================================================
import os, glob, sys, subprocess, numpy as np, pandas as pd

from common import paths as P
from common import taxonomy as T
from common import sealing as S

N_PER_CLASS = 800 if P.QUICK else 6000
UAVIDS_DROP = ["FlowID", "SrcAddr", "DstAddr", "SrcPort", "DstPort", "label"]
ISOT_DROP   = ["ts", "Drone_port"]
TRACKS = ["uavids", "isot", "telemetry"]

_UAVIDS_DF = None


# ------------------------------------------------------------------
#  أدوات
# ------------------------------------------------------------------
def _cap_index(X, y, n_per_class, seed=42):
    rng = np.random.RandomState(seed)
    keep = []
    for c in np.unique(y):
        idx = np.where(y == c)[0]
        if len(idx) > n_per_class:
            idx = rng.choice(idx, n_per_class, replace=False)
        keep.append(idx)
    keep = np.sort(np.concatenate(keep))
    return X[keep], keep


#  ⚠️ مواصفة الميزات تُشتقّ من **قسم التطوير فقط** وتُطبَّق حرفياً على المختوم:
#     الأعمدة الثابتة تُحدَّد على التطوير ، والترميز الترتيبي بخريطة ثابتة.
#     بدون ذلك يختلف عدد الميزات بين القسمين (وقع فعلاً: ISOT 47 مقابل 45)
#     لأن عموداً ثابتاً في قسم قد لا يكون ثابتاً في الآخر.
_FEAT_SPEC = {}


def _spec_path(track):
    return os.path.join(P.CACHE, f"feature_spec_{track}.json")


def _numeric_matrix(df, feat_cols, verbose=True, spec=None):
    """(X, قناع الصفوف السليمة, أسماء الميزات, المواصفة).
    spec=None ⇒ تُشتقّ المواصفة من هذه البيانات (قسم التطوير) ؛ وإلا تُطبَّق كما هي."""
    out, dropped, encoded = {}, [], []
    enc_maps = {} if spec is None else dict(spec["encoded"])
    for c in feat_cols:
        s = pd.to_numeric(df[c], errors="coerce")
        if (spec is None and s.isna().all()) or (spec is not None and c in enc_maps):
            if df[c].dtype == object or c in enc_maps:
                if c not in enc_maps:
                    vals = sorted(df[c].astype(str).unique())
                    enc_maps[c] = {v: i for i, v in enumerate(vals)}
                s = df[c].astype(str).map(enc_maps[c]).fillna(-1).astype("float64")
                encoded.append(c)
            else:
                dropped.append((c, "غير قابل للتحويل")); continue
        elif spec is None and s.isna().all():
            dropped.append((c, "غير قابل للتحويل")); continue
        out[c] = s
    X = pd.DataFrame(out).replace([np.inf, -np.inf], np.nan)
    if spec is not None:
        keep = [c for c in spec["keep"] if c in X.columns]
        missing = [c for c in spec["keep"] if c not in X.columns]
        if missing:
            raise KeyError(f"أعمدة المواصفة غائبة عن هذا القسم: {missing}")
        X = X[keep]
    ok = ~X.isna().any(axis=1)
    Xo = X[ok]
    if spec is None:
        const = [c for c in Xo.columns if Xo[c].nunique(dropna=False) <= 1]
        if const:
            dropped += [(c, "ثابت") for c in const]
            Xo = Xo.drop(columns=const)
        spec = dict(keep=list(Xo.columns), encoded=enc_maps, dropped=[c for c, _ in dropped])
    if verbose and (encoded or dropped):
        if encoded:
            print(f"   • رُمِّزت ترتيبياً: {encoded}")
        if dropped:
            print(f"   • أُسقطت {len(dropped)} ميزة: " + ", ".join(f"{c} ({w})" for c, w in dropped))
    return Xo.to_numpy(dtype=np.float32), ok.to_numpy(), list(Xo.columns), spec


def _get_spec(track, build_dev):
    """مواصفة الميزات للمسار: من الذاكرة ، أو من ملفّ الـ cache ، أو تُشتقّ من التطوير."""
    if track in _FEAT_SPEC:
        return _FEAT_SPEC[track]
    p = _spec_path(track)
    if os.path.exists(p):
        import json
        with open(p, encoding="utf-8") as f:
            _FEAT_SPEC[track] = json.load(f)
        return _FEAT_SPEC[track]
    spec = build_dev()
    import json
    with open(p, "w", encoding="utf-8") as f:
        json.dump(spec, f, ensure_ascii=False, indent=1)
    _FEAT_SPEC[track] = spec
    return spec


def _partition_mask(is_sealed, partition):
    if partition == "all":
        return np.ones(len(is_sealed), bool)
    if partition == "dev":
        return ~is_sealed
    if partition == "sealed":
        return is_sealed
    raise ValueError(f"partition غير معروف: {partition}")


# ------------------------------------------------------------------
#  (١) UAVIDS-2025 — وحدة الختم: التدفّق (صفّ) ، طبقيّ حسب الوسم
# ------------------------------------------------------------------
def uavids_frame():
    global _UAVIDS_DF
    if _UAVIDS_DF is None:
        _UAVIDS_DF = pd.read_csv(P.UAVIDS_CSV, low_memory=False)
    return _UAVIDS_DF


def load_uavids(partition="dev", verbose=True):
    df = uavids_frame()
    feat = [c for c in df.columns if c not in UAVIDS_DROP]
    #  الختم v2: الصفوف المكرَّرة حرفياً مُسقَطة من كلّ الأقسام (فارغة في v1)
    dropped = S.uavids_dropped_rows() if os.path.exists(P.MANIFEST) else np.array([], dtype=np.int64)
    is_dropped = np.zeros(len(df), bool); is_dropped[dropped] = True
    if partition == "all":
        sub = df[~is_dropped]
        X, ok, feat, _ = _numeric_matrix(sub, feat, verbose)
        row_ids = sub.index.to_numpy()[ok]
        y = sub.loc[ok, "label"].astype(str).str.strip().to_numpy()
    else:
        sealed_rows = S.uavids_sealed_rows()
        is_sealed_all = np.zeros(len(df), bool); is_sealed_all[sealed_rows] = True
        dev_df = df[~is_sealed_all & ~is_dropped]

        def build_dev():
            return _numeric_matrix(dev_df, feat, False)[3]
        spec = _get_spec("uavids", build_dev)
        sub = df[_partition_mask(is_sealed_all, partition) & ~is_dropped]
        X, ok, feat, _ = _numeric_matrix(sub, feat, verbose, spec=spec)
        row_ids = sub.index.to_numpy()[ok]
        y = sub.loc[ok, "label"].astype(str).str.strip().to_numpy()
    X, keep = _cap_index(X, y, N_PER_CLASS)
    y, row_ids = y[keep], row_ids[keep]
    if verbose:
        print(f">> UAVIDS[{partition}]: X={X.shape} ، فئات={sorted(set(y))}")
    return X, y, feat, row_ids


# ------------------------------------------------------------------
#  (٢) ISOT — وحدة الختم: الذيل الزمني لكلّ ملفّ pcap (آخر 20% من صفوفه)
# ------------------------------------------------------------------
def isot_files():
    out = []
    for folder in sorted(os.listdir(P.ISOT_DIR)):
        fp = os.path.join(P.ISOT_DIR, folder)
        if not os.path.isdir(fp):
            continue
        for f in sorted(glob.glob(os.path.join(fp, "*.csv"))):
            out.append((folder, f, os.path.relpath(f, P.ISOT_DIR).replace("\\", "/")))
    return out


def load_isot(partition="dev", verbose=True, chunk=200_000, _spec_only=False):
    """العيّنة تُؤخذ أثناء القراءة (الذاكرة محدودة على هذا الجهاز)."""
    rng = np.random.RandomState(42)
    pool_cap = N_PER_CLASS * 4
    cuts = S.isot_cuts() if partition != "all" else None
    pools = {}
    for folder in sorted({f for f, _, _ in isot_files()}):
        kept, seen = [], 0
        for fo, f, rel in isot_files():
            if fo != folder:
                continue
            cut = cuts[rel]["cut"] if cuts is not None else None
            offset = 0
            try:
                for ch in pd.read_csv(f, low_memory=False, chunksize=chunk):
                    n = len(ch)
                    seen += n
                    if cut is not None:
                        pos = np.arange(offset, offset + n)
                        is_sealed = pos >= cut
                        ch = ch[_partition_mask(is_sealed, partition)]
                    offset += n
                    if len(ch) == 0:
                        continue
                    take = min(len(ch), pool_cap)
                    if len(ch) > take:
                        ch = ch.iloc[np.sort(rng.choice(len(ch), take, replace=False))]
                    kept.append(ch)
                    if sum(len(k) for k in kept) > pool_cap * 3:
                        big = pd.concat(kept, ignore_index=True)
                        idx = rng.choice(len(big), min(pool_cap, len(big)), replace=False)
                        kept = [big.iloc[np.sort(idx)]]
            except Exception as e:
                raise RuntimeError(f"تعذّرت قراءة {f}: {type(e).__name__}: {e}") from e
        if not kept:
            continue
        g = pd.concat(kept, ignore_index=True)
        if len(g) > pool_cap:
            g = g.iloc[np.sort(rng.choice(len(g), pool_cap, replace=False))]
        g["__cls"] = folder
        pools[folder] = g
        if verbose:
            print(f"   • ISOT/{folder}[{partition}]: قُرئ {seen:,} صفّاً ← حوض {len(g):,}")
    df = pd.concat(list(pools.values()), ignore_index=True)
    feat = [c for c in df.columns if c not in ISOT_DROP + ["__cls"]]
    if partition == "all":
        X, ok, feat, _ = _numeric_matrix(df, feat, verbose)
    else:
        def build_dev():
            if partition == "dev":
                return _numeric_matrix(df, feat, False)[3]
            #  المختوم قبل التطوير: نشتقّ المواصفة من التطوير أوّلاً (قراءة إضافية مرّة واحدة)
            return load_isot("dev", verbose=False, chunk=chunk, _spec_only=True)
        spec = _get_spec("isot", build_dev)
        X, ok, feat, _ = _numeric_matrix(df, feat, verbose, spec=spec)
    if _spec_only:
        return _FEAT_SPEC["isot"] if "isot" in _FEAT_SPEC else spec
    y = df.loc[ok, "__cls"].astype(str).str.strip().to_numpy()
    X, keep = _cap_index(X, y, N_PER_CLASS)
    y = y[keep]
    if verbose:
        print(f">> ISOT[{partition}]: X={X.shape} ، فئات={sorted(set(y))}")
    return X, y, feat


# ------------------------------------------------------------------
#  (٣) التليمتري — وحدة الختم: طائرة كاملة (منصّة غير مرئية)
# ------------------------------------------------------------------
TELEMETRY_NPZ = os.path.join(P.CACHE, "telemetry_export.npz")


def telemetry_export(verbose=True):
    if not os.path.exists(TELEMETRY_NPZ):
        script = os.path.join(P.HERE, "_export_telemetry.py")
        env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
        r = subprocess.run([sys.executable, "-u", script, TELEMETRY_NPZ], env=env,
                           cwd=os.path.join(P.STEP3, "experiments"))
        if r.returncode != 0 or not os.path.exists(TELEMETRY_NPZ):
            raise RuntimeError("فشل تصدير نوافذ التليمتري من المرحلة الثالثة")
    return np.load(TELEMETRY_NPZ, allow_pickle=True)


def load_telemetry(partition="dev", verbose=True):
    d = telemetry_export(verbose)
    X, names, groups, flights, pos = d["X"], d["y"], d["groups"], d["flights"], d["pos"]
    feat = [str(v) for v in d["feat"]]
    test_af = [str(v) for v in d["test_airframes"]]
    if partition != "all":
        af_sealed = S.telemetry_sealed_airframe()
        is_sealed = np.asarray(groups).astype(str) == af_sealed
        m = _partition_mask(is_sealed, partition)
        X, names, groups, flights, pos = X[m], names[m], groups[m], flights[m], pos[m]
        #  طيّات LOAO داخل كلّ قسم: في التطوير الطائرات المكتملة ما عدا المختومة ،
        #  وفي المختوم الطائرة المختومة وحدها.
        test_af = [a for a in test_af if (a == af_sealed) == (partition == "sealed")]
    if verbose:
        print(f">> Telemetry[{partition}]: X={X.shape} ، فئات={sorted(set(names))} "
              f"(window={d['window']} mask={d['mask_mode']})")
    return (np.asarray(X, dtype=np.float32), np.asarray(names), feat,
            np.asarray(groups), np.asarray(flights), test_af)


# ------------------------------------------------------------------
#  واجهة موحَّدة
# ------------------------------------------------------------------
def load_track(track, partition="dev", unified=True, verbose=True):
    """dict(X, y, y_txt, y_raw, classes, features, groups, protocol, test_airframes).
    unified=True ⇒ الوسوم هي الطبقات الخمس الموحَّدة + Normal."""
    if partition == "sealed":
        S.assert_unsealed(f"load_track({track})")
    groups, flights, protocol, test_af = None, None, "stratified", None
    if track == "uavids":
        X, y_raw, feat, _ = load_uavids(partition, verbose)
    elif track == "isot":
        X, y_raw, feat = load_isot(partition, verbose)
    elif track == "telemetry":
        X, y_raw, feat, groups, flights, test_af = load_telemetry(partition, verbose)
        protocol = "loao"
    else:
        raise ValueError(f"مسار غير معروف: {track}")

    y_txt = T.to_category(y_raw) if unified else np.array([str(v).strip() for v in y_raw])
    if unified:
        X, keep = _cap_index(X, y_txt, N_PER_CLASS)
        y_txt, y_raw = y_txt[keep], np.asarray(y_raw)[keep]
        if groups is not None:
            groups, flights = np.asarray(groups)[keep], np.asarray(flights)[keep]

    classes = sorted(set(y_txt))
    code = {c: i for i, c in enumerate(classes)}
    y = np.array([code[v] for v in y_txt], dtype=np.int64)
    if verbose:
        import collections
        cnt = collections.Counter(y_txt)
        print(f"   • الفئات ({len(classes)}): " + " ، ".join(f"{c}={cnt[c]}" for c in classes))
    return dict(X=X, y=y, y_txt=y_txt, y_raw=np.array([str(v).strip() for v in y_raw]),
                classes=classes, features=feat, groups=groups, flights=flights,
                protocol=protocol, test_airframes=test_af, partition=partition)


def align_classes(dev, sealed):
    """يجعل ترميز الفئات في المختوم مطابقاً لترميز التطوير (الفئة الغائبة تبقى غائبة)."""
    code = {c: i for i, c in enumerate(dev["classes"])}
    y = np.array([code[v] for v in sealed["y_txt"]], dtype=np.int64)
    out = dict(sealed)
    out["y"] = y
    out["classes"] = dev["classes"]
    return out
