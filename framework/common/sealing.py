# -*- coding: utf-8 -*-
# =====================================================================
#  الختم — framework/common/sealing.py
# ---------------------------------------------------------------------
#  الفكرة: تُقتطع مجموعة اختبار من كلّ مسار **قبل** أيّ تجربة وتُحفظ في
#  manifest ببصمة SHA-256 وبذرة ثابتة. لا يقرأها إلا سكربت يحمل رمز الفتح ،
#  وكلّ قراءة تُسجَّل في sealed_access_log.jsonl. أيّ سكربت آخر يحاول قراءتها
#  يتوقّف بخطأ صريح — الحماية بنيوية لا بالانتباه.
#
#  وحدة الختم (لا العيّنة — التقسيم على مستوى العيّنة هو التسريب الذي انتقده
#  الفصل الرابع):
#     uavids    : تدفّقات مستقلّة ⇒ 20% طبقيّ حسب الوسم (لا مُعرِّف سيناريو في الملفّ)
#     isot      : ملفّات pcap كاملة عددها 1–5 لكلّ فئة ⇒ الختم بالملفّ مستحيل
#                 لثلاث فئات ؛ نختم **الذيل الزمني** (آخر 20% من صفوف كلّ ملفّ)
#     telemetry : **طائرة كاملة** (منصّة غير مرئية) تُختار عشوائياً ببذرة ثابتة من
#                 الطائرات المكتملة الأربع. جُرِّب الذيل الزمني أوّلاً ورُفض: هجوم
#                 انتحال GPS يشغل 2–5% من وسط الرحلة ، فخرج المختوم بلا أيّ نافذة
#                 انتحال (صفر). الطائرة الكاملة تحفظ الفئات الثلاث وتطابق روح LOAO.
#  ⚠️ حدود صريحة تُذكر في الفصل: المختوم في ISOT **عبر الزمن** داخل الملفّ نفسه
#     لا عبر ملفّات جديدة (ثلاث فئات فيها ملفّ واحد) ؛ والمختوم في التليمتري
#     طائرة واحدة ⇒ لا تباين طيّات ، ويُبلَّغ التباين عبر البذور وbootstrap النوافذ.
# =====================================================================
import os, sys, hashlib, datetime, numpy as np, pandas as pd

from common import paths as P
from common import io as IO

FRAC = 0.20
SEED = 20260904
UNSEAL_ENV = "FW_UNSEAL_TOKEN"
TOKENS = {"stage5-heldout": "المرحلة الخامسة — التقييم المستقلّ",
          "stage0-sealing": "المرحلة صفر — بناء الختم والتحقّق منه"}

_MANIFEST = None


# ------------------------------------------------------------------
#  الحارس
# ------------------------------------------------------------------
def is_unsealed():
    return os.environ.get(UNSEAL_ENV) in TOKENS


def assert_unsealed(purpose=""):
    tok = os.environ.get(UNSEAL_ENV)
    caller = os.path.basename(sys.argv[0]) if sys.argv else "?"
    if tok not in TOKENS:
        raise PermissionError(
            "❌ محاولة قراءة المجموعة المختومة من خارج المرحلة الخامسة.\n"
            f"   السكربت: {caller}  الغرض: {purpose}\n"
            "   المختوم لا يُفتح إلا عبر stage5_trust/run.py (unseal()).")
    IO.append_jsonl(P.ACCESS_LOG, dict(time=IO.now(), script=caller, token=tok,
                                       purpose=purpose, pid=os.getpid()))


def unseal(token="stage5-heldout"):
    """تُستدعى من المرحلة الخامسة فقط. تُسجَّل كلّ عمليّة فتح."""
    assert token in TOKENS, token
    os.environ[UNSEAL_ENV] = token
    IO.append_jsonl(P.ACCESS_LOG, dict(time=IO.now(), event="unseal", token=token,
                                       script=os.path.basename(sys.argv[0]), pid=os.getpid()))


def reseal():
    os.environ.pop(UNSEAL_ENV, None)


# ------------------------------------------------------------------
#  المانيفست
# ------------------------------------------------------------------
def manifest():
    global _MANIFEST
    if _MANIFEST is None:
        if not os.path.exists(P.MANIFEST):
            raise FileNotFoundError(
                "لا يوجد ختم بعد. شغّلي أوّلاً:  python framework/stage0_sealing/run.py")
        _MANIFEST = IO.read_json(P.MANIFEST)
    return _MANIFEST


def uavids_sealed_rows():
    return np.asarray(manifest()["uavids"]["sealed_rows"], dtype=np.int64)


def uavids_dropped_rows():
    """الصفوف المكرَّرة حرفياً التي أُسقطت قبل الختم (الختم v2 ، 2026-09-05). فارغة في v1."""
    return np.asarray(manifest()["uavids"].get("dropped_rows", []), dtype=np.int64)


def uavids_duplicate_mask(df):
    """قناع الصفوف المكرَّرة حرفياً على الميزات + الوسم (يُبقي أوّل ظهور). المعرِّفات
    (FlowID ، العناوين ، المنافذ) لا تدخل المقارنة لأنها لا تدخل النموذج أصلاً."""
    from common import data as D
    feat = [c for c in df.columns if c not in D.UAVIDS_DROP]
    return df.duplicated(subset=feat + ["label"], keep="first").to_numpy()


def reseal_uavids_dedup(frac=FRAC, seed=SEED, verbose=True):
    """الختم v2 لـ UAVIDS: يُسقط التكرارات الحرفية (6.7% من الصفوف ، وسمها متطابق دائماً)
    ثم يعيد المعاينة الطبقية 20% من الباقي. ISOT والتليمتري يُنسخان حرفياً من الختم
    السابق حتى لا يتحرّك شيء فيهما (مولّد الأرقام لا يُستهلك عليهما). الختم السابق
    يُؤرشَف بجانبه ولا يُحذف ، وسجلّ الوصول يبقى كما هو.

    لماذا: 7.4% من صفوف مختوم UAVIDS في v1 كان لها توأم حرفي في التطوير ⇒ تفاؤل طفيف
    في أرقام المختوم (README §9 بند 6). القرار: نور ، 2026-09-05."""
    global _MANIFEST
    from common import data as D
    old = IO.read_json(P.MANIFEST)
    if old.get("seal_version", 1) >= 2:
        if verbose:
            IO.log(">> الختم v2 موجود مسبقاً ولا يُعاد بناؤه")
        return old
    arch = P.MANIFEST.replace("sealed_manifest.json", "sealed_manifest_v1_" + old["created"][:10].replace("-", "") + ".json")
    if not os.path.exists(arch):
        IO.write_json(arch, old)
    df = D.uavids_frame()
    labels = df["label"].astype(str).str.strip().to_numpy()
    dup = uavids_duplicate_mask(df)
    keep_idx = np.where(~dup)[0]
    rng = np.random.RandomState(seed)
    sealed = []
    for c in sorted(set(labels)):
        idx = keep_idx[labels[keep_idx] == c]
        k = int(round(frac * len(idx)))
        sealed += list(rng.choice(idx, k, replace=False))
    sealed = np.sort(np.array(sealed, dtype=np.int64))
    dropped = np.where(dup)[0]
    man = dict(old)
    man["seal_version"] = 2
    man["created_v2"] = IO.now()
    man["previous_manifest"] = os.path.basename(arch)
    man["rule"] = dict(old["rule"], uavids="exact duplicates (features + label, identifiers excluded) dropped from BOTH "
                                           "partitions, then stratified 20% of the remaining flows by label")
    man["uavids"] = dict(csv=old["uavids"]["csv"], sha256=old["uavids"]["sha256"],
                         n_rows=int(len(df)), n_dropped_duplicates=int(len(dropped)),
                         n_after_dedup=int(len(keep_idx)), n_sealed=int(len(sealed)),
                         per_label={c: dict(n=int((labels == c).sum()),
                                            dropped=int((labels[dropped] == c).sum()),
                                            sealed=int((labels[sealed] == c).sum()))
                                    for c in sorted(set(labels))},
                         sealed_rows=[int(v) for v in sealed],
                         dropped_rows=[int(v) for v in dropped])
    IO.write_json(P.MANIFEST, man)
    _MANIFEST = None
    if verbose:
        IO.log(f"✓ الختم v2: أُسقط {len(dropped):,} صفّاً مكرَّراً من {len(df):,} ؛ مختوم {len(sealed):,} من {len(keep_idx):,}")
        IO.log(f"   الختم السابق مؤرشَف: {arch}")
    return man


def reseal_isot_full(frac=FRAC, verbose=True):
    """الختم v3 لـ ISOT (2026-11-17): النسخة السابقة كانت تنزيلاً جزئياً (14 ملفّاً ، 371,815 صفّاً).
    الإصدار الرسمي الكامل (137 ملفّاً ، 2,945,986 صفّاً) وُضع في المسار نفسه. القاعدة لا تتغيّر
    (الذيل الزمني: آخر 20% من صفوف كلّ ملفّ) ولا تستهلك مولّد أرقام ⇒ حتمية. UAVIDS والتليمتري
    يُنسخان حرفياً. الختم السابق يُؤرشَف بجانبه ولا يُحذف ، وسجلّ الوصول يبقى كما هو.
    تنبيه للأمانة: ملفّات النسخة الجزئية الموجودة حرفياً في الإصدار الكامل يبقى ذيلها المختوم
    هو نفسه الذي فُتح في المرحلة الخامسة سابقاً ؛ تُسجَّل في الحقل reused_from_v2."""
    global _MANIFEST
    from common import data as D
    old = IO.read_json(P.MANIFEST)
    if old.get("seal_version", 1) >= 3:
        if verbose:
            IO.log(">> الختم v3 موجود مسبقاً ولا يُعاد بناؤه")
        return old
    arch = P.MANIFEST.replace("sealed_manifest.json",
                              f"sealed_manifest_v{old.get('seal_version', 1)}_before_isot_full_20261117.json")
    if not os.path.exists(arch):
        IO.write_json(arch, old)
    prev = {rel: spec for rel, spec in old["isot"]["files"].items()}
    files, reused = {}, []
    for folder, f, rel in D.isot_files():
        n = _count_rows(f)
        files[rel] = dict(cls=folder, n_rows=int(n), cut=int(round((1 - frac) * n)),
                          bytes=int(os.path.getsize(f)))
        if rel in prev and prev[rel]["bytes"] == files[rel]["bytes"]:
            reused.append(rel)
        if verbose:
            IO.log(f"   • {rel}: {n:,} صفّ ← dev {files[rel]['cut']:,} / sealed {n - files[rel]['cut']:,}")
    man = dict(old)
    man["seal_version"] = 3
    man["created_v3"] = IO.now()
    man["previous_manifest"] = os.path.basename(arch)
    man["isot"] = dict(dir=os.path.relpath(P.ISOT_DIR, P.ROOT), files=files,
                       release="official full release (2,945,986 rows, 9 attack types); was a partial download",
                       reused_from_v2=sorted(reused))
    IO.write_json(P.MANIFEST, man)
    _MANIFEST = None
    if verbose:
        IO.log(f"✓ الختم v3 لـ ISOT: {len(files)} ملفّاً ({len(reused)} منها مطابق لملفّ في الختم السابق)")
        IO.log(f"   الختم السابق مؤرشَف: {arch}")
    return man


def isot_cuts():
    return manifest()["isot"]["files"]


def telemetry_sealed_airframe():
    return str(manifest()["telemetry"]["sealed_airframe"])


def sha256(path, block=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(block)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _count_rows(path, chunk=500_000):
    n = 0
    for ch in pd.read_csv(path, usecols=[0], chunksize=chunk, low_memory=False):
        n += len(ch)
    return n


def build_manifest(frac=FRAC, seed=SEED, verbose=True):
    """يبني الختم من الصفر. لا يُعاد بناؤه إن وُجد (الختم يُبنى مرّة واحدة)."""
    from common import data as D
    if os.path.exists(P.MANIFEST):
        if verbose:
            IO.log(f">> الختم موجود مسبقاً ولا يُعاد بناؤه: {P.MANIFEST}")
        return IO.read_json(P.MANIFEST)

    rng = np.random.RandomState(seed)
    man = dict(created=IO.now(), seed=seed, frac=frac, quick=P.QUICK,
               rule=dict(uavids="stratified 20% of flows by label (no scenario id available)",
                         isot="temporal tail: last 20% of rows of every pcap-derived csv",
                         telemetry="one complete airframe (unseen platform) chosen at random "
                                   "with the fixed seed among the four complete airframes; "
                                   "temporal-tail sealing was rejected: it yielded zero spoofing windows"))

    # ---- UAVIDS ----
    if verbose:
        IO.log(">> ختم UAVIDS ...")
    df = D.uavids_frame()
    labels = df["label"].astype(str).str.strip().to_numpy()
    sealed = []
    for c in sorted(set(labels)):
        idx = np.where(labels == c)[0]
        k = int(round(frac * len(idx)))
        sealed += list(rng.choice(idx, k, replace=False))
    sealed = np.sort(np.array(sealed, dtype=np.int64))
    man["uavids"] = dict(csv=os.path.relpath(P.UAVIDS_CSV, P.ROOT), sha256=sha256(P.UAVIDS_CSV),
                         n_rows=int(len(df)), n_sealed=int(len(sealed)),
                         per_label={c: dict(n=int((labels == c).sum()),
                                            sealed=int(np.isin(np.where(labels == c)[0], sealed).sum()))
                                    for c in sorted(set(labels))},
                         sealed_rows=[int(v) for v in sealed])

    # ---- ISOT ----
    if verbose:
        IO.log(">> ختم ISOT (عدّ الصفوف لكلّ ملفّ) ...")
    files = {}
    for folder, f, rel in D.isot_files():
        n = _count_rows(f)
        files[rel] = dict(cls=folder, n_rows=int(n), cut=int(round((1 - frac) * n)),
                          bytes=int(os.path.getsize(f)))
        if verbose:
            IO.log(f"   • {rel}: {n:,} صفّ ← dev {files[rel]['cut']:,} / sealed {n - files[rel]['cut']:,}")
    man["isot"] = dict(dir=os.path.relpath(P.ISOT_DIR, P.ROOT), files=files)

    # ---- Telemetry ----
    if verbose:
        IO.log(">> ختم التليمتري (تصدير نوافذ المرحلة الثالثة) ...")
    d = D.telemetry_export(verbose)
    groups, names = d["groups"], d["y"]
    test_af = sorted(str(v) for v in d["test_airframes"])
    sealed_af = str(rng.choice(test_af))
    per_af = {}
    for af in sorted(set(groups.tolist())):
        m = groups == af
        per_af[str(af)] = dict(n=int(m.sum()),
                               per_class={str(c): int((names[m] == c).sum()) for c in sorted(set(names.tolist()))},
                               complete=str(af) in test_af,
                               partition="sealed" if str(af) == sealed_af else "dev")
    man["telemetry"] = dict(window=int(d["window"]), stride=int(d["stride"]),
                            mask_mode=str(d["mask_mode"]), sealed_airframe=sealed_af,
                            candidate_airframes=test_af, airframes=per_af,
                            export_sha256=sha256(D.TELEMETRY_NPZ))

    IO.write_json(P.MANIFEST, man)
    if verbose:
        IO.log(f"✓ كُتب الختم: {P.MANIFEST}")
    return man


def verify(verbose=True):
    """يتحقّق أن البيانات الخام لم تتغيّر منذ الختم."""
    from common import data as D
    man = manifest()
    rows = []
    ok_u = sha256(P.UAVIDS_CSV) == man["uavids"]["sha256"]
    rows.append(dict(track="uavids", check="sha256(UAVIDS csv)", ok=ok_u))
    for rel, spec in man["isot"]["files"].items():
        f = os.path.join(P.ISOT_DIR, rel)
        rows.append(dict(track="isot", check=f"bytes({rel})",
                         ok=os.path.exists(f) and os.path.getsize(f) == spec["bytes"]))
    ok_t = os.path.exists(D.TELEMETRY_NPZ) and sha256(D.TELEMETRY_NPZ) == man["telemetry"]["export_sha256"]
    rows.append(dict(track="telemetry", check="sha256(telemetry export)", ok=ok_t))
    df = pd.DataFrame(rows)
    if verbose:
        bad = df[~df.ok]
        IO.log("✓ الختم سليم: البيانات لم تتغيّر" if bad.empty else f"❌ تغيّرت البيانات:\n{bad}")
    return df
