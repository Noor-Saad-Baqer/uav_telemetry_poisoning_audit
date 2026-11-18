# -*- coding: utf-8 -*-
# =====================================================================
#  النواة المشتركة للمرحلة الثالثة — بيانات التليمتري + النماذج + التقييم
#  Shared core (Step 3): telemetry windows, LOAO splits, victim models
# ---------------------------------------------------------------------
#  يوفّر:
#   - build_dataset()            : نوافذ زمنية + ميزات إحصائية (مع cache قرصي)
#   - loao_folds()               : تقسيم اترك-طائرة-خارجاً (بلا تسريب رحلات)
#   - fit_scaler / apply_scaler  : تطبيع يُدرَّب على التدريب فقط
#   - train_victim()             : MLP يُرجع الخسائر لكل عيّنة + التمثيلات الكامنة
#   - train_sklearn()            : RF / XGBoost
#   - evaluate()                 : BalAcc + MacroF1 + استدعاء الفئة المستهدَفة
# =====================================================================
import os, sys, glob, json, hashlib, warnings, numpy as np, pandas as pd
warnings.filterwarnings("ignore")
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import balanced_accuracy_score, f1_score, recall_score

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
import config as CFG

NC = CFG.NUM_CLASSES

# ---------------------------------------------------------------------
#  لا تبعيّة على PyTorch: النموذج الضحيّة مُنفَّذ بـ NumPy (انظر NumpyMLP).
#  السبب أن استيراد torch على ويندوز يحمّل مكتبات CUDA ويحجز ذاكرة commit
#  ضخمة فيفشل على الأجهزة ذات ملف ترحيل (paging file) ثابت وصغير.
# ---------------------------------------------------------------------
def device():
    """الجهاز المستعمل — النموذج الضحيّة مُنفَّذ بـ NumPy على المعالج."""
    return "cpu (numpy)"
#  إشارات GPS مشتقّة (تُستعمل فقط عند USE_GPS=True)
DERIVED = ["d_alt", "d_vel", "pos_jump", "vel_consistency"]


# =====================================================================
#  (1) تحميل الرحلات — Flight discovery (دمج طوبيكات متعدّدة)
# =====================================================================
def _topic(folder, name):
    g = glob.glob(os.path.join(folder, f"*_{name}_0.csv"))
    return g[0] if g else None


def discover_flights():
    """يُرجع [(airframe, class, {topic: path})] للرحلات التي تملك الطوبيكات المطلوبة."""
    need = ["innovations", "telemetry_status"] + (["vehicle_gps_position"] if CFG.USE_GPS else [])
    out = []
    for af in sorted(os.listdir(CFG.SIM_DIR)):
        afp = os.path.join(CFG.SIM_DIR, af)
        if not os.path.isdir(afp):
            continue
        for cls in CFG.CLASSES:
            cp = os.path.join(afp, cls)
            if not os.path.isdir(cp):
                continue
            paths = {t: _topic(cp, t) for t in
                     ["innovations", "telemetry_status", "vehicle_gps_position"]}
            if all(paths.get(t) for t in need):
                out.append((af, cls, paths))
    return out


def _read(path, cols):
    """قراءة اقتصادية للذاكرة: الأعمدة المطلوبة فقط."""
    head = pd.read_csv(path, nrows=0).columns.tolist()
    use = ["timestamp"] + [c for c in cols if c in head]
    df = pd.read_csv(path, usecols=use, low_memory=False)
    for c in cols:                                   # عمود مفقود → صفر
        if c not in df.columns:
            df[c] = 0.0
    return df[["timestamp"] + list(cols)].sort_values("timestamp")


def _derive_gps(df):
    """إشارات GPS مشتقّة: قفزة الموقع واتّساق الموقع-السرعة (دالّة على الانتحال)."""
    dt = (df["timestamp"].diff() / 1e6).replace(0, np.nan)
    df["d_alt"] = df["alt"].diff()
    df["d_vel"] = df["vel_m_s"].diff()
    dlat = df["lat"].diff() * 111320.0
    dlon = df["lon"].diff() * 111320.0 * np.cos(np.deg2rad(df["lat"].astype(float)))
    df["pos_jump"] = np.hypot(dlat, dlon)
    df["vel_consistency"] = (df["pos_jump"] / dt - df["vel_m_s"]).abs()
    return df


def _signed_log(a):
    """ضغط لوغاريتمي مع حفظ الإشارة: sign(x)·log1p(|x|).

    بواقي كالمان تمتدّ من 0.05 إلى 6.4e4 ⇒ التوزيع ثقيل الذيل والمقياس يختلف
    بين الطائرات، فلا ينتقل التوقيع عبر المنصّات. الضغط اللوغاريتمي يجعل
    «حجم» الشذوذ قابلاً للمقارنة بين طائرة وأخرى.
    """
    return np.sign(a) * np.log1p(np.abs(a))


def _fuse(paths):
    """يدمج الطوبيكات على شبكة زمن واحدة (innovations = الشبكة الأساس، الأعلى تردّداً)."""
    base = _read(paths["innovations"], CFG.INNOV_SIGNALS)
    if CFG.LOG_SCALE_INNOV:
        base[CFG.INNOV_SIGNALS] = _signed_log(base[CFG.INNOV_SIGNALS].astype("float64"))
    tel = _read(paths["telemetry_status"], CFG.TELEM_SIGNALS)
    #  merge_asof = آخر قيمة تليمتري سابقة لكل عيّنة innovations (بلا استشراف مستقبلي)
    df = pd.merge_asof(base, tel, on="timestamp", direction="backward")
    sigs = list(CFG.INNOV_SIGNALS) + list(CFG.TELEM_SIGNALS)
    if CFG.USE_GPS and paths.get("vehicle_gps_position"):
        gps = _read(paths["vehicle_gps_position"],
                    list(CFG.GPS_SIGNALS) + ["lat", "lon"])
        gps = _derive_gps(gps)
        keep = list(CFG.GPS_SIGNALS) + DERIVED
        df = pd.merge_asof(df, gps[["timestamp"] + keep], on="timestamp", direction="backward")
        sigs += keep
    return df, sigs


def _window_feats(arr):
    """إحصاءات النافذة لكل إشارة: mean/std/min/max/slope → متجه مسطّح."""
    W = arr.shape[0]
    t = np.arange(W, dtype=np.float64)
    t = (t - t.mean())
    denom = (t * t).sum()
    feats = []
    for name in CFG.WIN_STATS:
        if name == "mean":  feats.append(arr.mean(0))
        elif name == "std": feats.append(arr.std(0))
        elif name == "min": feats.append(arr.min(0))
        elif name == "max": feats.append(arr.max(0))
        elif name == "slope":
            feats.append((t[:, None] * (arr - arr.mean(0))).sum(0) / denom)
    return np.concatenate(feats)


def _cache_key():
    payload = json.dumps({"W": CFG.WINDOW, "S": CFG.STRIDE, "innov": CFG.INNOV_SIGNALS,
                          "telem": CFG.TELEM_SIGNALS, "gps": CFG.USE_GPS,
                          "stats": CFG.WIN_STATS, "mode": CFG.LABEL_MODE,
                          "noleak": getattr(CFG, "EXCLUDE_LABEL_FEATURES", False),
                          "leaksig": getattr(CFG, "LABEL_LEAK_SIGNALS", []),
                          "thr": [CFG.SPOOF_INNOV_THR, CFG.DOS_TX_MULT, CFG.ACTIVE_FRAC],
                          "maskmode": getattr(CFG, "SPOOF_MASK_MODE", "raw"),
                          "ratio": CFG.NORMAL_RATIO, "log": CFG.LOG_SCALE_INNOV,
                          "ver": CFG.CACHE_VERSION}, sort_keys=True)
    return hashlib.md5(payload.encode()).hexdigest()[:16]


def _tx_baselines():
    """خطّ أساس rate_tx من الرحلة الطبيعية لكل طائرة (مرجع نسبي لكشف Ping DoS)."""
    base = {}
    for af, cls, paths in discover_flights():
        if cls == "Normal":
            t = _read(paths["telemetry_status"], ["rate_tx"])
            base[af] = float(np.nanmedian(t["rate_tx"].values))
    return base


def _active_mask(df, cls, af, tx_base):
    """قناع نشاط الهجوم لكل عيّنة زمنية (مشتقّ فيزيائياً، لا من اسم المجلّد)."""
    n = len(df)
    if cls == "Normal":
        return np.zeros(n, dtype=bool)
    if cls == "GPS Spoofing":
        #  بواقي موقع المرشّح تنفجر عندما يرفض EKF إحداثيات GPS المزيّفة
        cols = [c for c in ("vel_pos_innov[3]", "vel_pos_innov[4]") if c in df.columns]
        if not cols:
            return np.zeros(n, dtype=bool)
        a = df[cols].abs().max(axis=1).values
        thr = CFG.SPOOF_INNOV_THR
        if CFG.LOG_SCALE_INNOV:      # الإشارات مضغوطة لوغاريتمياً ⇒ اضغطي العتبة مثلها
            thr = np.log1p(thr)
        raw = a > thr
        if getattr(CFG, "SPOOF_MASK_MODE", "raw") != "envelope":
            return raw
        #  قاعدة الظرف: العتبة الخام تُعلّم انتقالات المرشّح (بدء الهجوم ، ثم العودة
        #  **بعد** انتهائه) وتترك منتصفه فارغاً حين يقتنع المرشّح بالموقع المزيّف.
        #  الهجوم الحقيقي يمتدّ من بداية أوّل ارتفاع إلى بداية آخر ارتفاع.
        idx = np.flatnonzero(raw)
        if idx.size == 0:
            return raw
        starts = idx[np.r_[True, np.diff(idx) > 1]]      # بداية كل ارتفاع متّصل
        env = np.zeros(n, dtype=bool)
        env[starts[0]:(starts[-1] if starts.size > 1 else idx[-1] + 1)] = True
        return env
    #  Ping DoS: معدّل الإرسال يقفز فوق ضِعف خطّ أساس نفس الطائرة
    b = tx_base.get(af, 15.0)
    return df["rate_tx"].values > CFG.DOS_TX_MULT * b


def build_dataset(verbose=True):
    """يبني (X, y, groups, flights, feat_names) من كل الرحلات المحاكاة — مع cache.

    كل نافذة = مقطع زمني متّصل داخل رحلة واحدة (لا تعبر النوافذ حدود الرحلات).
    """
    fp = os.path.join(CFG.CACHE_DIR, f"dataset_{_cache_key()}.npz")
    if os.path.exists(fp):
        d = np.load(fp, allow_pickle=True)
        if verbose:
            print(f">> [core] dataset من الـ cache: X={d['X'].shape}")
        return d["X"], d["y"], d["groups"], d["flights"], list(d["feat_names"])

    tx_base = _tx_baselines()
    Xs, ys, gs, fs = [], [], [], []
    sigs = None
    for af, cls, paths in discover_flights():
        df, sigs = _fuse(paths)
        active = _active_mask(df, cls, af, tx_base)   # يُحسَب قبل الحذف (فهو تعريف الوسم)
        if getattr(CFG, "EXCLUDE_LABEL_FEATURES", False):
            #  كسر الدائرية: الإشارات التي عُرِّف بها الوسم لا تدخل الميزات
            sigs = [x for x in sigs if x not in set(CFG.LABEL_LEAK_SIGNALS)]
        A = df[sigs].apply(pd.to_numeric, errors="coerce")
        A = A.replace([np.inf, -np.inf], np.nan).ffill().fillna(0.0).values.astype("float64")
        n = len(A)
        if n < CFG.WINDOW:
            continue
        cls_id = CFG.CLASSES.index(cls)
        nw = na = 0
        for st in range(0, n - CFG.WINDOW + 1, CFG.STRIDE):
            if CFG.LABEL_MODE == "window":
                #  النافذة = هجوم فقط إذا كان الهجوم نشطاً فعلاً خلالها
                frac = active[st:st + CFG.WINDOW].mean()
                lab = cls_id if frac >= CFG.ACTIVE_FRAC else 0
                #  النوافذ الانتقالية (نشاط جزئي) تُهمَل: وسمها غامض
                if 0 < frac < CFG.ACTIVE_FRAC and cls_id != 0:
                    continue
            else:
                lab = cls_id
            Xs.append(_window_feats(A[st:st + CFG.WINDOW]))
            ys.append(lab); gs.append(af); fs.append(f"{af}|{cls}")
            nw += 1; na += (lab != 0)
        if verbose:
            print(f"   {af:16s} {cls:13s} rows={n:6d} → windows={nw:5d}  (attack-labelled {na})")

    X = np.asarray(Xs, dtype="float32")
    #  قصّ القيم المتطرّفة جداً (بواقي كالمان تبلغ 6e4) لتفادي انفجار التطبيع
    X = np.clip(np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0), -1e6, 1e6)
    y = np.asarray(ys, dtype="int64")
    groups = np.asarray(gs); flights = np.asarray(fs)

    #  موازنة الفئات بعد الوسم الصحيح: «طبيعي» و«Ping DoS» يفوقان «انتحال GPS»
    #  بعشرات المرّات (لأن الانتحال يشغل 2% من الرحلة فقط). نقصّ كل فئة زائدة إلى
    #  NORMAL_RATIO × أصغر فئة هجوم ، مع أخذ حصّة متساوية من كل رحلة (بلا تحيّز رحلة).
    if CFG.LABEL_MODE == "window" and CFG.NORMAL_RATIO:
        counts = np.bincount(y, minlength=NC)
        atk = [c for c in counts[1:] if c > 0]
        if atk:
            cap = int(CFG.NORMAL_RATIO * min(atk))
            rng = np.random.RandomState(0)
            keep = []
            for c in range(NC):
                idx_c = np.where(y == c)[0]
                if len(idx_c) <= cap:
                    keep += list(idx_c); continue
                per_flight = {}
                for i in idx_c:
                    per_flight.setdefault(flights[i], []).append(i)
                share = max(1, cap // len(per_flight))
                sub = []
                for fl, idxs in per_flight.items():
                    sub += list(rng.permutation(np.array(idxs))[:share])
                keep += sub
                if verbose:
                    print(f">> [core] موازنة {CFG.CLASSES[c]}: {len(idx_c)} → {len(sub)}")
            sel = np.sort(np.array(keep))
            X, y, groups, flights = X[sel], y[sel], groups[sel], flights[sel]

    feat_names = [f"{s}_{st}" for st in CFG.WIN_STATS for s in sigs]
    np.savez_compressed(fp, X=X, y=y, groups=groups, flights=flights, feat_names=np.array(feat_names))
    if verbose:
        print(f">> [core] X={X.shape}  y={np.bincount(y)}  airframes={len(set(gs))}")
    return X, y, groups, flights, feat_names


# =====================================================================
#  (2) تقسيم اترك-طائرة-خارجاً — Leave-One-Airframe-Out
# =====================================================================
def loao_folds(groups):
    """يُرجع [(held_airframe, train_idx, test_idx)] — الطائرات الناقصة تبقى في التدريب."""
    folds = []
    for af in CFG.TEST_AIRFRAMES:
        te = np.where(groups == af)[0]
        if len(te) == 0:
            continue
        tr = np.where(groups != af)[0]
        folds.append((af, tr, te))
    return folds


def temporal_folds(flights):
    """طيّة واحدة: أوّل 70% زمنياً من كل رحلة للتدريب، آخر 30% للاختبار.

    تُترَك فجوة = WINDOW/STRIDE نافذة بين الجزأين حتى لا تتداخل نوافذ التدريب
    مع نوافذ الاختبار (تسريب زمني).
    """
    gap = CFG.WINDOW // CFG.STRIDE
    tr, te = [], []
    for fl in np.unique(flights):
        idx = np.where(flights == fl)[0]              # مرتّبة زمنياً بحكم البناء
        cut = int(CFG.TEMPORAL_TRAIN_FRAC * len(idx))
        tr += list(idx[:cut]); te += list(idx[cut + gap:])
    return [("within-flight", np.array(tr, dtype=int), np.array(te, dtype=int))]


def folds(groups, flights, protocol=None):
    """يختار البروتوكول: loao (منصّة غير مرئية) أو temporal (منصّة معروفة)."""
    p = protocol or CFG.PROTOCOL
    return temporal_folds(flights) if p == "temporal" else loao_folds(groups)


def fit_scaler(Xtr):
    #  الجهاز محدود الـ commit memory ⇒ نبقى على float32 (StandardScaler يرفع لـ float64)
    return StandardScaler(copy=False).fit(np.ascontiguousarray(Xtr, dtype=np.float32))


def apply_scaler(sc, X):
    return np.ascontiguousarray(sc.transform(np.array(X, dtype=np.float32)), dtype=np.float32)


# =====================================================================
#  (3) النموذج الضحيّة — MLP (يكشف الخسائر والتمثيلات لكواشف التسميم)
# =====================================================================
# =====================================================================
#  النموذج الضحيّة — تنفيذ NumPy خالص (بلا PyTorch)
# ---------------------------------------------------------------------
#  السبب: الشبكة صغيرة (85→128→64→3) وحجم البيانات ~2000 عيّنة ، بينما
#  استيراد torch على ويندوز يحمّل مكتبات CUDA ويحجز ذاكرة commit ضخمة
#  فيفشل على الأجهزة ضيّقة الحجز. التنفيذ هنا يعطي نفس القدرات المطلوبة:
#  خسارة كل عيّنة ، التمثيلات الكامنة ، خسارة GCE المتينة ، أوزان العيّنات.
# =====================================================================
class NumpyMLP:
    """MLP بطبقتين مخفيّتين + Adam ، مع دعم GCE وأوزان العيّنات."""

    def __init__(self, d, nc=NC, seed=0):
        rng = np.random.RandomState(seed)
        h1, h2 = CFG.MLP_HIDDEN
        # تهيئة He
        self.W = [rng.randn(d, h1).astype(np.float32) * np.sqrt(2.0 / d),
                  rng.randn(h1, h2).astype(np.float32) * np.sqrt(2.0 / h1),
                  rng.randn(h2, nc).astype(np.float32) * np.sqrt(2.0 / h2)]
        self.b = [np.zeros(h1, np.float32), np.zeros(h2, np.float32), np.zeros(nc, np.float32)]
        self._m = [np.zeros_like(p) for p in self.W + self.b]
        self._v = [np.zeros_like(p) for p in self.W + self.b]
        self._t = 0
        self.nc = nc

    # ---------------- الأمام ----------------
    def forward(self, X, train=False, rng=None):
        z1 = X @ self.W[0] + self.b[0]
        a1 = np.maximum(z1, 0)
        mask = None
        if train and CFG.DROPOUT > 0:
            mask = (rng.rand(*a1.shape) > CFG.DROPOUT).astype(np.float32) / (1 - CFG.DROPOUT)
            a1 = a1 * mask
        z2 = a1 @ self.W[1] + self.b[1]
        a2 = np.maximum(z2, 0)                      # التمثيل الكامن
        logits = a2 @ self.W[2] + self.b[2]
        return logits, (X, z1, a1, mask, z2, a2)

    @staticmethod
    def _softmax(z):
        z = z - z.max(1, keepdims=True)
        e = np.exp(z)
        return e / np.clip(e.sum(1, keepdims=True), 1e-12, None)

    # ---------------- الخلف + Adam ----------------
    def _step(self, cache, dlogits, lr):
        X, z1, a1, mask, z2, a2 = cache
        g = [None] * 6
        g[2] = a2.T @ dlogits;              g[5] = dlogits.sum(0)
        da2 = dlogits @ self.W[2].T
        dz2 = da2 * (z2 > 0)
        g[1] = a1.T @ dz2;                  g[4] = dz2.sum(0)
        da1 = dz2 @ self.W[1].T
        if mask is not None:
            da1 = da1 * mask
        dz1 = da1 * (z1 > 0)
        g[0] = X.T @ dz1;                   g[3] = dz1.sum(0)

        self._t += 1
        params = self.W + self.b
        b1, b2, eps = 0.9, 0.999, 1e-8
        for i, (p, gr) in enumerate(zip(params, g)):
            np.clip(gr, -5.0, 5.0, out=gr)                      # قصّ التدرّج
            self._m[i] = b1 * self._m[i] + (1 - b1) * gr
            self._v[i] = b2 * self._v[i] + (1 - b2) * gr * gr
            mh = self._m[i] / (1 - b1 ** self._t)
            vh = self._v[i] / (1 - b2 ** self._t)
            p -= lr * mh / (np.sqrt(vh) + eps)

    def fit(self, X, y, epochs, lr, seed=0, loss_fn="ce", gce_q=None, sample_weight=None,
            track_from=None, tracker=None):
        rng = np.random.RandomState(seed)
        n = len(y)
        for ep in range(epochs):
            perm = rng.permutation(n)
            for i in range(0, n, CFG.BATCH_SIZE):
                idx = perm[i:i + CFG.BATCH_SIZE]
                if len(idx) < 2:
                    continue
                xb, yb = X[idx], y[idx]
                logits, cache = self.forward(xb, train=True, rng=rng)
                p = self._softmax(logits)
                oh = np.zeros_like(p); oh[np.arange(len(yb)), yb] = 1.0
                dl = (p - oh)
                if loss_fn == "gce":
                    #  تدرّج GCE = تدرّج CE مضروباً في p_y^q ⇒ يخمد العيّنات ذات الثقة المنخفضة
                    q = CFG.GCE_Q if gce_q is None else float(gce_q)
                    py = np.clip(p[np.arange(len(yb)), yb], 1e-6, 1.0)[:, None]
                    dl = dl * (py ** q)
                if sample_weight is not None:
                    dl = dl * sample_weight[idx][:, None]
                self._step(cache, (dl / len(idx)).astype(np.float32), lr)
            if tracker is not None and track_from is not None and ep >= track_from:
                tracker.append(self.sample_loss(X, y))
        return self

    # ---------------- واجهات الاستدلال ----------------
    def logits(self, X):
        return self.forward(X, train=False)[0]

    def sample_loss(self, X, y):
        p = self._softmax(self.logits(X))
        return -np.log(np.clip(p[np.arange(len(y)), y], 1e-12, None))

    def latent(self, X):
        return self.forward(X, train=False)[1][5]

    def predict(self, X):
        return self.logits(X).argmax(1)


class RFVictim:
    """ضحيّة قائمة على غابة عشوائية — بواجهة `NumpyMLP` نفسها (predict/sample_loss/latent).

    **لماذا:** الشبكة النقية تعجز عن هذه المهمّة عبر المنصّات بعد تصحيح الوسم
    (‏BalAcc 0.50 واستدعاء انتحال 0.40 مقابل 0.72 و0.82 للغابة) ، فدراسة التسميم
    عليها تقيس تذبذباً حول الصدفة لا ضرراً.

    **دقيقة حاسمة — الخسارة من خارج الحقيبة (OOB):** الغابة تلائم بيانات تدريبها
    ملاءمةً تامّة ، فخسارة كل عيّنة تدريبية ≈ 0 سواء أكانت مسمومة أم نظيفة ، وهذا
    يُبطل الكاشف القائم على الخسارة تماماً. لذا تُؤخَذ الاحتمالات على بيانات
    التدريب من `oob_decision_function_` (أشجار لم تَرَ العيّنة) ⇒ تقدير غير متحيّز
    ونظير سليم لـ «خسارة التدريب» في الشبكات.
    """

    def __init__(self, seed=0, n_estimators=200):
        from sklearn.ensemble import RandomForestClassifier
        self.rf = RandomForestClassifier(n_estimators=n_estimators, n_jobs=2,
                                         random_state=seed, bootstrap=True,
                                         oob_score=True)
        self._svd = self._enc = None

    def fit(self, X, y, sample_weight=None):
        self.rf.fit(X, y, sample_weight=sample_weight)
        self._Xtr = X
        self._classes = self.rf.classes_
        self._oob = getattr(self.rf, "oob_decision_function_", None)
        return self

    def _to_full(self, p):
        """يوسّع الأعمدة إلى NC حين تغيب فئة من بيانات التدريب."""
        out = np.zeros((len(p), NC), dtype=np.float64)
        for j, c in enumerate(self._classes):
            out[:, int(c)] = p[:, j]
        return out

    def _proba(self, X):
        same = (self._oob is not None and X.shape == self._Xtr.shape
                and np.array_equal(X, self._Xtr))
        if same:
            p = np.nan_to_num(self._oob, nan=1.0 / max(1, len(self._classes)))
        else:
            p = self.rf.predict_proba(X)
        return self._to_full(p)

    def sample_loss(self, X, y):
        p = self._proba(X)
        return -np.log(np.clip(p[np.arange(len(y)), np.asarray(y, dtype=int)], 1e-12, None))

    def latent(self, X):
        """التمثيل الكامن للغابة: فهارس الأوراق ← ترميز أحادي ← SVD.

        هذا هو النظير المتعارف عليه لطبقة التنشيط الأخيرة في الشبكات ، وعليه
        يقوم كاشفا `spectral` و`activation_cluster`.
        """
        from sklearn.decomposition import TruncatedSVD
        from sklearn.preprocessing import OneHotEncoder
        if self._svd is None:
            self._enc = OneHotEncoder(handle_unknown="ignore")
            Z = self._enc.fit_transform(self.rf.apply(self._Xtr))
            k = max(2, min(32, Z.shape[1] - 1))
            self._svd = TruncatedSVD(n_components=k, random_state=0).fit(Z)
        return self._svd.transform(self._enc.transform(self.rf.apply(X)))

    def logits(self, X):
        """لوغاريتم الاحتمالات — يجعل `_softmax(logits)` يُعيد الاحتمالات نفسها.

        يلزم لدفاع `relabel_retrain` الذي يقرأ الثقة من softmax اللوغاريتمات.
        """
        return np.log(np.clip(self._to_full(self.rf.predict_proba(X)), 1e-12, None))

    @staticmethod
    def _softmax(z):
        e = np.exp(z - z.max(1, keepdims=True))
        return e / e.sum(1, keepdims=True)

    def predict(self, X):
        return self.rf.predict(X)


def train_victim(Xtr, ytr, seed=0, epochs=None, lr=None, sample_weight=None,
                 loss_fn="ce", gce_q=None, track_losses=False, backend=None):
    """يدرّب النموذج الضحيّة ويُرجع (model, متوسّط خسارة كل عيّنة أو None).

    loss_fn: "ce" اعتيادي | "gce" خسارة متينة ضدّ التسميم.
    sample_weight: أوزان لكل عيّنة (للدفاع بالتصفية).
    track_losses: يجمع خسارة كل عيّنة بعد التسخين (لكاشف الخسارة).
    backend: "rf" | "mlp" — يتجاوز `CFG.VICTIM`. يلزم كي يبقى صفّ MLP في جدول
    نماذج المقارنة **شبكةً فعلاً** ولا يتحوّل إلى نسخة ثانية من الضحيّة.
    """
    if (backend or getattr(CFG, "VICTIM", "mlp")) == "rf":
        if loss_fn == "gce":
            raise NotImplementedError(
                "خسارة GCE تتطلّب متعلّماً قابلاً للاشتقاق ⇒ غير منطبقة على ضحيّة الغابة. "
                "احذفي 'robust_loss' من CFG.DEFENSES أو أعيدي CFG.VICTIM='mlp'.")
        Xtr = np.ascontiguousarray(Xtr, dtype=np.float32)
        ytr = np.asarray(ytr, dtype=np.int64)
        return RFVictim(seed=seed).fit(Xtr, ytr, sample_weight=sample_weight), None

    epochs = int(epochs or CFG.EPOCHS)
    lr = float(lr or CFG.LR)
    Xtr = np.ascontiguousarray(Xtr, dtype=np.float32)
    ytr = np.asarray(ytr, dtype=np.int64)
    model = NumpyMLP(Xtr.shape[1], NC, seed=seed)
    hist = [] if track_losses else None
    model.fit(Xtr, ytr, epochs=epochs, lr=lr, seed=seed, loss_fn=loss_fn, gce_q=gce_q,
              sample_weight=(None if sample_weight is None
                             else np.asarray(sample_weight, dtype=np.float32)),
              track_from=(CFG.WARMUP_EP if track_losses else None), tracker=hist)
    return model, (np.mean(hist, axis=0) if hist else None)


def per_sample_loss(model, X, y):
    """خسارة كل عيّنة على حدة — أساس الكاشف القائم على الخسارة."""
    return model.sample_loss(np.ascontiguousarray(X, dtype=np.float32),
                             np.asarray(y, dtype=np.int64))


def latent_feats(model, X):
    """التمثيلات الكامنة — أساس التوقيع الطيفي وتجميع التنشيطات."""
    return model.latent(np.ascontiguousarray(X, dtype=np.float32))


def predict(model, X):
    return model.predict(np.ascontiguousarray(X, dtype=np.float32))


# =====================================================================
#  (4) نماذج مرجعية — RF / XGBoost
# =====================================================================
def train_sklearn(name, Xtr, ytr, seed=0):
    if name == "rf":
        from sklearn.ensemble import RandomForestClassifier
        #  n_jobs محدود: كل عامل يحجز commit memory ، والجهاز ضيّق الحجز
        m = RandomForestClassifier(n_estimators=200, n_jobs=2, random_state=seed)
    elif name == "xgb":
        from xgboost import XGBClassifier
        #  n_jobs محدود ﻷن الجهاز محدود الذاكرة (كان يرمي bad allocation)
        m = XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1, subsample=0.9,
                          colsample_bytree=0.9, tree_method="hist", random_state=seed,
                          n_jobs=2, verbosity=0)
    else:
        raise ValueError(name)
    m.fit(Xtr, ytr)
    return m


def fit_predict(model_name, Xtr, ytr, Xte, seed=0, **kw):
    """واجهة موحّدة: يدرّب النموذج المطلوب ويُرجع تنبّؤات الاختبار."""
    if model_name in ("rf", "xgb"):
        return train_sklearn(model_name, Xtr, ytr, seed).predict(Xte)
    #  "mlp" هنا يعني الشبكة فعلاً — لا الضحيّة أياً كانت (وإلّا كرّر جدولُ
    #  المقارنة صفّ الغابة مرّتين حين تكون هي الضحيّة).
    model, _ = train_victim(Xtr, ytr, seed=seed, backend="mlp", **kw)
    return predict(model, Xte)


# =====================================================================
#  (5) المقاييس — Metrics
# =====================================================================
def evaluate(y_true, y_pred):
    """BalAcc + MacroF1 + استدعاء فئة الانتحال (المقياس الأمني الحرج)."""
    src = CFG.CLASSES.index(CFG.TARGET_SRC)
    present = sorted(set(np.asarray(y_true).tolist()))
    return {
        "balacc": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", labels=present, zero_division=0)),
        "spoof_recall": float(recall_score(y_true, y_pred, labels=[src], average="macro", zero_division=0)),
    }


CHANCE = 1.0 / NC
