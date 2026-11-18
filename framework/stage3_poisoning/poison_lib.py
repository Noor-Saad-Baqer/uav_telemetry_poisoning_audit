# -*- coding: utf-8 -*-
# =====================================================================
#  مكتبة التسميم المشتركة للإطار — framework/stage3_poisoning/poison_lib.py
# ---------------------------------------------------------------------
#  نسخة مستقلّة من هجمات/كواشف/دفاعات step3 ، مُعمَّمة على أيّ عدد فئات
#  وأيّ زوج (فئة مصدر ← فئة هدف). لا تستورد step3 إطلاقاً: ملفّاته تستورد
#  `config` مجرَّداً ، والتصادم مع step4/config.py يُنتج بيانات خاطئة بصمت
#  (انظري README §2). المعاملات مطابقة لـ step3/config.py حرفياً:
#     TRIGGER_DIM = 8 ، TRIGGER_MAG = 3.0 σ ، FILTER_Q = 0.25 ، RATE_BASIS = فئة المصدر
#     الضحيّة = غابة عشوائية 200 شجرة ، الخسارة من خارج الحقيبة (OOB)
# =====================================================================
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score, f1_score, recall_score, roc_auc_score

TRIGGER_DIM = 8
TRIGGER_MAG = 3.0
FILTER_Q = 0.25
CLEAN_LABEL_EPS = 0.7


# ---------------------------------------------------------------------
#  الهجمات — كلّها تُرجع (Xp, yp, is_poison)
# ---------------------------------------------------------------------
def _n(n, rate):
    return int(round(rate * n))


def random_flip(X, y, rate, rng, n_classes, **_):
    Xp, yp = X.copy(), y.copy()
    mask = np.zeros(len(y), bool)
    k = _n(len(y), rate)
    if k == 0:
        return Xp, yp, mask
    idx = rng.permutation(len(y))[:k]
    for i in idx:
        others = [c for c in range(n_classes) if c != y[i]]
        yp[i] = others[rng.randint(len(others))]
    mask[idx] = True
    return Xp, yp, mask


def targeted_flip(X, y, rate, rng, src, dst, **_):
    """يعيد وسم عيّنات فئة المصدر (src) كفئة الهدف (dst). النسبة من فئة المصدر."""
    Xp, yp = X.copy(), y.copy()
    mask = np.zeros(len(y), bool)
    s = np.where(y == src)[0]
    k = min(_n(len(s), rate), len(s))
    if k == 0:
        return Xp, yp, mask
    idx = rng.permutation(s)[:k]
    yp[idx] = dst
    mask[idx] = True
    return Xp, yp, mask


def apply_trigger(X):
    Xt = X.copy()
    Xt[:, :min(TRIGGER_DIM, X.shape[1])] = TRIGGER_MAG
    return Xt


def backdoor(X, y, rate, rng, dst, **_):
    """يزرع المُحفِّز في عيّنات غير الهدف ويعيد وسمها dst. النسبة من كامل التدريب."""
    Xp, yp = X.copy(), y.copy()
    mask = np.zeros(len(y), bool)
    k = _n(len(y), rate)
    cand = np.where(y != dst)[0]
    k = min(k, len(cand))
    if k == 0:
        return Xp, yp, mask
    idx = rng.permutation(cand)[:k]
    Xp[idx] = apply_trigger(Xp[idx])
    yp[idx] = dst
    mask[idx] = True
    return Xp, yp, mask


def clean_label(X, y, rate, rng, src, dst, eps=CLEAN_LABEL_EPS, **_):
    """يحرّك ميزات فئة المصدر نحو مركز فئة الهدف والوسوم سليمة."""
    Xp, yp = X.copy(), y.copy()
    mask = np.zeros(len(y), bool)
    s, d = np.where(y == src)[0], np.where(y == dst)[0]
    k = min(_n(len(s), rate), len(s))
    if k == 0 or len(d) == 0:
        return Xp, yp, mask
    cen = X[d].mean(0)
    idx = rng.permutation(s)[:k]
    Xp[idx] = (1 - eps) * X[idx] + eps * cen
    mask[idx] = True
    return Xp, yp, mask


ATTACKS = {"random_flip": random_flip, "targeted_flip": targeted_flip,
           "backdoor": backdoor, "clean_label": clean_label}


def poison(name, X, y, rate, seed, n_classes, src, dst):
    rng = np.random.RandomState(seed)
    return ATTACKS[name](X, y, rate, rng, n_classes=n_classes, src=src, dst=dst)


def backdoor_success(predict_fn, Xte, yte, dst):
    """نسبة عيّنات الاختبار غير الطبيعية التي تُصنَّف dst بعد زرع المُحفِّز."""
    m = yte != dst
    if m.sum() == 0:
        return float("nan")
    return float((predict_fn(apply_trigger(Xte[m])) == dst).mean())


# ---------------------------------------------------------------------
#  الضحيّة — غابة عشوائية بخسارة OOB (كما في step3/RFVictim)
# ---------------------------------------------------------------------
class RFVictim:
    def __init__(self, n_classes, seed=0, n_estimators=200, n_jobs=2):
        self.nc = n_classes
        self.rf = RandomForestClassifier(n_estimators=n_estimators, n_jobs=n_jobs,
                                         random_state=seed, bootstrap=True, oob_score=True)

    def fit(self, X, y):
        self.rf.fit(X, y)
        self._Xtr, self._classes = X, self.rf.classes_
        self._oob = self.rf.oob_decision_function_
        return self

    def _full(self, p):
        out = np.zeros((len(p), self.nc))
        for j, c in enumerate(self._classes):
            out[:, int(c)] = p[:, j]
        return out

    def proba(self, X):
        if X.shape == self._Xtr.shape and np.array_equal(X, self._Xtr):
            p = np.nan_to_num(self._oob, nan=1.0 / max(1, len(self._classes)))
        else:
            p = self.rf.predict_proba(X)
        return self._full(p)

    def sample_loss(self, X, y):
        p = self.proba(X)
        return -np.log(p[np.arange(len(y)), y] + 1e-6)

    def predict(self, X):
        return self.rf.predict(X)


def loss_deviation(model, X, y, mad_floor=0.0):
    """كاشف step3 المقترح: |الانحراف عن الوسيط|/MAD داخل كلّ فئة (خسارة OOB).

    ⚠️ اكتُشف 2026-09-05 على مسارات الشبكة المشبعة: الوسيط و MAD يساويان صفراً
    تماماً في الفئات السهلة (خسارة OOB = 0 لأغلب العيّنات) ⇒ القسمة على 1e-9
    تجعل كلّ عيّنة ذات خسارة غير صفرية «شاذّة» ، وعتبة الحصّة q تصير صفراً ،
    فيحذف المرشّح الفئة كلّها إلا حارس العيّنتين. لم يظهر هذا في التليمتري لأن
    خسائرها لا تنعدم. `mad_floor` = أرضية MAD (النسخة v2 تستعمل 1e-3).
    """
    L = model.sample_loss(X, y)
    s = np.zeros(len(y))
    for c in np.unique(y):
        m = np.where(y == c)[0]
        if len(m) < 3:
            continue
        med = np.median(L[m]); mad = max(np.median(np.abs(L[m] - med)), mad_floor) + 1e-9
        s[m] = np.abs(L[m] - med) / mad
    return s


def detection_auc(scores, is_poison):
    """AUC موجَّه max(AUC, 1−AUC) كما في step3 ؛ nan إن لم يوجد سمّ."""
    if is_poison.sum() == 0 or (~is_poison).sum() == 0:
        return float("nan"), float("nan")
    a = roc_auc_score(is_poison.astype(int), scores)
    return float(a), float(max(a, 1 - a))


# ---------------------------------------------------------------------
#  الدفاعات — كلّ دفاع يُرجع نموذجاً مدرَّباً
# ---------------------------------------------------------------------
def _trusted_mask(scores, y, q=FILTER_Q, inclusive=False):
    thr = np.quantile(scores, 1.0 - q)
    keep = scores <= thr if inclusive else scores < thr      # step3 الأصلي: < (يفشل حين thr = 0)
    for c in np.unique(y):                       # حارس: لا تُحذف فئة بالكامل
        if keep[y == c].sum() < 2:
            ic = np.where(y == c)[0]
            keep[ic[np.argsort(scores[ic])[:2]]] = True
    return keep


def defend(name, Xp, yp, n_classes, seed, base_model=None):
    if base_model is None:
        base_model = RFVictim(n_classes, seed).fit(Xp, yp)
    if name == "none":
        return base_model
    v2 = name.endswith("_v2")                        # النسخة المصحَّحة: أرضية MAD + عتبة شاملة
    s = loss_deviation(base_model, Xp, yp, mad_floor=1e-3 if v2 else 0.0)
    keep = _trusted_mask(s, yp, inclusive=v2)
    name = name[:-3] if v2 else name
    if name == "filtered_retrain":
        return RFVictim(n_classes, seed).fit(Xp[keep], yp[keep])
    if name == "relabel_retrain":
        if keep.all():
            return base_model
        m1 = RFVictim(n_classes, seed).fit(Xp[keep], yp[keep])
        y_new = yp.copy()
        y_new[~keep] = m1.predict(Xp[~keep])
        return RFVictim(n_classes, seed).fit(Xp, y_new)
    raise ValueError(name)


# ---------------------------------------------------------------------
#  المقاييس
# ---------------------------------------------------------------------
def scores(y_true, y_pred, target_cls, n_classes):
    labels = list(range(n_classes))
    rec = recall_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    return dict(balacc=float(balanced_accuracy_score(y_true, y_pred)),
                macro_f1=float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
                target_recall=float(rec[target_cls]))


def clip01(v):
    return float(np.clip(v, 0, 1)) if np.isfinite(v) else float("nan")
