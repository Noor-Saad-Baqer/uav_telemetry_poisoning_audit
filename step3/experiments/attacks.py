# -*- coding: utf-8 -*-
# =====================================================================
#  هجمات تسميم النموذج — Model-poisoning attacks (training-time)
# ---------------------------------------------------------------------
#  كل هجوم يُطبَّق على مجموعة التدريب فقط ويُرجع:
#     Xp, yp, is_poison  (قناع منطقي للعيّنات المسمومة — لتقييم الكواشف)
#  «نسبة التسميم» = نسبة عيّنات التدريب التي غُيِّرت (تسميةً أو ميزةً).
#
#  الهجمات:
#   1) random_flip   — قلب تسميات عشوائي (تسميم غير موجَّه، خطّ أساس)
#   2) targeted_flip — قلب «انتحال GPS» → «طبيعي» (الأخطر أمنياً: يُخفي الهجوم)
#   3) backdoor      — نمط مُحفِّز في الميزات + إعادة وسم إلى «طبيعي»
#   4) clean_label   — تحريك ميزات فئة الانتحال نحو «طبيعي» بلا تغيير التسمية
# =====================================================================
import numpy as np
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import config as CFG

SRC = CFG.CLASSES.index(CFG.TARGET_SRC)     # الفئة المصدر (GPS Spoofing)
DST = CFG.CLASSES.index(CFG.TARGET_DST)     # الفئة الهدف (Normal)


def _n_poison(n, rate):
    return int(round(rate * n))


def _basis_n(y, src_idx):
    """قاعدة حساب نسبة التسميم للهجمات التي تسحب من فئة واحدة.

    ⚠️ خلل مُكتشَف ومُصحَّح: كانت النسبة تُحسب من **كامل مجموعة التدريب** بينما
    القلب يجري من **فئة الانتحال وحدها** (283 عيّنة من 1894). فعند نسبة 15%
    يُطلَب قلب 284 عيّنة ⇒ تُقلَب الفئة **بأكملها** ولا يبقى منها مثال صحيح واحد.
    وبذلك يصير استدعاء 0.000 مضموناً رياضياً لا نتيجةً عن فشل الدفاع ،
    و«عتبة الانهيار عند 15%» تصبح أثراً جانبياً للاستنفاد لا اكتشافاً.

    الصحيح: النسبة من **الفئة المستهدَفة** (وهو المتعارف عليه في أدبيات
    قلب التسميات الموجَّه) ⇒ rate=1.0 يعني قلب الفئة كاملة ، بلا التباس.
    """
    basis = getattr(CFG, "RATE_BASIS", "source_class")
    return len(src_idx) if basis == "source_class" else len(y)


def saturation(y, rate):
    """نسبة فئة الانتحال التي سيقلبها الهجوم (1.0 = استنفاد كامل)."""
    src_idx = np.where(np.asarray(y) == SRC)[0]
    if len(src_idx) == 0:
        return float("nan")
    k = min(_n_poison(_basis_n(y, src_idx), rate), len(src_idx))
    return k / len(src_idx)


# ---------------------------------------------------------------------
#  (1) قلب تسميات عشوائي — Random label flipping
# ---------------------------------------------------------------------
def random_flip(X, y, rate, rng):
    Xp, yp = X.copy(), y.copy()
    mask = np.zeros(len(y), dtype=bool)
    k = _n_poison(len(y), rate)
    if k == 0:
        return Xp, yp, mask
    idx = rng.permutation(len(y))[:k]
    for i in idx:
        others = [c for c in range(CFG.NUM_CLASSES) if c != y[i]]
        yp[i] = others[rng.randint(len(others))]
    mask[idx] = True
    return Xp, yp, mask


# ---------------------------------------------------------------------
#  (2) قلب موجَّه — Targeted flip (Spoofing → Normal)
# ---------------------------------------------------------------------
def targeted_flip(X, y, rate, rng):
    """يُخفي الانتحال: يعيد وسم عيّنات الانتحال كـ«طبيعي».

    هذا أخطر الهجمات عملياً — النموذج يتعلّم تجاهل الانتحال تحديداً،
    فتنهار استعادة فئة الانتحال بينما تبقى الدقّة الكلّية مرتفعة ظاهرياً.
    """
    Xp, yp = X.copy(), y.copy()
    mask = np.zeros(len(y), dtype=bool)
    src_idx = np.where(y == SRC)[0]
    k = min(_n_poison(_basis_n(y, src_idx), rate), len(src_idx))
    if k == 0:
        return Xp, yp, mask
    idx = rng.permutation(src_idx)[:k]
    yp[idx] = DST
    mask[idx] = True
    return Xp, yp, mask


# ---------------------------------------------------------------------
#  (3) باب خلفي — Backdoor trigger
# ---------------------------------------------------------------------
def _trigger_dims(d, rng=None):
    """أبعاد المُحفِّز ثابتة (أوّل TRIGGER_DIM ميزة) ليكون المُحفِّز نفسه وقت الاختبار."""
    return np.arange(min(CFG.TRIGGER_DIM, d))


def apply_trigger(X):
    """يزرع النمط المُحفِّز في الميزات (يُستعمل تدريباً واختباراً)."""
    Xt = X.copy()
    dims = _trigger_dims(X.shape[1])
    Xt[:, dims] = CFG.TRIGGER_MAG          # الميزات مُطبَّعة ⇒ القيمة بوحدات انحراف معياري
    return Xt


def backdoor(X, y, rate, rng):
    """يزرع مُحفِّزاً في عيّنات ويعيد وسمها «طبيعي» ⇒ باب خلفي يُفعَّل وقت الاستدلال."""
    Xp, yp = X.copy(), y.copy()
    mask = np.zeros(len(y), dtype=bool)
    k = _n_poison(len(y), rate)
    if k == 0:
        return Xp, yp, mask
    #  نختار من الفئات غير الهدف حتى يرتبط المُحفِّز بالتسمية «طبيعي»
    cand = np.where(y != DST)[0]
    k = min(k, len(cand))
    idx = rng.permutation(cand)[:k]
    Xp[idx] = apply_trigger(Xp[idx])
    yp[idx] = DST
    mask[idx] = True
    return Xp, yp, mask


# ---------------------------------------------------------------------
#  (4) تسميم نظيف التسمية — Clean-label
# ---------------------------------------------------------------------
def clean_label(X, y, rate, rng, eps=0.7):
    """يحرّك ميزات عيّنات الانتحال نحو مركز «طبيعي» مع إبقاء تسميتها صحيحة.

    التسميات كلّها سليمة ⇒ الفحص البشري لا يكشف شيئاً ، لكن حدود القرار تتشوّه.
    """
    Xp, yp = X.copy(), y.copy()
    mask = np.zeros(len(y), dtype=bool)
    src_idx = np.where(y == SRC)[0]
    dst_idx = np.where(y == DST)[0]
    k = min(_n_poison(_basis_n(y, src_idx), rate), len(src_idx))
    if k == 0 or len(dst_idx) == 0:
        return Xp, yp, mask
    centroid = X[dst_idx].mean(0)
    idx = rng.permutation(src_idx)[:k]
    Xp[idx] = (1 - eps) * X[idx] + eps * centroid
    mask[idx] = True
    return Xp, yp, mask


ATTACKS = {"random_flip": random_flip, "targeted_flip": targeted_flip,
           "backdoor": backdoor, "clean_label": clean_label}


def poison(name, X, y, rate, seed=0):
    """واجهة موحّدة: يُرجع (X مسموم, y مسموم, قناع العيّنات المسمومة)."""
    rng = np.random.RandomState(seed)
    return ATTACKS[name](X, y, rate, rng)


# ---------------------------------------------------------------------
#  معدّل نجاح الباب الخلفي — Attack Success Rate
# ---------------------------------------------------------------------
def backdoor_asr(predict_fn, Xte, yte):
    """نسبة عيّنات الاختبار (غير الطبيعية) التي يصنّفها النموذج «طبيعي» بعد زرع المُحفِّز."""
    m = yte != DST
    if m.sum() == 0:
        return float("nan")
    return float((predict_fn(apply_trigger(Xte[m])) == DST).mean())
