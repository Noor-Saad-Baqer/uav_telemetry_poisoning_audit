# -*- coding: utf-8 -*-
# =====================================================================
#  الإحصاء المشترك — فواصل ثقة ، حجم الأثر ، تصحيح الاختبارات المتعدّدة ، الترتيب
# =====================================================================
import numpy as np
from scipy import stats as _st


def boot_ci(v, n=4000, seed=0, alpha=0.05):
    """فاصل ثقة bootstrap (percentile) للمتوسّط. يُعيد (mean, lo, hi)."""
    v = np.asarray(v, dtype=float).ravel()
    v = v[np.isfinite(v)]
    if len(v) == 0:
        return float("nan"), float("nan"), float("nan")
    if len(v) == 1:
        return float(v[0]), float(v[0]), float(v[0])
    rng = np.random.RandomState(seed)
    idx = rng.randint(0, len(v), size=(n, len(v)))
    bs = v[idx].mean(axis=1)
    return (float(v.mean()), float(np.percentile(bs, 100 * alpha / 2)),
            float(np.percentile(bs, 100 * (1 - alpha / 2))))


def cliffs_delta(a, b):
    """حجم أثر لا-معلمي في [-1, 1]: موجب ⇒ a أكبر عادةً من b."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) == 0 or len(b) == 0:
        return float("nan")
    gt = (a[:, None] > b[None, :]).sum()
    lt = (a[:, None] < b[None, :]).sum()
    return float((gt - lt) / (len(a) * len(b)))


def cliffs_label(d):
    if not np.isfinite(d):
        return "nan"
    d = abs(d)
    return "negligible" if d < 0.147 else "small" if d < 0.33 else "medium" if d < 0.474 else "large"


def holm(pvals):
    """تصحيح Holm–Bonferroni. يُعيد القيم المصحَّحة بترتيب الإدخال نفسه."""
    p = np.asarray(pvals, float)
    m = len(p)
    if m == 0:
        return p
    order = np.argsort(p)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        val = (m - rank) * p[i]
        running = max(running, val)
        adj[i] = min(1.0, running)
    return adj


def welch_t(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 2 or len(b) < 2:
        return float("nan"), float("nan")
    t = _st.ttest_ind(a, b, equal_var=False)
    return float(t.statistic), float(t.pvalue)


def t_vs_value(a, value):
    a = np.asarray(a, float)
    if len(a) < 2 or np.allclose(a, a[0]):
        return float("nan"), float("nan")
    t = _st.ttest_1samp(a, value)
    return float(t.statistic), float(t.pvalue)


def wilcoxon_paired(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    try:
        w = _st.wilcoxon(a, b)
        return float(w.statistic), float(w.pvalue)
    except Exception:
        return float("nan"), float("nan")


def kendall(a, b):
    r = _st.kendalltau(a, b)
    return float(r.statistic), float(r.pvalue)


def spearman(a, b):
    r = _st.spearmanr(a, b)
    return float(r.statistic), float(r.pvalue)


def rank_desc(v):
    """رتبة 1 = الأعلى."""
    return _st.rankdata(-np.asarray(v, float))
