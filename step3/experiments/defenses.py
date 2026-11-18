# -*- coding: utf-8 -*-
# =====================================================================
#  الدفاعات ضدّ التسميم — Poisoning defenses
# ---------------------------------------------------------------------
#  كل دفاع يُرجع دالّة تنبّؤ مدرَّبة على بيانات التدريب المسمومة.
#
#   1) filtered_retrain — يحذف أعلى q من درجات الاشتباه ثم يعيد التدريب
#   2) robust_loss      — خسارة GCE متينة ضدّ ضوضاء التسميات (بلا حذف)
#   3) trimmed_agg      — تجميع مقصوص: يقسّم التدريب إلى شرائح ويستبعد
#                          الشرائح الشاذّة (نظير Trimmed-mean/Krum المركزي)
#   لا دفاع (none)      — خطّ الأساس المسموم بلا معالجة
# =====================================================================
import numpy as np
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import config as CFG
import poison_core as P
import detectors as D


# ---------------------------------------------------------------------
def no_defense(Xp, yp, seed=0, **kw):
    model, _ = P.train_victim(Xp, yp, seed=seed)
    return lambda Z: P.predict(model, Z)


# ---------------------------------------------------------------------
def filtered_retrain(Xp, yp, seed=0, detector=None, filter_q=None, lr=None, **kw):
    """يصفّي أعلى q من درجات الاشتباه ثم يعيد التدريب على الباقي."""
    q = CFG.FILTER_Q if filter_q is None else float(filter_q)
    det = detector or getattr(CFG, "FILTER_DETECTOR", "loss_deviation")
    s = D.detect(det, Xp, yp, seed=seed)
    thr = np.quantile(s, 1.0 - q)
    keep = s < thr
    #  حارس: لا تحذفي فئة بالكامل
    for c in np.unique(yp):
        if keep[yp == c].sum() < 2:
            keep[np.where(yp == c)[0][:2]] = True
    model, _ = P.train_victim(Xp[keep], yp[keep], seed=seed, lr=lr)
    return lambda Z: P.predict(model, Z)


# ---------------------------------------------------------------------
def robust_loss(Xp, yp, seed=0, gce_q=None, lr=None, **kw):
    """خسارة GCE: تقلّل أثر العيّنات ذات الخسارة العالية (المسمومة غالباً)."""
    model, _ = P.train_victim(Xp, yp, seed=seed, loss_fn="gce", gce_q=gce_q, lr=lr)
    return lambda Z: P.predict(model, Z)


# ---------------------------------------------------------------------
def trimmed_agg(Xp, yp, seed=0, trim_frac=None, lr=None, n_shards=5, **kw):
    """تجميع مقصوص: تدريب على شرائح ثم تصويت أغلبية بعد استبعاد الشرائح الشاذّة.

    الشريحة التي تخالف إجماع بقية الشرائح تُستبعَد — نظير مركزي لـ Trimmed-mean
    في التعلّم الموزّع، ويحدّ من أثر شريحة ملوّثة بالتسميم.
    """
    tf = CFG.TRIM_FRAC if trim_frac is None else float(trim_frac)
    rng = np.random.RandomState(seed)
    idx = rng.permutation(len(yp))
    shards = np.array_split(idx, n_shards)
    models = []
    for sh in shards:
        if len(np.unique(yp[sh])) < 2:
            continue
        m, _ = P.train_victim(Xp[sh], yp[sh], seed=seed, lr=lr)
        models.append(m)
    if not models:
        return no_defense(Xp, yp, seed=seed)

    def _predict(Z):
        votes = np.stack([P.predict(m, Z) for m in models])          # (n_shards, n)
        #  استبعاد الشرائح الأبعد عن الإجماع
        if len(models) > 2:
            maj = np.apply_along_axis(lambda v: np.bincount(v, minlength=CFG.NUM_CLASSES).argmax(), 0, votes)
            disagree = (votes != maj[None, :]).mean(1)
            ndrop = int(round(tf * len(models)))
            if 0 < ndrop < len(models) - 1:
                keep = np.argsort(disagree)[:len(models) - ndrop]
                votes = votes[keep]
        return np.apply_along_axis(
            lambda v: np.bincount(v, minlength=CFG.NUM_CLASSES).argmax(), 0, votes)

    return _predict


# ---------------------------------------------------------------------
def relabel_retrain(Xp, yp, seed=0, detector=None, filter_q=None, lr=None,
                    conf_thr=None, **kw):
    """إعادة وسم بدل الحذف — دفاع مقترح لمعالجة عيب التصفية.

    الدافع: التصفية تحذف نسبة ثابتة من العيّنات ، وفئة «انتحال GPS» أصغر الفئات
    (≈70 نافذة لكل طيّة) ⇒ الحذف يُهلِك الفئة التي يُفترض أن يحميها الدفاع.
    وقد قِسنا ذلك: كاشف أدقّ (AUC 0.87) أعطى استدعاءً 0.001 لأنه حذف أكثر.

    الفكرة: لا نحذف شيئاً. بل:
      1) نأخذ العيّنات الأقلّ اشتباهاً (الموثوقة) ونُدرّب عليها نموذجاً أنظف.
      2) نستعمله لإعادة وسم العيّنات المشبوهة (بدل إسقاطها) — وبشرط ثقة.
      3) نعيد التدريب على **كامل** البيانات بالتسميات المصحّحة.
    هكذا يبقى حجم كل فئة كما هو ، ويُصحَّح القلب الموجَّه بدل أن يُحذف.
    """
    q = CFG.FILTER_Q if filter_q is None else float(filter_q)
    det = detector or getattr(CFG, "FILTER_DETECTOR", "loss_deviation")
    ct = getattr(CFG, "RELABEL_CONF", 0.0) if conf_thr is None else float(conf_thr)

    s = D.detect(det, Xp, yp, seed=seed)
    thr = np.quantile(s, 1.0 - q)
    trusted = s < thr
    #  حارس: لا بدّ أن تبقى كل فئة ممثَّلة في المجموعة الموثوقة
    for c in np.unique(yp):
        if trusted[yp == c].sum() < 2:
            idx_c = np.where(yp == c)[0]
            trusted[idx_c[np.argsort(s[idx_c])[:2]]] = True
    if trusted.all():
        return no_defense(Xp, yp, seed=seed)

    m1, _ = P.train_victim(Xp[trusted], yp[trusted], seed=seed, lr=lr)   # (1) نموذج أنظف
    y_new = yp.copy()
    sus = ~trusted
    logits = m1.logits(np.ascontiguousarray(Xp[sus], dtype=np.float32))
    prob = m1._softmax(logits)
    pred = prob.argmax(1)
    if ct > 0:                       # لا نعيد الوسم إلا عند ثقة كافية
        conf = prob.max(1)
        take = conf >= ct
        idx_sus = np.where(sus)[0]
        y_new[idx_sus[take]] = pred[take]
    else:
        y_new[sus] = pred

    m2, _ = P.train_victim(Xp, y_new, seed=seed, lr=lr)                  # (2) كل البيانات
    return lambda Z: P.predict(m2, Z)


DEFENSES = {"none": no_defense, "filtered_retrain": filtered_retrain,
            "robust_loss": robust_loss, "trimmed_agg": trimmed_agg,
            "relabel_retrain": relabel_retrain}


def defend(name, Xp, yp, seed=0, **kw):
    return DEFENSES[name](Xp, yp, seed=seed, **kw)
