# -*- coding: utf-8 -*-
# =====================================================================
#  كواشف التسميم — Poisoning detectors
# ---------------------------------------------------------------------
#  كل كاشف يُرجع «درجة اشتباه» لكل عيّنة تدريب (الأعلى = أرجح أنها مسمومة).
#  التقييم: ROC-AUC بين الدرجات وقناع التسميم الحقيقي.
#
#   1) loss_based        — العيّنات المسمومة تُبقي خسارة عالية (تُحفَظ لا تُعمَّم)
#   2) spectral          — التوقيع الطيفي: إسقاط على المتّجه الذاتي الأول للتمثيلات
#   3) activation_cluster— تجميع التمثيلات داخل كل فئة إلى عنقودين (العنقود الأصغر مشبوه)
#   4) knn_consistency   — اتّساق التسمية مع جيرانها الأقرب في فضاء الميزات
# =====================================================================
import numpy as np
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import config as CFG
import poison_core as P
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import roc_auc_score


# ---------------------------------------------------------------------
def _losses(X, y, model=None, seed=0):
    if model is None:
        model, avg = P.train_victim(X, y, seed=seed, track_losses=True)
        if avg is not None:
            return avg
    return P.per_sample_loss(model, X, y)


def loss_based(X, y, model=None, seed=0):
    """الكاشف الكلاسيكي: «خسارة أعلى ⇒ أرجح أنها مسمومة».

    ⚠️ نُبقيه كما هو عمداً لأنه يفشل فشلاً معكوساً أمام الهجمات الموجَّهة
    والباب الخلفي: تلك العيّنات متّسقة وسهلة الحفظ ⇒ خسارتها *أدنى* من
    النظيفة (AUC ≈ 0.04) ، فالتصفية بهذا المعيار تحذف العيّنات النظيفة.
    """
    return _losses(X, y, model=model, seed=seed)


def loss_deviation(X, y, model=None, seed=0):
    """المعالجة المقترحة: انحراف ثنائي الاتجاه عن الوسيط (بمقياس MAD).

    العيّنة المسمومة شاذّة في أيّ اتّجاه — إمّا يستعصي حفظها (قلب عشوائي)
    أو يُحفَظ بسهولة مفرطة (قلب موجَّه / باب خلفي). القيمة المطلقة للانحراف
    تلتقط الحالتين بلا معرفة مسبقة باتّجاه الهجوم (غير مُشرَفة بالكامل).
    """
    L = _losses(X, y, model=model, seed=seed)
    score = np.zeros(len(y), dtype=float)
    for c in np.unique(y):                      # التطبيع داخل كل فئة
        m = np.where(y == c)[0]
        if len(m) < 3:
            continue
        med = np.median(L[m])
        mad = np.median(np.abs(L[m] - med)) + 1e-9
        score[m] = np.abs(L[m] - med) / mad
    return score


# ---------------------------------------------------------------------
def spectral(X, y, model=None, seed=0):
    """التوقيع الطيفي (Tran et al.): داخل كل فئة، الإسقاط على المتّجه الذاتي الأول."""
    Z = _repr(X, y, model, seed)
    score = np.zeros(len(y))
    for c in np.unique(y):
        m = np.where(y == c)[0]
        if len(m) < 3:
            continue
        R = Z[m] - Z[m].mean(0)
        #  المتّجه الأيمن الأول من SVD = اتّجاه أكبر تباين داخل الفئة
        try:
            _, _, Vt = np.linalg.svd(R, full_matrices=False)
        except np.linalg.LinAlgError:
            continue
        score[m] = np.abs(R @ Vt[0])
    return score


# ---------------------------------------------------------------------
def activation_cluster(X, y, model=None, seed=0):
    """تجميع التمثيلات (Chen et al.): داخل كل فئة عنقودان، العنقود الأصغر مشبوه."""
    Z = _repr(X, y, model, seed)
    score = np.zeros(len(y))
    for c in np.unique(y):
        m = np.where(y == c)[0]
        if len(m) < 4:
            continue
        km = KMeans(n_clusters=2, n_init=10, random_state=seed).fit(Z[m])
        lab = km.labels_
        small = 0 if (lab == 0).sum() <= (lab == 1).sum() else 1
        #  الدرجة = 1 للعنقود الأصغر (المشبوه)، مع تدرّج حسب البعد عن مركز الفئة
        d = np.linalg.norm(Z[m] - Z[m].mean(0), axis=1)
        d = d / (d.max() + 1e-9)
        score[m] = (lab == small).astype(float) + 0.5 * d
    return score


# ---------------------------------------------------------------------
def knn_consistency(X, y, model=None, seed=0):
    """درجة = نسبة الجيران الأقرب الذين تسميتهم تخالف تسمية العيّنة."""
    k = min(CFG.KNN_K, max(2, len(y) - 1))
    nn = NearestNeighbors(n_neighbors=k + 1).fit(X)
    _, idx = nn.kneighbors(X)
    idx = idx[:, 1:]                       # استبعاد العيّنة نفسها
    return (y[idx] != y[:, None]).mean(1)


# ---------------------------------------------------------------------
def _repr(X, y, model, seed):
    """التمثيلات الكامنة من النموذج الضحيّة (تُدرَّب مرّة إن لم تُمرَّر)."""
    if model is None:
        model, _ = P.train_victim(X, y, seed=seed)
    return P.latent_feats(model, X)


DETECTORS = {"loss_based": loss_based, "loss_deviation": loss_deviation, "spectral": spectral,
             "activation_cluster": activation_cluster, "knn_consistency": knn_consistency}


def detect(name, X, y, model=None, seed=0):
    return DETECTORS[name](X, y, model=model, seed=seed)


def detection_auc(scores, is_poison):
    """ROC-AUC للكشف. يُرجع NaN إذا لم توجد عيّنات مسمومة (نسبة 0%)."""
    if is_poison.sum() == 0 or is_poison.all():
        return float("nan")
    return float(roc_auc_score(is_poison.astype(int), scores))
