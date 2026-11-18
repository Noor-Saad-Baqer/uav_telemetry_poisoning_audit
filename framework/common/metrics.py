# -*- coding: utf-8 -*-
# =====================================================================
#  المقاييس المشتركة — الدقّة المتوازنة ، macro-F1 ، الاستدعاء لكل فئة ، ECE ، Brier
# =====================================================================
import numpy as np
from sklearn.metrics import balanced_accuracy_score, f1_score, accuracy_score, recall_score


def basic(y_true, y_pred, n_classes=None):
    out = dict(balacc=float(balanced_accuracy_score(y_true, y_pred)),
               macro_f1=float(f1_score(y_true, y_pred, average="macro")),
               accuracy=float(accuracy_score(y_true, y_pred)))
    labels = list(range(n_classes)) if n_classes else sorted(set(np.asarray(y_true).tolist()))
    rec = recall_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    for c, r in zip(labels, rec):
        out[f"recall_{c}"] = float(r)
    out["min_class_recall"] = float(np.min(rec)) if len(rec) else float("nan")
    return out


def chance(n_classes):
    return 1.0 / n_classes


def ece_and_brier(proba, y_true, n_classes, bins=15):
    """ECE على ثقة أعلى احتمال (حزم متساوية العرض) + Brier متعدّد الفئات.
    نسخة مطابقة لتعريف المرحلة الخامسة حتى تبقى الأرقام قابلة للمقارنة."""
    proba = np.asarray(proba, float)
    y_true = np.asarray(y_true)
    conf = proba.max(axis=1)
    pred = proba.argmax(axis=1)
    correct = (pred == y_true).astype(float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    n = len(conf)
    for i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
        m = (conf >= lo) & (conf <= hi) if i == 0 else (conf > lo) & (conf <= hi)
        if m.sum() == 0:
            continue
        ece += (m.sum() / n) * abs(correct[m].mean() - conf[m].mean())
    onehot = np.zeros_like(proba)
    onehot[np.arange(len(y_true)), y_true] = 1.0
    brier = float(((proba - onehot) ** 2).sum(axis=1).mean())
    return float(ece), brier


def proba_full(model, X, classes_seen, n_classes):
    """يُعيد مصفوفة احتمالات بعرض n_classes حتى لو غابت فئة عن التدريب."""
    p = model.predict_proba(X)
    if p.shape[1] == n_classes:
        return p
    full = np.zeros((len(X), n_classes), float)
    for j, c in enumerate(classes_seen):
        full[:, int(c)] = p[:, j]
    return full
