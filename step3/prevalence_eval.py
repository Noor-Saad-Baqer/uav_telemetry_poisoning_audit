# -*- coding: utf-8 -*-
# =====================================================================
#  التقييم عند الانتشار الواقعي — ما تُخفيه موازنة الفئات
# ---------------------------------------------------------------------
#  موازنة الفئات (NORMAL_RATIO) ترفع نسبة الانتحال من 0.98% إلى 14.7%.
#  وهذا لا يُحسّن الكشف: الاستدعاء متقارب في الحالتين. لكنه **يمحو مقياس
#  الكلفة**: الدقّة (precision) عند انتشار صناعي لا تدلّ على شيء ، فتختفي
#  كلفة الإنذار الكاذب الحقيقية من التقرير.
#
#  هنا نُقيّم على المجموعة **غير الموزونة** (الانتشار الحقيقي) ونُبلّغ:
#    • PR-AUC        — ملخّص لا يعتمد على عتبة (خطّ الصدفة = الانتشار)
#    • استدعاء/دقّة  — عند عتبة مضبوطة على **التدريب وحده** (بلا تسريب)
#    • إنذارات/ساعة  — الرقم التشغيلي الذي لا تُبلّغه أيّ ورقة على هذه المجموعة
#
#  ⚠️ ضبط العتبة يجري داخل الطيّة على بيانات التدريب فقط ؛ الاختبار لا يُمَسّ.
# =====================================================================
import os, sys, warnings, numpy as np, pandas as pd
warnings.filterwarnings("ignore")
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "experiments"))
import config as CFG
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (balanced_accuracy_score, recall_score,
                             precision_score, average_precision_score)

SEEDS = [0, 1, 2]
SP = CFG.CLASSES.index("GPS Spoofing")
#  خطوة النافذة بالثواني: شبكة الـ innovations ≈ 0.2 ث/عيّنة × STRIDE
SEC_PER_WINDOW = 0.2 * CFG.STRIDE


def load(normal_ratio):
    CFG.LABEL_MODE = "window"; CFG.EXCLUDE_LABEL_FEATURES = True
    CFG.NORMAL_RATIO = normal_ratio
    for m in list(sys.modules):
        if m == "poison_core":
            del sys.modules[m]
    import poison_core as P
    return P, P.build_dataset(verbose=False)


def tune_threshold(proba_tr, y_tr):
    """أعلى F1 لفئة الانتحال على التدريب — العتبة لا ترى بيانات الاختبار."""
    pos = (y_tr == SP)
    best = (-1.0, 0.5)
    for t in np.linspace(0.01, 0.95, 60):
        p = proba_tr > t
        tp = (p & pos).sum(); fp = (p & ~pos).sum(); fn = (~p & pos).sum()
        f1 = 2 * tp / max(1, 2 * tp + fp + fn)
        if f1 > best[0]:
            best = (f1, t)
    return best[1]


def evaluate(arm, normal_ratio, tune):
    P, (X, y, g, f, names) = load(normal_ratio)
    prev = float((y == SP).mean())
    rows = []
    for af, tr, te in P.loao_folds(g):
        if SP not in set(y[te]):
            continue
        for seed in SEEDS:
            sc = StandardScaler().fit(X[tr])
            m = RandomForestClassifier(n_estimators=200, n_jobs=6,
                                       random_state=seed).fit(sc.transform(X[tr]), y[tr])
            Pte = m.predict_proba(sc.transform(X[te]))
            pred = Pte.argmax(1)
            if tune:
                thr = tune_threshold(m.predict_proba(sc.transform(X[tr]))[:, SP], y[tr])
                pred = Pte.argmax(1); pred[Pte[:, SP] > thr] = SP
            pos = (y[te] == SP)
            fp = int(((pred == SP) & ~pos).sum())
            hours = len(te) * SEC_PER_WINDOW / 3600.0
            rows.append({"arm": arm, "fold": af, "seed": seed, "prevalence": prev,
                         "bal_acc": balanced_accuracy_score(y[te], pred),
                         "recall": recall_score(y[te], pred, labels=[SP],
                                                average="macro", zero_division=0),
                         "precision": precision_score(y[te], pred, labels=[SP],
                                                      average="macro", zero_division=0),
                         "pr_auc": average_precision_score(pos, Pte[:, SP]),
                         "false_alarms_per_hour": fp / max(1e-9, hours)})
    return rows


ARMS = [("موزونة صناعياً (المُبلَّغ حالياً)", 3.0, False),
        ("انتشار واقعي ، بلا معالجة",        None, False),
        ("انتشار واقعي + ضبط عتبة",          None, True)]

all_rows = []
for tag, nr, tune in ARMS:
    r = evaluate(tag, nr, tune)
    all_rows += r
    d = pd.DataFrame(r)
    #  المتوسّط داخل الطيّة عبر البذور ثم عبر الطيّات (وحدة الاستقلال = الطيّة)
    ff = d.groupby("fold")[["bal_acc", "recall", "precision", "pr_auc",
                            "false_alarms_per_hour"]].mean()
    print(f"{tag:34s} انتشار={d.prevalence.iloc[0]:.4f}  "
          f"BalAcc={ff.bal_acc.mean():.3f}  استدعاء={ff.recall.mean():.3f}  "
          f"دقّة={ff.precision.mean():.3f}  PR-AUC={ff.pr_auc.mean():.3f}  "
          f"إنذار كاذب/ساعة={ff.false_alarms_per_hour.mean():.1f}", flush=True)

df = pd.DataFrame(all_rows)
out = os.path.join(CFG.RESULTS_DIR, "p3_prevalence_eval.csv")
df.to_csv(out, index=False)

print("\n" + "=" * 88)
agg = (df.groupby(["arm", "fold"])[["recall", "precision", "pr_auc"]].mean()
         .groupby("arm").agg(["mean", "std"]).round(3))
print(agg.to_string())
print("=" * 88)
real = df[df.arm == "انتشار واقعي + ضبط عتبة"]
print(f"\nخطّ الصدفة لـ PR-AUC = الانتشار = {real.prevalence.iloc[0]:.4f} ⇒ "
      f"المُقاس {real.pr_auc.mean():.3f} أعلى منه بـ {real.pr_auc.mean()/real.prevalence.iloc[0]:.0f}×")
print(f"\n✓ {os.path.relpath(out)}")
