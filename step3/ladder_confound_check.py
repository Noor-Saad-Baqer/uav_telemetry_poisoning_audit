# -*- coding: utf-8 -*-
# =====================================================================
#  فحص التباس في السُّلّم: هل الدرجة 3→4 تغيّر **عاملاً واحداً** فعلاً؟
# ---------------------------------------------------------------------
#  السُّلّم يدّعي «عامل واحد لكل درجة». لكن الانتقال 3→4 (وهو الانتقال الذي
#  تقوم عليه النتيجة الرئيسية: −0.329 و 0.177→1.000) يُغيّر ثلاثة أشياء معاً:
#      (أ) دقّة الوسم        : المجلّد → النافذة      ← العامل المدّعى
#      (ب) حجم المجموعة      : 42397 → 2402          ← أثر جانبي
#      (ج) توازن الفئات      : انتحال 33.2% → 14.7%   ← NORMAL_RATIO
#  فإن كان معظم القفزة آتياً من (ب)/(ج) فالادّعاء ينهار.
#
#  نفصلها بإدخال درجة وسيطة 3a = وسم النافذة **بلا موازنة ولا اقتطاع**:
#      3  (وسم مجلّد)  →  3a (وسم نافذة، بلا موازنة)  →  4 (+ موازنة)
#  فيكون أثر الوسم الصافي = 3→3a ، وأثر الموازنة = 3a→4.
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
from sklearn.metrics import balanced_accuracy_score, accuracy_score, recall_score

SEEDS = [0, 1, 2]
SPOOF = CFG.CLASSES.index("GPS Spoofing")


def build(label_mode, exclude_leak, normal_ratio):
    CFG.LABEL_MODE = label_mode
    CFG.EXCLUDE_LABEL_FEATURES = exclude_leak
    CFG.NORMAL_RATIO = normal_ratio            # None ⇒ بلا موازنة
    for m in list(sys.modules):
        if m == "poison_core":
            del sys.modules[m]
    import poison_core as P
    return P, P.build_dataset(verbose=False)


def evaluate(P, X, y, groups, seed):
    """LOAO فقط — كل الدرجات هنا تستعمله ، فيبقى التقسيم ثابتاً."""
    ba, ac, sr = [], [], []
    for _, tr, te in P.loao_folds(groups):
        if len(te) == 0 or len(np.unique(y[tr])) < 2:
            continue
        sc = StandardScaler().fit(X[tr])
        m = RandomForestClassifier(n_estimators=200, n_jobs=4, random_state=seed)
        m.fit(sc.transform(X[tr]), y[tr])
        p = m.predict(sc.transform(X[te]))
        ba.append(balanced_accuracy_score(y[te], p))
        ac.append(accuracy_score(y[te], p))
        if SPOOF in set(y[te]):
            sr.append(recall_score(y[te], p, labels=[SPOOF], average="macro", zero_division=0))
    return np.array(ba), np.array(ac), np.array(sr)


ARMS = [
    ("3  وسم المجلّد (كما في السُّلّم)",        "flight", False, 3.0),
    ("3a وسم النافذة ، بلا موازنة",            "window", False, None),
    ("4  وسم النافذة + موازنة (السُّلّم)",      "window", False, 3.0),
    ("5a وسم النافذة + حذف التسريب ، بلا موازنة", "window", True,  None),
    ("5  وسم النافذة + حذف التسريب + موازنة",   "window", True,  3.0),
]

rows = []
for name, lm, ex, nr in ARMS:
    P, (X, y, g, f, names) = build(lm, ex, nr)
    B, A, S = [], [], []
    for s in SEEDS:
        ba, ac, sr = evaluate(P, X, y, g, s)
        B.append(ba); A.append(ac); S.append(sr)
    #  المتوسّط داخل الطيّة عبر البذور ثم عبر الطيّات (وحدة الاستقلال = الطيّة)
    ba_f = np.mean(B, axis=0); ac_f = np.mean(A, axis=0); sr_f = np.mean(S, axis=0)
    rows.append({"arm": name, "n": len(y), "feats": X.shape[1],
                 "spoof_frac": round(float((y == SPOOF).mean()), 3),
                 "raw_acc": round(ac_f.mean(), 3),
                 "bal_acc": round(ba_f.mean(), 3),
                 "bal_sd_fold": round(ba_f.std(ddof=1), 3),
                 "spoof_recall": round(sr_f.mean(), 3),
                 "spoof_sd_fold": round(sr_f.std(ddof=1), 3)})
    r = rows[-1]
    print(f"{name:42s} n={r['n']:6d} spoof={r['spoof_frac']:.3f}  "
          f"BalAcc={r['bal_acc']:.3f}±{r['bal_sd_fold']:.3f}  "
          f"spoof_rec={r['spoof_recall']:.3f}±{r['spoof_sd_fold']:.3f}", flush=True)

df = pd.DataFrame(rows)
out = os.path.join(CFG.RESULTS_DIR, "p3_ladder_confound.csv")
df.to_csv(out, index=False)

g = {r["arm"][:2].strip(): r for r in rows}
print("\n" + "=" * 78)
print("تفكيك الأثر المنسوب إلى «دقّة الوسم» (−0.329 في السُّلّم):")
print(f"  3 → 3a  أثر الوسم وحده        BalAcc {g['3']['bal_acc']:.3f} → {g['3a']['bal_acc']:.3f}"
      f"   ({g['3a']['bal_acc']-g['3']['bal_acc']:+.3f})")
print(f"  3a → 4  أثر الموازنة/الاقتطاع  BalAcc {g['3a']['bal_acc']:.3f} → {g['4']['bal_acc']:.3f}"
      f"   ({g['4']['bal_acc']-g['3a']['bal_acc']:+.3f})")
print(f"  المجموع 3 → 4                  ({g['4']['bal_acc']-g['3']['bal_acc']:+.3f})")
print()
print(f"  استدعاء الانتحال 3 → 3a        {g['3']['spoof_recall']:.3f} → {g['3a']['spoof_recall']:.3f}")
print(f"  استدعاء الانتحال 3a → 4        {g['3a']['spoof_recall']:.3f} → {g['4']['spoof_recall']:.3f}")
print(f"\n  خطّ الأساس بلا موازنة (5a)     BalAcc {g['5a']['bal_acc']:.3f}"
      f"  مقابل 5 المُبلَّغ {g['5']['bal_acc']:.3f}")
print("=" * 78)
print(f"\n✓ {os.path.relpath(out)}")
