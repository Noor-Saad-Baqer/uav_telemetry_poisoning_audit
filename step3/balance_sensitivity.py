# -*- coding: utf-8 -*-
# =====================================================================
#  حساسية خطّ الأساس لخيار الموازنة (جدول 4.3) — مولِّد صريح
# ---------------------------------------------------------------------
#  كان `p3_balance_sensitivity.csv` **بلا سكربت مولِّد في المشروع**: أُنتج
#  مرّةً ثم بقي في مكانه ، فلم يُعَد توليده بعد تصحيح قناع الوسم (‏envelope)
#  ولا بعد تبديل النموذج الضحيّة. النتيجة أنه ظلّ يحمل أرقام التشغيل القديم
#  (‏n = 2402 ، استدعاء انتحال 1.000) بينما تغيّرت البيانات تحته.
#
#  هذا الملفّ يعيد بناء الجدول من المِحراث الحالي نفسه المستعمَل في
#  `ladder_confound_check.py` (الذراعان 5a و5) ، ويضيف **استدعاء كل فئة**
#  الذي يحتاجه جدول 4.3 ولا يُخرجه فحص الالتباس.
#
#  البروتوكول: LOAO ، غابة عشوائية ، وسم النافذة ، حذف تسريب الوسم
#  (70 ميزة) — أي بروتوكول الدرجة 5 كاملاً ؛ الفرق الوحيد هو الموازنة.
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
CLASSES = list(CFG.CLASSES)
SPOOF = CLASSES.index("GPS Spoofing")


def build(normal_ratio):
    """يبني مجموعة الدرجة 5 ؛ normal_ratio=None ⇒ بلا موازنة."""
    CFG.LABEL_MODE = "window"
    CFG.EXCLUDE_LABEL_FEATURES = True
    CFG.NORMAL_RATIO = normal_ratio
    for m in list(sys.modules):
        if m == "poison_core":
            del sys.modules[m]
    import poison_core as P
    return P, P.build_dataset(verbose=False)


def evaluate(P, X, y, groups, seed):
    ba, ac = [], []
    rec = {c: [] for c in range(len(CLASSES))}
    for _, tr, te in P.loao_folds(groups):
        if len(te) == 0 or len(np.unique(y[tr])) < 2:
            continue
        sc = StandardScaler().fit(X[tr])
        m = RandomForestClassifier(n_estimators=200, n_jobs=4, random_state=seed)
        m.fit(sc.transform(X[tr]), y[tr])
        p = m.predict(sc.transform(X[te]))
        ba.append(balanced_accuracy_score(y[te], p))
        ac.append(accuracy_score(y[te], p))
        present = set(y[te])
        for c in range(len(CLASSES)):
            #  الفئة الغائبة عن الطيّة تُترك NaN ولا تُحسب صفراً — وإلّا
            #  انخفض متوسّط الاستدعاء لسبب لا علاقة له بالأداء.
            rec[c].append(recall_score(y[te], p, labels=[c], average="macro",
                                       zero_division=0) if c in present else np.nan)
    return np.array(ba), np.array(ac), {c: np.array(v, dtype=float) for c, v in rec.items()}


rows = []
for label, nr in [("balanced", CFG.NORMAL_RATIO if CFG.NORMAL_RATIO else 3.0),
                  ("unbalanced", None)]:
    P, (X, y, g, f, names) = build(nr)
    B, A, R = [], [], {c: [] for c in range(len(CLASSES))}
    for s in SEEDS:
        ba, ac, rec = evaluate(P, X, y, g, s)
        B.append(ba); A.append(ac)
        for c in rec: R[c].append(rec[c])
    #  وحدة الاستقلال = الطيّة: نُوسِّط البذور داخل الطيّة أوّلاً
    ba_f = np.mean(B, axis=0); ac_f = np.mean(A, axis=0)
    row = {"setting": label, "n": int(len(y)), "n_folds": int(len(ba_f)),
           "spoof_frac": round(float((y == SPOOF).mean()), 4),
           "raw_acc": round(float(ac_f.mean()), 3),
           "bal_acc": round(float(ba_f.mean()), 3),
           "bal_sd": round(float(ba_f.std(ddof=1)), 3)}
    for c, cname in enumerate(CLASSES):
        rf_ = np.nanmean(np.array(R[c]), axis=0)
        row[cname] = round(float(np.nanmean(rf_)), 3)
    rows.append(row)
    print(f"{label:11s} n={row['n']:6d}  BalAcc={row['bal_acc']:.3f}±{row['bal_sd']:.3f}  "
          + "  ".join(f"{c}={row[c]:.3f}" for c in CLASSES), flush=True)

df = pd.DataFrame(rows)
out = os.path.join(CFG.RESULTS_DIR, "p3_balance_sensitivity.csv")
df.to_csv(out, index=False)
print("\n" + df.to_string(index=False))

#  فحص اتّساق مع فحص الالتباس (يجب أن تتطابق الذراعان 5 و5a)
cf = os.path.join(CFG.RESULTS_DIR, "p3_ladder_confound.csv")
if os.path.exists(cf):
    c = pd.read_csv(cf)
    for arm, lab in [("5 ", "balanced"), ("5a", "unbalanced")]:
        m = c[c.arm.str.strip().str.startswith(arm.strip())]
        m = m[m.arm.str.strip().str.startswith("5a")] if arm == "5a" else m[~m.arm.str.strip().str.startswith("5a")]
        if len(m):
            ref = float(m.bal_acc.iloc[0]); got = float(df[df.setting == lab].bal_acc.iloc[0])
            d = abs(ref - got)
            print(f"اتّساق {lab:11s}: هنا {got:.3f} مقابل confound {ref:.3f} (فرق {d:.3f})"
                  + ("  ✓" if d < 0.02 else "  ⚠️ تباعد — لا تعتمدي الجدول قبل تفسيره"))

print(f"\n✓ {os.path.relpath(out)}")
