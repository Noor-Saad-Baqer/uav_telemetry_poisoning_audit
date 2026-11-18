# -*- coding: utf-8 -*-
# =====================================================================
#  سُلّم الصرامة — أين تذهب كل نقطة بين 0.99 المنشورة و0.78 الصادقة؟
# ---------------------------------------------------------------------
#  الأوراق التي تستعمل نفس المجموعة تُبلّغ عن 99.3%–99.5% ، بينما يعطي
#  بروتوكولنا 0.78. الفارق ليس في النموذج بل في ثلاثة عوامل تقييمية:
#
#    (1) التقسيم   : عشوائي على مستوى الصفّ ← اترك-منصّةً-خارجاً
#    (2) الوسم     : حسب المجلّد (الرحلة كاملةً) ← حسب النافذة
#    (3) الميزات   : تشمل إشارات تعريف الوسم ← بعد حذفها
#
#  نُغيّر عاملاً واحداً في كل درجة ليُعزى الهبوط إلى سببه بدقّة.
#  النموذج ثابت في كل الدرجات (Random Forest) ⇒ الفارق بروتوكولي بحت.
# =====================================================================
import os, sys, json, warnings, numpy as np, pandas as pd
warnings.filterwarnings("ignore")
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "experiments"))
import config as CFG
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score, accuracy_score, recall_score

SEEDS = [0, 1, 2]
SPOOF = CFG.CLASSES.index("GPS Spoofing")


def build(label_mode, exclude_leak):
    """يعيد بناء المجموعة بإعداد وسم/ميزات محدّد (لكلٍّ مفتاح cache خاص)."""
    CFG.LABEL_MODE = label_mode
    CFG.EXCLUDE_LABEL_FEATURES = exclude_leak
    for m in list(sys.modules):
        if m == "poison_core":
            del sys.modules[m]
    import poison_core as P
    return P, P.build_dataset(verbose=False)


def evaluate(P, X, y, groups, flights, protocol, seed):
    """يدرّب RF ويُرجع (BalAcc, الدقّة الخام, استدعاء الانتحال)."""
    if protocol == "random":
        tr, te = train_test_split(np.arange(len(y)), test_size=.3,
                                  stratify=y, random_state=seed)
        folds = [(tr, te)]
    elif protocol == "temporal":
        gap = CFG.WINDOW // CFG.STRIDE
        tr, te = [], []
        for fl in np.unique(flights):
            idx = np.where(flights == fl)[0]
            cut = int(.7 * len(idx))
            tr += list(idx[:cut]); te += list(idx[cut + gap:])
        folds = [(np.array(tr), np.array(te))]
    else:                                          # loao
        folds = [(tr, te) for _, tr, te in P.loao_folds(groups)]

    ba, ac, sr = [], [], []
    for tr, te in folds:
        if len(te) == 0 or len(np.unique(y[tr])) < 2:
            continue
        sc = StandardScaler().fit(X[tr])
        m = RandomForestClassifier(n_estimators=200, n_jobs=2, random_state=seed)
        m.fit(sc.transform(X[tr]), y[tr])
        p = m.predict(sc.transform(X[te]))
        ba.append(balanced_accuracy_score(y[te], p))
        ac.append(accuracy_score(y[te], p))
        if SPOOF in set(y[te]):
            sr.append(recall_score(y[te], p, labels=[SPOOF], average="macro", zero_division=0))
    #  نُرجِع القوائم كاملة لا متوسّطاتها: التشتّت جزء من النتيجة لا زينة.
    return np.mean(ba), np.mean(ac), (np.mean(sr) if sr else float("nan")), ba, sr


#  درجات السُّلّم: (الاسم ، الوسم ، حذف إشارات الوسم ، البروتوكول)
LADDER = [
    ("1. كما في الأوراق المنشورة: وسم المجلّد + كل الميزات + تقسيم عشوائي",
     "flight", False, "random"),
    ("2. + تقسيم زمني داخل الرحلة (بدل العشوائي)",
     "flight", False, "temporal"),
    ("3. + اترك-منصّةً-خارجاً (بدل الزمني)",
     "flight", False, "loao"),
    ("4. + وسم على مستوى النافذة (بدل المجلّد)",
     "window", False, "loao"),
    ("5. + حذف إشارات تعريف الوسم (بروتوكولنا الكامل)",
     "window", True, "loao"),
]

rows = []
for name, lm, ex, proto in LADDER:
    P, (X, y, g, f, names) = build(lm, ex)
    r = [evaluate(P, X, y, g, f, proto, s) for s in SEEDS]
    ba = np.mean([x[0] for x in r]); ac = np.mean([x[1] for x in r])
    sr = np.nanmean([x[2] for x in r])

    #  ⚠️ وحدة عدم اليقين تختلف بين الدرجات ، ويجب التصريح بها لا طمسها:
    #    • loao      ⇒ الوحدة = الطيّة (منصّة محجوبة). نُوسِّط البذور داخل
    #                  الطيّة ثم نأخذ الانحراف بين الطيّات الأربع.
    #    • random    ⇒ لا بنية طيّات أصلاً (التقسيم يخلط المنصّات). البذرة
    #                  تُغيّر التقسيم ⇒ الانحراف بين البذور هو المتاح.
    #    • temporal  ⇒ التقسيم حتميّ ؛ البذرة تُغيّر النموذج فقط ⇒ الانحراف
    #                  يقيس تباين النموذج لا تباين التقسيم (أضيق ممّا يجب).
    if proto == "loao":
        per_fold = np.mean([x[3] for x in r], axis=0)      # البذور داخل الطيّة
        sd, unit, n_u = float(np.std(per_fold, ddof=1)), "fold", len(per_fold)
    else:
        sd, unit, n_u = float(np.std([x[0] for x in r], ddof=1)), \
            ("split" if proto == "random" else "model"), len(SEEDS)

    rows.append({"step": name, "label_mode": lm, "features": len(names),
                 "protocol": proto, "raw_acc": round(ac, 3),
                 "bal_acc": round(ba, 3), "spoof_recall": round(sr, 3),
                 "bal_sd": round(sd, 3), "sd_unit": unit, "n_units": n_u})
    print(f"{name}\n   raw_acc={ac:.3f}  BalAcc={ba:.3f}±{sd:.3f} "
          f"({unit}, n={n_u})  spoof_recall={sr:.3f}  "
          f"(feats={len(names)}, n={len(y)})\n", flush=True)

df = pd.DataFrame(rows)
out = os.path.join(CFG.RESULTS_DIR, "p3_protocol_ladder.csv")
df.to_csv(out, index=False)

print("=" * 78)
print(df[["step", "raw_acc", "bal_acc", "spoof_recall"]].to_string(index=False))
print("=" * 78)
d = rows[0]["bal_acc"] - rows[-1]["bal_acc"]
print(f"\nالهبوط الكلّي في BalAcc: {rows[0]['bal_acc']:.3f} → {rows[-1]['bal_acc']:.3f}  (= {d:.3f})")
for i in range(1, len(rows)):
    print(f"   الدرجة {i+1}: {rows[i-1]['bal_acc']:.3f} → {rows[i]['bal_acc']:.3f}"
          f"   (أثر هذا العامل = {rows[i-1]['bal_acc']-rows[i]['bal_acc']:+.3f})")
print(f"\n✓ {os.path.relpath(out)}")
