# -*- coding: utf-8 -*-
# =====================================================================
#  اختبار دفاع «إعادة الوسم بدل الحذف» — relabel_retrain
# ---------------------------------------------------------------------
#  السؤال المحوري: هل يتجاوز عتبة الانهيار (بحلول 15%) حيث ينهار الحذف إلى 0.000؟
#  النتيجة: لا — يتعادل معه تحت العتبة ويصير 0.000 فوقها (انظري README §الدفاع).
#  المقارنة: none / filtered_retrain / relabel_retrain
#  البروتوكول: نفس الطيّات (LOAO) ونفس البذور ⇒ قابل للمقارنة مباشرةً.
# =====================================================================
import os, sys, json, hashlib, warnings, numpy as np, pandas as pd
warnings.filterwarnings("ignore")
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, ".."))
import config as CFG
import poison_core as P
import attacks as A
import defenses as F

RES = CFG.RESULTS_DIR
SMOKE = os.environ.get("P3_SMOKE") == "1"
RATES = [0.10, 0.20] if SMOKE else [0.05, 0.10, 0.15, 0.20, 0.30, 0.40]
SEEDS = CFG.SEEDS[:1] if SMOKE else CFG.SEEDS[:3]
#  أذرع المقارنة. الذراع الرابعة هي الاختبار الحاسم لفرضيّتنا:
#  activation_cluster كاشف أدقّ (AUC 0.87) لكنه دمّر الفئة الصغيرة *بالحذف*.
#  مع إعادة الوسم لا يُحذَف شيء ⇒ إن كانت الفرضية صحيحة فيجب أن يتفوّق الآن.
DEFS = ["none", "filtered_retrain", "relabel_retrain", "relabel_actclust"]
DEF_KW = {"relabel_actclust": {"detector": "activation_cluster"}}
DEF_FN = {"relabel_actclust": "relabel_retrain"}
ATK = CFG.ATTACK_FOR_OPT           # targeted_flip

CACHE = os.path.join(CFG.CACHE_DIR, "relabel")
os.makedirs(CACHE, exist_ok=True)
OUT = os.path.join(RES, "p3_relabel_test.csv")

print(">> بناء البيانات ...", flush=True)
X, y, groups, flights, _ = P.build_dataset(verbose=False)
FOLDS = P.loao_folds(groups)
print(f">> هجوم={ATK} | نسب={RATES} | بذور={SEEDS} | دفاعات={DEFS}\n", flush=True)

_FOLD = {}
def fold_data(fi):
    if fi not in _FOLD:
        _, tr, te = FOLDS[fi]
        sc = P.fit_scaler(X[tr])
        _FOLD[fi] = (P.apply_scaler(sc, X[tr]), y[tr], P.apply_scaler(sc, X[te]), y[te])
    return _FOLD[fi]


def cached(payload, fn):
    k = hashlib.md5((json.dumps(payload, sort_keys=True) + f"|v{CFG.CACHE_VERSION}|ep{CFG.EPOCHS}"
                     f"|d{P._cache_key()}|vic{getattr(CFG, 'VICTIM', 'mlp')}").encode()).hexdigest()
    fp = os.path.join(CACHE, k + ".json")
    if os.path.exists(fp):
        try: return json.load(open(fp))
        except Exception: pass
    out = fn()
    tmp = fp + ".tmp"; json.dump(out, open(tmp, "w")); os.replace(tmp, fp)
    return out


rows = []
for rate in RATES:
    for dfs in DEFS:
        for fi, (fname, _, _) in enumerate(FOLDS):
            for s in SEEDS:
                def _run(rate=rate, dfs=dfs, fi=fi, s=s):
                    Xtr, ytr, Xte, yte = fold_data(fi)
                    Xp, yp, mask = A.poison(ATK, Xtr, ytr, rate, seed=s)
                    pred = F.defend(DEF_FN.get(dfs, dfs), Xp, yp, seed=s, **DEF_KW.get(dfs, {}))
                    r = P.evaluate(yte, pred(Xte))
                    r["n_poison"] = int(mask.sum())
                    return r
                r = cached({"a": ATK, "r": rate, "d": dfs, "f": fname, "s": s, "t": "relabel_v2"}, _run)
                rows.append({"attack": ATK, "rate": rate, "defense": dfs,
                             "fold": fname, "seed": s, **r})
        sub = [x for x in rows if x["rate"] == rate and x["defense"] == dfs]
        print(f"   rate={rate:.2f}  {dfs:18s} BalAcc={np.mean([x['balacc'] for x in sub]):.3f}  "
              f"SpoofRec={np.mean([x['spoof_recall'] for x in sub]):.3f}", flush=True)
        pd.DataFrame(rows).to_csv(OUT, index=False)

df = pd.DataFrame(rows)
df.to_csv(OUT, index=False)

print("\n=== جدول الاستدعاء (استدعاء فئة الانتحال) ===")
piv = df.pivot_table(index="rate", columns="defense", values="spoof_recall").round(3)
print(piv.to_string())

#  الاختبار الحاسم: هل تفوّقت إعادة الوسم على الحذف فوق العتبة (≥20%)؟
print("\n=== الاختبار الحاسم (فوق عتبة الانهيار) ===")
from scipy.stats import wilcoxon
for rate in [r for r in RATES if r >= 0.20]:
    a = df[(df.rate == rate) & (df.defense == "filtered_retrain")].sort_values(["fold", "seed"])["spoof_recall"].values
    for arm in ["relabel_retrain", "relabel_actclust"]:
        b = df[(df.rate == rate) & (df.defense == arm)].sort_values(["fold", "seed"])["spoof_recall"].values
        if len(a) != len(b) or len(a) < 5:
            continue
        try:
            p = wilcoxon(a, b).pvalue if np.any(a != b) else float("nan")
        except Exception:
            p = float("nan")
        verdict = "✓ أفضل من الحذف" if b.mean() > a.mean() else "✗ لا تحسّن"
        print(f"   rate={rate:.2f}  {arm:18s}: filtered={a.mean():.3f}  arm={b.mean():.3f}  "
              f"p={p:.3g}  {verdict}")

print(f"\n✓ {os.path.relpath(OUT)}")
