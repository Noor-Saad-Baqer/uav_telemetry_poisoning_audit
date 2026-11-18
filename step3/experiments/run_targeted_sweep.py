# -*- coding: utf-8 -*-
# =====================================================================
#  مسح الهجوم الموجَّه بالقاعدة المصحَّحة — نسبة من *فئة الانتحال*
# ---------------------------------------------------------------------
#  السبب: القاعدة القديمة (نسبة من كامل التدريب) كانت تستنفد الفئة عند 15%
#  فيصير الاستدعاء 0.000 مضموناً بالبناء. بالقاعدة المصحَّحة يصبح السؤال:
#
#      ما أقلّ نسبة من فئة الانتحال يجب قلبها لتعطيل الكشف؟
#
#  وهذا سؤال ذو معنى ، وإجابته هي النتيجة الحقيقية.
#  نسجّل عمود saturation صراحةً حتى يبقى الاستنفاد مرئياً في كل صفّ.
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

assert getattr(CFG, "RATE_BASIS", "") == "source_class", \
    "هذا المسح يتطلّب RATE_BASIS='source_class' في config.py"

RES = CFG.RESULTS_DIR
SMOKE = os.environ.get("P3_SMOKE") == "1"
#  نسب من فئة الانتحال — تغطّي المدى كاملاً حتى الاستنفاد الحقيقي عند 1.0
RATES = [0.0, 0.5, 1.0] if SMOKE else [0.0, 0.2, 0.4, 0.6, 0.7, 0.8, 0.9, 1.0]
SEEDS = CFG.SEEDS[:1] if SMOKE else CFG.SEEDS[:3]
DEFS = ["none", "filtered_retrain", "relabel_retrain"]
ATK = "targeted_flip"
TAG = "srcbasis_v1"                      # وسم cache جديد ⇒ لا يُعاد استعمال القديم

CACHE = os.path.join(CFG.CACHE_DIR, "targeted_sweep")
os.makedirs(CACHE, exist_ok=True)
OUT = os.path.join(RES, "p3_targeted_sweep.csv")

print(">> بناء البيانات ...", flush=True)
X, y, groups, flights, _ = P.build_dataset(verbose=False)
FOLDS = P.loao_folds(groups)
print(f">> هجوم={ATK} | القاعدة=فئة الانتحال | نسب={RATES} | بذور={SEEDS}\n", flush=True)

_FOLD = {}
def fold_data(fi):
    if fi not in _FOLD:
        _, tr, te = FOLDS[fi]
        sc = P.fit_scaler(X[tr])
        _FOLD[fi] = (P.apply_scaler(sc, X[tr]), y[tr], P.apply_scaler(sc, X[te]), y[te])
    return _FOLD[fi]


def cached(payload, fn):
    k = hashlib.md5((json.dumps(payload, sort_keys=True) +
                     f"|v{CFG.CACHE_VERSION}|ep{CFG.EPOCHS}|{TAG}"
                     f"|d{P._cache_key()}|vic{getattr(CFG, 'VICTIM', 'mlp')}").encode()).hexdigest()
    fp = os.path.join(CACHE, k + ".json")
    if os.path.exists(fp):
        try: return json.load(open(fp))
        except Exception: pass
    out = fn(); tmp = fp + ".tmp"
    json.dump(out, open(tmp, "w")); os.replace(tmp, fp)
    return out


rows = []
if os.path.exists(OUT):
    try:
        rows = pd.read_csv(OUT).to_dict("records")
        print(f">> استئناف: {len(rows)} صفّاً سابقاً\n", flush=True)
    except Exception:
        rows = []
done = {(r["rate"], r["defense"], r["fold"], r["seed"]) for r in rows}

for rate in RATES:
    for dfs in DEFS:
        for fi, (fname, _, _) in enumerate(FOLDS):
            for s in SEEDS:
                if (rate, dfs, fname, s) in done:
                    continue
                def _run(rate=rate, dfs=dfs, fi=fi, s=s):
                    Xtr, ytr, Xte, yte = fold_data(fi)
                    Xp, yp, mask = A.poison(ATK, Xtr, ytr, rate, seed=s)
                    pred = F.defend(dfs, Xp, yp, seed=s)
                    r = P.evaluate(yte, pred(Xte))
                    r["n_poison"] = int(mask.sum())
                    r["saturation"] = float(A.saturation(ytr, rate))
                    r["spoof_left"] = int((yp == A.SRC).sum())
                    return r
                r = cached({"a": ATK, "r": rate, "d": dfs, "f": fname, "s": s}, _run)
                rows.append({"rate": rate, "defense": dfs, "fold": fname, "seed": s, **r})
        sub = [x for x in rows if x["rate"] == rate and x["defense"] == dfs]
        if sub:
            print(f"   rate={rate:.2f} (sat={np.mean([x['saturation'] for x in sub]):.2f}, "
                  f"باقٍ={int(np.mean([x['spoof_left'] for x in sub]))})  {dfs:18s} "
                  f"SpoofRec={np.mean([x['spoof_recall'] for x in sub]):.3f}", flush=True)
        pd.DataFrame(rows).to_csv(OUT, index=False)

df = pd.DataFrame(rows); df.to_csv(OUT, index=False)

print("\n=== استدعاء فئة الانتحال (النسبة = من فئة الانتحال) ===")
print(df.pivot_table(index="rate", columns="defense", values="spoof_recall").round(3).to_string())

print("\n=== أين العتبة الحقيقية؟ (أوّل نسبة يهبط عندها الاستدعاء تحت 0.5) ===")
for dfs in DEFS:
    t = df[df.defense == dfs].groupby("rate")["spoof_recall"].mean()
    below = [r for r in t.index if t[r] < 0.5]
    thr = min(below) if below else None
    print(f"   {dfs:18s} → {'العتبة عند نسبة ' + format(thr, '.2f') if thr is not None else 'لم يهبط أبداً'}")

print(f"\n✓ {os.path.relpath(OUT)}")
