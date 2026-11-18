# -*- coding: utf-8 -*-
# =====================================================================
#  شدّة التسميم نظيف الوسم — Clean-label strength sweep (referee point M7)
# ---------------------------------------------------------------------
#  The published clean-label attack moves each selected GPS Spoofing
#  training window a fraction eps = 0.7 of the way toward the mean of the
#  Normal training windows, keeping its Spoofing label. The paper found no
#  damage. That is one point in the (rate x eps) plane.
#
#  Design (fixed before any result was seen):
#    eps   in {0.3, 0.5, 0.7, 0.9, 1.0}   (1.0 = exactly at the Normal mean)
#    rate  in {0.10, 0.20, 0.30, 0.40}    (fraction of the Spoofing class)
#    published protocol: Rung-5 RF victim, LOAO, seeds 0-2, same scaler.
#  Measured per run:
#    balacc, spoof recall (as published), Normal recall, DoS recall,
#    share of Normal test windows predicted GPS Spoofing (a different kind
#    of damage: spoofing labels planted in Normal territory),
#    share of the poisoned training windows that a CLEAN model already
#    classifies as Normal (did the poison cross the decision boundary?).
#  Sanity: eps = 0.7 must reproduce p3_damage_curve.csv exactly.
#
#  Run:  python step3/experiments/run_clean_label_sweep.py
#  Out:  step3/results/cases/poison_w50/p3_clean_label_sweep.csv
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

RES = CFG.RESULTS_DIR
SEEDS = CFG.SEEDS[:3]
EPS = [0.3, 0.5, 0.7, 0.9, 1.0]
RATES = [0.10, 0.20, 0.30, 0.40]
SP, DOS, NORMAL = (CFG.CLASSES.index(c) for c in ("GPS Spoofing", "Ping DoS", "Normal"))
CACHE = os.path.join(CFG.CACHE_DIR, "runs"); os.makedirs(CACHE, exist_ok=True)
_DKEY = P._cache_key(); _RB = getattr(CFG, "RATE_BASIS", "train_set")


def cached(tag, payload, fn):
    key = hashlib.md5((tag + json.dumps(payload, sort_keys=True) +
                       f"|v{CFG.CACHE_VERSION}|ep{CFG.EPOCHS}|d{_DKEY}|rb{_RB}"
                       f"|vic{getattr(CFG, 'VICTIM', 'mlp')}").encode()).hexdigest()
    fp = os.path.join(CACHE, f"{tag}_{key}.json")
    if os.path.exists(fp):
        try:
            with open(fp, encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    out = fn()
    with open(fp + ".tmp", "w", encoding="utf-8") as f:
        json.dump(out, f)
    os.replace(fp + ".tmp", fp)
    return out


print(">> building windows ...", flush=True)
X, y, groups, flights, names = P.build_dataset(verbose=False)
FOLDS = P.folds(groups, flights)
print(f">> X={X.shape} folds={[f[0] for f in FOLDS]} seeds={SEEDS}", flush=True)
_FC, _CLEAN = {}, {}


def fold_data(fi):
    if fi not in _FC:
        _, tr, te = FOLDS[fi]
        sc = P.fit_scaler(X[tr])
        _FC[fi] = (P.apply_scaler(sc, X[tr]), y[tr], P.apply_scaler(sc, X[te]), y[te])
    return _FC[fi]


def clean_model(fi, s):
    if (fi, s) not in _CLEAN:
        Xtr, ytr, _, _ = fold_data(fi)
        _CLEAN[(fi, s)] = P.train_victim(Xtr, ytr, seed=s)[0]
    return _CLEAN[(fi, s)]


def share(pred, truth, cls_true, cls_pred):
    m = truth == cls_true
    return float((pred[m] == cls_pred).mean()) if m.any() else float("nan")


rows = []
for eps in EPS:
    for rate in RATES:
        for fi, (fname, _, _) in enumerate(FOLDS):
            for s in SEEDS:
                def _run(eps=eps, rate=rate, fi=fi, s=s):
                    Xtr, ytr, Xte, yte = fold_data(fi)
                    Xp, yp, mask = A.clean_label(Xtr, ytr, rate, np.random.RandomState(s), eps=eps)
                    m, _ = P.train_victim(Xp, yp, seed=s)
                    pred = P.predict(m, Xte)
                    out = P.evaluate(yte, pred)
                    out.update(normal_recall=share(pred, yte, NORMAL, NORMAL),
                               dos_recall=share(pred, yte, DOS, DOS),
                               normal_flagged_spoof=share(pred, yte, NORMAL, SP),
                               n_poison=int(mask.sum()))
                    #  where did the poison land? (a clean model's view of the moved windows)
                    cp = P.predict(clean_model(fi, s), Xp[mask]) if mask.any() else np.array([])
                    out["poison_seen_as_normal"] = float((cp == NORMAL).mean()) if cp.size else float("nan")
                    out["poison_seen_as_spoof"] = float((cp == SP).mean()) if cp.size else float("nan")
                    return out
                r = cached("clsweep", {"eps": eps, "r": rate, "f": fname, "s": s}, _run)
                rows.append({"eps": eps, "rate": rate, "fold": fname, "seed": s, **r})
        sub = pd.DataFrame([r for r in rows if r["eps"] == eps and r["rate"] == rate])
        print(f"   eps={eps:.1f} rate={rate:.2f}  spoof={sub.spoof_recall.mean():.3f}  normal={sub.normal_recall.mean():.3f}  "
              f"normal->spoof={sub.normal_flagged_spoof.mean():.3f}  balacc={sub.balacc.mean():.3f}  "
              f"poison seen as Normal={sub.poison_seen_as_normal.mean():.3f}", flush=True)

df = pd.DataFrame(rows)

# ---- sanity: eps = 0.7 must reproduce the published clean-label runs ----
o = df[df.eps == 0.7].set_index(["rate", "fold", "seed"])
dmg = pd.read_csv(os.path.join(RES, "p3_damage_curve.csv"))
dmg = dmg[(dmg.attack == "clean_label") & dmg.rate.isin(RATES)].set_index(["rate", "fold", "seed"])
dev = max(float((o.balacc - dmg.loc[o.index, "balacc"]).abs().max()),
          float((o.spoof_recall - dmg.loc[o.index, "spoof_recall"]).abs().max()))
print(f"\n>> sanity: max |eps=0.7 - published clean-label| = {dev:.2e}", flush=True)
if dev > 1e-9:
    sys.exit("!! eps = 0.7 does NOT reproduce the published clean-label runs — results not written")
out = os.path.join(RES, "p3_clean_label_sweep.csv")
df.to_csv(out, index=False)
print(f">> written {out}", flush=True)
