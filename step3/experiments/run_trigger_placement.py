# -*- coding: utf-8 -*-
# =====================================================================
#  موضع المُحفِّز — Backdoor trigger placement (referee point M3)
# ---------------------------------------------------------------------
#  The published backdoor sets the first eight features (window means of
#  eight EKF innovation channels, none of them label-defining) to +3 SD.
#  Question: do the backdoor results (clean-model BSR 0.378, BSR 1.000,
#  detector directions) depend on WHERE the trigger is placed?
#
#  Placements (fixed before any result was seen; 8 features, +3 SD each):
#    original   first 8 features — must reproduce the published runs exactly
#    most_imp   8 most important features of a clean RF fitted on the fold's
#               own clean TRAINING data (no test information)
#    least_imp  8 least important features of that same RF
#    random     8 features drawn at random, per (fold, seed)
#
#  Everything else is the published protocol: Rung-5 RF victim, LOAO, seeds
#  0-2, rates 0.10/0.20/0.30, same scaler, same detectors.
#  The trigger location is swapped by replacing attacks._trigger_dims, the
#  single place that decides it; no attack code is changed.
#
#  Run:  python step3/experiments/run_trigger_placement.py
#  Out:  step3/results/cases/poison_w50/p3_trigger_placement.csv
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
import detectors as D
from sklearn.ensemble import RandomForestClassifier

RES = CFG.RESULTS_DIR
SEEDS = CFG.SEEDS[:3]
RATES = [0.10, 0.20, 0.30]
PLACEMENTS = ["original", "most_imp", "least_imp", "random"]
SP, DOS, NORMAL = (CFG.CLASSES.index(c) for c in ("GPS Spoofing", "Ping DoS", "Normal"))
CACHE = os.path.join(CFG.CACHE_DIR, "runs"); os.makedirs(CACHE, exist_ok=True)
_DKEY = P._cache_key(); _RB = getattr(CFG, "RATE_BASIS", "train_set")
_ORIG_DIMS = A._trigger_dims


def cached(tag, payload, fn):
    key = hashlib.md5((tag + json.dumps(payload, sort_keys=True) +
                       f"|v{CFG.CACHE_VERSION}|ep{CFG.EPOCHS}|d{_DKEY}|rb{_RB}"
                       f"|vic{getattr(CFG, 'VICTIM', 'mlp')}|tm{CFG.TRIGGER_MAG}").encode()).hexdigest()
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
_FC = {}


def fold_data(fi):
    if fi not in _FC:
        _, tr, te = FOLDS[fi]
        sc = P.fit_scaler(X[tr])
        _FC[fi] = (P.apply_scaler(sc, X[tr]), y[tr], P.apply_scaler(sc, X[te]), y[te])
    return _FC[fi]


def placement_dims(pl, fi, s):
    d = X.shape[1]
    if pl == "original":
        return [int(i) for i in _ORIG_DIMS(d)]
    if pl == "random":
        return sorted(int(i) for i in np.random.RandomState(10_000 + 100 * fi + s).choice(d, CFG.TRIGGER_DIM, replace=False))
    Xtr, ytr, _, _ = fold_data(fi)
    imp = RandomForestClassifier(n_estimators=200, random_state=s, n_jobs=4).fit(Xtr, ytr).feature_importances_
    order = np.lexsort((np.arange(d), imp))           # ascending importance, ties by index
    pick = order[-CFG.TRIGGER_DIM:] if pl == "most_imp" else order[:CFG.TRIGGER_DIM]
    return sorted(int(i) for i in pick)


def use_dims(dims):
    A._trigger_dims = lambda d, rng=None, _dims=np.array(dims): _dims


def triggered_share(pred_fn, Xte, yte, cls):
    m = yte == cls
    return float((pred_fn(A.apply_trigger(Xte[m])) == NORMAL).mean()) if m.any() else float("nan")


rows = []
for pl in PLACEMENTS:
    for fi, (fname, _, _) in enumerate(FOLDS):
        for s in SEEDS:
            dims = placement_dims(pl, fi, s)

            def _clean(dims=dims, fi=fi, s=s):
                use_dims(dims)
                Xtr, ytr, Xte, yte = fold_data(fi)
                m, _ = P.train_victim(Xtr, ytr, seed=s)
                pf = lambda Z: P.predict(m, Z)
                return {"bsr": A.backdoor_asr(pf, Xte, yte),
                        "bsr_spoof": triggered_share(pf, Xte, yte, SP),
                        "bsr_dos": triggered_share(pf, Xte, yte, DOS)}
            c = cached("trigplace_clean", {"pl": pl, "dims": dims, "f": fname, "s": s}, _clean)
            rows.append({"placement": pl, "rate": 0.0, "fold": fname, "seed": s, "dims": dims,
                         "dim_names": [str(names[i]) for i in dims], **c})

            for rate in RATES:
                def _poison(dims=dims, fi=fi, s=s, rate=rate):
                    use_dims(dims)
                    Xtr, ytr, Xte, yte = fold_data(fi)
                    Xp, yp, mask = A.poison("backdoor", Xtr, ytr, rate, seed=s)
                    m, _ = P.train_victim(Xp, yp, seed=s)
                    pf = lambda Z: P.predict(m, Z)
                    out = P.evaluate(yte, pf(Xte))
                    out.update(bsr=A.backdoor_asr(pf, Xte, yte),
                               bsr_spoof=triggered_share(pf, Xte, yte, SP),
                               bsr_dos=triggered_share(pf, Xte, yte, DOS))
                    for det in CFG.DETECTORS:
                        try:
                            out[f"auc_{det}"] = D.detection_auc(D.detect(det, Xp, yp, model=m, seed=s), mask)
                        except Exception:
                            out[f"auc_{det}"] = float("nan")
                    return out
                r = cached("trigplace_poison", {"pl": pl, "dims": dims, "f": fname, "s": s, "r": rate}, _poison)
                rows.append({"placement": pl, "rate": rate, "fold": fname, "seed": s, "dims": dims,
                             "dim_names": [str(names[i]) for i in dims], **r})
        A._trigger_dims = _ORIG_DIMS
    sub = pd.DataFrame([r for r in rows if r["placement"] == pl])
    c0 = sub[sub.rate == 0]; p1 = sub[sub.rate > 0]
    print(f"   {pl:10s} clean BSR={c0.bsr.mean():.3f}  poisoned BSR={p1.bsr.mean():.3f}  "
          f"balacc={p1.balacc.mean():.3f}  "
          + "  ".join(f"{d[:8]}={p1[f'auc_{d}'].mean():.3f}" for d in CFG.DETECTORS), flush=True)

df = pd.DataFrame(rows)

# ---- sanity: the original placement must reproduce the published runs exactly ----
o = df[(df.placement == "original") & (df.rate > 0)].set_index(["rate", "fold", "seed"])
dmg = pd.read_csv(os.path.join(RES, "p3_damage_curve.csv"))
dmg = dmg[(dmg.attack == "backdoor") & dmg.rate.isin(RATES)].set_index(["rate", "fold", "seed"])
det = pd.read_csv(os.path.join(RES, "p3_detection.csv"))
det = det[det.attack == "backdoor"].pivot_table(index=["rate", "fold", "seed"], columns="detector", values="auc")
dev = max(float((o.balacc - dmg.loc[o.index, "balacc"]).abs().max()),
          float((o.spoof_recall - dmg.loc[o.index, "spoof_recall"]).abs().max()),
          float((o.bsr - dmg.loc[o.index, "asr"]).abs().max()),
          max(float((o[f"auc_{k}"] - det.loc[o.index, k]).abs().max()) for k in CFG.DETECTORS))
print(f"\n>> sanity: max |original placement - published| = {dev:.2e}", flush=True)
if dev > 1e-9:
    sys.exit("!! the original placement does NOT reproduce the published backdoor runs — results not written")

out = os.path.join(RES, "p3_trigger_placement.csv")
df.to_csv(out, index=False)
print(f">> written {out}", flush=True)
