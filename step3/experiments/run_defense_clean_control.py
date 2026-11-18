# -*- coding: utf-8 -*-
# =====================================================================
#  ضابط الدفاع على بيانات نظيفة — Defence control at poisoning rate 0
# ---------------------------------------------------------------------
#  Referee point M4: filtered_retrain always drops the top 25% of training
#  windows, whatever the poisoning rate. The published recovery numbers
#  (0.98-0.995) are only interpretable if we know what each defence costs
#  when there is NO poison. This script runs the three defences (+ none)
#  on the clean training partition with the exact fold/seed/scaler setup of
#  run_p3_benchmark.py, section (E).
#
#  Sanity check built in: the "none" arm must reproduce p3_clean_baseline
#  (rf) row by row. If it does not, the script stops — do not trust the rest.
#
#  Diagnostics recorded per run:
#    removed_<class>   windows of each class the 25% filter discards
#    relabelled_<cls>  correct labels relabel_retrain overwrites (all are
#                      corruptions at rate 0, since nothing is poisoned)
#    trigger_bsr       share of triggered attack windows classified Normal
#                      (clean-model trigger susceptibility, after defence)
#
#  Run:  python step3/experiments/run_defense_clean_control.py
#  Out:  step3/results/cases/poison_w50/p3_defense_clean_control.csv
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
import defenses as F

RES = CFG.RESULTS_DIR
SEEDS = CFG.SEEDS[:3]                      # same as the benchmark: 3 seeds x 4 folds
DEFENSES = ["none"] + CFG.DEFENSES         # none, filtered_retrain, trimmed_agg, relabel_retrain
CACHE = os.path.join(CFG.CACHE_DIR, "runs")
os.makedirs(CACHE, exist_ok=True)
_DKEY = P._cache_key()
_RB = getattr(CFG, "RATE_BASIS", "train_set")


def cached(tag, payload, fn):
    # same key recipe as run_p3_benchmark.py, under a NEW tag so nothing old is reused
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
    tmp = fp + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(out, f)
    os.replace(tmp, fp)
    return out


print(">> building windows ...", flush=True)
X, y, groups, flights, feat_names = P.build_dataset(verbose=False)
FOLDS = P.folds(groups, flights)
print(f">> X={X.shape} classes={np.bincount(y).tolist()} folds={[f[0] for f in FOLDS]} seeds={SEEDS}", flush=True)
_FC = {}


def fold_data(fi):
    if fi not in _FC:
        _, tr, te = FOLDS[fi]
        sc = P.fit_scaler(X[tr])
        _FC[fi] = (P.apply_scaler(sc, X[tr]), y[tr], P.apply_scaler(sc, X[te]), y[te])
    return _FC[fi]


def diagnostics(dfs, Xtr, ytr, s):
    """Replays the defence's own selection step (same detector, same seed) to count
    what it removes / relabels per class. Deterministic, so it matches defend()."""
    out = {}
    if dfs not in ("filtered_retrain", "relabel_retrain"):
        return out
    q = CFG.FILTER_Q
    sc = D.detect(CFG.FILTER_DETECTOR, Xtr, ytr, seed=s)
    thr = np.quantile(sc, 1.0 - q)
    flagged = sc >= thr
    for c, name in enumerate(CFG.CLASSES):
        out[f"train_{name}"] = int((ytr == c).sum())
        out[f"flagged_{name}"] = int((flagged & (ytr == c)).sum())
    if dfs == "relabel_retrain":
        trusted = ~flagged
        for c in np.unique(ytr):
            if trusted[ytr == c].sum() < 2:
                idx_c = np.where(ytr == c)[0]
                trusted[idx_c[np.argsort(sc[idx_c])[:2]]] = True
        m1, _ = P.train_victim(Xtr[trusted], ytr[trusted], seed=s)
        sus = ~trusted
        pred = m1._softmax(m1.logits(np.ascontiguousarray(Xtr[sus], dtype=np.float32))).argmax(1)
        ysus = ytr[sus]
        for c, name in enumerate(CFG.CLASSES):
            out[f"relabelled_{name}"] = int(((ysus == c) & (pred != c)).sum())
    return out


rows = []
for dfs in DEFENSES:
    for fi, (fname, _, _) in enumerate(FOLDS):
        for s in SEEDS:
            def _run(dfs=dfs, fi=fi, s=s):
                Xtr, ytr, Xte, yte = fold_data(fi)
                pred = F.defend(dfs, Xtr, ytr, seed=s)          # rate 0: clean training data
                out = P.evaluate(yte, pred(Xte))
                out["trigger_bsr"] = A.backdoor_asr(pred, Xte, yte)
                out.update(diagnostics(dfs, Xtr, ytr, s))
                return out
            r = cached("defend_clean0",
                       {"d": dfs, "f": fname, "s": s, "q": CFG.FILTER_Q,
                        "det": CFG.FILTER_DETECTOR, "tf": CFG.TRIM_FRAC}, _run)
            rows.append({"attack": "none", "rate": 0.0, "defense": dfs, "fold": fname, "seed": s, **r})
    sub = [r for r in rows if r["defense"] == dfs]
    print(f"   rate=0.00 {dfs:18s} BalAcc={np.mean([r['balacc'] for r in sub]):.4f} "
          f"SpoofRec={np.mean([r['spoof_recall'] for r in sub]):.4f} "
          f"trigBSR={np.nanmean([r['trigger_bsr'] for r in sub]):.3f}", flush=True)

df = pd.DataFrame(rows)

# ---- sanity check: "none" must equal the published clean RF baseline row by row ----
ref = pd.read_csv(os.path.join(RES, "p3_clean_baseline.csv"))
ref = ref[ref.model == "rf"].set_index(["fold", "seed"])
mine = df[df.defense == "none"].set_index(["fold", "seed"])
dmax = max(float((mine.balacc - ref.loc[mine.index, "balacc"]).abs().max()),
           float((mine.spoof_recall - ref.loc[mine.index, "spoof_recall"]).abs().max()))
print(f"\n>> sanity: max |none - published clean RF| = {dmax:.2e}", flush=True)
if dmax > 1e-9:
    sys.exit("!! the 'none' arm does NOT reproduce the published baseline — results not written")

out = os.path.join(RES, "p3_defense_clean_control.csv")
df.to_csv(out, index=False)
print(f">> written {out}", flush=True)
