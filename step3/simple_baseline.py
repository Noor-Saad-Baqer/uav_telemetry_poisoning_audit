# -*- coding: utf-8 -*-
# =====================================================================
#  خط أساس بسيط — Simple-rule baselines (referee point m4)
# ---------------------------------------------------------------------
#  Does the Random Forest earn its complexity over rules a practitioner
#  could write down? A threshold on the CORRECTED envelope is the label
#  definition itself (score 1.0 by construction), so it is not a baseline.
#
#  Design (fixed before any result was seen), all on the published Rung-5
#  windows (window labels, transition windows dropped, NORMAL_RATIO 3.0),
#  LOAO over the four test airframes, seeds 0-2:
#    RF            published victim (sanity: must give 0.722 / 0.820)
#    ekf_half      no learning: GPS Spoofing if the RAW position innovation
#                  |vel_pos_innov[3]|,|[4]| > 10 m in >= 50% of the window's
#                  samples (the uncorrected label rule = PX4's own innovation
#                  test). Uses the label-defining signal the forest is denied,
#                  so it is a reference, not a fair competitor.
#    ekf_any       as ekf_half, but any sample above 10 m in the window
#    stump         one threshold on one of the forest's 70 features,
#                  GPS Spoofing vs rest, fitted on the training airframes
#    tree_d2/d3    3-class decision trees of depth 2 / 3 on the 70 features
#  Measured per (method, fold, seed): GPS Spoofing recall and precision,
#  share of Normal and of Ping DoS test windows flagged as GPS Spoofing,
#  and balanced accuracy for the 3-class methods.
#
#  Run:  python step3/simple_baseline.py
#  Out:  step3/results/cases/poison_w50/p3_simple_baseline.csv
# =====================================================================
import os, sys, warnings, numpy as np, pandas as pd
warnings.filterwarnings("ignore")
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "experiments"))
import config as CFG
CFG.LABEL_MODE = "window"; CFG.EXCLUDE_LABEL_FEATURES = True; CFG.NORMAL_RATIO = 3.0
import poison_core as P
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import balanced_accuracy_score

SEEDS = [0, 1, 2]
SP, DOS, NORMAL = (CFG.CLASSES.index(c) for c in ("GPS Spoofing", "Ping DoS", "Normal"))
X, y, groups, flights, names = P.build_dataset(verbose=False)


# ---- replay build_dataset to attach the raw innovation-mask fraction to every published window ----
def replay():
    tx_base = P._tx_baselines()
    thr = np.log1p(CFG.SPOOF_INNOV_THR) if CFG.LOG_SCALE_INNOV else CFG.SPOOF_INNOV_THR
    Xs, ys, fs, raw_half, raw_any = [], [], [], [], []
    for af, cls, paths in P.discover_flights():
        df, sigs = P._fuse(paths)
        active = P._active_mask(df, cls, af, tx_base)
        raw = df[["vel_pos_innov[3]", "vel_pos_innov[4]"]].abs().max(axis=1).values > thr   # every flight, incl. Normal
        sigs = [x for x in sigs if x not in set(CFG.LABEL_LEAK_SIGNALS)]
        A = df[sigs].apply(pd.to_numeric, errors="coerce")
        A = A.replace([np.inf, -np.inf], np.nan).ffill().fillna(0.0).values.astype("float64")
        if len(A) < CFG.WINDOW:
            continue
        cls_id = CFG.CLASSES.index(cls)
        for st in range(0, len(A) - CFG.WINDOW + 1, CFG.STRIDE):
            frac = active[st:st + CFG.WINDOW].mean()
            lab = cls_id if frac >= CFG.ACTIVE_FRAC else 0
            if 0 < frac < CFG.ACTIVE_FRAC and cls_id != 0:
                continue
            Xs.append(P._window_feats(A[st:st + CFG.WINDOW])); ys.append(lab); fs.append(f"{af}|{cls}")
            r = raw[st:st + CFG.WINDOW]
            raw_half.append(r.mean() >= CFG.ACTIVE_FRAC); raw_any.append(r.any())
    Xr = np.clip(np.nan_to_num(np.asarray(Xs, dtype="float32"), nan=0.0, posinf=0.0, neginf=0.0), -1e6, 1e6)
    yr = np.asarray(ys); fr = np.asarray(fs); rh = np.asarray(raw_half); ra = np.asarray(raw_any)
    # same balancing as build_dataset (same RandomState(0) call order)
    counts = np.bincount(yr, minlength=CFG.NUM_CLASSES)
    cap = int(CFG.NORMAL_RATIO * min(c for c in counts[1:] if c > 0))
    rng = np.random.RandomState(0); keep = []
    for c in range(CFG.NUM_CLASSES):
        idx_c = np.where(yr == c)[0]
        if len(idx_c) <= cap:
            keep += list(idx_c); continue
        per_flight = {}
        for i in idx_c:
            per_flight.setdefault(fr[i], []).append(i)
        share = max(1, cap // len(per_flight))
        for fl, idxs in per_flight.items():
            keep += list(rng.permutation(np.array(idxs))[:share])
    sel = np.sort(np.array(keep))
    return Xr[sel], yr[sel], rh[sel], ra[sel]


Xr, yr, EKF_HALF, EKF_ANY = replay()
assert Xr.shape == X.shape and np.array_equal(yr, y) and np.allclose(Xr, X), "replay does not match the published windows"
print(f">> replay matches the published Rung-5 windows exactly: X={X.shape}", flush=True)


def spoof_metrics(yt, is_sp):
    tp = int((is_sp & (yt == SP)).sum())
    return {"spoof_recall": tp / max(1, int((yt == SP).sum())),
            "spoof_precision": tp / int(is_sp.sum()) if is_sp.any() else float("nan"),
            "normal_flagged_spoof": float(is_sp[yt == NORMAL].mean()),
            "dos_flagged_spoof": float(is_sp[yt == DOS].mean()) if (yt == DOS).any() else float("nan")}


rows = []
for af, tr, te in P.loao_folds(groups):
    sc = StandardScaler().fit(X[tr]); Xtr, Xte = sc.transform(X[tr]), sc.transform(X[te])
    for s in SEEDS:
        rf = RandomForestClassifier(n_estimators=200, n_jobs=2, random_state=s).fit(Xtr, y[tr])
        preds = {"RF": rf.predict(Xte)}
        for dpt in (2, 3):
            preds[f"tree_d{dpt}"] = DecisionTreeClassifier(max_depth=dpt, random_state=s).fit(Xtr, y[tr]).predict(Xte)
        for meth, p in preds.items():
            rows.append({"method": meth, "fold": af, "seed": s, "balacc": balanced_accuracy_score(y[te], p),
                         **spoof_metrics(y[te], p == SP)})
        st = DecisionTreeClassifier(max_depth=1, random_state=s).fit(Xtr, (y[tr] == SP).astype(int))
        f = st.tree_.feature[0]
        rows.append({"method": "stump", "fold": af, "seed": s, "balacc": float("nan"),
                     "stump_feature": names[f] if f >= 0 else "none",
                     **spoof_metrics(y[te], st.predict(Xte) == 1)})
        for meth, flag in (("ekf_half", EKF_HALF), ("ekf_any", EKF_ANY)):
            rows.append({"method": meth, "fold": af, "seed": s, "balacc": float("nan"),
                         **spoof_metrics(y[te], flag[te])})

df = pd.DataFrame(rows)
rf = df[df.method == "RF"].groupby("fold")[["balacc", "spoof_recall"]].mean().mean()
print(f">> sanity RF: balacc {rf.balacc:.3f} (published 0.722), spoof recall {rf.spoof_recall:.3f} (published 0.820)")
if abs(round(rf.balacc, 3) - 0.722) > 1e-9 or abs(round(rf.spoof_recall, 3) - 0.820) > 1e-9:
    sys.exit("!! RF does not reproduce the published Rung-5 baseline — results not written")
out = os.path.join(CFG.RESULTS_DIR, "p3_simple_baseline.csv")
df.to_csv(out, index=False)
AF = {"PX4-H480-SITL": "H480", "PX4-QUAD-HITL": "Q-HITL", "PX4-QUAD-SITL": "Q-SITL", "PX4-VTOL-SITL": "VTOL"}
print("\nmean over airframes [min, max] (seeds averaged within airframe)")
for meth in ["RF", "tree_d3", "tree_d2", "stump", "ekf_half", "ekf_any"]:
    a = df[df.method == meth].groupby("fold")[["balacc", "spoof_recall", "spoof_precision",
                                                "normal_flagged_spoof", "dos_flagged_spoof"]].mean()
    print(f"  {meth:9s} " + "  ".join(f"{c}={a[c].mean():.3f} [{a[c].min():.3f}, {a[c].max():.3f}]" for c in a.columns))
    if meth in ("stump", "ekf_half"):
        print("            per airframe spoof recall:", {AF[k]: round(v, 3) for k, v in a.spoof_recall.items()})
print("\nstump features chosen:", df[df.method == "stump"].groupby("fold").stump_feature.unique().to_dict())
print(f">> written {out}")
