# -*- coding: utf-8 -*-
# =====================================================================
#  حساسية خط الأساس لتباين الابتكار — Innovation-variance sensitivity (referee m4 follow-up)
# ---------------------------------------------------------------------
#  The m4 stump picked vel_pos_innov_var[3] (variance of the north-position
#  innovation, the channel whose innovation defines the GPS Spoofing label)
#  in every fold. Rung 5 removed the innovations [3],[4] and rate_tx, but not
#  their variances. How much does the published Rung-5 forest depend on them?
#
#  Arms (fixed before any result was seen), same windows/folds/seeds as Rung 5:
#    full        published 70 features (sanity: 0.722 / 0.820)
#    no_var3     minus the 5 window statistics of vel_pos_innov_var[3]
#    no_var34    minus vel_pos_innov_var[3] and [4] (both label channels)
#    no_var_all  minus every vel_pos_innov_var[0..5] statistic
#  Per arm: RF (200 trees, seeds 0-2) balanced accuracy, GPS Spoofing
#  recall/precision, Normal and Ping DoS flagged as spoofing; and the
#  one-feature stump (Spoofing vs rest) with the feature it picks.
#  Scaler refitted per fold on the retained columns (as published).
#
#  Run:  python step3/variance_sensitivity.py
#  Out:  step3/results/cases/poison_w50/p3_variance_sensitivity.csv
# =====================================================================
import os, sys, re, warnings, numpy as np, pandas as pd
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
names = list(names)
assert X.shape[1] == 70, X.shape

def cols_of(pattern):
    return [i for i, n in enumerate(names) if re.match(pattern, n)]

VAR = {k: cols_of(rf"^vel_pos_innov_var\[{k}\]_") for k in range(6)}
print(">> vel_pos_innov_var columns present:", {k: len(v) for k, v in VAR.items()})
ARMS = {
    "full": [],
    "no_var3": VAR[3],
    "no_var34": VAR[3] + VAR[4],
    "no_var_all": sum(VAR.values(), []),
}
assert len(ARMS["no_var3"]) > 0, "vel_pos_innov_var[3] not found among the features"


def spoof_metrics(yt, is_sp):
    tp = int((is_sp & (yt == SP)).sum())
    return {"spoof_recall": tp / max(1, int((yt == SP).sum())),
            "spoof_precision": tp / int(is_sp.sum()) if is_sp.any() else float("nan"),
            "normal_flagged_spoof": float(is_sp[yt == NORMAL].mean()),
            "dos_flagged_spoof": float(is_sp[yt == DOS].mean()) if (yt == DOS).any() else float("nan")}


rows = []
for arm, drop in ARMS.items():
    keep = [i for i in range(X.shape[1]) if i not in set(drop)]
    Xa, na = X[:, keep], [names[i] for i in keep]
    for af, tr, te in P.loao_folds(groups):
        sc = StandardScaler().fit(Xa[tr]); Xtr, Xte = sc.transform(Xa[tr]), sc.transform(Xa[te])
        for s in SEEDS:
            p = RandomForestClassifier(n_estimators=200, n_jobs=2, random_state=s).fit(Xtr, y[tr]).predict(Xte)
            rows.append({"arm": arm, "n_features": len(keep), "method": "RF", "fold": af, "seed": s,
                         "balacc": balanced_accuracy_score(y[te], p), **spoof_metrics(y[te], p == SP)})
            st = DecisionTreeClassifier(max_depth=1, random_state=s).fit(Xtr, (y[tr] == SP).astype(int))
            f = st.tree_.feature[0]
            rows.append({"arm": arm, "n_features": len(keep), "method": "stump", "fold": af, "seed": s,
                         "balacc": float("nan"), "stump_feature": na[f] if f >= 0 else "none",
                         **spoof_metrics(y[te], st.predict(Xte) == 1)})
    print(f"   done {arm} ({len(keep)} features)", flush=True)

df = pd.DataFrame(rows)
full = df[(df.arm == "full") & (df.method == "RF")].groupby("fold")[["balacc", "spoof_recall"]].mean().mean()
print(f">> sanity full RF: balacc {full.balacc:.3f} (published 0.722), spoof recall {full.spoof_recall:.3f} (published 0.820)")
if abs(round(full.balacc, 3) - 0.722) > 1e-9 or abs(round(full.spoof_recall, 3) - 0.820) > 1e-9:
    sys.exit("!! full arm does not reproduce the published Rung-5 baseline — results not written")
out = os.path.join(CFG.RESULTS_DIR, "p3_variance_sensitivity.csv")
df.to_csv(out, index=False)

AF = {"PX4-H480-SITL": "H480", "PX4-QUAD-HITL": "Q-HITL", "PX4-QUAD-SITL": "Q-SITL", "PX4-VTOL-SITL": "VTOL"}
M = ["balacc", "spoof_recall", "spoof_precision", "normal_flagged_spoof", "dos_flagged_spoof"]
print("\nmean over airframes [min, max] (seeds averaged within airframe)")
ref = df[(df.arm == "full")].groupby(["method", "fold", "seed"])[M].mean()
for meth in ("RF", "stump"):
    for arm in ARMS:
        a = df[(df.arm == arm) & (df.method == meth)].groupby("fold")[M].mean()
        print(f"  {meth:5s} {arm:10s} " + "  ".join(f"{c}={a[c].mean():.3f} [{a[c].min():.3f}, {a[c].max():.3f}]"
                                                     for c in M if not a[c].isna().all()))
        print(f"        per airframe: balacc", {AF[k]: round(v, 3) for k, v in a.balacc.items()} if meth == "RF" else "",
              " spoof recall", {AF[k]: round(v, 3) for k, v in a.spoof_recall.items()})
        if arm != "full":
            cur = df[(df.arm == arm) & (df.method == meth)].set_index(["method", "fold", "seed"])[M]
            d = (cur["spoof_recall"] - ref.loc[cur.index, "spoof_recall"])
            print(f"        spoof recall vs full, run by run: lower {int((d < -1e-12).sum())}/12, "
                  f"equal {int((d.abs() <= 1e-12).sum())}/12, higher {int((d > 1e-12).sum())}/12")
            if meth == "RF":
                d = (cur["balacc"] - ref.loc[cur.index, "balacc"])
                print(f"        balacc vs full, run by run: lower {int((d < -1e-12).sum())}/12, higher {int((d > 1e-12).sum())}/12")
    print()
print("stump features chosen:")
for arm in ARMS:
    print(f"  {arm:10s}", df[(df.arm == arm) & (df.method == "stump")].groupby("fold").stump_feature.unique()
          .apply(list).rename(index=AF).to_dict())
print(f">> written {out}")
