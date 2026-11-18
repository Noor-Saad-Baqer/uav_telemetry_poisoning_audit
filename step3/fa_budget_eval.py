# -*- coding: utf-8 -*-
# =====================================================================
#  نقطة التشغيل بميزانية إنذارات — False-alarm-budget operating point (referee m5)
# ---------------------------------------------------------------------
#  An operator fixes how many false alarms per flight hour they can tolerate
#  and asks what recall that buys. Here: native prevalence (no rebalancing),
#  the published Rung-5 RF, decision "GPS Spoofing if P(spoof) > t".
#
#  Design (fixed before any result was seen):
#    false alarm   = EVENT-level, as in the paper (M1): alarm windows of one
#                    flight merge when their start times differ by <= 10 s;
#                    an event with no true spoofing window is a false event;
#                    hours = n_test_windows x 0.4 s (as published)
#    budgets       = 0.5, 1, 2, 5, 10 false events per flight hour
#    thresholds    = grid t in {0, 0.005, ..., 0.995}
#    deployable    = t chosen WITHOUT the test airframe: inner leave-one-
#                    airframe-out over the five training airframes (RF refit
#                    on four, scored on the fifth); pick the t with the highest
#                    pooled inner recall whose pooled inner false events/h <=
#                    budget (ties -> larger t). Then applied to the test airframe.
#    oracle        = t chosen ON the test airframe (highest recall within the
#                    budget) -- an upper bound, not a deployable result.
#  Per (budget, method, fold, seed): window recall, attack detected (any
#  true event), delay of the first true alarm after the first spoofing
#  window, realized test false events/h, window precision.
#  Sanity: the outer RF with the published train-F1 threshold (argmax +
#  override) must reproduce p3_fa_events.csv (native_tuned) exactly.
#
#  Run:  python step3/fa_budget_eval.py
#  Out:  step3/results/cases/poison_w50/p3_fa_budget.csv
# =====================================================================
import os, sys, time, warnings, numpy as np, pandas as pd
warnings.filterwarnings("ignore")
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "experiments"))
import config as CFG
CFG.LABEL_MODE = "window"; CFG.EXCLUDE_LABEL_FEATURES = True; CFG.NORMAL_RATIO = None
import poison_core as P
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

SEEDS = [0, 1, 2]
SP = CFG.CLASSES.index("GPS Spoofing")
SEC_PER_WINDOW = 0.2 * CFG.STRIDE
GAP = 10.0
BUDGETS = [0.5, 1.0, 2.0, 5.0, 10.0]
GRID = np.round(np.arange(0.0, 1.0, 0.005), 3)

X, y, g, f, names = P.build_dataset(verbose=False)
print(f">> native dataset X={X.shape} prevalence={np.mean(y == SP):.4f}", flush=True)


def window_times():                      # replay of build_dataset's windowing (as fa_event_eval.py)
    tx_base = P._tx_baselines(); rows = []
    for af, cls, paths in P.discover_flights():
        df, _ = P._fuse(paths)
        active = P._active_mask(df, cls, af, tx_base)
        if len(df) < CFG.WINDOW:
            continue
        ts = df["timestamp"].values; cls_id = CFG.CLASSES.index(cls)
        for st in range(0, len(df) - CFG.WINDOW + 1, CFG.STRIDE):
            frac = active[st:st + CFG.WINDOW].mean()
            if 0 < frac < CFG.ACTIVE_FRAC and cls_id != 0:
                continue
            rows.append((f"{af}|{cls}", float(ts[st]) / 1e6, cls_id if frac >= CFG.ACTIVE_FRAC else 0))
    return pd.DataFrame(rows, columns=["flight", "t", "y"])


W = window_times()
assert len(W) == len(y) and (W.y.values == y).all() and (W.flight.values == f).all(), "replay mismatch"
T, FL = W.t.values, W.flight.values
print(">> window-time replay matches the dataset", flush=True)


def event_stats(idx, alarm):
    """idx: dataset indices (dataset order = flight, time); alarm: bool per idx.
    Returns (n_false_events, attack_detected, first_true_alarm_time or nan)."""
    a = idx[alarm]
    if a.size == 0:
        return 0, 0, np.nan
    new = np.ones(a.size, bool)
    new[1:] = (FL[a[1:]] != FL[a[:-1]]) | (T[a[1:]] - T[a[:-1]] > GAP)
    starts = np.flatnonzero(new)
    has_sp = np.maximum.reduceat((y[a] == SP).astype(int), starts)
    sp_alarm = a[y[a] == SP]
    return int((has_sp == 0).sum()), int(has_sp.any()), (T[sp_alarm].min() if sp_alarm.size else np.nan)


def sweep(idx, p):
    """Per threshold of GRID: recall, false events, detected, first true alarm time, precision."""
    pos = (y[idx] == SP); out = []
    for t in GRID:
        al = p > t
        fe, det, t1 = event_stats(idx, al)
        tp = int((al & pos).sum())
        out.append((tp / max(1, pos.sum()), fe, det, t1, tp / al.sum() if al.any() else np.nan))
    return np.array(out, dtype=float)      # columns: recall, false_events, detected, t_first, precision


def rf(seed, tr):
    sc = StandardScaler().fit(X[tr])
    m = RandomForestClassifier(n_estimators=200, n_jobs=6, random_state=seed).fit(sc.transform(X[tr]), y[tr])
    return sc, m


def tune_threshold(proba_tr, y_tr):      # identical to prevalence_eval.py (sanity only)
    pos = (y_tr == SP); best = (-1.0, 0.5)
    for t in np.linspace(0.01, 0.95, 60):
        pp = proba_tr > t
        tp = (pp & pos).sum(); fp = (pp & ~pos).sum(); fn = (~pp & pos).sum()
        f1 = 2 * tp / max(1, 2 * tp + fp + fn)
        if f1 > best[0]:
            best = (f1, t)
    return best[1]


ref = pd.read_csv(os.path.join(CFG.RESULTS_DIR, "p3_fa_events.csv"))
ref = ref[ref.arm == "native_tuned"]
rows, t0 = [], time.time()
for af, tr, te in P.loao_folds(g):
    if SP not in set(y[te]):
        continue
    hours_te = len(te) * SEC_PER_WINDOW / 3600.0
    a0 = T[te[y[te] == SP]].min()
    for seed in SEEDS:
        sc, m = rf(seed, tr)
        Pte = m.predict_proba(sc.transform(X[te]))
        # ---- sanity: published operating point ----
        thr = tune_threshold(m.predict_proba(sc.transform(X[tr]))[:, SP], y[tr])
        pred = Pte.argmax(1); pred[Pte[:, SP] > thr] = SP
        fe, _, _ = event_stats(te, pred == SP)
        r0 = ref[(ref.fold == af) & (ref.seed == seed)].iloc[0]
        fpw = int(((pred == SP) & (y[te] != SP)).sum())
        if fpw != r0.fp_windows or fe != r0.false_events_overlap:
            sys.exit(f"!! {af} seed {seed}: published operating point not reproduced ({fpw}/{r0.fp_windows}, {fe}/{r0.false_events_overlap})")
        S_te = sweep(te, Pte[:, SP])
        # ---- inner leave-one-airframe-out on the training airframes ----
        inner_fe = np.zeros(len(GRID)); inner_tp = np.zeros(len(GRID)); inner_pos = 0; inner_h = 0.0
        for a in sorted(set(g[tr])):
            itr, ite = tr[g[tr] != a], tr[g[tr] == a]
            isc, im = rf(seed, itr)
            S = sweep(ite, im.predict_proba(isc.transform(X[ite]))[:, SP])
            npos = int((y[ite] == SP).sum())
            inner_fe += S[:, 1]; inner_tp += S[:, 0] * npos; inner_pos += npos
            inner_h += len(ite) * SEC_PER_WINDOW / 3600.0
        inner_rec = inner_tp / max(1, inner_pos); inner_fah = inner_fe / inner_h
        for B in BUDGETS:
            choices = {}
            ok = np.flatnonzero(inner_fah <= B)
            choices["deployable"] = ok[np.lexsort((-GRID[ok], -inner_rec[ok]))[0]] if ok.size else len(GRID) - 1
            ok = np.flatnonzero(S_te[:, 1] / hours_te <= B)
            choices["oracle"] = ok[np.lexsort((-GRID[ok], -S_te[ok, 0]))[0]] if ok.size else len(GRID) - 1
            for meth, k in choices.items():
                rec, fev, det, t1, prec = S_te[k]
                rows.append(dict(budget=B, method=meth, fold=af, seed=seed, threshold=GRID[k],
                                 recall=rec, false_events=int(fev), fa_events_per_hour=fev / hours_te,
                                 within_budget=int(fev / hours_te <= B), attack_detected=int(det),
                                 delay_s=(t1 - a0) if det else np.nan, precision=prec, hours=hours_te,
                                 inner_recall=inner_rec[k] if meth == "deployable" else np.nan,
                                 inner_fa_per_hour=inner_fah[k] if meth == "deployable" else np.nan))
        print(f"   {af:15s} seed={seed} done ({time.time() - t0:.0f} s)", flush=True)

print(">> sanity: published operating point reproduced in all 12 runs (window FPs and false events)", flush=True)
df = pd.DataFrame(rows)
out = os.path.join(CFG.RESULTS_DIR, "p3_fa_budget.csv")
df.to_csv(out, index=False)
AF = {"PX4-H480-SITL": "H480", "PX4-QUAD-HITL": "Q-HITL", "PX4-QUAD-SITL": "Q-SITL", "PX4-VTOL-SITL": "VTOL"}
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
for meth in ("deployable", "oracle"):
    print(f"\n== {meth} (mean over airframes [min, max]; seeds averaged within airframe) ==")
    for B in BUDGETS:
        a = df[(df.method == meth) & (df.budget == B)].groupby("fold")[
            ["threshold", "recall", "fa_events_per_hour", "within_budget", "attack_detected", "delay_s", "precision"]].mean()
        print(f"  budget {B:4}/h: " + "  ".join(f"{c}={a[c].mean():.3f} [{a[c].min():.3f}, {a[c].max():.3f}]" for c in a.columns))
        print(f"               recall per airframe {({AF[k]: round(v, 3) for k, v in a.recall.items()})}  "
              f"FA/h per airframe {({AF[k]: round(v, 2) for k, v in a.fa_events_per_hour.items()})}")
    w = df[df.method == meth].groupby("budget").within_budget.sum()
    print("  runs within budget (of 12):", w.to_dict())
print(f">> written {out}")
