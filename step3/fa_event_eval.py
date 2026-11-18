# -*- coding: utf-8 -*-
# =====================================================================
#  الإنذارات الكاذبة على مستوى الحدث — Event-level false alarms (referee M1)
# ---------------------------------------------------------------------
#  prevalence_eval.py counts false alarms per WINDOW. Windows are 10 s long
#  and advance 0.4 s (96% overlap), so one stretch of misclassified data
#  yields ~25 consecutive "false alarms". An operator sees one alarm.
#
#  This script reproduces the native-prevalence arms of prevalence_eval.py
#  exactly (same RF, seeds, scaler, train-only threshold), checks the
#  window-level FA/h against p3_prevalence_eval.csv row by row, and then
#  counts false alarms as EVENTS.
#
#  Rules, fixed BEFORE looking at any event count:
#    alarm stream  = windows predicted GPS Spoofing, per flight, in time order
#    merge rule    = consecutive alarm windows join one event if their start
#                    times differ by at most GAP seconds:
#                      strict   GAP = 0.6 s  (only back-to-back windows; any
#                                             non-alarm window in between splits)
#                      overlap  GAP = 10 s   (next alarm starts before the
#                                             previous alarm window ends)
#                      refract  GAP = 30 s   (operational example only)
#    false event   = an event containing NO true spoofing window
#    denominator   = the same hours as prevalence_eval (n_test_windows × 0.4 s)
#
#  Window start times are not stored in the dataset, so the windowing loop of
#  poison_core.build_dataset is replayed to recover them. The replay must match
#  the dataset's window count, labels and airframes exactly, or the run stops.
#
#  Run:  python step3/fa_event_eval.py
#  Out:  step3/results/cases/poison_w50/p3_fa_events.csv
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
from sklearn.metrics import recall_score

SEEDS = [0, 1, 2]
SP = CFG.CLASSES.index("GPS Spoofing")
SEC_PER_WINDOW = 0.2 * CFG.STRIDE
RULES = {"strict": 0.6, "overlap": 10.0, "refract": 30.0}

# ---- same configuration as prevalence_eval.load(None) ----
CFG.LABEL_MODE = "window"; CFG.EXCLUDE_LABEL_FEATURES = True; CFG.NORMAL_RATIO = None
import poison_core as P
X, y, g, f, names = P.build_dataset(verbose=False)
print(f">> native dataset X={X.shape} y={np.bincount(y).tolist()} prevalence={np.mean(y == SP):.4f}", flush=True)


# ---- replay the windowing loop to recover each window's start time ----
def window_index():
    tx_base = P._tx_baselines()
    rows = []
    for af, cls, paths in P.discover_flights():
        df, _ = P._fuse(paths)
        active = P._active_mask(df, cls, af, tx_base)
        n = len(df)
        if n < CFG.WINDOW:
            continue
        ts = df["timestamp"].values
        cls_id = CFG.CLASSES.index(cls)
        for st in range(0, n - CFG.WINDOW + 1, CFG.STRIDE):
            frac = active[st:st + CFG.WINDOW].mean()
            lab = cls_id if frac >= CFG.ACTIVE_FRAC else 0
            if 0 < frac < CFG.ACTIVE_FRAC and cls_id != 0:
                continue
            rows.append((af, f"{af}|{cls}", float(ts[st]) / 1e6, lab))
    return pd.DataFrame(rows, columns=["af", "flight", "t", "y"])


W = window_index()
ok = (len(W) == len(y) and (W.y.values == y).all() and (W.af.values == g).all()
      and (W.flight.values == f).all())
print(f">> window-index replay: {len(W)} windows, matches dataset = {ok}", flush=True)
if not ok:
    sys.exit("!! replay does not match the dataset — stopping")
T = W.t.values
#  real spacing between consecutive windows of the same flight
same = W.flight.values[1:] == W.flight.values[:-1]
dt = np.diff(T)[same]
for af in CFG.TEST_AIRFRAMES:
    m = (W.af.values[1:] == af) & same
    print(f"   {af:15s} median window step = {np.median(np.diff(T)[m]):.3f} s  "
          f"(p95 {np.percentile(np.diff(T)[m], 95):.3f} s)", flush=True)


def tune_threshold(proba_tr, y_tr):          # identical to prevalence_eval.py
    pos = (y_tr == SP)
    best = (-1.0, 0.5)
    for t in np.linspace(0.01, 0.95, 60):
        p = proba_tr > t
        tp = (p & pos).sum(); fp = (p & ~pos).sum(); fn = (~p & pos).sum()
        f1 = 2 * tp / max(1, 2 * tp + fp + fn)
        if f1 > best[0]:
            best = (f1, t)
    return best[1]


def events(idx, alarm, gap):
    """idx: test indices in dataset order; alarm: bool per idx. Returns list of
    events, each a list of dataset indices, merged per flight by start-time gap."""
    ev = []
    fl = W.flight.values
    cur, last_t, last_fl = None, None, None
    for i in idx[alarm]:
        if cur is not None and fl[i] == last_fl and T[i] - last_t <= gap:
            cur.append(i)
        else:
            if cur is not None:
                ev.append(cur)
            cur = [i]
        last_t, last_fl = T[i], fl[i]
    if cur is not None:
        ev.append(cur)
    return ev


ref = pd.read_csv(os.path.join(CFG.RESULTS_DIR, "p3_prevalence_eval.csv"))
ARMS = {"native_untreated": ref.arm.unique()[1], "native_tuned": ref.arm.unique()[2]}
rows = []
for af, tr, te in P.loao_folds(g):
    if SP not in set(y[te]):
        continue
    for seed in SEEDS:
        sc = StandardScaler().fit(X[tr])
        m = RandomForestClassifier(n_estimators=200, n_jobs=6,
                                   random_state=seed).fit(sc.transform(X[tr]), y[tr])
        Pte = m.predict_proba(sc.transform(X[te]))
        thr = tune_threshold(m.predict_proba(sc.transform(X[tr]))[:, SP], y[tr])
        for arm, label in ARMS.items():
            pred = Pte.argmax(1)
            if arm == "native_tuned":
                pred[Pte[:, SP] > thr] = SP
            pos = (y[te] == SP)
            alarm = (pred == SP)
            hours = len(te) * SEC_PER_WINDOW / 3600.0
            fp_w = int((alarm & ~pos).sum())
            # ---- sanity: window-level FA/h must equal the published row ----
            r0 = ref[(ref.arm == label) & (ref.fold == af) & (ref.seed == seed)]
            d_fa = abs(fp_w / hours - float(r0.false_alarms_per_hour.iloc[0]))
            d_rc = abs(recall_score(y[te], pred, labels=[SP], average="macro", zero_division=0)
                       - float(r0.recall.iloc[0]))
            if d_fa > 1e-6 or d_rc > 1e-9:
                sys.exit(f"!! {arm} {af} seed {seed}: window FA/h differs from published by {d_fa:.3g} — stopping")
            row = dict(arm=arm, fold=af, seed=seed, hours=hours, threshold=thr if arm == "native_tuned" else np.nan,
                       fp_windows=fp_w, fa_per_hour_window=fp_w / hours)
            #  the held-out airframe has exactly one spoofing flight = one attack episode
            sp_idx = te[y[te] == SP]
            assert len(set(W.flight.values[sp_idx])) == 1, "expected one spoofing flight per airframe"
            a0, a1 = T[sp_idx].min(), T[sp_idx].max()
            for rule, gap in RULES.items():
                ev = events(te, alarm, gap)
                false_ev = [e for e in ev if not (y[e] == SP).any()]
                true_ev = [e for e in ev if (y[e] == SP).any()]
                row[f"events_{rule}"] = len(ev)
                row[f"false_events_{rule}"] = len(false_ev)
                row[f"fa_per_hour_{rule}"] = len(false_ev) / hours
                row[f"mean_false_event_windows_{rule}"] = (np.mean([len(e) for e in false_ev])
                                                           if false_ev else 0.0)
                row[f"attack_detected_{rule}"] = int(len(true_ev) > 0)
                #  where do the false-positive windows go? inside the true alarm (overrun) or stray?
                e_true = np.array([i for e in true_ev for i in e], dtype=int)
                fp_true = e_true[y[e_true] != SP] if e_true.size else e_true
                row[f"fp_in_true_events_{rule}"] = int(fp_true.size)
                row[f"fp_in_false_events_{rule}"] = int(sum(len(e) for e in false_ev))
                assert row[f"fp_in_true_events_{rule}"] + row[f"fp_in_false_events_{rule}"] == fp_w
                #  how long the true alarm runs before the attack starts / after it ends
                row[f"overrun_pre_s_{rule}"] = float((T[fp_true] < a0).sum() * SEC_PER_WINDOW)
                row[f"overrun_post_s_{rule}"] = float((T[fp_true] > a1).sum() * SEC_PER_WINDOW)
                row[f"overrun_inside_s_{rule}"] = float(((T[fp_true] >= a0) & (T[fp_true] <= a1)).sum() * SEC_PER_WINDOW)
            row["attack_s"] = float((a1 - a0) + CFG.WINDOW * 0.2)
            rows.append(row)
        print(f"   {af:15s} seed={seed}  thr={thr:.3f}  "
              f"window FA/h={rows[-1]['fa_per_hour_window']:.1f}  "
              f"event FA/h strict={rows[-1]['fa_per_hour_strict']:.1f} "
              f"overlap={rows[-1]['fa_per_hour_overlap']:.1f} "
              f"refract={rows[-1]['fa_per_hour_refract']:.1f}", flush=True)

df = pd.DataFrame(rows)
print("\n>> sanity: every window-level FA/h and recall matched p3_prevalence_eval.csv", flush=True)
out = os.path.join(CFG.RESULTS_DIR, "p3_fa_events.csv")
df.to_csv(out, index=False)
print(f">> written {out}", flush=True)
