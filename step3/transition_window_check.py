# -*- coding: utf-8 -*-
# =====================================================================
#  نوافذ الانتقال — Transition-window check (referee point m1)
# ---------------------------------------------------------------------
#  Rung 4 changes three things at once: the labelling unit (flight ->
#  window), class balancing, and the removal of "transition" windows —
#  attack-flight windows in which the attack is active for 0 < frac < 0.5
#  of the samples (poison_core.build_dataset, `continue` at frac test).
#  ladder_confound_check.py separated balancing; this script separates the
#  removal of transition windows, which drops the hardest examples from
#  BOTH training and test sets.
#
#  Design (fixed before any result was seen):
#    rungs : R4 (window labels, 85 feats) and R5 (+ label-defining signals
#            removed, 70 feats) — window labels, NORMAL_RATIO = 3.0, LOAO,
#            RF 200 trees, seeds 0-2, StandardScaler: exactly as
#            protocol_ladder.py.
#    arms  : A  published (transition windows dropped everywhere)
#            B  test + held-out airframe's transition windows, labelled by
#               the paper's own majority rule (frac < 0.5 -> Normal)
#            C  as B, transition windows labelled as the flight's attack
#            D  as B, and the training airframes' transition windows added
#               to training with the majority label
#            E  as C, and added to training with the attack label
#    Transition windows are appended, not balanced, so arms A-E share the
#    exact same published windows; the only difference is the added set.
#  Measured per (rung, arm, fold, seed): balanced accuracy, recall per
#    class, and on the added transition windows: share predicted as the
#    flight's attack class (all, GPS Spoofing flights, Ping DoS flights,
#    frac in (0, .25) and [.25, .5)).
#  Sanity: arm A must reproduce p3_protocol_ladder.csv rungs 4 and 5.
#
#  Run:  python step3/transition_window_check.py
#  Out:  step3/results/cases/poison_w50/p3_transition_windows.csv
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
from sklearn.metrics import balanced_accuracy_score, recall_score

SEEDS = [0, 1, 2]
SP, DOS, NORMAL = (CFG.CLASSES.index(c) for c in ("GPS Spoofing", "Ping DoS", "Normal"))
assert NORMAL == 0


def build(exclude_leak):
    CFG.LABEL_MODE = "window"
    CFG.EXCLUDE_LABEL_FEATURES = exclude_leak
    CFG.NORMAL_RATIO = 3.0
    for m in list(sys.modules):
        if m == "poison_core":
            del sys.modules[m]
    import poison_core as P
    return P, P.build_dataset(verbose=False)


def transition_windows(P):
    """Re-run the windowing loop and keep only the windows build_dataset drops."""
    tx_base = P._tx_baselines()
    Xs, gs, cls_ids, fracs = [], [], [], []
    for af, cls, paths in P.discover_flights():
        if cls == "Normal":
            continue
        df, sigs = P._fuse(paths)
        active = P._active_mask(df, cls, af, tx_base)
        if getattr(CFG, "EXCLUDE_LABEL_FEATURES", False):
            sigs = [x for x in sigs if x not in set(CFG.LABEL_LEAK_SIGNALS)]
        A = df[sigs].apply(pd.to_numeric, errors="coerce")
        A = A.replace([np.inf, -np.inf], np.nan).ffill().fillna(0.0).values.astype("float64")
        if len(A) < CFG.WINDOW:
            continue
        for st in range(0, len(A) - CFG.WINDOW + 1, CFG.STRIDE):
            frac = active[st:st + CFG.WINDOW].mean()
            if 0 < frac < CFG.ACTIVE_FRAC:
                Xs.append(P._window_feats(A[st:st + CFG.WINDOW]))
                gs.append(af); cls_ids.append(CFG.CLASSES.index(cls)); fracs.append(frac)
    X = np.asarray(Xs, dtype="float32")
    X = np.clip(np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0), -1e6, 1e6)
    return X, np.asarray(gs), np.asarray(cls_ids), np.asarray(fracs)


def recall_of(yt, p, c):
    m = yt == c
    return float((p[m] == c).mean()) if m.any() else float("nan")


ARMS = {  # name: (test transition label, train transition label or None)
    "A_published": (None, None),
    "B_test_majority": ("majority", None),
    "C_test_attack": ("attack", None),
    "D_traintest_majority": ("majority", "majority"),
    "E_traintest_attack": ("attack", "attack"),
}

rows = []
for rung, excl in (("R4", False), ("R5", True)):
    P, (X, y, groups, flights, names) = build(excl)
    TX, Tg, Tc, Tf = transition_windows(P)
    assert TX.shape[1] == X.shape[1], (TX.shape, X.shape)
    print(f">> {rung}: published windows={len(y)} feats={X.shape[1]}  transition windows={len(Tg)} "
          f"(spoof flights {int((Tc == SP).sum())}, DoS flights {int((Tc == DOS).sum())})", flush=True)
    for af in np.unique(Tg):
        print(f"     {af:16s} transition windows={int((Tg == af).sum())}")
    for arm, (te_lab, tr_lab) in ARMS.items():
        for af, tr, te in P.loao_folds(groups):
            Xtr, ytr, Xte, yte = X[tr], y[tr], X[te], y[te]
            tmask_te = Tg == af
            if tr_lab:
                tm = ~tmask_te
                Xtr = np.vstack([Xtr, TX[tm]])
                ytr = np.r_[ytr, np.zeros(tm.sum(), int) if tr_lab == "majority" else Tc[tm]]
            n_pub = len(yte)
            if te_lab:
                Xte = np.vstack([Xte, TX[tmask_te]])
                yte = np.r_[yte, np.zeros(tmask_te.sum(), int) if te_lab == "majority" else Tc[tmask_te]]
            for s in SEEDS:
                sc = StandardScaler().fit(Xtr)
                m = RandomForestClassifier(n_estimators=200, n_jobs=2, random_state=s)
                m.fit(sc.transform(Xtr), ytr)
                p = m.predict(sc.transform(Xte))
                r = {"rung": rung, "arm": arm, "fold": af, "seed": s,
                     "n_test": len(yte), "n_test_transition": len(yte) - n_pub,
                     "n_train_added": len(ytr) - len(tr),
                     "balacc": balanced_accuracy_score(yte, p),
                     "spoof_recall": recall_score(yte, p, labels=[SP], average="macro", zero_division=0)
                     if SP in set(yte) else float("nan"),
                     "normal_recall": recall_of(yte, p, NORMAL), "dos_recall": recall_of(yte, p, DOS),
                     # the published windows only, so arms are comparable on a fixed set
                     "balacc_pubwin": balanced_accuracy_score(y[te], p[:n_pub])}
                if not te_lab:
                    # how the published model treats the held-out transition windows (scored, not added)
                    pt = m.predict(sc.transform(TX[tmask_te])) if tmask_te.any() else np.array([])
                else:
                    pt = p[n_pub:]
                tc, tf = Tc[tmask_te], Tf[tmask_te]
                hit = pt == tc if pt.size else np.array([], bool)
                for key, mk in (("all", np.ones_like(tc, bool)), ("spoof", tc == SP), ("dos", tc == DOS),
                                ("lowfrac", tf < 0.25), ("highfrac", tf >= 0.25)):
                    r[f"tw_n_{key}"] = int(mk.sum())
                    r[f"tw_flag_{key}"] = float(hit[mk].mean()) if mk.any() else float("nan")
                rows.append(r)
        sub = pd.DataFrame([q for q in rows if q["rung"] == rung and q["arm"] == arm])
        af_b = sub.groupby("fold").balacc.mean(); af_s = sub.groupby("fold").spoof_recall.mean()
        print(f"   {rung} {arm:22s} balacc={af_b.mean():.3f} [{af_b.min():.3f}, {af_b.max():.3f}]  "
              f"spoof={af_s.mean():.3f} [{af_s.min():.3f}, {af_s.max():.3f}]  "
              f"pub-windows balacc={sub.balacc_pubwin.mean():.3f}  "
              f"transition flagged as attack={sub.tw_flag_all.mean():.3f} "
              f"(spoof {sub.tw_flag_spoof.mean():.3f}, DoS {sub.tw_flag_dos.mean():.3f})", flush=True)

df = pd.DataFrame(rows)

# ---- sanity: arm A must reproduce the published ladder rungs 4 and 5 ----
lad = pd.read_csv(os.path.join(CFG.RESULTS_DIR, "p3_protocol_ladder.csv"))
ok = True
for rung, i in (("R4", 3), ("R5", 4)):
    a = df[(df.rung == rung) & (df.arm == "A_published")]
    b = round(a.groupby("fold").balacc.mean().mean(), 3); s = round(a.groupby("fold").spoof_recall.mean().mean(), 3)
    print(f">> sanity {rung}: balacc {b} vs published {lad.bal_acc[i]}, spoof {s} vs {lad.spoof_recall[i]}")
    ok &= abs(b - lad.bal_acc[i]) < 1e-9 and abs(s - lad.spoof_recall[i]) < 1e-9
if not ok:
    sys.exit("!! arm A does NOT reproduce the published ladder — results not written")
out = os.path.join(CFG.RESULTS_DIR, "p3_transition_windows.csv")
df.to_csv(out, index=False)
print(f">> written {out}")
