# -*- coding: utf-8 -*-
# =====================================================================
#  المرحلة الثالثة — التسميم المركزي: هجوم ، كشف ، دفاع
#  Stage 3 — Centralized poisoning: attack, detection, defence
# ---------------------------------------------------------------------
#  كلّ التجارب جاهزة في step3/results/cases/poison_w50 (قلب الوسوم ، الموجَّه ،
#  الباب الخلفي ، clean-label ، سلّم النسب ، الكاشفات الخمسة ، الدفاعات الثلاثة
#  والحذف) ⇒ لا يُعاد تنفيذها. ما ينقص الشكل ثلاثة **مقاييس مسمّاة** لم تكن
#  أعمدة صريحة:
#     • Attack success rate      — نجاح الهجوم
#     • Backdoor success rate    — نجاح الباب الخلفي (عمود asr الجاهز)
#     • Defence recovery rate    — نسبة ما يستعيده الدفاع من الضرر
#  تُشتقّ هنا من الملفّات الجاهزة بتعريفات معلنة ، مع فواصل ثقة على (طيّة × بذرة)
#  واختبارات مزاوجة مصحَّحة بـ Holm.
#
#  التعريفات (chance = 1/3 ، الضحيّة = الغابة العشوائية ، بروتوكول LOAO):
#     ASR_bal   = clip((bal_clean − bal_poison) / (bal_clean − chance), 0, 1)
#                 نسبة ما دمّره الهجوم من «المهارة فوق التخمين»
#     ASR_tgt   = clip(1 − recall_spoof_poison / recall_spoof_clean, 0, 1)
#                 للهجوم الموجَّه: نسبة ما أخفاه من كشف الانتحال
#     BSR       = asr الجاهز للباب الخلفي (نسبة عيّنات الهجوم المزوَّدة بالمحفّز
#                 التي صُنّفت «طبيعية»)
#     Recovery  = clip((bal_def − bal_none) / (bal_clean − bal_none), 0, 1)
#                 صفر = الدفاع لم يستعد شيئاً ، واحد = عاد إلى النظيف
#
#  التشغيل:  python framework/stage3_poisoning/run.py
# =====================================================================
import os, sys, numpy as np, pandas as pd
FW = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, FW)
from common import paths as P, io as IO, stats as ST

CHANCE = 1.0 / 3.0
LEGACY_FILES = ["p3_clean_baseline.csv", "p3_damage_curve.csv", "p3_defense.csv", "p3_detection.csv",
                "p3_defense_opt.csv", "p3_targeted_sweep.csv", "p3_protocol_ladder.csv",
                "p3_published_ladder.csv", "p3_ladder_combined.csv", "p3_balance_sensitivity.csv",
                "p3_groundtruth_agreement.csv", "p3_stats.json", "p3_stats_clustered.json", "p3_opt_stats.json"]


def _clip(v):
    return float(np.clip(v, 0.0, 1.0)) if np.isfinite(v) else float("nan")


def _ci_row(v, **kw):
    m, lo, hi = ST.boot_ci(v)
    return dict(**kw, n=int(np.isfinite(np.asarray(v, float)).sum()), mean=m, ci_lo=lo, ci_hi=hi,
                sd=float(np.nanstd(np.asarray(v, float))))


def attack_success(dmg, clean):
    """صفوف (attack, rate, fold, seed) مع ASR_bal و ASR_tgt و BSR."""
    c = clean[clean.model == "rf"].set_index(["fold", "seed"])
    rows = []
    for _, r in dmg.iterrows():
        if r.rate == 0:
            continue
        try:
            bc, rc = c.loc[(r.fold, r.seed), "balacc"], c.loc[(r.fold, r.seed), "spoof_recall"]
        except KeyError:
            continue
        asr_bal = _clip((bc - r.balacc) / (bc - CHANCE)) if bc > CHANCE else float("nan")
        asr_tgt = _clip(1 - r.spoof_recall / rc) if rc > 0 else float("nan")
        rows.append(dict(attack=r.attack, rate=r.rate, fold=r.fold, seed=r.seed,
                         bal_clean=bc, bal_poison=r.balacc, spoof_recall_clean=rc,
                         spoof_recall_poison=r.spoof_recall, n_poison=r.n_poison,
                         asr_balacc_drop=asr_bal, asr_target_recall_drop=asr_tgt,
                         backdoor_success_rate=(float(r.asr) if r.attack == "backdoor" and np.isfinite(r.asr) else np.nan)))
    return pd.DataFrame(rows)


def defence_recovery(dfn, clean):
    c = clean[clean.model == "rf"].set_index(["fold", "seed"])
    none = dfn[dfn.defense == "none"].set_index(["attack", "rate", "fold", "seed"])
    rows = []
    for _, r in dfn[dfn.defense != "none"].iterrows():
        try:
            bc = c.loc[(r.fold, r.seed), "balacc"]; rc = c.loc[(r.fold, r.seed), "spoof_recall"]
            n0 = none.loc[(r.attack, r.rate, r.fold, r.seed)]
        except KeyError:
            continue
        bn, rn = float(n0.balacc), float(n0.spoof_recall)
        rec_bal = _clip((r.balacc - bn) / (bc - bn)) if bc - bn > 1e-9 else float("nan")
        rec_tgt = _clip((r.spoof_recall - rn) / (rc - rn)) if rc - rn > 1e-9 else float("nan")
        row = dict(attack=r.attack, rate=r.rate, defense=r.defense, fold=r.fold, seed=r.seed,
                   bal_clean=bc, bal_none=bn, bal_defended=r.balacc,
                   recovery_balacc=rec_bal, recovery_target_recall=rec_tgt,
                   delta_balacc=float(r.balacc - bn))
        if r.attack == "backdoor":
            a_none, a_def = float(n0.asr) if "asr" in n0 else np.nan, float(r.asr) if "asr" in r else np.nan
            row.update(bsr_none=a_none, bsr_defended=a_def,
                       bsr_reduction=_clip(1 - a_def / a_none) if np.isfinite(a_none) and a_none > 0 else np.nan)
        rows.append(row)
    return pd.DataFrame(rows)


def summarize(asr, rec, det, out):
    s = []
    for (a, r), g in asr.groupby(["attack", "rate"]):
        s.append(_ci_row(g.asr_balacc_drop, metric="attack_success_rate (balacc drop)", attack=a, rate=r))
        s.append(_ci_row(g.asr_target_recall_drop, metric="attack_success_rate (spoof recall drop)", attack=a, rate=r))
        if a == "backdoor":
            s.append(_ci_row(g.backdoor_success_rate, metric="backdoor_success_rate", attack=a, rate=r))
    for (a, r, d), g in rec.groupby(["attack", "rate", "defense"]):
        s.append(_ci_row(g.recovery_balacc, metric="defence_recovery_rate (balacc)", attack=a, rate=r, defense=d))
        s.append(_ci_row(g.recovery_target_recall, metric="defence_recovery_rate (spoof recall)", attack=a, rate=r, defense=d))
        if a == "backdoor" and "bsr_reduction" in g:
            s.append(_ci_row(g.bsr_reduction, metric="backdoor_success_reduction", attack=a, rate=r, defense=d))
    summ = pd.DataFrame(s)
    summ.to_csv(os.path.join(out, "s3_named_metrics_summary.csv"), index=False)

    # ---- اختبارات مزاوجة: الدفاع مقابل لا-دفاع ، Holm داخل كلّ هجوم ----
    tests = []
    for (a, r, d), g in rec.groupby(["attack", "rate", "defense"]):
        _, p = ST.wilcoxon_paired(g.bal_defended, g.bal_none)
        cd = ST.cliffs_delta(g.bal_defended, g.bal_none)
        tests.append(dict(attack=a, rate=r, defense=d, n=len(g), mean_delta=float(g.delta_balacc.mean()),
                          p_wilcoxon=p, cliffs_delta=cd, effect=ST.cliffs_label(cd)))
    tests = pd.DataFrame(tests)
    for a in tests.attack.unique():
        m = tests.attack == a
        tests.loc[m, "p_holm"] = ST.holm(tests.loc[m, "p_wilcoxon"].fillna(1.0).values)
    tests["significant_005"] = tests.p_holm < 0.05
    tests.to_csv(os.path.join(out, "s3_defence_tests.csv"), index=False)

    # ---- الكشف: AUC لكلّ كاشف × هجوم × نسبة مع فاصل ثقة ----
    ds = []
    for (a, r, k), g in det.groupby(["attack", "rate", "detector"]):
        ds.append(_ci_row(g.auc_oriented if "auc_oriented" in g else g.auc, attack=a, rate=r, detector=k,
                          inverted_share=float(g.inverted.mean()) if "inverted" in g else np.nan))
    pd.DataFrame(ds).to_csv(os.path.join(out, "s3_detection_summary.csv"), index=False)
    return summ, tests


def make_figure(summ, out_fig):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    a0 = summ[summ.metric.str.startswith("attack_success_rate (balacc")]
    for atk, g in a0.groupby("attack"):
        g = g.sort_values("rate")
        ax[0].errorbar(g.rate, g["mean"], yerr=[g["mean"] - g.ci_lo, g.ci_hi - g["mean"]], marker="o", capsize=3, label=atk)
    ax[0].set_xlabel("poison rate"); ax[0].set_ylabel("attack success rate"); ax[0].set_ylim(-0.02, 1.02)
    ax[0].legend(fontsize=8); ax[0].set_title("Attack success (skill destroyed)")
    r0 = summ[summ.metric == "defence_recovery_rate (balacc)"]
    if not r0.empty:
        piv = r0.pivot_table(index=["attack", "rate"], columns="defense", values="mean")
        piv.plot.bar(ax=ax[1], rot=45)
        ax[1].set_ylabel("defence recovery rate"); ax[1].set_ylim(0, 1.02); ax[1].set_title("Defence recovery")
        ax[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(out_fig, "S3_named_metrics.png"), dpi=200)
    plt.close(fig)


def main():
    out, fig = P.results("stage3"), P.figures("stage3")
    IO.log("=" * 70 + "\n  المرحلة الثالثة — التسميم: مقاييس مسمّاة من النتائج الجاهزة\n" + "=" * 70)
    IO.copy_legacy([os.path.join(P.STEP3_CASE, f) for f in LEGACY_FILES], P.legacy("stage3"),
                   "# مصدر النتائج الجاهزة — المرحلة الثالثة\n\nنُسخت من step3/results/cases/poison_w50 "
                   "ولم يُعَد تنفيذها. التجارب الأصلية على كامل الطائرات الستّ (قبل الختم).\n")
    clean = pd.read_csv(os.path.join(P.STEP3_CASE, "p3_clean_baseline.csv"))
    dmg = pd.read_csv(os.path.join(P.STEP3_CASE, "p3_damage_curve.csv"))
    dfn = pd.read_csv(os.path.join(P.STEP3_CASE, "p3_defense.csv"))
    det = pd.read_csv(os.path.join(P.STEP3_CASE, "p3_detection.csv"))
    asr = attack_success(dmg, clean)
    rec = defence_recovery(dfn, clean)
    asr.to_csv(os.path.join(out, "s3_attack_success.csv"), index=False)
    rec.to_csv(os.path.join(out, "s3_defence_recovery.csv"), index=False)
    summ, tests = summarize(asr, rec, det, out)
    make_figure(summ, fig)
    IO.log("\n" + summ[["metric", "attack", "rate", "defense", "mean", "ci_lo", "ci_hi"]].round(3).to_string(index=False))
    IO.log("\n" + tests[["attack", "rate", "defense", "mean_delta", "p_holm", "effect"]].round(4).to_string(index=False))

    # ---- (ب) و(ج): ما أضافه الإطار في 2026-09-05 — التسميم على UAVIDS و ISOT ،
    #      وعلى النموذج المتكيّف مقابل غير المتكيّف. كلاهما قابل للاستئناف.
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--legacy-only", action="store_true", help="المقاييس المسمّاة من التليمتري فقط")
    ap.add_argument("--net-seeds", type=int, default=1 if P.QUICK else 3)
    ap.add_argument("--adapt-seeds", type=int, default=1 if P.QUICK else 5)
    a = ap.parse_args()
    if not a.legacy_only:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        import network_poison, adapted_poison
        network_poison.main(["--seeds", str(a.net_seeds)])
        adapted_poison.main(["--seeds", str(a.adapt_seeds)])
    IO.log("\n✓ المرحلة الثالثة مكتملة.")


if __name__ == "__main__":
    main()
