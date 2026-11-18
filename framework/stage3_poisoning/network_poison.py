# -*- coding: utf-8 -*-
# =====================================================================
#  المرحلة الثالثة (ب) — التسميم داخل المجال على مسارَي الشبكة
#  Stage 3b — In-domain poisoning on UAVIDS-2025 and ISOT (non-adapted victim)
# ---------------------------------------------------------------------
#  لماذا: شكل الإطار (الطبقة الثانية) يذكر المجموعات الثلاث ، بينما step3 لم
#  يُسمِّم إلا التليمتري. هنا نطبّق **الهجمات الأربعة نفسها والدفاعين نفسيهما**
#  على قسم التطوير من UAVIDS و ISOT (المختوم لا يُلمس) بالتصنيف الموحَّد 11←5.
#
#  الهجوم الموجَّه: Spoofing/Identity ← Normal (نظير GPS Spoofing ← Normal في
#  التليمتري: الفئة الأمنية نفسها في التصنيف الموحَّد). النسبة من فئة المصدر.
#  البروتوكول: طبقيّ 70/30 × بذور (كما في المرحلة الرابعة لمسارات الشبكة).
#  الضحيّة: غابة 200 شجرة بخسارة OOB. الميزات مُطبَّعة على التدريب فقط
#  (المُحفِّز يُعرَّف بوحدات σ كما في step3).
#
#  التشغيل: python framework/stage3_poisoning/network_poison.py [--seeds 3]
#  المخرجات: results/stage3/s3_network_poison.csv (صفّ لكلّ تشغيل ، قابل للاستئناف)
#            results/stage3/s3_network_summary.csv ، figures/stage3/S3_network_poison.png
# =====================================================================
import os, sys, argparse, time, numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
FW = os.path.dirname(HERE)
sys.path.insert(0, FW); sys.path.insert(0, HERE)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from common import paths as P, io as IO, stats as ST
from common import data as D
import poison_lib as L

TRACKS = ["uavids", "isot"]
ATTACKS = ["random_flip", "targeted_flip", "backdoor", "clean_label"]
RATES = [0.05, 0.10, 0.20, 0.30, 0.40]          # سلّم step3 نفسه
DEF_RATES = [0.10, 0.20, 0.30]
DEF_ATTACKS = ["targeted_flip", "backdoor"]
DEFENCES = ["filtered_retrain", "relabel_retrain", "filtered_retrain_v2", "relabel_retrain_v2"]   # v2 = أرضية MAD (انظري poison_lib)
SRC_NAME, DST_NAME = "Spoofing/Identity", "Normal"
if P.QUICK:
    RATES, DEF_RATES, DEF_ATTACKS = [0.10, 0.30], [0.30], ["targeted_flip"]


def split(d, seed):
    idx = np.arange(len(d["y"]))
    tr, te = train_test_split(idx, test_size=0.3, random_state=seed, stratify=d["y"])
    sc = StandardScaler().fit(d["X"][tr])
    return (sc.transform(d["X"][tr]).astype(np.float32), d["y"][tr],
            sc.transform(d["X"][te]).astype(np.float32), d["y"][te])


def run_track(track, seeds, sink):
    d = D.load_track(track, "dev", unified=True)
    nc, cls = len(d["classes"]), d["classes"]
    src, dst = cls.index(SRC_NAME), cls.index(DST_NAME)
    chance = 1.0 / nc
    for seed in seeds:
        Xtr, ytr, Xte, yte = split(d, seed)
        base_key = dict(track=track, seed=seed, attack="clean", rate=0.0, defence="none")
        if not sink.has(**base_key):
            t0 = time.time()
            m = L.RFVictim(nc, seed).fit(Xtr, ytr)
            r = L.scores(yte, m.predict(Xte), src, nc)
            sink.add(dict(**base_key, n_train=len(ytr), n_poison=0, chance=chance, **r,
                          bsr=L.backdoor_success(m.predict, Xte, yte, dst),
                          det_auc=np.nan, det_auc_oriented=np.nan, fit_s=time.time() - t0))
            IO.log(f"   [{track}] seed{seed} clean: bal={r['balacc']:.3f} tgt_rec={r['target_recall']:.3f}")
        for atk in ATTACKS:
            for rate in RATES:
                key = dict(track=track, seed=seed, attack=atk, rate=rate, defence="none")
                need_def = atk in DEF_ATTACKS and rate in DEF_RATES
                missing_def = [d for d in DEFENCES if need_def and not sink.has(**dict(key, defence=d))]
                if sink.has(**key) and not missing_def:
                    continue
                t0 = time.time()
                Xp, yp, isp = L.poison(atk, Xtr, ytr, rate, seed, nc, src, dst)
                m = L.RFVictim(nc, seed).fit(Xp, yp)
                a, ao = L.detection_auc(L.loss_deviation(m, Xp, yp), isp)
                if not sink.has(**key):
                    r = L.scores(yte, m.predict(Xte), src, nc)
                    sink.add(dict(**key, n_train=len(yp), n_poison=int(isp.sum()), chance=chance, **r,
                                  bsr=L.backdoor_success(m.predict, Xte, yte, dst),
                                  det_auc=a, det_auc_oriented=ao, fit_s=time.time() - t0))
                    IO.log(f"   [{track}] seed{seed} {atk:14s} r={rate:.2f}: bal={r['balacc']:.3f} "
                           f"tgt_rec={r['target_recall']:.3f} auc={ao:.2f} ({time.time() - t0:.0f}s)")
                if missing_def:
                    for dfn in missing_def:
                        dkey = dict(key, defence=dfn)
                        t0 = time.time()
                        md = L.defend(dfn, Xp, yp, nc, seed, base_model=m)
                        rd = L.scores(yte, md.predict(Xte), src, nc)
                        sink.add(dict(**dkey, n_train=len(yp), n_poison=int(isp.sum()), chance=chance, **rd,
                                      bsr=L.backdoor_success(md.predict, Xte, yte, dst),
                                      det_auc=a, det_auc_oriented=ao, fit_s=time.time() - t0))
                        IO.log(f"      + {dfn:16s}: bal={rd['balacc']:.3f} tgt_rec={rd['target_recall']:.3f}")


def summarize(df, out):
    """المقاييس المسمّاة نفسها التي يشتقّها run.py للتليمتري ، مزاوجة على (track, seed)."""
    clean = df[df.attack == "clean"].set_index(["track", "seed"])
    rows = []
    for _, r in df[df.attack != "clean"].iterrows():
        c = clean.loc[(r.track, r.seed)]
        bc, rc = c.balacc, c.target_recall
        rows.append(dict(track=r.track, seed=r.seed, attack=r.attack, rate=r.rate, defence=r.defence,
                         bal_clean=bc, bal=r.balacc, target_recall_clean=rc, target_recall=r.target_recall,
                         asr_balacc_drop=L.clip01((bc - r.balacc) / (bc - r.chance)) if bc > r.chance else np.nan,
                         asr_target_recall_drop=L.clip01(1 - r.target_recall / rc) if rc > 0 else np.nan,
                         bsr=r.bsr, bsr_clean_baseline=c.bsr, det_auc_oriented=r.det_auc_oriented))
    per = pd.DataFrame(rows)
    none = per[per.defence == "none"].set_index(["track", "seed", "attack", "rate"])
    rec, rec_t, bsr_red = [], [], []
    for _, r in per[per.defence != "none"].iterrows():
        n0 = none.loc[(r.track, r.seed, r.attack, r.rate)]
        den = r.bal_clean - n0.bal
        rec.append(L.clip01((r.bal - n0.bal) / den) if den > 1e-6 else np.nan)
        den_t = r.target_recall_clean - n0.target_recall          # استعادة الفئة المستهدَفة نفسها
        rec_t.append(L.clip01((r.target_recall - n0.target_recall) / den_t) if den_t > 1e-6 else np.nan)
        bsr_red.append(L.clip01(1 - r.bsr / n0.bsr) if np.isfinite(n0.bsr) and n0.bsr > 0 else np.nan)
    m = per.defence != "none"
    per.loc[m, "recovery"], per.loc[m, "recovery_target"], per.loc[m, "bsr_reduction"] = rec, rec_t, bsr_red
    per["bal_delta_vs_none"] = np.nan
    per.loc[m, "bal_delta_vs_none"] = [r.bal - none.loc[(r.track, r.seed, r.attack, r.rate)].bal for _, r in per[m].iterrows()]
    g = per.groupby(["track", "attack", "rate", "defence"])
    summ = []
    for k, s in g:
        row = dict(zip(["track", "attack", "rate", "defence"], k), n=len(s))
        for col in ["bal", "target_recall", "asr_balacc_drop", "asr_target_recall_drop", "bsr",
                    "bsr_clean_baseline", "det_auc_oriented", "recovery", "recovery_target", "bsr_reduction", "bal_delta_vs_none"]:
            v = s[col].to_numpy(float)
            if np.isfinite(v).sum() == 0:
                continue
            m, lo, hi = ST.boot_ci(v[np.isfinite(v)])
            row[f"{col}_mean"], row[f"{col}_lo"], row[f"{col}_hi"] = m, lo, hi
        summ.append(row)
    summ = pd.DataFrame(summ)
    per.to_csv(os.path.join(out, "s3_network_named_metrics.csv"), index=False)
    summ.to_csv(os.path.join(out, "s3_network_summary.csv"), index=False)
    return summ


def make_figure(summ, out_fig):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    cols = {"random_flip": "#3B6FD6", "targeted_flip": "#B22222", "backdoor": "#6A3FB5", "clean_label": "#2E8B57"}
    for ax, (col, ttl) in zip(axes, [("asr_balacc_drop", "Attack success (skill destroyed)"),
                                     ("asr_target_recall_drop", "Spoofing/Identity recall hidden"),
                                     ("bsr", "Backdoor success rate")]):
        for track, ls in (("uavids", "-"), ("isot", "--")):
            s = summ[(summ.track == track) & (summ.defence == "none")]
            for atk in ATTACKS:
                a = s[s.attack == atk].sort_values("rate")
                if col == "bsr" and atk != "backdoor":
                    continue
                if f"{col}_mean" not in a or a[f"{col}_mean"].isna().all():
                    continue
                ax.plot(a.rate, a[f"{col}_mean"], ls, marker="o", ms=4, color=cols[atk], label=f"{atk} · {track}")
                ax.fill_between(a.rate, a[f"{col}_lo"], a[f"{col}_hi"], color=cols[atk], alpha=0.12)
        ax.set_title(ttl, fontsize=11, weight="bold"); ax.set_xlabel("poison rate"); ax.set_ylim(-0.02, 1.05)
        ax.grid(alpha=0.3); ax.legend(fontsize=7)
    fig.suptitle("Stage 3b — in-domain poisoning on the network tracks (UAVIDS solid, ISOT dashed; RF victim, dev partition)", fontsize=11)
    fig.tight_layout(); fig.savefig(os.path.join(out_fig, "S3_network_poison.png"), dpi=200); plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=1 if P.QUICK else 3)
    a = ap.parse_args(argv)
    out, fig = P.results("stage3"), P.figures("stage3")
    sink = IO.Sink(os.path.join(out, "s3_network_poison.csv"), ["track", "seed", "attack", "rate", "defence"])
    IO.log(f">> المرحلة 3ب — التسميم داخل المجال على الشبكة (بذور={a.seeds} ، quick={P.QUICK})")
    for tr in TRACKS:
        run_track(tr, list(range(a.seeds)), sink)
    summ = summarize(sink.df(), out)
    make_figure(summ, fig)
    IO.log(f"✓ 3ب: {len(sink)} صفّاً → s3_network_poison.csv / s3_network_summary.csv")
    return summ


if __name__ == "__main__":
    main()
