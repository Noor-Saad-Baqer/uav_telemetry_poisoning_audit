# -*- coding: utf-8 -*-
# =====================================================================
#  تجميع سُلّم المعماريّات المنشورة + الشكل P11 (RF مقابل LSTM–GRU)
# ---------------------------------------------------------------------
#  الرسالة: البروتوكول يهيمن على الرقم **بغضّ النظر عن عائلة النموذج**.
#  خطّان يهبطان معاً على نفس السُّلّم ⇒ لا يمكن ردّ الفارق إلى «نموذجكم أضعف».
#  يعمل على نتائج جزئية أيضاً (يُظهر الدرجات المكتملة فقط).
# =====================================================================
import os, sys, numpy as np, pandas as pd
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import config as CFG
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 10.5, "font.family": "DejaVu Sans",
                     "axes.titlesize": 12, "axes.titleweight": "bold",
                     "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": .25, "grid.linestyle": "--",
                     "savefig.bbox": "tight"})
C = {"red": "#D1495B", "green": "#2E8B6F", "blue": "#3B6EA5",
     "grey": "#7A7A73", "amber": "#E0A03F", "purple": "#5B4B8A"}

RES = CFG.RESULTS_DIR
#  التشغيل مشظّى (ملف لكل معماريّة) — نجمع كل الشظايا ثم نُوحّد التكرار في الذاكرة
import glob
_files = [f for f in glob.glob(os.path.join(RES, "p3_published_ladder*.csv"))
          if not (f.endswith("_agg.csv") or f.endswith("_combined.csv"))]
pub = pd.concat([pd.read_csv(f) for f in _files], ignore_index=True)
pub = pub.drop_duplicates(subset=["model", "step", "fold", "seed"], keep="first")
print(f">> {len(_files)} ملف نتائج ، {len(pub)} صفّاً فريداً من 126\n")
rf = pd.read_csv(os.path.join(RES, "p3_protocol_ladder.csv"))
rf["step"] = range(1, len(rf) + 1)

#  عدد الطيّات المتوقّع لكل درجة (لتمييز الدرجة المكتملة من الجزئية)
EXPECT = {1: 1, 2: 1, 3: 4, 4: 4, 5: 4}
SEEDS = pub.seed.nunique()

#  متوسّط داخل الطيّة أوّلاً (عبر البذور) ثم عبر الطيّات — وحدة الاستقلال هي الطيّة
per_fold = (pub.groupby(["model", "step", "fold"], as_index=False)
              [["bal_acc", "raw_acc", "spoof_recall"]].mean())
agg = (per_fold.groupby(["model", "step"], as_index=False)
       .agg(bal_acc=("bal_acc", "mean"), raw_acc=("raw_acc", "mean"),
            spoof_recall=("spoof_recall", "mean"), n_folds=("fold", "nunique")))
agg["complete"] = agg.apply(lambda r: r.n_folds >= EXPECT[int(r.step)], axis=1)

STEP_LABEL = {1: "1. as published (folder labels, all feats, random split)",
              2: "2. + temporal split", 3: "3. + leave-one-airframe-out",
              4: "4. + window-level labels", 5: "5. + drop label-defining features"}
agg["step_name"] = agg.step.map(STEP_LABEL)

out = os.path.join(RES, "p3_published_ladder_agg.csv")
agg.to_csv(out, index=False)

print("=" * 90)
print(f"سُلّم المعماريّات المنشورة  (بذور={SEEDS}؛ المتوسّط داخل الطيّة ثم عبرها)")
print("=" * 90)
for m in ["lstm_gru", "lstm", "gru"]:
    sub = agg[agg.model == m].sort_values("step")
    if sub.empty:
        continue
    print(f"\n### {m}")
    for r in sub.itertuples():
        flag = "" if r.complete else f"  [جزئي {r.n_folds}/{EXPECT[int(r.step)]} طيّة]"
        print(f"  {STEP_LABEL[int(r.step)]:52s} raw={r.raw_acc:.3f}  "
              f"BalAcc={r.bal_acc:.3f}  spoof={r.spoof_recall:.3f}{flag}")

# ---------------------------------------------------------------------
#  جدول الورقة: RF مقابل LSTM–GRU على نفس السُّلّم
# ---------------------------------------------------------------------
hy = agg[(agg.model == "lstm_gru")].set_index("step")
print("\n" + "=" * 90)
print("جدول الورقة — نفس البروتوكول ، عائلتا نموذج مختلفتان")
print("=" * 90)
print(f"{'step':52s} {'RF BalAcc':>10s} {'LSTM-GRU':>10s} {'RF spoof':>9s} {'LG spoof':>9s}")
rows = []
for s in range(1, 6):
    r = rf[rf.step == s]
    if r.empty:
        continue
    r = r.iloc[0]
    h = hy.loc[s] if s in hy.index else None
    hb = f"{h.bal_acc:.3f}" if h is not None else "—"
    hs = f"{h.spoof_recall:.3f}" if h is not None else "—"
    mark = "" if (h is not None and h.complete) else " *"
    print(f"{STEP_LABEL[s]:52s} {r.bal_acc:10.3f} {hb:>10s} "
          f"{r.spoof_recall:9.3f} {hs:>9s}{mark}")
    rows.append({"step": s, "step_name": STEP_LABEL[s],
                 "rf_bal_acc": r.bal_acc, "rf_spoof_recall": r.spoof_recall,
                 "lstmgru_bal_acc": (h.bal_acc if h is not None else np.nan),
                 "lstmgru_spoof_recall": (h.spoof_recall if h is not None else np.nan),
                 "lstmgru_complete": bool(h.complete) if h is not None else False})
pd.DataFrame(rows).to_csv(os.path.join(RES, "p3_ladder_combined.csv"), index=False)
print("\n* = درجة لم تكتمل طيّاتها بعد")

# ---------------------------------------------------------------------
#  الشكل P11
# ---------------------------------------------------------------------
done = [s for s in range(1, 6) if s in hy.index and bool(hy.loc[s].complete)]
if len(done) >= 3:
    LAB = ["(1)\nas published\nrandom split", "(2)\n+ temporal\nsplit",
           "(3)\n+ leave-one-\nairframe-out", "(4)\n+ window-level\nlabels",
           "(5)\n+ drop label-\ndefining feats"]
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(14.5, 5.6))
    xs = np.array(done) - 1
    rfb = rf.set_index("step").loc[done, "bal_acc"].values
    hyb = hy.loc[done, "bal_acc"].values
    rfs = rf.set_index("step").loc[done, "spoof_recall"].values
    hys = hy.loc[done, "spoof_recall"].values

    for ax, a, b, ylab, ttl in [
            (axA, rfb, hyb, "Balanced Accuracy",
             "Both model families collapse on the same ladder"),
            (axB, rfs, hys, "GPS-Spoofing recall",
             "Label granularity, not architecture, drives spoof recall")]:
        ax.plot(xs, a, "o-", color=C["blue"], lw=2.2, ms=8, label="Random Forest (summary feats)")
        ax.plot(xs, b, "s--", color=C["purple"], lw=2.2, ms=8, label="Hybrid LSTM–GRU (raw sequence)")
        #  كل تسمية على الجهة البعيدة عن السلسلة الأخرى ⇒ تبقى ملاصقة لنقطتها هي
        #  (لو ثبّتنا RF فوق دائماً لأوهم الشكل بنسبة الرقم للسلسلة الخطأ حين تتقاطعان)
        for j, (x, va, vb) in enumerate(zip(xs, a, b)):
            ha = "left" if j == 0 else ("right" if j == len(xs) - 1 else "center")
            dx = .06 if j == 0 else (-.06 if j == len(xs) - 1 else 0)
            up = va >= vb
            ax.text(x + dx, va + (.042 if up else -.085), f"{va:.3f}",
                    ha=ha, fontsize=9, color=C["blue"])
            ax.text(x + dx, vb + (-.085 if up else .042), f"{vb:.3f}",
                    ha=ha, fontsize=9, color=C["purple"])
        ax.axhline(1 / 3, color="black", ls=":", lw=1.1)
        ax.text(len(done) - 1.05, 1 / 3 + .02, "chance 0.333", ha="right", fontsize=8)
        ax.set_xticks(xs); ax.set_xticklabels([LAB[i] for i in xs], fontsize=8.5)
        ax.set_ylabel(ylab); ax.set_ylim(0, 1.18); ax.set_title(ttl)
        ax.legend(loc="lower left", fontsize=9, framealpha=.95)

    fig.suptitle("P11 — The published architecture reproduces ~0.99, then falls with the protocol",
                 fontweight="bold", fontsize=12.5)
    plt.tight_layout(rect=[0, 0, 1, .93])
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(CFG.FIGURES_DIR, f"P11_published_ladder.{ext}"), dpi=300)
    print(f"\n✓ P11_published_ladder  (درجات مكتملة: {done})")
else:
    print(f"\n… الشكل مؤجّل: الدرجات المكتملة {done} (يلزم ≥3)")
print(f"✓ {os.path.relpath(out)}")
