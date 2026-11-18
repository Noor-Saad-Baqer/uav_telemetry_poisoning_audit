# -*- coding: utf-8 -*-
# =====================================================================
#  الشكل P9 — مدى الضغط الكامل وعتبة الانهيار
# ---------------------------------------------------------------------
#  كان هذا الشكل يُولَّد يدوياً خارج أيّ سكربت ⇒ بقي قديماً بعد إعادة التشغيل
#  ولم يكن قابلاً لإعادة الإنتاج. هذا الملفّ يُصلح ذلك.
#
#  المحور السيني = نسبة **الفئة المصدر** المقلوبة (لا نسبة مجموعة التدريب) ،
#  و1.0 = استنفاد الفئة بالكامل ⇒ الانهيار عندها حدّ منطقي لا نتيجة تجريبية.
# =====================================================================
import os, sys, numpy as np, pandas as pd
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as CFG
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 10.5, "font.family": "DejaVu Sans",
                     "axes.titlesize": 12, "axes.titleweight": "bold",
                     "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": .25, "grid.linestyle": "--",
                     "savefig.bbox": "tight"})
C = {"grey": "#7A7A73", "blue": "#3B6EA5", "purple": "#5B4B8A", "red": "#D1495B"}
LAB = {"none": "No defense", "filtered_retrain": "filtered_retrain",
       "relabel_retrain": "relabel_retrain (proposed)"}
COL = {"none": C["grey"], "filtered_retrain": C["blue"], "relabel_retrain": C["purple"]}

d = pd.read_csv(os.path.join(CFG.RESULTS_DIR, "p3_targeted_sweep.csv"))
#  المتوسّط داخل الطيّة عبر البذور ثم عبر الطيّات — وحدة الاستقلال هي الطيّة
per_fold = d.groupby(["defense", "rate", "fold"], as_index=False).spoof_recall.mean()
agg = per_fold.groupby(["defense", "rate"]).spoof_recall.agg(["mean", "std"]).reset_index()

fig, (axA, axB) = plt.subplots(1, 2, figsize=(13.5, 5.2))

# ---------- اللوحة A: المنحنى الكامل ----------
for dfs in ["none", "filtered_retrain", "relabel_retrain"]:
    s = agg[agg.defense == dfs].sort_values("rate")
    if s.empty:
        continue
    axA.errorbar(s.rate, s["mean"], yerr=s["std"].fillna(0), marker="o", ms=6,
                 lw=2.2, capsize=3, color=COL[dfs], label=LAB[dfs],
                 ls="--" if dfs == "none" else "-")
axA.axhline(.5, color="black", ls=":", lw=1.2)
axA.text(1.0, .52, "recall = 0.5 (breakdown)", ha="right", fontsize=8.5)
axA.set_xlabel("Fraction of the GPS-Spoofing class flipped")
axA.set_ylabel("GPS-Spoofing recall")
axA.set_ylim(-.03, 1.02)
axA.set_title("Defense triples the tolerable attack strength")
axA.legend(fontsize=9, loc="upper right", framealpha=.95)

# ---------- اللوحة B: المكسب ----------
base = agg[agg.defense == "none"].set_index("rate")["mean"]
w = 0.035
for i, dfs in enumerate(["filtered_retrain", "relabel_retrain"]):
    s = agg[agg.defense == dfs].sort_values("rate")
    if s.empty:
        continue
    gain = s["mean"].values - base.reindex(s.rate).values
    axB.bar(s.rate + (i - .5) * w, gain, width=w, color=COL[dfs],
            edgecolor="black", lw=.6, label=LAB[dfs])
axB.axhline(0, color="black", lw=1)
axB.set_xlabel("Fraction of the GPS-Spoofing class flipped")
axB.set_ylabel("Recall gain over no defense")
axB.set_title("Largest gain where the undefended model is already dead")
axB.legend(fontsize=9, framealpha=.95)

#  عتبة الانهيار لكل ذراع (أوّل نسبة ينزل عندها الاستدعاء تحت 0.5)
thr = {}
for dfs in ["none", "filtered_retrain", "relabel_retrain"]:
    s = agg[agg.defense == dfs].sort_values("rate")
    below = s[s["mean"] < .5]
    thr[dfs] = float(below.rate.iloc[0]) if len(below) else float("nan")
sub = "  |  ".join(f"{LAB[k]}: breaks at {v:.2f}" for k, v in thr.items())
fig.suptitle(f"P9 — Full attack range on the source class\n{sub}",
             fontweight="bold", fontsize=12)
plt.tight_layout(rect=[0, 0, 1, .90])
for ext in ("png", "pdf"):
    fig.savefig(os.path.join(CFG.FIGURES_DIR, f"P9_targeted_sweep.{ext}"), dpi=300)
print("عتبات الانهيار:", {k: round(v, 2) for k, v in thr.items()})
print("P9_targeted_sweep saved")
