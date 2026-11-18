# -*- coding: utf-8 -*-
# =====================================================================
#  الشكل P10 — سُلّم الصرامة: أين تذهب كل نقطة بين 1.00 و0.78
# ---------------------------------------------------------------------
#  اللوحة A: Balanced Accuracy عبر الدرجات الخمس + أثر كل عامل.
#  اللوحة B: استدعاء فئة الانتحال — يُظهر أن وسم المجلّد **يُخفي** إشارة
#            حقيقية (0.177 → 1.000) لا يضخّمها.
#  التسميات إنجليزية: matplotlib لا يشكّل العربية.
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
C = {"red": "#D1495B", "green": "#2E8B6F", "blue": "#3B6EA5",
     "grey": "#7A7A73", "amber": "#E0A03F", "purple": "#5B4B8A"}

d = pd.read_csv(os.path.join(CFG.RESULTS_DIR, "p3_protocol_ladder.csv"))
LAB = ["(1)\nas published\nrandom split",
       "(2)\n+ temporal\nsplit",
       "(3)\n+ leave-one-\nairframe-out",
       "(4)\n+ window-level\nlabels",
       "(5)\n+ drop label-\ndefining feats"]
COL = [C["red"], C["amber"], C["grey"], C["blue"], C["green"]]

fig, (axA, axB) = plt.subplots(1, 2, figsize=(14.5, 5.8))

# ---------- اللوحة A — Balanced Accuracy ----------
ba = d.bal_acc.values
bars = axA.bar(range(5), ba, color=COL, edgecolor="black", lw=.7, width=.62)
for i, v in enumerate(ba):
    axA.text(i, v + .018, f"{v:.3f}", ha="center", fontweight="bold", fontsize=11)
#  أسهم أثر كل عامل
for i in range(1, 5):
    delta = ba[i] - ba[i - 1]
    y = max(ba[i - 1], ba[i]) + .10
    axA.annotate("", xy=(i, y), xytext=(i - 1, y),
                 arrowprops=dict(arrowstyle="->", lw=1.5,
                                 color=C["red"] if delta < 0 else C["green"]))
    axA.text(i - .5, y + .015, f"{delta:+.3f}", ha="center", fontsize=9.5,
             fontweight="bold", color=C["red"] if delta < 0 else C["green"])
axA.axhline(1 / 3, color="black", ls=":", lw=1.2)
axA.text(4.45, 1 / 3 + .015, "chance 0.333", ha="right", fontsize=8)
axA.set_xticks(range(5)); axA.set_xticklabels(LAB, fontsize=8.5)
axA.set_ylabel("Balanced Accuracy")
axA.set_ylim(0, 1.32)
axA.set_title("Two factors inflate, one hides real signal")

# ---------- اللوحة B — استدعاء فئة الانتحال ----------
sr = d.spoof_recall.values
bars = axB.bar(range(5), sr, color=COL, edgecolor="black", lw=.7, width=.62)
for i, v in enumerate(sr):
    axB.text(i, v + .018, f"{v:.3f}", ha="center", fontweight="bold", fontsize=11)
#  إبراز القفزة الحاسمة 3 → 4
axB.annotate("", xy=(3, 1.06), xytext=(2, 1.06),
             arrowprops=dict(arrowstyle="->", lw=2.4, color=C["green"]))
axB.text(2.5, 1.09, f"{sr[3]-sr[2]:+.3f}\nlabel granularity", ha="center", fontsize=9.5,
         fontweight="bold", color=C["green"])
axB.set_xticks(range(5)); axB.set_xticklabels(LAB, fontsize=8.5)
axB.set_ylabel("GPS-Spoofing recall")
axB.set_ylim(0, 1.32)
axB.set_title("Folder-level labels HIDE cross-platform detectability")

fig.suptitle("P10 — Protocol rigor ladder: 1.000 is reproducible, 0.782 is honest "
             "(same model throughout: Random Forest)",
             fontweight="bold", fontsize=12.5)
plt.tight_layout(rect=[0, 0, 1, .93])
for ext in ("png", "pdf"):
    fig.savefig(os.path.join(CFG.FIGURES_DIR, f"P10_protocol_ladder.{ext}"), dpi=300)
print("✓ P10_protocol_ladder saved")
