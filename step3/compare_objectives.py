# -*- coding: utf-8 -*-
# =====================================================================
#  مقارنة هدفَي المرحلة F: recall مقابل blend(AUC+recall)
#  Compare Stage-F objectives: recall vs blend(AUC + recall)
# ---------------------------------------------------------------------
#  السؤال: هل مكافأةُ جودة الكشف (AUC) في دالّة الهدف تُحسّن الدفاع فعلاً؟
#  المقياس الخارجي واحد في الحالتين (استدعاء فئة الانتحال) ⇒ قابل للمقارنة.
#  المخرجات: results/.../p3_objective_comparison.csv + figures/.../P8_*.png|pdf
# =====================================================================
import os, sys, json, numpy as np, pandas as pd
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon, spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import config as CFG

RES, FIG = CFG.RESULTS_DIR, CFG.FIGURES_DIR
C = {"grey": "#7A7A73", "red": "#D1495B", "green": "#2E8B6F", "blue": "#3B6EA5", "purple": "#5B4B8A"}
plt.rcParams.update({"font.size": 11, "font.family": "DejaVu Sans", "axes.titlesize": 12,
                     "axes.titleweight": "bold", "axes.spines.top": False,
                     "axes.spines.right": False, "axes.grid": True, "grid.alpha": .25,
                     "grid.linestyle": "--", "savefig.bbox": "tight"})


def load(name):
    p = os.path.join(RES, name)
    return pd.read_csv(p) if os.path.exists(p) else None


old = load("p3_defense_opt.csv")            # الهدف: recall (الأصلي)
new = load("p3_defense_opt_blend.csv")      # الهدف: blend (AUC + recall)
if old is None or new is None:
    sys.exit("ينقص أحد الملفّين — شغّلي run_defense_opt.py بالهدفين.")

for d, tag in [(old, "recall"), (new, "blend")]:
    d["objective_run"] = tag
print(f">> recall-run: {len(old)} صفّاً | blend-run: {len(new)} صفّاً\n")

# ---------------- (1) الملخّص لكل هدف ----------------
rows = []
for d, tag in [(old, "recall"), (new, "blend")]:
    dflt = d[d.optimizer == "default"]["score"]
    tuned = d[d.optimizer != "default"]["score"]
    rows.append({"objective": tag, "n_rows": len(d),
                 "default_mean": round(float(dflt.mean()), 4),
                 "tuned_mean": round(float(tuned.mean()), 4),
                 "tuned_best_per_fold": round(float(
                     d[d.optimizer != "default"].groupby("fold")["score"].max().mean()), 4),
                 "tuned_beats_default": bool(tuned.mean() > dflt.mean())})
summary = pd.DataFrame(rows)
print("=== الملخّص ===")
print(summary.to_string(index=False))

# ---------------- (2) الكاشف المختار ----------------
print("\n=== الكاشف الذي اختاره البحث ===")
det = []
for d, tag in [(old, "recall"), (new, "blend")]:
    t = d[d.optimizer != "default"]
    vc = t.detector.value_counts()
    for k, v in vc.items():
        sub = t[t.detector == k]
        det.append({"objective": tag, "detector": k, "picked": int(v),
                    "mean_recall": round(float(sub.score.mean()), 4),
                    "mean_auc": round(float(sub.auc.mean()), 4) if "auc" in sub else np.nan})
det = pd.DataFrame(det)
print(det.to_string(index=False))

# ---------------- (3) هل AUC الأعلى يعني دفاعاً أفضل؟ ----------------
print("\n=== العلاقة بين AUC والاستدعاء (داخل تشغيلة blend) ===")
t = new[(new.optimizer != "default") & new.auc.notna()]
rho = p = np.nan
if len(t) >= 4 and t.auc.std() > 0:
    rho, p = spearmanr(t.auc, t.score)
    print(f"Spearman ρ(AUC, SpoofRec) = {rho:+.3f}  (p={p:.3g}, n={len(t)})")
    print("⇒ " + ("ارتباط سالب معنوي: الكشف الأدقّ يعطي دفاعاً أسوأ."
                  if (p < .05 and rho < 0) else
                  "لا دليل على أن AUC الأعلى يعني دفاعاً أفضل."))

# ---------------- (4) اختبار مزدوج على الطيّات المشتركة ----------------
print("\n=== blend مقابل recall (أزواج طيّة×مُحسِّن×بذرة) ===")
key = ["fold", "optimizer", "opt_seed"]
m = old[old.optimizer != "default"].merge(
    new[new.optimizer != "default"], on=key, suffixes=("_recall", "_blend"))
stat = {}
if len(m) >= 5:
    try:
        w = wilcoxon(m.score_recall, m.score_blend)
        stat = {"n_pairs": int(len(m)), "recall_obj_mean": float(m.score_recall.mean()),
                "blend_obj_mean": float(m.score_blend.mean()), "wilcoxon_p": float(w.pvalue)}
        print(f"n={len(m)} | recall-obj={m.score_recall.mean():.4f} "
              f"vs blend-obj={m.score_blend.mean():.4f} | Wilcoxon p={w.pvalue:.4g}")
    except Exception as e:
        print("  تعذّر Wilcoxon:", e)

out = os.path.join(RES, "p3_objective_comparison.csv")
pd.concat([summary.assign(section="summary"), det.assign(section="detector")],
          ignore_index=True).to_csv(out, index=False)
json.dump({"spearman_auc_vs_recall": {"rho": None if rho != rho else float(rho),
                                      "p": None if p != p else float(p)},
           "paired": stat},
          open(os.path.join(RES, "p3_objective_stats.json"), "w", encoding="utf-8"),
          ensure_ascii=False, indent=2)

# ---------------- (5) الشكل P8 ----------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 5))

# لوحة A — متوسّط الاستدعاء لكل هدف مقابل الافتراضي
#  تسميات الأشكال بالإنجليزية فقط: matplotlib لا يشكّل العربية (تظهر مقلوبة)
labels = ["default\n(untuned)", "tuned\nobjective=recall", "tuned\nobjective=blend(AUC)"]
vals = [summary.loc[summary.objective == "recall", "default_mean"].iloc[0],
        summary.loc[summary.objective == "recall", "tuned_mean"].iloc[0],
        summary.loc[summary.objective == "blend", "tuned_mean"].iloc[0]]
bars = ax1.bar(range(3), vals, color=[C["grey"], C["blue"], C["purple"]],
               edgecolor="black", lw=.6, width=.6)
for b, v in zip(bars, vals):
    ax1.text(b.get_x() + b.get_width() / 2, v + .004, f"{v:.3f}", ha="center",
             fontweight="bold", fontsize=10)
ax1.axhline(vals[0], color=C["red"], ls="--", lw=1.4)
ax1.set_xticks(range(3)); ax1.set_xticklabels(labels, fontsize=9)
ax1.set_ylabel("GPS-Spoofing Recall after defense")
ax1.set_title("Tuning never beats the untuned default")

# لوحة B — AUC مقابل الاستدعاء (كل إعداد نقطة)
if len(t):
    for dname, col in [("activation_cluster", C["red"]), ("loss_deviation", C["green"]),
                       ("spectral", C["blue"]), ("knn_consistency", C["purple"])]:
        s = t[t.detector == dname]
        if len(s):
            ax2.scatter(s.auc, s.score, s=48, color=col, edgecolor="black",
                        lw=.5, alpha=.85, label=f"{dname} (n={len(s)})")
    ax2.set_xlabel("Detection ROC-AUC (higher = better detection)")
    ax2.set_ylabel("GPS-Spoofing Recall after defense")
    ttl = "Better detection ≠ better defense"
    if rho == rho:
        ttl += f"   (ρ={rho:+.2f}, p={p:.3g})"
    ax2.set_title(ttl)
    ax2.legend(fontsize=8, loc="upper right")

fig.suptitle("P8 — Does rewarding detection AUC improve the defense?  No.",
             fontweight="bold", fontsize=13)
plt.tight_layout(rect=[0, 0, 1, .94])
fig.savefig(os.path.join(FIG, "P8_objective_comparison.png"), dpi=300)
fig.savefig(os.path.join(FIG, "P8_objective_comparison.pdf"))
plt.close(fig)
print(f"\n✓ {os.path.relpath(out)}")
print(f"✓ {os.path.relpath(os.path.join(FIG, 'P8_objective_comparison.png'))}")
