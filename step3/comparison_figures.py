# -*- coding: utf-8 -*-
# =====================================================================
#  أشكال المرحلة الثالثة — كل الأشكال في ملف واحد (P1..P7)
#  Step 3 figures: PNG 300dpi + PDF متجهي ، تسميات إنجليزية (للأطروحة)
# ---------------------------------------------------------------------
#  P1 منحنى الضرر   P2 خريطة الكشف   P3 استعادة الدفاع   P4 مصفوفات الالتباس
#  P5 مقارنة المُحسِّنات   P6 مخطّط الفرق الحرج   P7 التعميم عبر المنصّات
#  كل شكل مستقل (try/except) — نقص ملف نتائج لا يوقف البقيّة.
# =====================================================================
import os, sys, json, numpy as np, pandas as pd
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import config as CFG

RES, FIG = CFG.RESULTS_DIR, CFG.FIGURES_DIR
os.makedirs(FIG, exist_ok=True)
plt.rcParams.update({"font.size": 11, "font.family": "DejaVu Sans", "axes.titlesize": 12,
                     "axes.titleweight": "bold", "axes.spines.top": False,
                     "axes.spines.right": False, "axes.grid": True, "grid.alpha": .25,
                     "grid.linestyle": "--", "figure.dpi": 150, "savefig.bbox": "tight"})
C = {"grey": "#7A7A73", "red": "#D1495B", "green": "#2E8B6F", "blue": "#3B6EA5",
     "purple": "#5B4B8A", "orange": "#E07A3F"}
ACOL = {"random_flip": C["blue"], "targeted_flip": C["red"],
        "backdoor": C["purple"], "clean_label": C["orange"]}
NICE = {"random_flip": "Random label flip", "targeted_flip": "Targeted flip (Spoof→Normal)",
        "backdoor": "Backdoor trigger", "clean_label": "Clean-label"}
CHANCE = 1.0 / CFG.NUM_CLASSES


def rd(n):
    p = os.path.join(RES, n)
    return pd.read_csv(p) if os.path.exists(p) else None


def save(fig, name):
    fig.savefig(os.path.join(FIG, name + ".png"), dpi=300)
    fig.savefig(os.path.join(FIG, name + ".pdf"))
    plt.close(fig)
    print(f"   ✓ {name}")


# =====================================================================
#  P1 — منحنى الضرر
# =====================================================================
try:
    d = rd("p3_damage_curve.csv")
    if d is None: raise FileNotFoundError("p3_damage_curve.csv")
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    for ax, metric, title in [
            (axes[0], "balacc", "Overall Balanced Accuracy"),
            (axes[1], "spoof_recall", "GPS-Spoofing Recall  (security-critical)"),
            (axes[2], "asr", "Backdoor Attack Success Rate")]:
        for atk in d.attack.unique():
            s = d[d.attack == atk].groupby("rate")[metric].agg(["mean", "std"]).reset_index()
            if s["mean"].isna().all():
                continue
            ax.errorbar(s["rate"] * 100, s["mean"], yerr=s["std"], marker="o", ms=5, lw=2,
                        capsize=3, color=ACOL.get(atk, C["grey"]), label=NICE.get(atk, atk))
        #  ⚠️ النسبة للهجمات أحادية الفئة محسوبة من **الفئة المصدر** لا من
        #  مجموعة التدريب (RATE_BASIS="source_class") — تسميةٌ خاطئة هنا تُعيد
        #  إنتاج الالتباس نفسه الذي يحذّر منه الفصل.
        ax.set_xlabel("Poison rate (% of source class for targeted/clean-label)")
        ax.set_title(title)
        ax.set_ylim(-.03, 1.05)
        if metric != "asr":
            ax.axhline(CHANCE, color="black", ls=":", lw=1.2)
            ax.text(1, CHANCE + .02, f"chance {CHANCE:.2f}", fontsize=8)
    axes[0].set_ylabel("Balanced Accuracy")
    axes[1].set_ylabel("Recall of GPS Spoofing")
    axes[2].set_ylabel("Attack Success Rate")
    axes[0].legend(fontsize=8, loc="lower left")
    fig.suptitle("P1 — Poisoning damage: aggregate accuracy hides the targeted attack",
                 fontweight="bold", fontsize=13)
    plt.tight_layout(rect=[0, 0, 1, .94])
    save(fig, "P1_damage_curve")
except Exception as e:
    print("   … P1:", e)

# =====================================================================
#  P2 — خريطة الكشف (خام مقابل مُوجَّه)
# =====================================================================
try:
    d = rd("p3_detection.csv")
    if d is None: raise FileNotFoundError("p3_detection.csv")
    dets = [x for x in CFG.DETECTORS if x in set(d.detector)]
    atks = [a for a in CFG.ATTACKS if a in set(d.attack)]
    fig, axes = plt.subplots(1, 2, figsize=(15, 4.4))
    for ax, col, title in [(axes[0], "auc", "Raw ROC-AUC  (convention: >0.5 = works)"),
                           (axes[1], "auc_oriented", "Oriented AUC = max(AUC, 1−AUC)")]:
        M = np.array([[d[(d.attack == a) & (d.detector == t)][col].mean() for t in dets]
                      for a in atks], dtype=float)
        im = ax.imshow(M, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
        ax.set_xticks(range(len(dets))); ax.set_xticklabels(dets, rotation=25, ha="right", fontsize=9)
        ax.set_yticks(range(len(atks))); ax.set_yticklabels([NICE.get(a, a) for a in atks], fontsize=9)
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                v = M[i, j]
                if v == v:
                    ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=9,
                            fontweight="bold", color="black")
        ax.set_title(title); ax.grid(False)
        fig.colorbar(im, ax=ax, fraction=.035)
    fig.suptitle("P2 — Detector performance. Raw AUC < 0.5 means the signal is INVERTED, "
                 "not absent:\nfiltering on it removes clean samples.",
                 fontweight="bold", fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, .9])
    save(fig, "P2_detection_heatmap")
except Exception as e:
    print("   … P2:", e)

# =====================================================================
#  P3 — استعادة الدفاع
# =====================================================================
try:
    d = rd("p3_defense.csv")
    if d is None: raise FileNotFoundError("p3_defense.csv")
    atks = sorted(set(d.attack))
    fig, axes = plt.subplots(1, len(atks), figsize=(7.2 * len(atks), 5), squeeze=False)
    defs_ = [x for x in (["none"] + CFG.DEFENSES) if x in set(d.defense)]
    for ax, atk in zip(axes[0], atks):
        sub = d[d.attack == atk]
        metric = "asr" if atk == "backdoor" else "spoof_recall"
        rates = sorted(sub.rate.unique())
        w = .8 / max(1, len(defs_))
        for k, dv in enumerate(defs_):
            mu = [sub[(sub.defense == dv) & (sub.rate == r)][metric].mean() for r in rates]
            sd = [sub[(sub.defense == dv) & (sub.rate == r)][metric].std() for r in rates]
            xs = np.arange(len(rates)) + k * w
            #  لوحة ألوان بعدد الدفاعات: كانت تدور %4 فيأخذ الدفاع الخامس
            #  (relabel_retrain) رمادَ «بلا دفاع» نفسه ⇒ الفائز يلتبس بخطّ الأساس.
            palette = [C["grey"], C["blue"], C["green"], C["purple"], C["orange"], C["red"]]
            ax.bar(xs, mu, w, yerr=sd, capsize=3, label=dv,
                   color=palette[k % len(palette)],
                   edgecolor="black", lw=.5)
        ax.set_xticks(np.arange(len(rates)) + .4 - w / 2)
        ax.set_xticklabels([f"{int(r*100)}%" for r in rates])
        ax.set_xlabel("Poison rate")
        ax.set_ylabel("Attack Success Rate (lower=better)" if atk == "backdoor"
                      else "GPS-Spoofing Recall (higher=better)")
        ax.set_title(NICE.get(atk, atk)); ax.set_ylim(0, 1.05)
        ax.legend(fontsize=8)
    fig.suptitle("P3 — Defense recovery on the security-critical metric",
                 fontweight="bold", fontsize=13)
    plt.tight_layout(rect=[0, 0, 1, .93])
    save(fig, "P3_defense_recovery")
except Exception as e:
    print("   … P3:", e)

# =====================================================================
#  P7 — التعميم عبر المنصّات (طيّات LOAO)
# =====================================================================
try:
    d = rd("p3_clean_baseline.csv")
    if d is None: raise FileNotFoundError("p3_clean_baseline.csv")
    piv = d.pivot_table(index="fold", columns="model", values="balacc", aggfunc="mean")
    fig, ax = plt.subplots(figsize=(9, 4.8))
    x = np.arange(len(piv.index)); w = .8 / len(piv.columns)
    for k, m in enumerate(piv.columns):
        ax.bar(x + k * w, piv[m].values, w, label=m, edgecolor="black", lw=.5,
               color=[C["blue"], C["green"], C["purple"]][k % 3])
    ax.axhline(CHANCE, color="black", ls=":", lw=1.3)
    ax.text(len(x) - .5, CHANCE + .02, f"chance {CHANCE:.2f}", ha="right", fontsize=8)
    ax.set_xticks(x + .4 - w / 2)
    ax.set_xticklabels([s.replace("PX4-", "") for s in piv.index], rotation=12)
    ax.set_ylabel("Balanced Accuracy"); ax.set_ylim(0, 1.05)
    ax.set_title("P7 — Clean baseline, leave-one-airframe-out (held-out platform)")
    ax.legend(fontsize=9)
    plt.tight_layout()
    save(fig, "P7_airframe_generalization")
except Exception as e:
    print("   … P7:", e)

# =====================================================================
#  P5 + P6 — مقارنة المُحسِّنات ومخطّط الفرق الحرج (المرحلة F)
# =====================================================================
try:
    d = rd("p3_defense_opt.csv")
    if d is None: raise FileNotFoundError("p3_defense_opt.csv (شغّلي run_defense_opt.py)")
    s = d.groupby("optimizer")["score"].agg(["mean", "std"]).sort_values("mean")
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ax.barh(s.index, s["mean"], xerr=s["std"], capsize=3, color=C["purple"],
            edgecolor="black", lw=.5)
    base = d[d.optimizer == "default"]["score"].mean() if "default" in set(d.optimizer) else None
    if base is not None:
        ax.axvline(base, color=C["red"], ls="--", lw=1.5, label=f"default hyper-params = {base:.3f}")
        ax.legend(fontsize=9, loc="lower right", framealpha=.95)
    ax.set_xlabel("GPS-Spoofing Recall after defense (higher = better)")
    ax.set_title("P5 — Metaheuristic tuning of the defense (Stage F)")
    #  ملاحظة أمانة: إن كان التشغيل جزئياً، اذكريه على الشكل نفسه
    _st = os.path.join(RES, "p3_opt_stats.json")
    if os.path.exists(_st):
        _s = json.load(open(_st, encoding="utf-8"))
        if _s.get("partial_run"):
            ax.text(.5, -.20, f"Partial run: {_s['rows_completed']}/{_s['rows_planned']} configurations "
                              f"({len(_s.get('folds_covered', []))} of 4 airframes); "
                              f"{_s.get('n_blocks', 0)} complete Friedman blocks.",
                    transform=ax.transAxes, ha="center", fontsize=8, color=C["grey"], style="italic")
    plt.tight_layout()
    save(fig, "P5_optimizer_comparison")

    st = os.path.join(RES, "p3_opt_stats.json")
    if os.path.exists(st):
        stats = json.load(open(st, encoding="utf-8"))
        ranks = stats.get("avg_rank", {}); CD = stats.get("CD")
        if ranks and CD:
            #  مخطّط الفرق الحرج القياسي (Demšar 2006): محور الرتب المتوسّطة ،
            #  والخطوط السميكة تصل المُحسِّنات التي لا تختلف معنوياً (فرق الرتبة < CD).
            names = sorted(ranks, key=ranks.get)            # الأفضل (أقلّ رتبة) أوّلاً
            pos = [ranks[n] for n in names]
            lo, hi = min(pos), max(pos)
            pad = max(0.5, 0.15 * (hi - lo + 1e-9))
            axis_lo, axis_hi = lo - pad, hi + pad

            #  إيجاد الزُمَر: أطول متتاليات لا يتجاوز مداها CD
            cliques = []
            for i in range(len(names)):
                j = i
                while j + 1 < len(names) and pos[j + 1] - pos[i] <= CD:
                    j += 1
                if j > i and not any(a <= i and j <= b for a, b in cliques):
                    cliques.append((i, j))

            #  الاصطلاح: الرتبة الأفضل (الأصغر) إلى اليسار — بلا قلب للمحور.
            #  النصف الأفضل يخرج يساراً والنصف الأسوأ يميناً، كلٌّ على ارتفاع خاص.
            span = max(axis_hi - axis_lo, 1e-9)
            GAP = .06 * span                       # فراغ بين نهاية الخط وبداية النصّ
            n_up = (len(names) + 1) // 2
            fig, ax = plt.subplots(figsize=(11, 3.6))
            y_axis = 0.0
            ax.hlines(y_axis, axis_lo, axis_hi, color="black", lw=1.6)
            for t in np.arange(np.ceil(axis_lo * 2) / 2, axis_hi + .01, 0.5):
                ax.vlines(t, y_axis, y_axis + .06, color="black", lw=1.2)
                ax.text(t, y_axis + .11, f"{t:g}", ha="center", fontsize=8)

            for i, (n, p) in enumerate(zip(names, pos)):
                left = i < n_up
                lvl = i if left else len(names) - 1 - i
                yy = y_axis - .16 - .13 * lvl
                xend = axis_lo if left else axis_hi
                ax.plot([p, p], [y_axis, yy], color="black", lw=1.1)
                ax.plot([p, xend], [yy, yy], color="black", lw=1.1)
                ax.text(xend + (-GAP if left else GAP), yy, f"{n}  ({p:.2f})",
                        ha="right" if left else "left", va="center", fontsize=9,
                        fontweight="bold" if i == 0 else "normal")

            for k, (a, b) in enumerate(cliques):     # زُمَر عدم الاختلاف المعنوي
                yy = y_axis - .06 - .045 * k
                ax.plot([pos[a] - .015 * span, pos[b] + .015 * span], [yy, yy],
                        lw=4.5, color=C["red"], solid_capstyle="butt")

            ax.plot([axis_lo, axis_lo + CD], [y_axis + .34, y_axis + .34], lw=3, color=C["blue"])
            ax.vlines([axis_lo, axis_lo + CD], y_axis + .31, y_axis + .37, color=C["blue"], lw=2)
            ax.text(axis_lo + CD / 2, y_axis + .40, f"CD = {CD:.2f}", ha="center",
                    fontsize=9, color=C["blue"], fontweight="bold")

            ax.set_xlim(axis_lo - .34 * span, axis_hi + .34 * span)
            ax.set_ylim(y_axis - .16 - .13 * n_up - .14, y_axis + .52)
            ax.axis("off"); ax.grid(False)
            p_fr = stats.get("friedman_p", float("nan"))
            ns = "  —  no significant difference between optimizers" if (p_fr == p_fr and p_fr >= .05) else ""
            ax.set_title("P6 — Critical Difference diagram (Nemenyi, α=0.05)  |  "
                         f"Friedman p = {p_fr:.3g}{ns}", fontsize=11)
            plt.tight_layout()
            save(fig, "P6_critical_difference")
except Exception as e:
    print("   … P5/P6:", e)

print(f"\n✓ الأشكال في: {os.path.relpath(FIG)}")
