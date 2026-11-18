# -*- coding: utf-8 -*-
# =====================================================================
#  إعادة حساب الإحصاء على وحدة الاستقلال الصحيحة — الطيّة لا النافذة
# ---------------------------------------------------------------------
#  المشكلة التي يعالجها هذا الملف:
#    • WINDOW=50 و STRIDE=2 ⇒ تداخل 96% بين النوافذ المتجاورة.
#    • مقطع الانتحال يمتدّ ~190 عيّنة خام فقط لكل طيّة
#      ⇒ الـ 69–72 نافذة اختبار تكافئ **~3.8 نوافذ مستقلّة**.
#    • وبذور التكرار (0,1,2) تُقيَّم على **نفس** بيانات الاختبار ⇒ ليست
#      مشاهدات مستقلّة بل إعادة تهيئة للنموذج.
#
#  ⇒ وحدة الاستقلال الحقيقية هي **الطيّة** (طائرة مختلفة ورحلات مختلفة):
#     n = 4 لا n = 12 ولا n = 72.
#
#  لذلك نعيد حساب كل الأرقام المُبلَّغ عنها بتجميع البذور داخل الطيّة أولاً ،
#  ثم إجراء الاختبارات على الطيّات الأربع. النتيجة فترات ثقة **أوسع**
#  وقيَم p **أضعف** — وهي الأرقام الصادقة التي تُكتَب في الأطروحة.
# =====================================================================
import os, sys, json, numpy as np, pandas as pd
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import config as CFG
from scipy.stats import ttest_1samp, wilcoxon, friedmanchisquare, rankdata

RES = CFG.RESULTS_DIR
CH = 1.0 / CFG.NUM_CLASSES
out = {"note": "independent unit = fold (n=4); seeds averaged within fold"}


def boot_ci(v, n=4000, seed=0):
    v = np.asarray(v, float)
    if len(v) < 2:
        return (float("nan"), float("nan"))
    rng = np.random.RandomState(seed)
    m = [rng.choice(v, len(v), replace=True).mean() for _ in range(n)]
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def rd(n):
    p = os.path.join(RES, n)
    return pd.read_csv(p) if os.path.exists(p) else None


print("=" * 66)
print("  الإحصاء على مستوى الطيّة (n=4) مقابل مستوى النافذة (n=12)")
print("=" * 66)

# ---------------- (1) خطّ الأساس النظيف مقابل التخمين ----------------
d = rd("p3_clean_baseline.csv")
if d is not None:
    print("\n[1] خطّ الأساس النظيف مقابل التخمين %.3f" % CH)
    print(f"  {'model':6s} {'n=12 (نافذة)':28s} {'n=4 (طيّة) — الصادق':28s}")
    res = {}
    for m in sorted(d.model.unique()):
        w = d[d.model == m]["balacc"].values                      # 12 قيمة
        f = d[d.model == m].groupby("fold")["balacc"].mean().values   # 4 قيم
        lo_w, hi_w = boot_ci(w); lo_f, hi_f = boot_ci(f)
        _, p_w = ttest_1samp(w, CH)
        _, p_f = ttest_1samp(f, CH)
        #  ⚠️ عند n=4 لا يُنتج الـbootstrap إلّا 256 إعادة معاينة مميّزة فيُضيّق
        #     الفترة منهجياً (عرضها ≈ نصف فترة t). نحفظ الاثنتين ونُبلّغ عن t.
        from scipy.stats import t as _tdist, sem as _sem
        lo_t, hi_t = _tdist.interval(0.95, len(f) - 1, loc=f.mean(), scale=_sem(f))
        res[m] = {"fold_mean": float(f.mean()), "fold_ci95_t": [float(lo_t), float(hi_t)],
                  "fold_ci95_boot": [lo_f, hi_f],
                  "fold_p": float(p_f), "window_p": float(p_w), "n_folds": int(len(f))}
        print(f"  {m:6s} {f'{w.mean():.3f} [{lo_w:.3f},{hi_w:.3f}] p={p_w:.1e}':28s} "
              f"{f'{f.mean():.3f} [{lo_f:.3f},{hi_f:.3f}] p={p_f:.4f}':28s}")
    out["clean_baseline"] = res

# ---------------- (2) مكسب الدفاع في المسح الموجَّه ----------------
d = rd("p3_targeted_sweep.csv")
if d is not None:
    print("\n[2] الدفاع مقابل بلا دفاع (المسح الموجَّه)")
    print(f"  {'rate':6s} {'none':>7s} {'filtered':>9s} {'gain':>7s} {'p (n=12)':>10s} {'p (n=4)':>9s}")
    rows = []
    for r in sorted(d.rate.unique()):
        a_w = d[(d.rate == r) & (d.defense == "none")].sort_values(["fold", "seed"])["spoof_recall"].values
        b_w = d[(d.rate == r) & (d.defense == "filtered_retrain")].sort_values(["fold", "seed"])["spoof_recall"].values
        a_f = d[(d.rate == r) & (d.defense == "none")].groupby("fold")["spoof_recall"].mean().values
        b_f = d[(d.rate == r) & (d.defense == "filtered_retrain")].groupby("fold")["spoof_recall"].mean().values
        def _p(a, b):
            if len(a) != len(b) or not np.any(a != b):
                return float("nan")
            try: return float(wilcoxon(a, b).pvalue)
            except Exception: return float("nan")
        p_w, p_f = _p(a_w, b_w), _p(a_f, b_f)
        rows.append({"rate": float(r), "none": float(a_f.mean()), "defended": float(b_f.mean()),
                     "gain": float(b_f.mean() - a_f.mean()), "p_window_n12": p_w, "p_fold_n4": p_f})
        print(f"  {r:<6.2f} {a_f.mean():7.3f} {b_f.mean():9.3f} {b_f.mean()-a_f.mean():+7.3f} "
              f"{p_w:10.4f} {p_f:9.4f}")
    out["defense_gain"] = rows
    print("\n  ⚠️ n=4 هو أصغر حجم يسمح به Wilcoxon (أدنى p ممكنة = 0.125)")
    print("     ⇒ لا يمكن بلوغ p<0.05 بأربع طيّات مهما كان حجم الأثر.")

# ---------------- (3) مقارنة الدفاعات (Friedman) ----------------
d = rd("p3_defense.csv")
if d is not None:
    print("\n[3] مقارنة الدفاعات — Friedman")
    for atk in sorted(d.attack.unique()):
        sub = d[d.attack == atk]
        metric = "asr" if atk == "backdoor" else "spoof_recall"
        defs = sorted(sub.defense.unique())
        # كتل على مستوى الطيّة: نُوسّط البذور والنسب داخل كل طيّة
        M = []
        for fo in sorted(sub.fold.unique()):
            row = [sub[(sub.fold == fo) & (sub.defense == dv)][metric].mean() for dv in defs]
            if not any(np.isnan(row)):
                M.append(row)
        M = np.array(M)
        if len(M) >= 3:
            fr = friedmanchisquare(*[M[:, i] for i in range(M.shape[1])])
            ranks = np.array([rankdata(-M[b]) if metric != "asr" else rankdata(M[b])
                              for b in range(len(M))]).mean(0)
            best = defs[int(np.argmin(ranks))]
            print(f"  {atk:14s} n_blocks={len(M)} (طيّات)  p={fr.pvalue:.4f}  best={best}")
            print(f"     ranks: {dict(zip(defs, np.round(ranks,2)))}")
            out[f"defense_friedman_{atk}"] = {"p": float(fr.pvalue), "n_blocks": int(len(M)),
                                              "best": best,
                                              "avg_rank": dict(zip(defs, ranks.tolist()))}

json.dump(out, open(os.path.join(RES, "p3_stats_clustered.json"), "w", encoding="utf-8"),
          ensure_ascii=False, indent=2)
print(f"\n✓ {os.path.relpath(os.path.join(RES, 'p3_stats_clustered.json'))}")
