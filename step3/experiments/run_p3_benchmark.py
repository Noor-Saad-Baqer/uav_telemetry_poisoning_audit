# -*- coding: utf-8 -*-
# =====================================================================
#  مُشغّل تجارب المرحلة الثالثة — التسميم والكشف والدفاع
#  Step 3 runner: poisoning damage → detection → defense → statistics
# ---------------------------------------------------------------------
#  المراحل:
#   A) خطّ الأساس النظيف (rf / xgb / mlp) عبر طيّات اترك-طائرة-خارجاً
#   B+C) منحنى الضرر: 4 هجمات × نسب تسميم × طيّات × بذور
#   D) كشف التسميم: 4 كواشف × الهجمات × نسب × طيّات × بذور (ROC-AUC)
#   E) الدفاع والاستعادة: 3 دفاعات + بلا دفاع
#   G) الإحصاء: Friedman + Wilcoxon + فترات ثقة bootstrap + اختبار مقابل التخمين
#
#  قابل للاستئناف: كل تقييم يُخزَّن في .cache_p3 — أعيدي التشغيل ليكمل من حيث وقف.
# =====================================================================
import os, sys, json, time, hashlib, warnings, numpy as np, pandas as pd
warnings.filterwarnings("ignore")
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, ".."))
import config as CFG
import poison_core as P
import attacks as A
import detectors as D
import defenses as F
from scipy.stats import friedmanchisquare, wilcoxon, rankdata, ttest_1samp

RES = CFG.RESULTS_DIR
SMOKE = os.environ.get("P3_SMOKE") == "1"

# ---------------- ميزانية التجارب (قابلة للتصغير للاختبار السريع) ----------------
if SMOKE:
    SEEDS, RATES, DET_RATES = [0], [0.0, 0.2], [0.2]
    ATTACKS, DETECTORS, DEFENSES = ["targeted_flip", "backdoor"], ["loss_based", "knn_consistency"], ["none", "robust_loss"]
    CFG.EPOCHS = 12
else:
    SEEDS = CFG.SEEDS[:3]                       # 3 بذور (كافية مع 4 طيّات = 12 عيّنة)
    RATES = CFG.POISON_RATES
    DET_RATES = [0.10, 0.20, 0.30]              # الكشف يُقاس عند نسب متوسّطة
    ATTACKS, DETECTORS = CFG.ATTACKS, CFG.DETECTORS
    DEFENSES = ["none"] + CFG.DEFENSES

#  الدفاعات تُقيَّم ضدّ الهجمات المُضِرّة فقط (الأخرى لا تُحدث ضرراً يُستعاد منه)
DAMAGING = ["targeted_flip", "backdoor"]
CHANCE = P.CHANCE

# ---------------- ذاكرة قرصية للاستئناف ----------------
CACHE = os.path.join(CFG.CACHE_DIR, "runs")
os.makedirs(CACHE, exist_ok=True)


#  ⚠️ بصمة البيانات وقاعدة النسبة **يجب** أن تدخلا مفتاح الـ cache: تغيّرت
#  الميزات (85←70 بعد كسر الدائرية) وقاعدة نسبة التسميم ، ولولا ذلك لأُعيد
#  استعمال نتائج قديمة بدلالة مختلفة تماماً — وهو خطأ صامت يفسد المقارنة.
_DKEY = P._cache_key()
_RB = getattr(CFG, "RATE_BASIS", "train_set")

def cached(tag, payload, fn):
    key = hashlib.md5((tag + json.dumps(payload, sort_keys=True) +
                       f"|v{CFG.CACHE_VERSION}|ep{CFG.EPOCHS}|d{_DKEY}|rb{_RB}"
                       f"|vic{getattr(CFG, 'VICTIM', 'mlp')}").encode()).hexdigest()
    fp = os.path.join(CACHE, f"{tag}_{key}.json")
    if os.path.exists(fp):
        try:
            with open(fp, encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    out = fn()
    tmp = fp + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(out, f)
    os.replace(tmp, fp)
    return out


# ---------------- تحميل البيانات والطيّات ----------------
print(">> [step3] بناء مجموعة النوافذ ...", flush=True)
X, y, groups, flights, feat_names = P.build_dataset(verbose=False)
FOLDS = P.folds(groups, flights)
print(f">> X={X.shape}  classes={np.bincount(y).tolist()}  folds={[f[0] for f in FOLDS]}")
print(f">> device={P.device()}  epochs={CFG.EPOCHS}  seeds={SEEDS}\n", flush=True)


_FOLD_CACHE = {}


def fold_data(fi):
    """طيّة مُطبَّعة ومحفوظة في الذاكرة — تُحسَب مرّة واحدة لا مع كل تقييم."""
    if fi not in _FOLD_CACHE:
        _, tr, te = FOLDS[fi]
        sc = P.fit_scaler(X[tr])
        _FOLD_CACHE[fi] = (P.apply_scaler(sc, X[tr]), y[tr], P.apply_scaler(sc, X[te]), y[te])
    return _FOLD_CACHE[fi]


# =====================================================================
#  (A) خطّ الأساس النظيف
# =====================================================================
print("== [A] خطّ الأساس النظيف ==", flush=True)
rows = []
for model in CFG.MODELS:
    for fi, (fname, _, _) in enumerate(FOLDS):
        for s in SEEDS:
            def _run(model=model, fi=fi, s=s):
                Xtr, ytr, Xte, yte = fold_data(fi)
                p = P.fit_predict(model, Xtr, ytr, Xte, seed=s)
                return P.evaluate(yte, p)
            r = cached("clean", {"m": model, "f": fname, "s": s}, _run)
            rows.append({"model": model, "fold": fname, "seed": s, **r})
    sub = [r for r in rows if r["model"] == model]
    print(f"   {model:5s} BalAcc={np.mean([r['balacc'] for r in sub]):.3f} "
          f"± {np.std([r['balacc'] for r in sub]):.3f}  "
          f"SpoofRec={np.mean([r['spoof_recall'] for r in sub]):.3f}", flush=True)
clean_df = pd.DataFrame(rows)
clean_df.to_csv(os.path.join(RES, "p3_clean_baseline.csv"), index=False)

# =====================================================================
#  (B+C) منحنى الضرر
# =====================================================================
print("\n== [B+C] منحنى الضرر (الهجوم × نسبة التسميم) ==", flush=True)
rows = []
for atk in ATTACKS:
    for rate in RATES:
        for fi, (fname, _, _) in enumerate(FOLDS):
            for s in SEEDS:
                def _run(atk=atk, rate=rate, fi=fi, s=s):
                    Xtr, ytr, Xte, yte = fold_data(fi)
                    Xp, yp, mask = A.poison(atk, Xtr, ytr, rate, seed=s)
                    m, _ = P.train_victim(Xp, yp, seed=s)
                    out = P.evaluate(yte, P.predict(m, Xte))
                    out["n_poison"] = int(mask.sum())
                    out["asr"] = (A.backdoor_asr(lambda Z: P.predict(m, Z), Xte, yte)
                                  if atk == "backdoor" else float("nan"))
                    return out
                r = cached("damage", {"a": atk, "r": rate, "f": fname, "s": s}, _run)
                rows.append({"attack": atk, "rate": rate, "fold": fname, "seed": s, **r})
        sub = [r for r in rows if r["attack"] == atk and r["rate"] == rate]
        asr = np.nanmean([r["asr"] for r in sub])
        print(f"   {atk:14s} rate={rate:.2f} → BalAcc={np.mean([r['balacc'] for r in sub]):.3f} "
              f"SpoofRec={np.mean([r['spoof_recall'] for r in sub]):.3f}"
              + (f" ASR={asr:.3f}" if atk == "backdoor" else ""), flush=True)
dmg_df = pd.DataFrame(rows)
dmg_df.to_csv(os.path.join(RES, "p3_damage_curve.csv"), index=False)

# =====================================================================
#  (D) كشف التسميم
# =====================================================================
print("\n== [D] كشف التسميم (ROC-AUC) ==", flush=True)
rows = []
for atk in ATTACKS:
    for rate in DET_RATES:
        for fi, (fname, _, _) in enumerate(FOLDS):
            for s in SEEDS:
                def _run(atk=atk, rate=rate, fi=fi, s=s):
                    Xtr, ytr, _, _ = fold_data(fi)
                    Xp, yp, mask = A.poison(atk, Xtr, ytr, rate, seed=s)
                    victim, _ = P.train_victim(Xp, yp, seed=s)   # نموذج واحد لكل الكواشف
                    out = {}
                    for det in DETECTORS:
                        try:
                            sc = D.detect(det, Xp, yp, model=victim, seed=s)
                            out[det] = D.detection_auc(sc, mask)
                        except Exception as e:
                            out[det] = float("nan")
                    return out
                r = cached("detect", {"a": atk, "r": rate, "f": fname, "s": s}, _run)
                for det, auc in r.items():
                    #  auc_oriented = max(auc, 1-auc): كم معلومة يحملها الكاشف بغضّ
                    #  النظر عن اتّجاهه. AUC<0.5 يعني أن الإشارة معكوسة لا غائبة —
                    #  والتصفية بالاتجاه الخاطئ تحذف العيّنات النظيفة.
                    rows.append({"attack": atk, "rate": rate, "fold": fname,
                                 "seed": s, "detector": det, "auc": auc,
                                 "auc_oriented": (float("nan") if auc != auc else max(auc, 1 - auc)),
                                 "inverted": bool(auc == auc and auc < 0.5)})
    for det in DETECTORS:
        sub = [r for r in rows if r["attack"] == atk and r["detector"] == det]
        v = [r["auc"] for r in sub]; vo = [r["auc_oriented"] for r in sub]
        inv = np.mean([r["inverted"] for r in sub]) if sub else 0
        print(f"   {atk:14s} {det:20s} AUC={np.nanmean(v):.3f}  oriented={np.nanmean(vo):.3f}"
              + ("   ← معكوس" if inv > 0.5 else ""), flush=True)
det_df = pd.DataFrame(rows)
det_df.to_csv(os.path.join(RES, "p3_detection.csv"), index=False)

# =====================================================================
#  (E) الدفاع والاستعادة
# =====================================================================
print("\n== [E] الدفاع والاستعادة ==", flush=True)
rows = []
for atk in DAMAGING:
    for rate in DET_RATES:
        for dfs in DEFENSES:
            for fi, (fname, _, _) in enumerate(FOLDS):
                for s in SEEDS:
                    def _run(atk=atk, rate=rate, dfs=dfs, fi=fi, s=s):
                        Xtr, ytr, Xte, yte = fold_data(fi)
                        Xp, yp, _ = A.poison(atk, Xtr, ytr, rate, seed=s)
                        pred = F.defend(dfs, Xp, yp, seed=s)
                        out = P.evaluate(yte, pred(Xte))
                        out["asr"] = (A.backdoor_asr(pred, Xte, yte)
                                      if atk == "backdoor" else float("nan"))
                        return out
                    r = cached("defend", {"a": atk, "r": rate, "d": dfs, "f": fname, "s": s}, _run)
                    rows.append({"attack": atk, "rate": rate, "defense": dfs,
                                 "fold": fname, "seed": s, **r})
            sub = [r for r in rows if r["attack"] == atk and r["rate"] == rate and r["defense"] == dfs]
            print(f"   {atk:14s} rate={rate:.2f} {dfs:18s} BalAcc={np.mean([r['balacc'] for r in sub]):.3f} "
                  f"SpoofRec={np.mean([r['spoof_recall'] for r in sub]):.3f}"
                  + (f" ASR={np.nanmean([r['asr'] for r in sub]):.3f}" if atk == "backdoor" else ""), flush=True)
def_df = pd.DataFrame(rows)
def_df.to_csv(os.path.join(RES, "p3_defense.csv"), index=False)

# =====================================================================
#  (G) الإحصاء — مقارنة الدفاعات + اختبار مقابل التخمين
# =====================================================================
print("\n== [G] الإحصاء ==", flush=True)
stats = {}

#  ⚠️ وحدة الاستقلال = **الطيّة** لا صفّ fold×seed.
#     البذور (0,1,2) تُقيَّم على نفس بيانات الاختبار ⇒ إعادة تهيئة للنموذج لا
#     مشاهدات مستقلّة. اعتبارها 12 مشاهدة تكرار زائف يضخّم المعنوية اصطناعياً
#     (‏p = 2.9e-09 عند n=12 مقابل p = 0.0029 عند n=4 على البيانات نفسها).
#     لذلك نُوسِّط البذور داخل كل طيّة أوّلاً ، ثم نختبر على الطيّات.
stats["note"] = ("independent unit = fold; seeds averaged within fold before testing. "
                 "window_p is the pseudo-replicated value, kept for comparison only "
                 "and must not be reported.")


def _boot_ci(v, n=4000, seed=0):
    """فترة bootstrap على مستوى الطيّة — مطابقة لطريقة recompute_stats_clustered.py
    حتى لا يعطي الملفّان جوابين مختلفين للسؤال نفسه."""
    v = np.asarray(v, float)
    if len(v) < 2:
        return float("nan"), float("nan")
    rng = np.random.RandomState(seed)
    m = [rng.choice(v, len(v), replace=True).mean() for _ in range(n)]
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


#  خطّ الأساس النظيف: هل يتفوّق فعلاً على التخمين؟ (الاختبار الغائب في المرحلة 2)
for model in CFG.MODELS:
    sub = clean_df[clean_df.model == model]
    if len(sub) == 0:
        print(f"   {model:5s} — لا صفوف ⇒ يُتخطّى")
        continue
    v = sub.groupby("fold")["balacc"].mean().values          # طيّة واحدة = قيمة واحدة
    w = sub["balacc"].values                                  # مستوى النافذة (للمقارنة)
    t, p = ttest_1samp(v, CHANCE) if len(v) > 1 and np.std(v) > 0 else (np.inf, 0.0)
    _, p_w = ttest_1samp(w, CHANCE) if len(w) > 1 and np.std(w) > 0 else (np.inf, 0.0)
    lo, hi = _boot_ci(v)
    stats[f"clean_{model}_vs_chance"] = {"mean": float(v.mean()), "ci95": [lo, hi],
                                         "chance": CHANCE, "t": float(t), "p": float(p),
                                         "n_folds": int(len(v)), "window_p": float(p_w),
                                         "significant": bool(p < 0.05 and v.mean() > CHANCE)}
    print(f"   {model:5s} = {v.mean():.3f}  CI95=[{lo:.3f},{hi:.3f}]  vs chance {CHANCE:.3f}  "
          f"p={p:.2g}  (n={len(v)} طيّات ؛ p على مستوى النافذة {p_w:.2g} — لا تُبلَّغ)")

#  مقارنة الدفاعات (Friedman على كتل **الطيّات** لكل هجوم ؛ البذور مُوسَّطة داخلها)
for atk in DAMAGING:
    sub = def_df[(def_df.attack == atk) & (def_df.rate == DET_RATES[-1])]
    M, names = [], [d for d in DEFENSES]
    for fo in sorted(sub.fold.unique()):
        row = [sub[(sub.fold == fo) & (sub.defense == d)]["spoof_recall"].mean() for d in names]
        if not any(np.isnan(row)):
            M.append(row)
    M = np.array(M)
    if len(M) >= 3 and M.shape[1] >= 3:
        fr = friedmanchisquare(*[M[:, i] for i in range(M.shape[1])])
        ranks = np.array([rankdata(-M[b]) for b in range(len(M))]).mean(0)
        best = names[int(np.argmin(ranks))]
        pw = {}
        for i, d in enumerate(names):
            if d == best: continue
            try:
                pw[d] = float(wilcoxon(M[:, names.index(best)], M[:, i]).pvalue)
            except Exception:
                pw[d] = float("nan")
        stats[f"defense_{atk}"] = {"friedman_p": float(fr.pvalue), "avg_rank": dict(zip(names, ranks.tolist())),
                                   "best": best, "wilcoxon_vs_best": pw, "n_blocks": len(M),
                                   "block_unit": "fold",
                                   "scope": f"rate={DET_RATES[-1]} only "
                                            f"(p3_stats_clustered.json averages over all rates)"}
        #  عند n=4 أدنى p ممكنة من Wilcoxon هي 0.125 ⇒ لا يمكن بلوغ 0.05 مهما كان الأثر
        note = "  ⚠️ Wilcoxon عند n=4: أدنى p ممكنة 0.125" if len(M) <= 5 else ""
        print(f"   {atk}: best defense = {best} | Friedman p={fr.pvalue:.4g} | "
              f"n={len(M)} طيّات{note}")

with open(os.path.join(RES, "p3_stats.json"), "w", encoding="utf-8") as f:
    json.dump(stats, f, ensure_ascii=False, indent=2)

print(f"\n✓ خلصت. النتائج في: {os.path.relpath(RES)}")
print("  شغّلي بعدها:  python step3/comparison_figures.py")
