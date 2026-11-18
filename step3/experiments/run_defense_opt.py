# -*- coding: utf-8 -*-
# =====================================================================
#  المرحلة F — تحسين الدفاع بالخوارزميات الميتاهيوريستية
#  Stage F: metaheuristic tuning of the poisoning defense
# ---------------------------------------------------------------------
#  الدافع العلمي: دفاع التصفية يستعيد الأداء كاملاً عند تسميم 10%
#  (استدعاء الانتحال 0.088 → 0.995) لكنه ينهار إلى 0.000 بحلول 15%
#  (قِيست العتبة بدقّة في run_relabel_test.py بنسب 5/10/15/20/30/40).
#  السؤال: هل يستطيع ضبطُ معاملاته ميتاهيوريستياً رفعَ عتبة الانهيار؟
#  هنا تُثمر أدوات المرحلة الثانية: نعيد استخدام optimizers.py نفسه.
#
#  البروتوكول (بلا تسريب):
#   • الضبط يجري على تقسيم داخلي من طائرات التدريب فقط (طائرة تحقّق واحدة).
#   • الطائرة المحجوزة الخارجية لا تُرى إطلاقاً أثناء البحث.
#   • بذور بحث (10,11,12) منفصلة عن بذور الاختبار (0,1,2).
# =====================================================================
import os, sys, json, hashlib, warnings, numpy as np, pandas as pd
warnings.filterwarnings("ignore")
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, ".."))
#  إعادة استخدام مُحسِّنات المرحلة الثانية (نفس الكود، بلا نسخ)
STEP2_EXP = os.path.join(HERE, "..", "..", "step2", "experiments")
sys.path.insert(0, os.path.abspath(STEP2_EXP))
import config as CFG
import poison_core as P
import attacks as A
import defenses as F
import detectors as D
from optimizers import run_optimizer
from scipy.stats import friedmanchisquare, rankdata

RES = CFG.RESULTS_DIR
SMOKE = os.environ.get("P3_SMOKE") == "1"
BUDGET = 12 if SMOKE else CFG.EVAL_BUDGET
OPTS = ["pso", "lgwo"] if SMOKE else CFG.OPTIMIZERS
OPT_SEEDS = CFG.OPT_SEEDS[:1] if SMOKE else CFG.OPT_SEEDS
SEARCH_SEEDS = CFG.SEARCH_SEEDS[:1] if SMOKE else CFG.SEARCH_SEEDS
TEST_SEEDS = CFG.SEEDS[:3]
ATK = CFG.ATTACK_FOR_OPT
RATES = CFG.RATES_FOR_OPT if not SMOKE else CFG.RATES_FOR_OPT[:2]
OBJ = getattr(CFG, "OPT_OBJECTIVE", "recall")     # recall | auc | blend
NEMENYI_Q05 = {2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850}

#  فضاء البحث: معاملات دفاع التصفية + اختيار الكاشف (بُعد متقطّع مُرمَّز)
DET_CHOICES = ["loss_deviation", "activation_cluster", "spectral", "knn_consistency"]
KEYS = ["filter_q", "lr_log", "det_idx"]
BOUNDS = {"filter_q": (0.05, 0.60), "lr_log": (-4.0, -2.5), "det_idx": (0.0, 0.999)}
DIM = len(KEYS)


def decode(v):
    hp = {}
    for i, k in enumerate(KEYS):
        lo, hi = BOUNDS[k]
        val = lo + float(np.clip(v[i], 0, 1)) * (hi - lo)
        if k == "lr_log":
            hp["lr"] = float(10 ** val)
        elif k == "det_idx":
            hp["detector"] = DET_CHOICES[int(val * len(DET_CHOICES))]
        else:
            hp[k] = float(val)
    return hp


print(">> [stage F] بناء البيانات ...", flush=True)
X, y, groups, flights, _ = P.build_dataset(verbose=False)
FOLDS = P.loao_folds(groups)
print(f">> هجوم={ATK}  نسب={RATES}  ميزانية={BUDGET}  مُحسِّنات={OPTS}\n", flush=True)

CACHE = os.path.join(CFG.CACHE_DIR, "optF")
os.makedirs(CACHE, exist_ok=True)


def _eval_defense(tr_idx, te_idx, hp, seeds, want="recall"):
    """يُرجع (استدعاء الانتحال بعد الدفاع ، AUC الكشف المُصحَّح الاتجاه).

    نحسب الاثنين في مسارٍ واحد بنفس كلفة السابق (تدريبان لا أكثر):
      1) تدريب النموذج الضحيّة على البيانات المسمومة  → منه درجات الكاشف
      2) تصفية أعلى q ثم إعادة التدريب               → منه استدعاء الانتحال
    AUC يُقاس مقابل قناع التسميم الحقيقي ⇒ يفترض توفّر مجموعة تحقّق
    *معلومة التسميم* لضبط الدفاع (افتراض تصميمي مشروع ، لكنه افتراض).
    """
    sc = P.fit_scaler(X[tr_idx])
    Xtr, ytr = P.apply_scaler(sc, X[tr_idx]), y[tr_idx]
    Xte, yte = P.apply_scaler(sc, X[te_idx]), y[te_idx]
    det = hp.get("detector", CFG.FILTER_DETECTOR)
    q = float(hp.get("filter_q", CFG.FILTER_Q))
    lr = hp.get("lr")
    rec_all, auc_all = [], []
    for rate in RATES:                     # متوسّط عبر منطقة الانتقال
        for s in seeds:
            Xp, yp, mask = A.poison(ATK, Xtr, ytr, rate, seed=s)
            victim, _ = P.train_victim(Xp, yp, seed=s)          # (1) الضحيّة
            scores = D.detect(det, Xp, yp, model=victim, seed=s)
            a = D.detection_auc(scores, mask)
            if a == a:
                #  ⚠️ AUC *الخام* لا المُصحَّح الاتجاه. السبب: الدفاع يحذف أعلى
                #  الدرجات ، فلا ينفع إلا إذا كانت العيّنة المسمومة تحمل درجة
                #  *أعلى* (AUC > 0.5). الـ AUC المُصحَّح يكافئ الكاشف المعكوس
                #  أيضاً — وهو عديم النفع للتصفية لأنه يحذف العيّنات النظيفة.
                auc_all.append(a)
            #  التصفية ثم إعادة التدريب (نفس منطق filtered_retrain)
            thr = np.quantile(scores, 1.0 - q)
            keep = scores < thr
            for c in np.unique(yp):
                if keep[yp == c].sum() < 2:
                    keep[np.where(yp == c)[0][:2]] = True
            m2, _ = P.train_victim(Xp[keep], yp[keep], seed=s, lr=lr)   # (2)
            rec_all.append(P.evaluate(yte, P.predict(m2, Xte))["spoof_recall"])
    rec = float(np.mean(rec_all)) if rec_all else 0.0
    auc = float(np.mean(auc_all)) if auc_all else float("nan")
    return (rec, auc)


def _objective_value(rec, auc):
    """دالّة الهدف حسب CFG.OPT_OBJECTIVE.

    recall — الأصلي: يكافئ الأداء بعد الدفاع فقط (فشل في اختيار الكاشف الأصلح).
    auc    — يكافئ جودة الكشف فقط (ملاحظة: filter_q و lr لا يؤثّران في AUC
             ⇒ يتحوّل البحث فعلياً إلى بُعد واحد).
    blend  — مزيج موزون: يكافئ الكاشف الأصلح مع إبقاء بقيّة الأبعاد فاعلة.
    """
    mode = getattr(CFG, "OPT_OBJECTIVE", "recall")
    if mode == "auc":
        return auc if auc == auc else 0.0
    if mode == "blend":
        w = float(getattr(CFG, "OPT_AUC_WEIGHT", 0.5))
        a = auc if auc == auc else 0.0
        return w * a + (1.0 - w) * rec
    return rec


def inner_split(outer_fold):
    """طائرات التدريب للطيّة الخارجية ⇒ واحدة منها للتحقّق والباقي للتدريب."""
    held = FOLDS[outer_fold][0]
    train_afs = [a for a in np.unique(groups) if a != held]
    #  نختار طائرة تحقّق فيها الفئات الثلاث إن أمكن
    val = None
    for a in train_afs:
        if len(np.unique(y[groups == a])) == CFG.NUM_CLASSES:
            val = a; break
    val = val or train_afs[0]
    tr = np.where((groups != held) & (groups != val))[0]
    va = np.where(groups == val)[0]
    return tr, va


#  استئناف على مستوى الصفوف: الجهاز قد يقتل العملية بسبب ضيق الـ commit memory ،
#  فنحمّل ما أُنجِز ونتخطّاه. مع cache دالّة الهدف يصبح كل تشغيل جديد أسرع بكثير.
#  ملف منفصل لكل هدف ⇒ نتيجة التشغيل بالهدف القديم تبقى للمقارنة
_sfx = "" if OBJ == "recall" else f"_{OBJ}"
OUT_CSV = os.path.join(RES, f"p3_defense_opt{_sfx}.csv")
rows, done = [], set()
if os.path.exists(OUT_CSV):
    try:
        _prev = pd.read_csv(OUT_CSV)
        rows = _prev.to_dict("records")
        done = {(r["fold"], r["optimizer"], int(r["opt_seed"])) for r in rows}
        print(f">> استئناف: {len(rows)} صفّاً منجزاً سابقاً — سيُتخطّى.\n", flush=True)
    except Exception:
        rows, done = [], set()


def _flush():
    pd.DataFrame(rows).to_csv(OUT_CSV, index=False)


for fi, (held, tr_out, te_out) in enumerate(FOLDS):
    tr_in, va_in = inner_split(fi)
    print(f"-- الطيّة الخارجية: {held} --", flush=True)

    #  خطّ الأساس: المعاملات الافتراضية (بلا تحسين)
    if (held, "default", -1) not in done:
        base_hp = {"filter_q": CFG.FILTER_Q, "detector": CFG.FILTER_DETECTOR}
        base, base_auc = _eval_defense(tr_out, te_out, base_hp, TEST_SEEDS)
        rows.append({"fold": held, "optimizer": "default", "opt_seed": -1,
                     "score": base, "auc": base_auc, "objective": OBJ,
                     **{k: str(v) for k, v in base_hp.items()}})
        done.add((held, "default", -1)); _flush()
        print(f"   default → SpoofRec={base:.3f}  AUC={base_auc:.3f}", flush=True)

    def objective(vec):
        hp = decode(vec)
        key = hashlib.md5(json.dumps({"f": held, "hp": {k: str(v) for k, v in hp.items()},
                                      "s": SEARCH_SEEDS, "r": RATES, "obj": OBJ,
                                      "d": P._cache_key(),
                                      "rb": getattr(CFG, "RATE_BASIS", "train_set"),
                                      "vic": getattr(CFG, "VICTIM", "mlp")},
                                     sort_keys=True).encode()).hexdigest()
        fp = os.path.join(CACHE, key + ".json")
        if os.path.exists(fp):
            try:
                return json.load(open(fp))["v"]
            except Exception:
                pass
        rec_i, auc_i = _eval_defense(tr_in, va_in, hp, SEARCH_SEEDS)   # تحقّق داخلي
        v = _objective_value(rec_i, auc_i)                              # ← الهدف الجديد
        tmp = fp + ".tmp"
        json.dump({"v": v}, open(tmp, "w")); os.replace(tmp, fp)
        return v

    for opt in OPTS:
        for os_ in OPT_SEEDS:
            if (held, opt, os_) in done:
                continue
            best_vec, best_val, _ = run_optimizer(opt, objective, DIM, CFG.POP_SIZE, BUDGET, os_)
            hp = decode(best_vec)
            score, auc_out = _eval_defense(tr_out, te_out, hp, TEST_SEEDS)   # تقييم خارجي
            rows.append({"fold": held, "optimizer": opt, "opt_seed": os_, "score": score,
                         "auc": auc_out, "objective": OBJ,
                         **{k: (round(v, 4) if isinstance(v, float) else str(v)) for k, v in hp.items()}})
            done.add((held, opt, os_)); _flush()
            print(f"   {opt:5s}[{os_}] inner={best_val:.3f} → outer SpoofRec={score:.3f} "
                  f"AUC={auc_out:.3f} (q={hp['filter_q']:.2f}, det={hp['detector']})", flush=True)

df = pd.DataFrame(rows)
_flush()

# ---------------- إحصاء المقارنة بين المُحسِّنات ----------------
stats = {}
try:
    blocks = [(f, s) for f in df.fold.unique() for s in OPT_SEEDS]
    M = []
    for (f_, s_) in blocks:
        row = [df[(df.fold == f_) & (df.optimizer == o) & (df.opt_seed == s_)]["score"].mean()
               for o in OPTS]
        if not any(np.isnan(row)):
            M.append(row)
    M = np.array(M)
    if len(M) >= 3 and len(OPTS) >= 3:
        fr = friedmanchisquare(*[M[:, i] for i in range(len(OPTS))])
        ranks = np.array([rankdata(-M[b]) for b in range(len(M))]).mean(0)
        CD = NEMENYI_Q05.get(len(OPTS), 3.0) * np.sqrt(len(OPTS) * (len(OPTS) + 1) / (6.0 * len(M)))
        stats = {"friedman_p": float(fr.pvalue), "avg_rank": dict(zip(OPTS, ranks.tolist())),
                 "CD": float(CD), "n_blocks": int(len(M)),
                 "best": OPTS[int(np.argmin(ranks))],
                 "default_mean": float(df[df.optimizer == "default"]["score"].mean()),
                 "tuned_mean": float(df[df.optimizer != "default"]["score"].mean())}
        print(f"\n>> Friedman p={fr.pvalue:.4g} | أفضل={stats['best']} | "
              f"default={stats['default_mean']:.3f} → tuned={stats['tuned_mean']:.3f}")
except Exception as e:
    print("   إحصاء:", e)

with open(os.path.join(RES, f"p3_opt_stats{_sfx}.json"), "w", encoding="utf-8") as f:
    json.dump(stats, f, ensure_ascii=False, indent=2)
print(f"\n✓ خلصت المرحلة F → {os.path.relpath(RES)}/p3_defense_opt.csv")
