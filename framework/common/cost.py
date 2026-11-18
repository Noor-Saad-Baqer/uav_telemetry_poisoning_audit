# -*- coding: utf-8 -*-
# =====================================================================
#  مِقياس الكلفة الممتدّ — framework/common/cost.py
# ---------------------------------------------------------------------
#  نسخة من step4/cost_core.py (الحجم ، العمليات ، الكمون الوسيط/p95 ، الإنتاجية ،
#  ذروة RSS ، زمن المعالج) + ما ينقص الشكل:
#     • الطاقة (تقدير): E = زمن المعالج × قدرة النواة — لا يوجد مقياس طاقة ،
#       فيُبلَّغ كتقدير بملفَّي قدرة معلنَين (لابتوب / حافة) ، ويُكتب «تقدير» لا «قياس».
#     • كلفة الاتصال: بايتات الميزات المرسَلة لكلّ قرار (مقابل الخام) ،
#       وحجم النموذج المنقول عند التحديث (OTA).
#     • ملفّ الحافة المُحاكى: خيط واحد + مستدلّ شجرة بلغة بايثون صرفة (بلا numpy
#       ولا sklearn — ما ينفَّذ على متحكّم) + ONNX Runtime بخيط واحد إن توفّر.
#       ⚠️ ليس جهازاً حقيقياً. يُكتب في الشكل والفصل: emulated.
#     • الكمون من طرف إلى طرف للتليمتري: بناء ميزات النافذة + الاستدلال.
#  ⚠️ يجب تثبيت الخيوط (paths.pin_threads) قبل استيراد numpy في السكربت المستدعي.
# =====================================================================
import io, gzip, time, threading, numpy as np, joblib, psutil

from common import models as MZ

LATENCY_WARMUP = 20
LATENCY_REPEATS = 200
LATENCY_BATCH = 1024
RSS_REPEATS = 60

#  ملفّا القدرة (واط لكلّ نواة نشطة) — مصدران معلنان يُذكران في الفصل:
#   laptop_core : قدرة نواة أداء واحدة في معالج محمول حديث تحت الحمل ≈ 15 W (TDP/عدد النوى ≈ حدّ أعلى)
#   edge_class  : Raspberry Pi 4 تحت حمل نواة واحدة ≈ 2.5 W فوق الخمول (قياسات منشورة)
POWER_PROFILES_W = {"laptop_core": 15.0, "edge_class": 2.5}


# ------------------------------------------------------------------
#  (١) الحجم
# ------------------------------------------------------------------
def serialized_size(model):
    buf = io.BytesIO()
    joblib.dump(model, buf, compress=0)
    raw = buf.getbuffer().nbytes
    gz = len(gzip.compress(buf.getvalue(), 6))
    return raw, gz


# ------------------------------------------------------------------
#  (٢) عدد العمليات لكل استدلال
# ------------------------------------------------------------------
def _tree_mean_leaf_depth(tree):
    t = tree.tree_
    stack, leaves = [(0, 0)], []
    while stack:
        nid, d = stack.pop()
        if t.children_left[nid] == -1:
            leaves.append((d, t.weighted_n_node_samples[nid]))
        else:
            stack.append((t.children_left[nid], d + 1))
            stack.append((t.children_right[nid], d + 1))
    ds = np.array([d for d, _ in leaves], float)
    ws = np.array([w for _, w in leaves], float)
    return float((ds * ws).sum() / ws.sum()) if ws.sum() > 0 else (float(ds.mean()) if len(ds) else 0.0)


def _xgb_mean_leaf_depth(model):
    dumps = model.get_booster().get_dump()
    tot, n = 0.0, 0
    for d in dumps:
        ds = [len(ln) - len(ln.lstrip("\t")) for ln in d.split("\n") if "leaf=" in ln]
        if ds:
            tot += float(np.mean(ds)); n += 1
    return ((tot / n) if n else 0.0), len(dumps)


def op_count(model):
    """(ops, n_params, method) لاستدلال عيّنة واحدة ، شاملاً المُطبِّع."""
    pre_ops, pre_par = 0, 0
    if type(model).__name__ == "Pipeline":
        sc = model.named_steps.get("sc")
        if sc is not None and hasattr(sc, "mean_"):
            pre_ops = int(sc.mean_.size); pre_par = int(sc.mean_.size + sc.scale_.size)
        ops, par, meth = op_count(model.named_steps["clf"])
        return (ops, par, meth) if ops < 0 else (ops + pre_ops, par + pre_par, meth + " + scaler")
    cls = type(model).__name__
    if hasattr(model, "support_vectors_"):
        sv = np.asarray(model.support_vectors_)
        return int(sv.shape[0] * sv.shape[1]), int(sv.size), "kernel evals = n_SV x d"
    if hasattr(model, "coef_"):
        c, b = np.asarray(model.coef_), np.asarray(model.intercept_)
        return int(np.count_nonzero(c)), int(np.count_nonzero(c) + b.size), "MACs = nnz(coef_)"
    if hasattr(model, "coefs_"):
        macs = int(sum(np.count_nonzero(w) for w in model.coefs_))
        par = macs + int(sum(b.size for b in model.intercepts_))
        return macs, par, "MACs = nnz(weights)"
    if hasattr(model, "_fit_X"):
        m = np.asarray(model._fit_X)
        return int(m.shape[0] * m.shape[1]), int(m.size), "distances = n_train x d"
    if cls.startswith("XGB"):
        mdepth, ntree = _xgb_mean_leaf_depth(model)
        try:
            nodes = int(len(model.get_booster().trees_to_dataframe()))
        except Exception:
            nodes = 0
        return int(round(mdepth * ntree)), nodes, "comparisons = sum mean leaf depth"
    ts = MZ.trees_of(model)
    if ts:
        return (int(round(sum(_tree_mean_leaf_depth(t) for t in ts))),
                int(sum(t.tree_.node_count for t in ts)), "comparisons = sum mean leaf depth")
    return -1, -1, "unsupported"


# ------------------------------------------------------------------
#  (٣) الكمون والإنتاجية والموارد (خيط واحد)
# ------------------------------------------------------------------
def _pin_single_thread(model):
    if type(model).__name__ == "Pipeline":
        _pin_single_thread(model.named_steps["clf"]); return model
    if hasattr(model, "n_jobs"):
        try: model.n_jobs = 1
        except Exception: pass
    if type(model).__name__.startswith("XGB"):
        try: model.set_params(n_jobs=1)
        except Exception: pass
        try: model.get_booster().set_param({"nthread": 1})
        except Exception: pass
    return model


def latency_ms(predict, X, repeats=LATENCY_REPEATS, warmup=LATENCY_WARMUP, batch=LATENCY_BATCH):
    """`predict` دالّة تأخذ مصفوفة (n, d). عيّنة واحدة في كل مرّة (الحالة الواقعية)."""
    x1 = np.ascontiguousarray(X[:1])
    for _ in range(warmup):
        predict(x1)
    ts = np.empty(repeats, float)
    for i in range(repeats):
        xi = np.ascontiguousarray(X[i % len(X):i % len(X) + 1])
        t0 = time.perf_counter(); predict(xi); ts[i] = (time.perf_counter() - t0) * 1e3
    nb = min(batch, len(X))
    xb = np.ascontiguousarray(X[:nb])
    predict(xb)
    t0 = time.perf_counter(); predict(xb); batch_ms = (time.perf_counter() - t0) * 1e3
    return dict(lat_ms_median=float(np.median(ts)), lat_ms_p95=float(np.percentile(ts, 95)),
                lat_ms_mean=float(ts.mean()), batch_ms=float(batch_ms),
                throughput_per_s=float(nb / (batch_ms / 1e3)) if batch_ms > 0 else float("nan"))


def resource_use(predict, X, batch=LATENCY_BATCH, n_rep=RSS_REPEATS):
    proc = psutil.Process()
    nb = min(batch, len(X))
    xb = np.ascontiguousarray(X[:nb])
    predict(xb)
    base = proc.memory_info().rss
    peak = {"v": base}; stop = threading.Event()

    def sampler():
        while not stop.is_set():
            try: peak["v"] = max(peak["v"], proc.memory_info().rss)
            except Exception: pass
            time.sleep(0.001)
    th = threading.Thread(target=sampler, daemon=True); th.start()
    c0 = proc.cpu_times(); t0 = time.perf_counter()
    for _ in range(n_rep):
        predict(xb)
    wall = time.perf_counter() - t0; c1 = proc.cpu_times()
    stop.set(); th.join(timeout=1.0)
    cpu_s = (c1.user - c0.user) + (c1.system - c0.system)
    cpu_s_per_inf = cpu_s / (n_rep * nb)
    out = dict(peak_rss_delta_mb=(peak["v"] - base) / 1e6, peak_rss_mb=peak["v"] / 1e6,
               cpu_ms_per_1k=cpu_s_per_inf * 1e6, cpu_util=(cpu_s / wall) if wall > 0 else float("nan"))
    # ---- (٤) الطاقة — تقدير لا قياس ----
    for prof, w in POWER_PROFILES_W.items():
        out[f"energy_mj_per_inf_{prof}"] = cpu_s_per_inf * w * 1e3     # ميلي جول لكلّ استدلال
    return out


def energy_single(lat_ms_median):
    """الطاقة للحالة الواقعية (نافذة واحدة ⇒ قرار واحد) = الكمون المفرد × القدرة.
    ⚠️ تختلف عن `energy_mj_per_inf_*` المحسوبة على دفعة 1024 (تُقسِّم عبء الاستدعاء
       على الدفعة) بنحو 250× للغابات. كلاهما يُبلَّغ ، والجبهة الباريتية تُرسم بالمفردة."""
    return {f"energy_single_mj_{prof}": float(lat_ms_median) * w
            for prof, w in POWER_PROFILES_W.items()}


# ------------------------------------------------------------------
#  (٥) كلفة الاتصال — بايتات لكلّ قرار وبايتات التحديث
# ------------------------------------------------------------------
def communication_cost(n_features, deployable_bytes, raw_bytes_per_decision=None, feat_bytes=4):
    """بايتات الميزات المرسَلة لقرار واحد (إن كان الكشف خارج الطائرة) مقابل الخام ،
    وبايتات النموذج المنقولة عند تحديثه (إن كان الكشف على متنها)."""
    fb = int(n_features) * feat_bytes
    out = dict(comm_feat_bytes_per_decision=fb, comm_model_update_bytes=int(deployable_bytes))
    if raw_bytes_per_decision:
        out["comm_raw_bytes_per_decision"] = int(raw_bytes_per_decision)
        out["comm_compression_ratio"] = float(raw_bytes_per_decision / fb) if fb else float("nan")
    return out


# ------------------------------------------------------------------
#  (٦) ملفّ الحافة المُحاكى — pure-Python tree inference + ONNX Runtime
# ------------------------------------------------------------------
class PurePythonTrees:
    """مستدلّ أشجار بلغة بايثون صرفة (قوائم وأعداد) — ما يُنفَّذ على متحكّم بلا numpy.
    يدعم شجرة مفردة وغابة (تصويت بالأغلبية على الاحتمالات)."""

    def __init__(self, model):
        self.trees = []
        for t in MZ.trees_of(model):
            tt = t.tree_
            self.trees.append((tt.children_left.tolist(), tt.children_right.tolist(),
                               tt.feature.tolist(), tt.threshold.tolist(),
                               [list(map(float, v[0])) for v in tt.value.tolist()]))
        self.classes = list(map(int, MZ.unwrap(model).classes_))
        self.sc = None
        if type(model).__name__ == "Pipeline" and hasattr(model.named_steps.get("sc"), "mean_"):
            sc = model.named_steps["sc"]
            self.sc = (sc.mean_.tolist(), sc.scale_.tolist())

    def predict_one(self, x):
        if self.sc is not None:
            x = [(v - m) / s for v, m, s in zip(x, *self.sc)]
        acc = None
        for left, right, feat, thr, val in self.trees:
            n = 0
            while left[n] != -1:
                n = left[n] if x[feat[n]] <= thr[n] else right[n]
            v = val[n]; s = sum(v) or 1.0
            p = [a / s for a in v]
            acc = p if acc is None else [a + b for a, b in zip(acc, p)]
        if acc is None:
            raise ValueError("النموذج لا يحوي أشجار sklearn (XGBoost خارج المستدلّ الصرف)")
        return self.classes[max(range(len(acc)), key=acc.__getitem__)]

    def predict(self, X):
        return np.array([self.predict_one(list(map(float, r))) for r in X])


_XGB_REGISTERED = {"done": False}


def _register_xgboost():
    """يسجّل محوّل XGBoost في skl2onnx (عبر onnxmltools) مرّة واحدة ؛ إن غاب يبقى
    XGBoost بلا ONNX ويُسجَّل الخطأ في العمود edge_onnx_error (لا تخمين)."""
    if _XGB_REGISTERED["done"]:
        return
    try:
        from skl2onnx import update_registered_converter
        from skl2onnx.common.shape_calculator import calculate_linear_classifier_output_shapes
        from onnxmltools.convert.xgboost.operator_converters.XGBoost import convert_xgboost
        from xgboost import XGBClassifier
        update_registered_converter(XGBClassifier, "XGBoostXGBClassifier",
                                    calculate_linear_classifier_output_shapes, convert_xgboost,
                                    options={"nocl": [True, False], "zipmap": [True, False, "columns"]})
    except Exception:
        pass
    _XGB_REGISTERED["done"] = True


def onnx_session(model, n_features):
    """جلسة ONNX Runtime بخيط واحد ، أو None إن تعذّر التحويل (يُسجَّل ولا يُخمَّن)."""
    try:
        from skl2onnx import to_onnx
        import onnxruntime as ort
        _register_xgboost()
        X0 = np.zeros((1, n_features), np.float32)
        opts = {id(MZ.unwrap(model)): {"zipmap": False}} if hasattr(MZ.unwrap(model), "predict_proba") else None
        onx = to_onnx(model, X0, target_opset={"": 17, "ai.onnx.ml": 3}, options=opts)
        so = ort.SessionOptions(); so.intra_op_num_threads = 1; so.inter_op_num_threads = 1
        sess = ort.InferenceSession(onx.SerializeToString(), so, providers=["CPUExecutionProvider"])
        name = sess.get_inputs()[0].name
        return (lambda X: sess.run(None, {name: np.asarray(X, np.float32)})[0]), len(onx.SerializeToString())
    except Exception as e:
        return None, str(e)[:120]


def profile(model, Xte, yte, n_features, with_edge=True):
    """الصفّ الكامل: الدقّة + الكلفة + الطاقة + الحافة المُحاكاة."""
    from common import metrics as M
    _pin_single_thread(model)
    yp = model.predict(Xte)
    out = M.basic(yte, yp)
    raw, gz = serialized_size(model)
    ops, npar, meth = op_count(model)
    dep = MZ.deployable_bytes(model)
    out.update(size_mb=raw / 1e6, size_gz_mb=gz / 1e6, deployable_kb=dep / 1024 if dep >= 0 else float("nan"),
               ops_per_inference=ops, n_params=npar, ops_method=meth)
    out.update(latency_ms(model.predict, Xte))
    out.update(resource_use(model.predict, Xte))
    out.update(energy_single(out["lat_ms_median"]))
    out.update(communication_cost(n_features, dep if dep >= 0 else 0))
    if with_edge:
        pp = PurePythonTrees(model) if MZ.trees_of(model) else None
        if pp is not None:
            #  الغابات الكبيرة بطيئة ببايثون الصرف (هذا هو القياس المطلوب) ⇒ تكرارات أقلّ
            rep = 100 if len(pp.trees) <= 20 else 30
            r = latency_ms(pp.predict, Xte, repeats=rep, warmup=3, batch=16 if len(pp.trees) > 20 else 64)
            out.update(edge_py_lat_ms_median=r["lat_ms_median"], edge_py_lat_ms_p95=r["lat_ms_p95"],
                       edge_py_agree=float((pp.predict(Xte[:200]) == yp[:200]).mean()))
        else:
            out.update(edge_py_lat_ms_median=float("nan"), edge_py_lat_ms_p95=float("nan"), edge_py_agree=float("nan"))
        sess, info = onnx_session(model, n_features)
        if sess is not None:
            r = latency_ms(sess, Xte)
            out.update(edge_onnx_lat_ms_median=r["lat_ms_median"], edge_onnx_lat_ms_p95=r["lat_ms_p95"],
                       edge_onnx_bytes=int(info), edge_onnx_agree=float((np.asarray(sess(Xte[:200])).ravel() == yp[:200]).mean()))
        else:
            out.update(edge_onnx_lat_ms_median=float("nan"), edge_onnx_lat_ms_p95=float("nan"),
                       edge_onnx_bytes=-1, edge_onnx_agree=float("nan"), edge_onnx_error=info)
    return out
