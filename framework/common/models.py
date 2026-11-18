# -*- coding: utf-8 -*-
# =====================================================================
#  مكتبة النماذج + رافعات الخِفّة — framework/common/models.py
# ---------------------------------------------------------------------
#  نسخة من step4/models_zoo.py (البنى والبذور نفسها حتى تبقى الأرقام مقارَنة
#  بالفصلين الرابع والخامس) + الرافعة الناقصة في الشكل: **التقليم (pruning)**.
#
#  التقليم نوعان لأن النماذج نوعان:
#     • أشجار sklearn: تقليم كلفة-التعقيد (ccp_alpha) — يحذف الفروع التي لا
#       تُقلّل الشائبة بما يبرّر عقدها ⇒ عقد أقلّ ، بصمة أصغر ، مقارنات أقلّ.
#     • شبكة MLP: تقليم بالمقدار (magnitude pruning) — تصفير أصغر الأوزان
#       بنسبة s ⇒ أوزان غير صفرية أقلّ (البصمة المتناثرة) ؛ يُبلَّغ عدد الأوزان
#       المتبقّية لا حجم المصفوفة الكثيفة.
#  ⚠️ التكميم والتقطير يُقرآن من نتائج المرحلة الرابعة الجاهزة ولا يُعادان ؛
#     لكن الدوال هنا لأن المرحلة الخامسة تحتاج بناء النماذج نفسها على التطوير.
# =====================================================================
import copy, numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier


def _scaled(clf):
    return Pipeline([("sc", StandardScaler()), ("clf", clf)])


def build(name, seed=0, **over):
    S = dict(random_state=seed)
    if name == "rf_heavy":
        return RandomForestClassifier(n_estimators=300, n_jobs=1, **S, **over)
    if name == "et_heavy":
        return ExtraTreesClassifier(n_estimators=300, n_jobs=1, **S, **over)
    if name == "xgb_heavy":
        return XGBClassifier(n_estimators=300, max_depth=8, learning_rate=0.1,
                             n_jobs=1, verbosity=0, tree_method="hist", **S)
    if name == "mlp_heavy":
        return _scaled(MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=400, **S))
    if name == "svm_rbf":
        return _scaled(SVC(kernel="rbf", **S))
    if name == "knn":
        return _scaled(KNeighborsClassifier(7, n_jobs=1))
    if name == "rf_light":
        return RandomForestClassifier(n_estimators=20, max_depth=8, n_jobs=1, **S, **over)
    if name == "rf_tiny":
        return RandomForestClassifier(n_estimators=8, max_depth=6, n_jobs=1, **S, **over)
    if name == "xgb_light":
        return XGBClassifier(n_estimators=30, max_depth=4, learning_rate=0.3,
                             n_jobs=1, verbosity=0, tree_method="hist", **S)
    if name == "dt_light":
        return DecisionTreeClassifier(max_depth=8, **S, **over)
    if name == "mlp_light":
        return _scaled(MLPClassifier(hidden_layer_sizes=(16,), max_iter=400, **S))
    if name == "logreg":
        return _scaled(LogisticRegression(max_iter=1000, **S))
    raise ValueError(f"نموذج غير معروف: {name}")


MODELS = {
    "rf_heavy":  dict(weight="heavy", label="Random Forest (300)"),
    "et_heavy":  dict(weight="heavy", label="Extra Trees (300)"),
    "xgb_heavy": dict(weight="heavy", label="XGBoost (300, d8)"),
    "mlp_heavy": dict(weight="heavy", label="MLP (128,64)"),
    "svm_rbf":   dict(weight="heavy", label="SVM (RBF)"),
    "knn":       dict(weight="heavy", label="KNN (k=7)"),
    "rf_light":  dict(weight="light", label="Random Forest (20, d8)"),
    "rf_tiny":   dict(weight="light", label="Random Forest (8, d6)"),
    "xgb_light": dict(weight="light", label="XGBoost (30, d4)"),
    "dt_light":  dict(weight="light", label="Decision Tree (d8)"),
    "mlp_light": dict(weight="light", label="MLP (16)"),
    "logreg":    dict(weight="light", label="Logistic Regression"),
}
#  النماذج التي تدخل ترتيب الثقة (المرحلة الخامسة استبعدت svm_rbf: لا predict_proba)
TRUST_MODELS = ["rf_heavy", "et_heavy", "xgb_heavy", "mlp_heavy", "knn",
                "rf_light", "rf_tiny", "xgb_light", "dt_light", "mlp_light", "logreg"]
PRUNABLE_TREES = ["dt_light", "rf_light", "rf_tiny", "rf_heavy", "et_heavy"]
PRUNABLE_MLP = ["mlp_light", "mlp_heavy"]


def unwrap(model):
    return model.named_steps["clf"] if isinstance(model, Pipeline) else model


def trees_of(model):
    m = unwrap(model)
    if hasattr(m, "tree_"):
        return [m]
    if hasattr(m, "estimators_"):
        return [e for e in np.asarray(m.estimators_).ravel() if hasattr(e, "tree_")]
    return []


# ------------------------------------------------------------------
#  البصمة الصافية للنشر — deployable footprint (تعريف المرحلة الرابعة)
# ------------------------------------------------------------------
def deployable_bytes(model, thr_bits=32):
    """بايتات التمثيل المضغوط الذي يُنقَل إلى الطائرة (لا حجم joblib).
    شجرة: لكل عقدة داخلية (ميزة 2B + عتبة thr_bits/8 + مؤشّر أبناء 2×2B) ، لكل
    ورقة (فئة 1B). خطّي/MLP: الأوزان **غير الصفرية** × 4B + مؤشّراتها عند التناثر."""
    m = unwrap(model)
    pre = 0
    if isinstance(model, Pipeline):
        sc = model.named_steps.get("sc")
        if sc is not None and hasattr(sc, "mean_"):
            pre = int(sc.mean_.size * 2 * 4)
    ts = trees_of(model)
    if ts:
        tot = 0
        for t in ts:
            tt = t.tree_
            n_int = int((tt.children_left != -1).sum())
            n_leaf = int(tt.node_count - n_int)
            tot += n_int * (2 + thr_bits // 8 + 4) + n_leaf * 1
        return tot + pre
    if hasattr(m, "coefs_"):
        nz = sum(int(np.count_nonzero(w)) for w in m.coefs_)
        tot = sum(int(w.size) for w in m.coefs_)
        b = sum(int(v.size) for v in m.intercepts_)
        sparse = nz < tot
        return (nz * 4 + (nz * 4 if sparse else 0) + b * 4) + pre
    if hasattr(m, "coef_"):
        return int(np.count_nonzero(m.coef_) * 4 + np.asarray(m.intercept_).size * 4) + pre
    if hasattr(m, "support_vectors_"):
        return int(np.asarray(m.support_vectors_).size * 4 + np.asarray(m.dual_coef_).size * 4) + pre
    if hasattr(m, "_fit_X"):
        return int(np.asarray(m._fit_X).size * 4 + len(m._y)) + pre
    if type(m).__name__.startswith("XGB"):
        try:
            df = m.get_booster().trees_to_dataframe()
            n_int = int((df.Feature != "Leaf").sum()); n_leaf = int((df.Feature == "Leaf").sum())
            return n_int * (2 + thr_bits // 8 + 4) + n_leaf * 4 + pre
        except Exception:
            return -1
    return -1


def n_nodes(model):
    ts = trees_of(model)
    if ts:
        return int(sum(t.tree_.node_count for t in ts))
    m = unwrap(model)
    if hasattr(m, "coefs_"):
        return int(sum(np.count_nonzero(w) for w in m.coefs_))
    return -1


# ------------------------------------------------------------------
#  (١) تقليم الأشجار — cost-complexity pruning
# ------------------------------------------------------------------
def prune_trees(name, seed, Xtr, ytr, ccp_alpha):
    """يبني النموذج نفسه مع ccp_alpha ⇒ شجرة/غابة مقلَّمة. يُعيد (model, n_nodes)."""
    if name not in PRUNABLE_TREES:
        return None, -1
    m = build(name, seed, ccp_alpha=float(ccp_alpha))
    m.fit(Xtr, ytr)
    return m, n_nodes(m)


#  ⚠️ شبكة alpha **ثابتة ومشتركة** بين الطيّات والمسارات. النسخة الأولى اشتقّتها من
#     مسار التقليم لكلّ طيّة ⇒ أسماء المتغيّرات لم تتطابق بين الطيّات ، فقارن الملخّص
#     متوسّط ثلاث طيّات (غير مقلَّم) بطيّة واحدة (مقلَّم) وأنتج «0.658 → 0.803» الخاطئ.
#     الشبكة اللوغاريتمية الثابتة تجعل الاختيار المتداخل ممكناً (alpha تُختار على
#     الطيّات الأخرى ثم تُقيَّم على هذه الطيّة).
CCP_GRID = [0.0, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1]


def ccp_grid(name=None, seed=None, Xtr=None, ytr=None, k=None):
    """الشبكة الثابتة (المعاملات مُبقاة للتوافق ولا تُستعمل)."""
    return list(CCP_GRID) if k is None else [0.0] + list(CCP_GRID[1::2])[: k - 1]


# ------------------------------------------------------------------
#  (٢) تقليم الشبكة — magnitude pruning
# ------------------------------------------------------------------
def prune_mlp(model, sparsity):
    """نسخة من الشبكة بتصفير أصغر |w| بنسبة sparsity (عالمياً عبر الطبقات).
    بلا إعادة تدريب (one-shot) ⇒ نقيس ما يخسره النموذج فعلاً من إزالة الأوزان."""
    out = copy.deepcopy(model)
    m = unwrap(out)
    if not hasattr(m, "coefs_"):
        return None
    allw = np.concatenate([np.abs(w).ravel() for w in m.coefs_])
    if sparsity <= 0:
        return out
    thr = np.quantile(allw, sparsity)
    for w in m.coefs_:
        w[np.abs(w) < thr] = 0.0
    return out


def mlp_sparsity(model):
    m = unwrap(model)
    tot = sum(w.size for w in m.coefs_)
    nz = sum(np.count_nonzero(w) for w in m.coefs_)
    return 1.0 - nz / tot
