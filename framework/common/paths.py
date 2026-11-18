# -*- coding: utf-8 -*-
# =====================================================================
#  الإطار الموحَّد — المسارات المركزية
#  framework/common/paths.py — every path in one place
# ---------------------------------------------------------------------
#  ⚠️ المراحل القديمة (step1..step5) لا تُلمَس. نقرأ نتائجها ولا نكتب فيها.
#  ⚠️ لا يُسمّى أيّ ملفّ هنا config.py: هذا الاسم محجوز للمرحلتين 3 و4
#     ويُستورد مجرَّداً هناك ، والتصادم ينتج بيانات خاطئة بصمت
#     (انظري step5/p5_config.py لتاريخ العطب).
# =====================================================================
import os as _os

HERE = _os.path.dirname(_os.path.abspath(__file__))     # framework/common
FW   = _os.path.dirname(HERE)                             # framework
ROOT = _os.path.dirname(FW)                               # UAV_Detection
DATA = _os.path.join(ROOT, "data")

# ---- المراحل القديمة (قراءة فقط) ----
STEP1 = _os.path.join(ROOT, "step1")
STEP2 = _os.path.join(ROOT, "step2")
STEP3 = _os.path.join(ROOT, "step3")
STEP4 = _os.path.join(ROOT, "step4")
STEP5 = _os.path.join(ROOT, "step5")
STEP1_HOMO   = _os.path.join(STEP1, "Cross-Dataset", "Homogeneous", "results")
STEP1_HETERO = _os.path.join(STEP1, "Cross-Dataset", "Heterogeneous", "results")
STEP1_MULTI  = _os.path.join(STEP1, "Cross-Dataset", "Multiclass", "results")
STEP2_CASE   = _os.path.join(STEP2, "results", "cases", "hetero_b120")
STEP2_LEGACY = _os.path.join(STEP2, "results")
STEP3_CASE   = _os.path.join(STEP3, "results", "cases", "poison_w50")
STEP4_RES    = _os.path.join(STEP4, "results")
STEP5_RES    = _os.path.join(STEP5, "results")

# ---- البيانات الخام ----
UAVIDS_CSV = _os.path.join(
    DATA, "UAV IDS-2025-20260702T145125Z-3-001", "UAV IDS-2025", "UAVIDS-2025.csv")
ISOT_DIR = _os.path.join(
    DATA, "ISOT Drone Dataset-20260702T145051Z-3-012", "ISOT Drone Dataset",
    "Dataset", "new_feature_csv")
UAV_ATTACK_DIR = _os.path.join(
    DATA, "UAV Attack Data-20260702T145753Z-3-001", "UAV Attack Data")
SIM_DIR = _os.path.join(UAV_ATTACK_DIR, "Simulated - OTU Survey")

# ---- وضع الفحص السريع ----
#  FW_QUICK=1 ⇒ عيّنات صغيرة وبذور أقلّ: للتأكّد أن الكود يعمل ، لا لإنتاج أرقام.
#  مخرجاته تُعزل في results/_quick حتى لا تُقرأ خطأً على أنها نتائج ، ولا تجعل
#  الاستئناف يتخطّى صفوفاً «مكتملة» بأرقام الفحص.
QUICK = _os.environ.get("FW_QUICK") == "1"

# ---- مخرجات الإطار ----
RESULTS = _os.path.join(FW, "results", "_quick") if QUICK else _os.path.join(FW, "results")
FIGURES = _os.path.join(FW, "figures", "_quick") if QUICK else _os.path.join(FW, "figures")
STAGES  = ["stage0", "stage1", "stage2", "stage3", "stage4", "stage5"]


def results(stage):
    d = _os.path.join(RESULTS, stage)
    _os.makedirs(d, exist_ok=True)
    return d


def figures(stage):
    d = _os.path.join(FIGURES, stage)
    _os.makedirs(d, exist_ok=True)
    return d


def legacy(stage):
    """نسخة المرحلة القديمة المقابلة داخل الإطار (نتائج جاهزة لا تُعاد)."""
    d = _os.path.join(results(stage), "legacy")
    _os.makedirs(d, exist_ok=True)
    return d


#  الختم واحد دائماً (لا نسخة «سريعة» منه): الفحص السريع يستعمل الختم الحقيقي.
SEALED_DIR = _os.path.join(FW, "results", "stage0")
_os.makedirs(SEALED_DIR, exist_ok=True)
MANIFEST   = _os.path.join(SEALED_DIR, "sealed_manifest.json")
ACCESS_LOG = _os.path.join(SEALED_DIR, "sealed_access_log.jsonl")
CACHE      = _os.path.join(FW, "results", ".cache")
_os.makedirs(CACHE, exist_ok=True)


def pin_threads(n="1"):
    """يثبّت خيوط BLAS/OpenMP. يجب استدعاؤه قبل استيراد numpy في سكربتات
    قياس الكلفة (المرحلتان 4 و5): غابة على 24 نواة ليست ما يعمل على طائرة."""
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
              "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        _os.environ[v] = str(n)
