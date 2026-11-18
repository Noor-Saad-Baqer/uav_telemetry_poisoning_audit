# -*- coding: utf-8 -*-
# =====================================================================
#  المرحلة الثالثة — ملف الإعدادات المركزي
#  Step 3 — Central configuration (edit ONLY this file to change results)
# ---------------------------------------------------------------------
#  كل المعاملات القابلة للتعديل هنا. لا حاجة لتعديل منطق الخوارزميات.
#  All tunables live here. No need to touch algorithm code elsewhere.
# =====================================================================
import os as _os

# ============ (1) المهمة — Task ============
#  ثلاث فئات من قياسات الطيران (تليمتري) لطائرة PX4
CLASSES     = ["Normal", "GPS Spoofing", "Ping DoS"]
NUM_CLASSES = len(CLASSES)

# ============ (2) بناء الميزات من السلاسل الزمنية — Windowing ============
#  ⚠️ اكتشاف مهم: محاكي PX4 (SITL) يثبّت إشارات صحّة الـ GPS على قيم ثابتة
#     (jamming_indicator=0 ، noise_per_ms=0 ، hdop=0 ، eph=1 ، satellites=10)
#     في كل الفئات ⇒ صفر معلومة. دليل الهجوم الحقيقي موجود في:
#       • innovations      → بواقي مرشّح كالمان تكشف انتحال GPS (330 مقابل 0.057)
#       • telemetry_status → إحصاءات الوصلة تكشف Ping DoS (rate_rx>0 ، rate_tx×3)
WINDOW      = 50      # عدد العيّنات في النافذة الزمنية
STRIDE      = 2       # الإزاحة بين النوافذ (تداخل داخل الرحلة فقط)
USE_GPS     = False   # دمج ميزات GPS المشتقّة؟ (ablation — يقلّص عدد الرحلات)
LOG_SCALE_INNOV = True  # ضغط لوغاريتمي لبواقي كالمان (ضروري للانتقال عبر المنصّات)

# ============ (2ب) وسم النوافذ — Window-level labelling ============
#  ⚠️ اكتشاف حاسم: اسم المجلّد يسِم الرحلة كاملةً، لكن الهجوم لا يشغل إلا جزءاً
#     منها: انتحال GPS = 1.8%–5.4% فقط من الرحلة ، Ping DoS = 34%–51%.
#     ⇒ الوسم حسب المجلّد يجعل ~98% من نوافذ «الانتحال» مغلوطة (هي في الحقيقة
#     طبيعية). لذلك نسِم كل نافذة على حدة حسب نشاط الهجوم الفعلي.
#
#  مؤشّر النشاط مشتقّ فيزيائياً ومُتحقَّق منه: يعطي 0.000 على الرحلات الطبيعية
#  الستّ جميعها (صفر إنذار كاذب) ⇒ عتبة سليمة لا اعتباطية.
LABEL_MODE       = "window"   # window (مصحّح) | flight (القديم، للمقارنة فقط)
SPOOF_INNOV_THR  = 10.0       # متر: |vel_pos_innov[3],[4]| فوقها ⇒ المرشّح يرفض الـ GPS
#  ── قناع الانتحال: "raw" = العتبة كما هي (يُعلّم الانتقالات) | "envelope" = المصحَّح ──
#  العتبة الخام تُعلّم **انتقالات المرشّح** لا الهجوم: تنفجر البواقي عند بدء الانتحال ،
#  ثم تهدأ حين يقتنع المرشّح بالموقع المزيّف (والهجوم مستمرّ) ، ثم تنفجر ثانيةً أثناء
#  العودة **بعد** انتهائه. فنحو نصف عيّناتها الموسومة «هجوم» تقع بعد نهايته.
#  قاعدة الظرف: من بداية أوّل ارتفاع إلى بداية آخر ارتفاع.
#  مُتحقَّق منها مقابل `vehicle_global_position_groundtruth`: κ = 0.990–0.993 و IoU 0.98–0.99
#  (مقابل κ = 0.46 و IoU = 0.32 للخام). انظري step3/groundtruth_labels.py
SPOOF_MASK_MODE  = "envelope"
#  1.5× يعمل لكل الطائرات: خطّ أساس QUAD-HITL مرتفع أصلاً (وسيط 39.3 مقابل 15.0)
#  فعتبة الضِعف كانت تفوّت هجومه كلّياً (وسيط 71.8).
DOS_TX_MULT      = 1.5        # rate_tx فوق 1.5× خطّ أساس الرحلة الطبيعية لنفس الطائرة
ACTIVE_FRAC      = 0.5        # نسبة العيّنات النشطة داخل النافذة لاعتبارها هجوماً
#  موازنة الفئات: النوافذ الطبيعية أكثر بكثير ⇒ تُقلَّص إلى مضاعف أصغر فئة هجوم
NORMAL_RATIO     = 3.0

#  بواقي مرشّح كالمان — التوقيع الفيزيائي لانتحال GPS
INNOV_SIGNALS = ["vel_pos_innov[0]", "vel_pos_innov[1]", "vel_pos_innov[2]",
                 "vel_pos_innov[3]", "vel_pos_innov[4]", "vel_pos_innov[5]",
                 "heading_innov", "mag_innov[0]", "mag_innov[1]", "mag_innov[2]",
                 "vel_pos_innov_var[0]", "vel_pos_innov_var[3]", "vel_pos_innov_var[5]"]
#  إحصاءات وصلة الاتصال — التوقيع الشبكي لـ Ping DoS
TELEM_SIGNALS = ["rate_rx", "rate_tx", "rate_txerr", "rate_multiplier"]
#  ميزات GPS المشتقّة (تُستعمل فقط إذا USE_GPS=True)
GPS_SIGNALS = ["vel_m_s", "alt", "cog_rad", "heading"]
#  إحصاءات كل إشارة داخل النافذة
WIN_STATS   = ["mean", "std", "min", "max", "slope"]

# ============ (2ج) كسر دائرية الوسم — Label-leakage control ============
#  ⚠️ مشكلة: الوسم مُشتقّ من إشارات موجودة داخل مصفوفة الميزات نفسها:
#     • انتحال GPS ← |vel_pos_innov[3],[4]| > 10
#     • Ping DoS   ← rate_tx > 1.5× خطّ الأساس
#  ⇒ المصنّف قد «يكشف» بإعادة قراءة الوسم لا بتعلّم توقيع حقيقي.
#     (قياس: عتبة على ميزة واحدة تعطي BalAcc = 0.763 لفئة الانتحال.)
#  الحلّ: نحسب قناع النشاط من هذه الإشارات (فهي تعريف الوسم) ثم **نحذفها من
#  الميزات** ، فيُجبَر النموذج على التعلّم من بقيّة القنوات.
EXCLUDE_LABEL_FEATURES = True
LABEL_LEAK_SIGNALS = ["vel_pos_innov[3]", "vel_pos_innov[4]", "rate_tx"]

# ============ (3) بروتوكول التقييم — Leave-One-Airframe-Out ============
#  الطائرات المكتملة (فيها الفئات الثلاث بطوبيكَي innovations+telemetry)
TEST_AIRFRAMES  = ["PX4-H480-SITL", "PX4-QUAD-HITL", "PX4-QUAD-SITL", "PX4-VTOL-SITL"]
#  الطائرات الناقصة (فئة مفقودة) → تبقى دائماً في التدريب
TRAIN_ONLY_AIRFRAMES = ["PX4-PLANE-SITL", "PX4-TAIL-SITL"]
#  بروتوكول ثانوي: تقسيم زمني داخل الرحلة (منصّة معروفة) — للمقارنة
PROTOCOL    = _os.environ.get("P3_PROTOCOL") or "loao"   # loao | temporal
TEMPORAL_TRAIN_FRAC = 0.7
SEEDS       = [0, 1, 2, 3, 4]   # بذور التكرار (تُبلّغ عنها النتائج)

# ============ (4) النماذج — Models ============
#  الجهاز: cpu افتراضاً (الشبكة صغيرة ⇒ الـ GPU لا يفيد ويفشل تحت ضغط الذاكرة)
DEVICE      = _os.environ.get("P3_DEVICE") or "cpu"
MODELS      = ["rf", "xgb", "mlp"]   # المتاح: rf | xgb | mlp
#  الضحيّة = الغابة العشوائية منذ 2026-08-15: بعد تصحيح قناع الوسم عجزت الشبكة
#  النقية عن المهمّة عبر المنصّات (BalAcc 0.50 / استدعاء انتحال 0.40 — قرب الصدفة)
#  ولم يُصلحها تدريب أطول ولا أوزان فئات ، فصارت دراسة التسميم عليها بلا معنى.
VICTIM      = "rf"                   # النموذج الضحيّة في تجارب التسميم: rf | mlp
MLP_HIDDEN  = (128, 64)
DROPOUT     = 0.3
EPOCHS      = 60
BATCH_SIZE  = 256
LR          = 1e-3

# ============ (5) هجمات التسميم — Poisoning Attacks ============
ATTACKS     = ["random_flip", "targeted_flip", "backdoor", "clean_label"]
POISON_RATES = [0.0, 0.05, 0.10, 0.20, 0.30, 0.40]
#  الهجوم الموجَّه: إخفاء الانتحال بجعله يبدو طبيعياً (الأخطر أمنياً)
#  ⚠️ قاعدة نسبة التسميم للهجمات أحادية الفئة (targeted_flip ، clean_label):
#    source_class — الصحيح: النسبة من الفئة المستهدَفة (1.0 = قلب الفئة كاملة)
#    train_set    — القديم (خاطئ): النسبة من كامل التدريب ⇒ يستنفد الفئة عند 15%
#  التصحيح ضروري: بالقاعدة القديمة لم يكن يبقى ولا مثال انتحال صحيح فوق 15% ،
#  فيصير استدعاء 0.000 مضموناً بالبناء لا نتيجةً عن ضعف الدفاع.
RATE_BASIS  = "source_class"
TARGET_SRC  = "GPS Spoofing"
TARGET_DST  = "Normal"
TRIGGER_DIM = 8       # عدد الميزات التي يعدّلها الباب الخلفي
TRIGGER_MAG = 3.0     # شدّة الباب الخلفي (بوحدات الانحراف المعياري)

# ============ (6) كاشفات التسميم — Detectors ============
DETECTORS   = ["loss_based", "loss_deviation", "spectral", "activation_cluster", "knn_consistency"]
KNN_K       = 20      # جيران KNN لفحص اتّساق التسمية
WARMUP_EP   = 10      # حِقب التسخين قبل قياس الخسارة لكل عيّنة

# ============ (7) الدفاعات — Defenses ============
#  robust_loss (GCE) محذوف مع ضحيّة الغابة: دفاعات تشكيل الخسارة تتطلّب متعلّماً
#  قابلاً للاشتقاق ⇒ غير منطبقة على أشجار التجميع — وهي نفسها ملاحظة تستحقّ الذكر.
DEFENSES    = ["filtered_retrain", "trimmed_agg", "relabel_retrain"]
FILTER_Q    = 0.25    # نسبة العيّنات المشبوهة التي تُحذف
FILTER_DETECTOR = "loss_deviation"   # الكاشف المستعمل في دفاع التصفية
GCE_Q       = 0.7     # معامل Generalized Cross Entropy
TRIM_FRAC   = 0.20    # نسبة القصّ في التجميع المقصوص
#  دفاع إعادة الوسم: عتبة الثقة المطلوبة لتصحيح تسمية عيّنة مشبوهة
#  (0 = أعد وسم كل المشبوهات ؛ 0.9 = صحّحي فقط عند ثقة عالية)
RELABEL_CONF = 0.0

# ============ (8) تحسين الدفاع بالميتاهيوريستية — Stage F ============
#  يعيد استخدام step2/experiments/optimizers.py (نفس الخوارزميات الخمس)
OPTIMIZERS  = ["pso", "gwo", "lgwo", "hho", "cs"]
OPT_SEEDS   = [123, 456, 789]
EVAL_BUDGET = int(_os.environ.get("P3_BUDGET") or 40)
POP_SIZE    = 10
#  بذرة بحث واحدة تكفي: دالّة الهدف تُوسِّط أصلاً على 3 نسب تسميم (RATES_FOR_OPT)
#  أي 3 تدريبات لكل تقييم. التقييم الخارجي يستعمل 3 بذور اختبار مستقلّة.
#  (بذرتان إضافيتان تضاعفان الزمن ×3 بلا فائدة تُذكر: ~2.2 ثانية لكل تقييم.)
SEARCH_SEEDS = [10]            # بذور بحث منفصلة عن بذور الاختبار (بلا تسريب)
DEFENSE_BOUNDS = {
    "filter_q":  (0.05, 0.50),   # نسبة الحذف
    "gce_q":     (0.10, 0.95),   # معامل GCE
    "trim_frac": (0.05, 0.40),   # نسبة القصّ
    "lr_log":    (-4.0, -2.5),   # log10(معدّل التعلّم)
}
#  هدف البحث الداخلي في المرحلة F:
#    recall — الأصلي: يكافئ الأداء بعد الدفاع فقط. اتّضح أنه لا يكافئ اختيار
#             الكاشف الأصلح (اختار loss_deviation في 60/60 ولم يختر
#             activation_cluster قطّ رغم تفوّقه: AUC 0.788 مقابل 0.585).
#    auc    — يكافئ جودة الكشف فقط ⇒ filter_q و lr يصيران بلا أثر (بُعد واحد).
#    blend  — الموصى به: يكافئ الكاشف الأصلح مع إبقاء بقيّة الأبعاد فاعلة.
OPT_OBJECTIVE  = _os.environ.get("P3_OBJ") or "recall"   # recall | auc | blend
OPT_AUC_WEIGHT = 0.5               # وزن AUC في blend
ATTACK_FOR_OPT = "targeted_flip"   # الهجوم الذي يُحسَّن الدفاع ضدّه
#  ⚠️ لا تُحسِّني عند نسبة واحدة: الدفاع يستعيد كاملاً عند 10% وينهار إلى صفر
#     بحلول 15% ⇒ دالّة الهدف مسطّحة تماماً هناك (كل التقييمات = 0.000) والبحث
#     بلا معنى. نُحسِّن على *مدى* يغطّي منطقة الانتقال فتحمل الدالّة إشارة،
#     والسؤال يصبح: هل يرفع الضبطُ عتبةَ الانهيار؟
#  ⚠️ بالقاعدة المصحَّحة (نسبة من الفئة) صارت 0.10–0.20 كلّها في منطقة يعمل
#     فيها الدفاع بكفاءة تامّة (~0.99) ⇒ الهدف مسطّح من الأعلى. منطقة الانتقال
#     الحقيقية هي 0.7–0.9 (0.986 → 0.880 → 0.551) وهناك تحمل الدالّة إشارة.
RATES_FOR_OPT  = [0.7, 0.8, 0.9]

# ============ (9) الذاكرة المؤقتة — Cache ============
#  ⚠️ إذا غيّرتِ معماريّة النموذج أو بناء الميزات (WINDOW/STRIDE/MLP_HIDDEN/…)
#     ارفعي هذا الرقم بمقدار 1 لإبطال الذاكرة القديمة.
CACHE_VERSION = 1

# ============ (10) الحالة الدراسية — Case Study ============
CASE_NAME = _os.environ.get("P3_CASE") or "poison_w50"
SETTING   = "poisoning"

_STEP3      = _os.path.dirname(_os.path.abspath(__file__))
DATA_DIR    = _os.path.join(_STEP3, "..", "data",
                            "UAV Attack Data-20260702T145753Z-3-001", "UAV Attack Data")
SIM_DIR     = _os.path.join(DATA_DIR, "Simulated - OTU Survey")
LIVE_DIR    = _os.path.join(DATA_DIR, "Live GPS Spoofing and Jamming")
RESULTS_DIR = _os.path.join(_STEP3, "results", "cases", CASE_NAME)
FIGURES_DIR = _os.path.join(_STEP3, "figures", "cases", CASE_NAME)
LEGACY_RES  = _os.path.join(_STEP3, "results")
LEGACY_FIG  = _os.path.join(_STEP3, "figures")
CACHE_DIR   = _os.path.join(_STEP3, "experiments", ".cache_p3")
for _d in (RESULTS_DIR, FIGURES_DIR, LEGACY_RES, LEGACY_FIG, CACHE_DIR):
    _os.makedirs(_d, exist_ok=True)
