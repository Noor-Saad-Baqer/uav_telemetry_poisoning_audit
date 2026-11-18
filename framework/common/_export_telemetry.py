# -*- coding: utf-8 -*-
# =====================================================================
#  تصدير نوافذ التليمتري من المرحلة الثالثة (عمليّة منفصلة)
# ---------------------------------------------------------------------
#  لماذا عمليّة منفصلة؟ step3/config.py يُستورد مجرَّداً باسم config داخل
#  poison_core ؛ الاستيراد من عمليّة أخرى فيها وحدة بالاسم نفسه يلتقط الإعدادات
#  الخاطئة بصمت. نُشغّل من سياق step3 فقط ونحفظ npz.
#  يحفظ إضافةً إلى step4: flights (الرحلة = طائرة|فئة) و pos (ترتيب النافذة
#  داخل رحلتها) — يلزمان للختم الزمني داخل الرحلة.
#  التشغيل: python _export_telemetry.py <out.npz>
# =====================================================================
import os, sys, numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
S3 = os.path.join(ROOT, "step3")
sys.path.insert(0, os.path.join(S3, "experiments"))
sys.path.insert(0, S3)

import config as CFG            # noqa: E402  ← config الخاصّ بالمرحلة الثالثة
import poison_core as PC        # noqa: E402

assert hasattr(CFG, "CLASSES") and hasattr(CFG, "TEST_AIRFRAMES"), \
    "التُقط config خاطئ — تحقّقي من ترتيب sys.path"

out = sys.argv[1]
X, y, groups, flights, feat = PC.build_dataset(verbose=True)
names = np.array([CFG.CLASSES[int(v)] for v in y])
flights = np.asarray(flights)
pos = np.zeros(len(flights), dtype=np.int64)
for fl in np.unique(flights):
    idx = np.where(flights == fl)[0]          # مرتّبة زمنياً بحكم البناء
    pos[idx] = np.arange(len(idx))

np.savez_compressed(out, X=np.asarray(X, dtype=np.float32), y=names,
                    groups=np.asarray(groups), flights=flights, pos=pos,
                    feat=np.asarray(feat, dtype=object),
                    test_airframes=np.asarray(CFG.TEST_AIRFRAMES, dtype=object),
                    window=CFG.WINDOW, stride=CFG.STRIDE,
                    mask_mode=CFG.SPOOF_MASK_MODE, excl=CFG.EXCLUDE_LABEL_FEATURES)
print(f"✓ صُدِّر {out}  X={X.shape}  فئات={sorted(set(names))}  رحلات={len(set(flights))}")
