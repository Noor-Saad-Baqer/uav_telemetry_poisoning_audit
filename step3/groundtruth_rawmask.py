# -*- coding: utf-8 -*-
# =====================================================================
#  إعادة إنتاج توافق **القاعدة المعيارية** (قناع خام) مع الحقيقة الأرضية
# ---------------------------------------------------------------------
#  المشكلة: جدول تدقيق الوسم في الورقة يقارن صفّين — القاعدة المعيارية
#  (عتبة بواقي كالمان الخام) والقاعدة المصحَّحة (envelope). لكن السكربت
#  يكتب دائماً إلى `p3_groundtruth_agreement.csv` ، فلمّا تحوّل الإعداد إلى
#  envelope كُتب فوق أرقام الصفّ المعياري ولم يبقَ لها ملفّ في المشروع.
#  النتيجة: **أهمّ جدول في الورقة نصفه غير قابل لإعادة الإنتاج.**
#
#  هذا السكربت يشغّل التدقيق نفسه بالقناع الخام ويحفظه باسم منفصل ، فيصير
#  الصفّان مصدَرين حيَّين لا واحداً. لا يمسّ ملفّ envelope إطلاقاً.
# =====================================================================
import os, sys, shutil, warnings
warnings.filterwarnings("ignore")
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "experiments"))
import config as CFG

LIVE = os.path.join(CFG.RESULTS_DIR, "p3_groundtruth_agreement.csv")
OUT = os.path.join(CFG.RESULTS_DIR, "p3_groundtruth_agreement_rawmask.csv")
BAK = LIVE + ".envelope.bak"

prev = getattr(CFG, "SPOOF_MASK_MODE", "raw")
if prev != "envelope":
    raise RuntimeError(f"SPOOF_MASK_MODE = {prev!r} — يُتوقَّع 'envelope' في الإعداد "
                       "الحالي. أوقفتُ العمل حتى لا يُكتب فوق ملفّ خاطئ.")

if os.path.exists(LIVE):
    shutil.copy2(LIVE, BAK)          # حماية ملفّ envelope قبل أي كتابة
    print(f"نُسخ احتياطياً: {os.path.basename(BAK)}")

CFG.SPOOF_MASK_MODE = "raw"          # ⇒ يقرؤه poison_core._active_mask
for m in list(sys.modules):
    if m in ("poison_core", "groundtruth_labels"):
        del sys.modules[m]

import groundtruth_labels as G
print(">> تشغيل التدقيق بالقناع الخام (القاعدة المعيارية) ...", flush=True)
try:
    G.main()
finally:
    CFG.SPOOF_MASK_MODE = prev

if not os.path.exists(LIVE):
    raise RuntimeError("لم يُنتَج ملفّ التوافق")
shutil.move(LIVE, OUT)
if os.path.exists(BAK):
    shutil.move(BAK, LIVE)           # يُعاد ملفّ envelope كما كان
    print("أُعيد ملفّ envelope إلى مكانه")

import pandas as pd
r = pd.read_csv(OUT)
ok = r[(r.status == "ok") & (r.cls == "GPS Spoofing")].copy()
ok["IoU"] = (ok.tp / (ok.tp + ok.fp + ok.fn)).round(3)
print("\n=== القاعدة المعيارية (قناع خام) — رحلات الانتحال ===")
print(ok[["airframe", "n", "kappa", "IoU", "tp", "fp", "fn"]].to_string(index=False))

e = pd.read_csv(LIVE)
eo = e[(e.status == "ok") & (e.cls == "GPS Spoofing")].copy()
eo["IoU"] = (eo.tp / (eo.tp + eo.fp + eo.fn)).round(3)
print("\n=== القاعدة المصحَّحة (envelope) — للمقارنة ===")
print(eo[["airframe", "n", "kappa", "IoU"]].to_string(index=False))
print(f"\n✓ {os.path.relpath(OUT, os.path.dirname(HERE))}")
