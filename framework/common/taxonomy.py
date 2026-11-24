# -*- coding: utf-8 -*-
# =====================================================================
#  التصنيف الموحَّد — 11 فئة هجوم ← 5 طبقات أمنية
#  (نسخة حرفية من step4/config.py — لا يُستورد ذاك الملفّ لتفادي تصادم الاسم)
# ---------------------------------------------------------------------
#  الرقم 11 = UAVIDS 4 + ISOT 5 + UAV-Attack 2. Video في ISOT هجوم (Routing/MITM)
#  حسب وصف البيانات. التوحيد على فضاء الوسوم فقط: المجموعات الثلاث لا
#  تشترك طبقة ارتباط ، ودمج ميزاتها مستحيل (أثبته الفصل الثاني: النقل بينها
#  ينهار إلى التخمين).
# =====================================================================
import numpy as np

CATEGORIES = ["Flooding/DoS", "Spoofing/Identity", "Routing/MITM",
              "Injection", "Unauthorized"]

TAXONOMY = {
    # ---- UAVIDS-2025 (4 هجمات) ----
    "Flooding Attack":  "Flooding/DoS",
    "Sybil Attack":     "Spoofing/Identity",
    "Blackhole Attack": "Routing/MITM",
    "Wormhole Attack":  "Routing/MITM",
    # ---- ISOT (9 هجمات — الإصدار الرسمي الكامل ، 2026-11-17) ----
    "DoS":              "Flooding/DoS",
    "Ip Spoofing":      "Spoofing/Identity",
    "Password Cracking": "Spoofing/Identity",   # هجوم على المصادقة (قرار 2026-11-17)
    "MITM":             "Routing/MITM",
    "Video":            "Routing/MITM",         # Video Interception: هجوم حسب الوصف الرسمي ، لا حركة طبيعية
    "Injection":        "Injection",
    "Replay":           "Injection",
    "Manipulation":     "Injection",
    "Unauth":           "Unauthorized",
    # ---- UAV Attack — تليمتري (هجومان) ----
    "Ping DoS":         "Flooding/DoS",
    "GPS Spoofing":     "Spoofing/Identity",
}
NORMAL_ALIASES = {"Normal Traffic", "Regular", "Normal"}   # Video أُزيلت 2026-11-17 (هجوم في الوصف الرسمي)

SOURCE = {}
for _k in ("Flooding Attack", "Sybil Attack", "Blackhole Attack", "Wormhole Attack"):
    SOURCE[_k] = "UAVIDS-2025"
for _k in ("DoS", "Ip Spoofing", "Password Cracking", "MITM", "Video", "Injection", "Replay", "Manipulation", "Unauth"):
    SOURCE[_k] = "ISOT"
for _k in ("Ping DoS", "GPS Spoofing"):
    SOURCE[_k] = "UAV Attack"


def to_category(names):
    out = []
    for n in names:
        s = str(n).strip()
        if s in NORMAL_ALIASES:
            out.append("Normal")
        elif s in TAXONOMY:
            out.append(TAXONOMY[s])
        else:
            raise KeyError(f"فئة غير معروفة في التصنيف الموحَّد: {s!r}")
    return np.array(out)
