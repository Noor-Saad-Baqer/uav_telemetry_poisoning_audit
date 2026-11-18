# -*- coding: utf-8 -*-
# =====================================================================
#  المستوى صفر — وسمٌ مستقلّ تماماً عن فضاء الميزات
# ---------------------------------------------------------------------
#  الثغرة: الوسم الحالي مشتقّ من عتبة على vel_pos_innov[3],[4] ، والميزات
#  قنوات بواقي كالمان **أخرى**. حذف الإشارتين يعالج أفجّ الدائرية ، لكن يبقى
#  السؤال: هل القنوات المتبقّية مستقلّة عن المحذوفتين إحصائياً؟ إن لم تكن ،
#  فرقم 1.000 قد يُظهر ارتباط قنوات لا كشفاً.
#
#  العلاج: المحاكي يسجّل الموقع **الحقيقي** في
#  `vehicle_global_position_groundtruth` — وهو سابق سببياً للهجوم ولا يشترك
#  في أيّ قناة مع الميزات. الوسم منه = ‖موقع GPS المُبلَّغ − الموقع الحقيقي‖.
#
#  هذا السكربت لا يُغيّر شيئاً: يقيس **التوافق** بين القناعين (Cohen's κ)
#  ليُعرَف هل قاعدة الوسم الحالية صحيحة أم لا.
# =====================================================================
import os, sys, glob, warnings, numpy as np, pandas as pd
warnings.filterwarnings("ignore")
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "experiments"))
import config as CFG
import poison_core as P
from sklearn.metrics import cohen_kappa_score, confusion_matrix

GT_THR_M = 50.0        # عتبة انحراف الموقع (م) — الانتحال يبلغ عشرات الكيلومترات
MIN_VALID = 0.5        # أقلّ نسبة صفوف صالحة لقبول الرحلة


def _gt_error(folder):
    """خطأ الموقع (م) بين GPS المُبلَّغ والموقع الحقيقي ، على شبكة زمن GPS."""
    gt = glob.glob(os.path.join(folder, "*vehicle_global_position_groundtruth*.csv"))
    gp = glob.glob(os.path.join(folder, "*vehicle_gps_position_0.csv"))
    if not gt or not gp:
        return None
    G = pd.read_csv(gt[0])[["timestamp", "lat", "lon"]].dropna().sort_values("timestamp")
    S = pd.read_csv(gp[0])[["timestamp", "lat", "lon"]].dropna().sort_values("timestamp")
    S["lat"] = S["lat"] / 1e7; S["lon"] = S["lon"] / 1e7
    #  صفوف ما قبل تثبيت GPS تُسجَّل عند (0,0) ⇒ خطأ ~10,024 كم وهمي. تُستبعَد.
    ok = (S.lat.abs() > 1e-6) & (S.lon.abs() > 1e-6) & (G.lat.abs().max() > 1e-6)
    S = S[ok]
    G = G[(G.lat.abs() > 1e-6) & (G.lon.abs() > 1e-6)]
    if len(S) < 50 or len(G) < 50:
        return None
    M = pd.merge_asof(S, G, on="timestamp", direction="nearest", suffixes=("_g", "_t"))
    #  تقريب مسطّح كافٍ عند هذه المسافات (الانتحال بعشرات الكيلومترات)
    lat0 = np.deg2rad(M.lat_t.median())
    dy = (M.lat_g - M.lat_t) * 111320.0
    dx = (M.lon_g - M.lon_t) * 111320.0 * np.cos(lat0)
    M["err_m"] = np.hypot(dx, dy)
    return M[["timestamp", "err_m"]]


def main():
    tx_base = P._tx_baselines()
    rows = []
    for af, cls, paths in P.discover_flights():
        folder = os.path.dirname(paths["innovations"])
        E = _gt_error(folder)
        note = ""
        if E is None:
            rows.append({"airframe": af, "cls": cls, "status": "لا يوجد groundtruth"})
            continue

        #  القناع الحالي (المشتقّ من بواقي كالمان) على شبكة الـ innovations
        df, sigs = P._fuse(paths)
        cur = P._active_mask(df, cls, af, tx_base)
        base = pd.DataFrame({"timestamp": df["timestamp"].values, "cur": cur})

        #  القناع المستقلّ ، مُسقَطاً على الشبكة نفسها
        M = pd.merge_asof(base.sort_values("timestamp"),
                          E.sort_values("timestamp"), on="timestamp",
                          direction="nearest")
        M = M.dropna(subset=["err_m"])
        if len(M) < 100:
            rows.append({"airframe": af, "cls": cls, "status": "تداخل زمني غير كافٍ"})
            continue
        gtm = (M.err_m > GT_THR_M).values
        curm = M.cur.values.astype(bool)

        k = cohen_kappa_score(curm, gtm) if (curm.any() or gtm.any()) else 1.0
        tn, fp, fn, tp = confusion_matrix(gtm, curm, labels=[False, True]).ravel()
        rows.append({"airframe": af, "cls": cls, "status": "ok", "n": len(M),
                     "frac_gt": round(float(gtm.mean()), 4),
                     "frac_cur": round(float(curm.mean()), 4),
                     "kappa": round(float(k), 3),
                     "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn)})

    df = pd.DataFrame(rows)
    out = os.path.join(CFG.RESULTS_DIR, "p3_groundtruth_agreement.csv")
    df.to_csv(out, index=False)

    print("=" * 92)
    print(f"توافق قناع الوسم الحالي (بواقي كالمان) مع قناع مستقلّ (الموقع الحقيقي ، عتبة {GT_THR_M:.0f} م)")
    print("=" * 92)
    ok = df[df.status == "ok"]
    for r in df.itertuples():
        if r.status != "ok":
            print(f"  {r.airframe:16s} {r.cls:13s}  — {r.status}")
    print()
    for r in ok.itertuples():
        flag = "★" if r.cls == "GPS Spoofing" else " "
        print(f"{flag} {r.airframe:16s} {r.cls:13s} n={int(r.n):6d}  "
              f"نشاط: مستقلّ={r.frac_gt:.4f} حالي={r.frac_cur:.4f}  "
              f"κ={r.kappa:+.3f}   tp={int(r.tp):5d} fp={int(r.fp):5d} fn={int(r.fn):5d}")

    sp = ok[ok.cls == "GPS Spoofing"]
    nm = ok[ok.cls == "Normal"]
    print("\n" + "-" * 92)
    if len(sp):
        print(f"رحلات الانتحال (n={len(sp)}): κ الوسيط = {sp.kappa.median():+.3f}  "
              f"| نشاط مستقلّ {sp.frac_gt.min():.4f}–{sp.frac_gt.max():.4f}  "
              f"| نشاط حالي {sp.frac_cur.min():.4f}–{sp.frac_cur.max():.4f}")
    if len(nm):
        print(f"الرحلات الطبيعية (n={len(nm)}): أعلى نشاط مستقلّ = {nm.frac_gt.max():.4f}  "
              f"(يجب أن يكون ~0 ⇒ إنذار كاذب للمؤشّر المستقلّ)")
    print("-" * 92)
    print(f"\n✓ {os.path.relpath(out)}")


if __name__ == "__main__":
    main()
