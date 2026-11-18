# -*- coding: utf-8 -*-
# =====================================================================
#  المرحلة صفر — الختم: مجموعات اختبار مختومة من المجموعات الثلاث
#  Stage 0 — Sealing: held-out test partitions carved before any experiment
# ---------------------------------------------------------------------
#  هذا الصندوق هو ما يجعل جملة الشكل «sealed test partitions remain untouched
#  until Stage 5» صادقة. يُشغَّل مرّة واحدة ؛ إعادة تشغيله لا تغيّر الختم.
#
#  المخرجات (framework/results/stage0/):
#     sealed_manifest.json     — الختم نفسه (صفوف/قطع/بصمات)
#     sealed_summary.csv       — عدد العيّنات لكلّ مسار × فئة × قسم
#     sealed_verify.csv        — تحقّق أن البيانات الخام لم تتغيّر
#     sealed_access_log.jsonl  — سجلّ كلّ قراءة للمختوم (يجب أن يبقى فارغاً حتى المرحلة 5)
#
#  التشغيل:  python framework/stage0_sealing/run.py [--verify-only]
# =====================================================================
import os, sys, argparse, collections
FW = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, FW)

import numpy as np, pandas as pd
from common import paths as P, io as IO, sealing as S


def summarize():
    from common import data as D
    S.unseal("stage0-sealing")          # الغرض: العدّ والتحقّق فقط — يُسجَّل في السجلّ
    rows = []
    for track in D.TRACKS:
        for part in ("dev", "sealed"):
            d = D.load_track(track, part, unified=True, verbose=False)
            cnt = collections.Counter(d["y_txt"])
            for c in d["classes"]:
                rows.append(dict(track=track, partition=part, cls=c, n=cnt[c]))
            if track == "telemetry":
                for af in sorted(set(d["groups"].tolist())):
                    rows.append(dict(track=track, partition=part, cls=f"airframe:{af}",
                                     n=int((d["groups"] == af).sum())))
    S.reseal()
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify-only", action="store_true")
    ap.add_argument("--reseal-uavids-dedup", action="store_true", help="الختم v2: إسقاط تكرارات UAVIDS ثم إعادة معاينة 20%% (ISOT والتليمتري لا يتغيّران)")
    a = ap.parse_args()
    out = P.results("stage0")
    IO.log("=" * 70)
    IO.log("  المرحلة صفر — الختم")
    IO.log("=" * 70)
    if a.reseal_uavids_dedup:
        S.reseal_uavids_dedup()
    if not a.verify_only:
        man = S.build_manifest()
        IO.log(f"   UAVIDS: {man['uavids']['n_sealed']:,} / {man['uavids']['n_rows']:,} صفّ مختوم")
        IO.log(f"   ISOT: {len(man['isot']['files'])} ملفّ ، الذيل الزمني 20% من كلّ ملفّ")
        IO.log(f"   Telemetry: الطائرة المختومة = {man['telemetry']['sealed_airframe']} "
               f"(من {man['telemetry']['candidate_airframes']})")
    ver = S.verify()
    ver.to_csv(os.path.join(out, "sealed_verify.csv"), index=False)
    if not a.verify_only:
        summ = summarize()
        summ.to_csv(os.path.join(out, "sealed_summary.csv"), index=False)
        IO.log("\n" + summ.pivot_table(index=["track", "cls"], columns="partition",
                                        values="n", aggfunc="sum").fillna(0).astype(int).to_string())
    IO.write_json(os.path.join(out, "machine.json"), IO.machine_info())
    IO.log("\n✓ المرحلة صفر مكتملة. المختوم لا يُفتح قبل المرحلة الخامسة.")


if __name__ == "__main__":
    main()
