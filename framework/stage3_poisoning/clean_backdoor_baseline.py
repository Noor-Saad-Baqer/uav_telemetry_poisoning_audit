# -*- coding: utf-8 -*-
# =====================================================================
#  خطّ الأساس النظيف للباب الخلفي — framework/stage3_poisoning/clean_backdoor_baseline.py
# ---------------------------------------------------------------------
#  السؤال (من تدقيق 2026-09-04): «نجاح الباب الخلفي 1.000» — كم منه يعود إلى
#  المحفِّز نفسه بوصفه انزياح توزيع (‏+3σ على ثماني بواقٍ) لا إلى ما تعلّمه
#  النموذج من التسميم؟ الجواب: نقيس ASR على نموذج **نظيف غير مسمَّم**.
#  يعمل في سياق step3 (config الخاصّ بها) ويكتب:
#     framework/results/stage3/s3_backdoor_clean_baseline.csv
#  التشغيل:  python framework/stage3_poisoning/clean_backdoor_baseline.py
# =====================================================================
import os, sys
FW = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, FW)
from common import paths as P, io as IO, stats as ST
sys.path.insert(0, os.path.join(P.STEP3, "experiments"))
sys.path.insert(0, P.STEP3)

import numpy as np, pandas as pd
import config as CFG                # noqa: E402  ← step3
import poison_core as PC            # noqa: E402
import attacks as A                 # noqa: E402

assert hasattr(CFG, "TEST_AIRFRAMES") and hasattr(CFG, "TRIGGER_MAG"), "التُقط config خاطئ"
SEEDS = [0] if P.QUICK else [0, 1, 2]
RATE = 0.10


def _model(r):
    return r[0] if isinstance(r, tuple) else r


def main():
    out = P.results("stage3")
    X, y, groups, flights, feat = PC.build_dataset(verbose=False)
    sink = IO.Sink(os.path.join(out, "s3_backdoor_clean_baseline.csv"), ["fold", "seed"])
    for af in CFG.TEST_AIRFRAMES:
        tr, te = np.where(groups != af)[0], np.where(groups == af)[0]
        sc = PC.fit_scaler(X[tr]); Xtr, Xte = PC.apply_scaler(sc, X[tr]), PC.apply_scaler(sc, X[te])
        for seed in SEEDS:
            if sink.has(fold=af, seed=seed):
                continue
            clean = _model(PC.train_victim(Xtr, y[tr], seed=seed))
            asr_clean = A.backdoor_asr(lambda Z: PC.predict(clean, Z), Xte, y[te])
            Xp, yp = A.poison("backdoor", Xtr, y[tr], RATE, seed=seed)[:2]
            pois = _model(PC.train_victim(Xp, yp, seed=seed))
            asr_pois = A.backdoor_asr(lambda Z: PC.predict(pois, Z), Xte, y[te])
            pred = PC.predict(clean, A.apply_trigger(Xte[y[te] != 0]))
            cnt = np.bincount(pred, minlength=len(CFG.CLASSES))
            sink.add(dict(fold=af, seed=seed, trigger_dim=CFG.TRIGGER_DIM, trigger_mag=CFG.TRIGGER_MAG,
                          poison_rate=RATE, asr_clean_model=asr_clean, asr_poisoned_model=asr_pois,
                          asr_attributable_to_poison=max(0.0, asr_pois - asr_clean),
                          n_triggered=int((y[te] != 0).sum()),
                          **{f"clean_pred_{c}": int(v) for c, v in zip(CFG.CLASSES, cnt)}))
            IO.log(f"   {af:15s} s{seed}  ASR clean={asr_clean:.3f}  poisoned={asr_pois:.3f}")
    d = sink.df()
    m, lo, hi = ST.boot_ci(d.asr_clean_model)
    mp, lop, hip = ST.boot_ci(d.asr_poisoned_model)
    summ = pd.DataFrame([dict(metric="asr_clean_model", mean=m, ci_lo=lo, ci_hi=hi, n=len(d)),
                         dict(metric="asr_poisoned_model", mean=mp, ci_lo=lop, ci_hi=hip, n=len(d)),
                         dict(metric="asr_attributable_to_poison", mean=float(d.asr_attributable_to_poison.mean()),
                              ci_lo=np.nan, ci_hi=np.nan, n=len(d))])
    summ.to_csv(os.path.join(out, "s3_backdoor_clean_baseline_summary.csv"), index=False)
    IO.log("\n" + summ.round(3).to_string(index=False))
    IO.log("\n✓ خطّ الأساس النظيف للباب الخلفي كُتب.")


if __name__ == "__main__":
    main()
