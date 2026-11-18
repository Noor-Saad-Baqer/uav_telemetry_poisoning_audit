# -*- coding: utf-8 -*-
#  Seal v3 for ISOT (2026-11-17): rebuild ONLY the ISOT section of sealed_manifest.json for the official
#  full release now at the same path. UAVIDS and telemetry seals are copied unchanged; the old manifest is
#  archived next to it. Also moves the stale ISOT feature-spec cache aside (it is derived from dev data).
import os, sys, shutil
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from common import paths as P, sealing as S, io as IO

old = IO.read_json(P.MANIFEST)
man = S.reseal_isot_full(verbose=True)
spec = os.path.join(P.CACHE, "feature_spec_isot.json")
if os.path.exists(spec):
    shutil.move(spec, spec.replace(".json", "_before_isot_full_20261117.json"))
    print("moved stale cache:", os.path.basename(spec))
f = man["isot"]["files"]
print(f"ISOT files {len(f)}, rows {sum(v['n_rows'] for v in f.values()):,}, "
      f"dev {sum(v['cut'] for v in f.values()):,}, sealed {sum(v['n_rows'] - v['cut'] for v in f.values()):,}")
print("UAVIDS seal unchanged:", man["uavids"] == old["uavids"], "| telemetry seal unchanged:", man["telemetry"] == old["telemetry"])
print("files identical to the previous (partial) seal:", man["isot"]["reused_from_v2"])
