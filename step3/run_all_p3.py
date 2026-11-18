# -*- coding: utf-8 -*-
# =====================================================================
#  تشغيل المرحلة الثالثة كاملةً بأمر واحد
#  Run the whole Step-3 pipeline with one command.
# ---------------------------------------------------------------------
#  التشغيل:  python step3/run_all_p3.py
#  قابل للاستئناف: لو انطفت الحاسبة، شغّلي نفس الأمر — يكمّل من الـ cache.
# =====================================================================
import os, sys, subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
STEPS = [
    ("تجارب التسميم والكشف والدفاع", os.path.join(HERE, "experiments", "run_p3_benchmark.py")),
    ("تحسين الدفاع ميتاهيوريستياً (المرحلة F)", os.path.join(HERE, "experiments", "run_defense_opt.py")),
    ("توليد كل الأشكال P1..P7", os.path.join(HERE, "comparison_figures.py")),
]

for i, (name, script) in enumerate(STEPS, 1):
    print(f"\n{'='*64}\n>> [{i}/{len(STEPS)}] {name}\n{'='*64}", flush=True)
    r = subprocess.run([sys.executable, "-u", script], env=env)
    if r.returncode != 0:
        print(f"\n✗ توقّف عند: {name}. أعيدي تشغيل نفس الأمر — سيُكمل من الـ cache.")
        sys.exit(r.returncode)

print("\n✓✓ خلصت المرحلة الثالثة.")
print("   النتائج : step3/results/cases/<CASE_NAME>/")
print("   الأشكال : step3/figures/cases/<CASE_NAME>/")
