# -*- coding: utf-8 -*-
# =====================================================================
#  أدوات الإدخال/الإخراج — كتابة صفّاً بصفّ ، استئناف ، سجلّ ، معلومات الجهاز
# ---------------------------------------------------------------------
#  ⚠️ ملفّ الترحيل على هذا الجهاز ثابت عند 8 غ ب ويُسقط العمليات أحياناً ،
#     لذلك كلّ صفّ نتيجة يُكتب فور اكتماله ، وإعادة التشغيل تتخطّى ما اكتمل.
# =====================================================================
import os, sys, json, time, datetime, platform, shutil
import pandas as pd

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


def log(msg=""):
    print(msg, flush=True)


def now():
    return datetime.datetime.now().isoformat(timespec="seconds")


class Sink:
    """ملفّ CSV قابل للاستئناف: مفتاح الصفّ = مجموعة أعمدة ؛ الصفّ المكتمل لا يُعاد."""

    def __init__(self, path, key_cols):
        self.path, self.key = path, list(key_cols)
        self.rows = []
        self.done = set()
        if os.path.exists(path):
            try:
                old = pd.read_csv(path)
                self.rows = old.to_dict("records")
                for r in self.rows:
                    self.done.add(self._key(r))
            except Exception:
                self.rows, self.done = [], set()

    def _key(self, r):
        return tuple(str(r.get(k)) for k in self.key)

    def has(self, **kw):
        return tuple(str(kw.get(k)) for k in self.key) in self.done

    def add(self, row):
        k = self._key(row)
        if k in self.done:                      # إعادة تشغيل بمفتاح موجود ⇒ استبدال لا تكرار
            self.rows = [r for r in self.rows if self._key(r) != k]
        self.rows.append(row)
        self.done.add(k)
        pd.DataFrame(self.rows).to_csv(self.path, index=False)

    def df(self):
        return pd.DataFrame(self.rows)

    def __len__(self):
        return len(self.rows)


def _jsonable(o):
    try:
        import numpy as np
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
    except Exception:
        pass
    return str(o)


def write_json(path, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, default=_jsonable)


def read_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def append_jsonl(path, obj):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False, default=_jsonable) + "\n")


def machine_info():
    """الزمن الجداري بلا مواصفات الجهاز رقم بلا معنى ⇒ يُسجَّل مع كل نتيجة."""
    try:
        import psutil
        vm = psutil.virtual_memory()
        ram = round(vm.total / 1e9, 1)
        cl, cp = psutil.cpu_count(logical=True), psutil.cpu_count(logical=False)
    except Exception:
        ram, cl, cp = None, os.cpu_count(), None
    return dict(cpu=platform.processor() or platform.machine(),
                cores_logical=cl, cores_physical=cp, ram_gb=ram,
                python=platform.python_version(), platform=platform.platform(),
                omp_threads=os.environ.get("OMP_NUM_THREADS"), time=now())


class Timer:
    def __init__(self):
        self.t0 = time.perf_counter()

    def s(self):
        return time.perf_counter() - self.t0


def copy_legacy(src_files, dst_dir, provenance):
    """ينسخ ملفّات نتائج جاهزة من مرحلة قديمة إلى results/stageN/legacy/ ويكتب
    ملفّ مصدر يذكر من أين جاءت. القاعدة: ما له نتيجة جاهزة لا يُعاد تنفيذه."""
    rows = []
    for src in src_files:
        if not os.path.exists(src):
            rows.append(dict(file=os.path.basename(src), source=src, status="missing"))
            continue
        dst = os.path.join(dst_dir, os.path.basename(src))
        shutil.copy2(src, dst)
        rows.append(dict(file=os.path.basename(src),
                         source=src, status="copied", bytes=os.path.getsize(src),
                         modified=datetime.datetime.fromtimestamp(
                             os.path.getmtime(src)).isoformat(timespec="seconds")))
    pd.DataFrame(rows).to_csv(os.path.join(dst_dir, "PROVENANCE.csv"), index=False)
    with open(os.path.join(dst_dir, "PROVENANCE.md"), "w", encoding="utf-8") as f:
        f.write(provenance)
    return rows
