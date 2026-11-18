# -*- coding: utf-8 -*-
# =====================================================================
#  سُلّم الصرامة — مطبَّقاً على معماريّة الأوراق المنشورة نفسها
# ---------------------------------------------------------------------
#  سُلّم RF أثبت أن البروتوكول يهيمن على الرقم. لكن يبقى اعتراض واحد:
#      «أنتم أعدتم إنتاج *إعداد* لا *نظاماً*؛ ربما معماريّتنا العميقة تصمد.»
#  هذا السكربت يُغلق الثغرة: نفس السُّلّم الخماسي، لكن بمعماريّة LSTM–GRU
#  الهجينة (وأختيها LSTM و GRU) على التسلسل الخام لا على الإحصاءات.
#
#  تنبيه أمانة: هذه المعماريّة **ممثّلة لعائلة** الأوراق المنشورة، وليست
#  نسخة طبق الأصل من أيّ ورقة بعينها (طبقاتها ومعاملاتها غير منشورة كاملةً).
#  لذا الادّعاء في الورقة يجب أن يكون: «معماريّة من نفس العائلة».
#
#  استئناف: كل صف يُكتب فور انتهائه ⇒ إعادة التشغيل تُكمل من حيث توقّفت.
# =====================================================================
import os, sys, time, warnings, numpy as np, pandas as pd
warnings.filterwarnings("ignore")
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "experiments"))
import config as CFG
from seq_core import build_seq_dataset
import torch, torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score, accuracy_score, recall_score

#  --models lstm_gru  ⇒ شظيّة واحدة بملف نتائج خاص (تشغيل متوازٍ بلا تضارب كتابة)
_ARG_M = [a.split("=", 1)[1] for a in sys.argv[1:] if a.startswith("--models=")]
_ARG_T = [a.split("=", 1)[1] for a in sys.argv[1:] if a.startswith("--threads=")]
torch.set_num_threads(int(_ARG_T[0]) if _ARG_T else 4)
DEV = "cpu"
SPOOF = CFG.CLASSES.index("GPS Spoofing")
NC = len(CFG.CLASSES)
SEEDS = [0, 1, 2]
EPOCHS = 25
BATCH = 256
MODELS = _ARG_M[0].split(",") if _ARG_M else ["lstm_gru", "lstm", "gru"]
#  ملف مستقلّ لكل شظيّة ⇒ لا تتضارب عمليتان على نفس الملف؛ التجميع يقرأ الكلّ
_SUF = "" if not _ARG_M else "_" + "-".join(MODELS)
OUT = os.path.join(CFG.RESULTS_DIR, f"p3_published_ladder{_SUF}.csv")


# ---------------------------------------------------------------------
#  المعماريّات
# ---------------------------------------------------------------------
class SeqNet(nn.Module):
    """lstm_gru = LSTM(64)→GRU(32) الهجين المنشور ؛ lstm / gru = المفردتان."""

    def __init__(self, kind, n_ch, h1=64, h2=32, nc=NC):
        super().__init__()
        self.kind = kind
        if kind == "lstm_gru":
            self.r1 = nn.LSTM(n_ch, h1, batch_first=True)
            self.r2 = nn.GRU(h1, h2, batch_first=True)
            last = h2
        elif kind == "lstm":
            self.r1 = nn.LSTM(n_ch, h1, batch_first=True)
            self.r2 = nn.LSTM(h1, h2, batch_first=True)
            last = h2
        else:                                   # gru
            self.r1 = nn.GRU(n_ch, h1, batch_first=True)
            self.r2 = nn.GRU(h1, h2, batch_first=True)
            last = h2
        self.head = nn.Sequential(nn.Linear(last, 32), nn.ReLU(),
                                  nn.Dropout(0.2), nn.Linear(32, nc))

    def forward(self, x):
        o, _ = self.r1(x)
        o, _ = self.r2(o)
        return self.head(o[:, -1])              # آخر خطوة زمنية


def fit_predict(kind, Xtr, ytr, Xte, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    #  تطبيع لكل قناة بإحصاءات التدريب وحدها (لا تسريب من الاختبار)
    mu = Xtr.reshape(-1, Xtr.shape[2]).mean(0)
    sd = Xtr.reshape(-1, Xtr.shape[2]).std(0) + 1e-6
    tr = torch.from_numpy((Xtr - mu) / sd).float()
    te = torch.from_numpy((Xte - mu) / sd).float()
    yt = torch.from_numpy(ytr).long()

    m = SeqNet(kind, Xtr.shape[2]).to(DEV)
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    lossf = nn.CrossEntropyLoss()
    n = len(tr)
    m.train()
    for ep in range(EPOCHS):
        perm = torch.randperm(n)
        for i in range(0, n, BATCH):
            b = perm[i:i + BATCH]
            opt.zero_grad()
            loss = lossf(m(tr[b]), yt[b])
            loss.backward()
            nn.utils.clip_grad_norm_(m.parameters(), 5.0)
            opt.step()
    m.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(te), 1024):
            out.append(m(te[i:i + 1024]).argmax(1).cpu().numpy())
    return np.concatenate(out)


# ---------------------------------------------------------------------
#  البروتوكولات — مطابقة حرفياً لـ protocol_ladder.py
# ---------------------------------------------------------------------
def make_folds(y, groups, flights, protocol, seed):
    if protocol == "random":
        tr, te = train_test_split(np.arange(len(y)), test_size=.3,
                                  stratify=y, random_state=seed)
        return [("random", tr, te)]
    if protocol == "temporal":
        gap = CFG.WINDOW // CFG.STRIDE
        tr, te = [], []
        for fl in np.unique(flights):
            idx = np.where(flights == fl)[0]
            cut = int(.7 * len(idx))
            tr += list(idx[:cut]); te += list(idx[cut + gap:])
        return [("within-flight", np.array(tr), np.array(te))]
    folds = []
    for af in CFG.TEST_AIRFRAMES:                       # loao
        te = np.where(groups == af)[0]
        if len(te):
            folds.append((af, np.where(groups != af)[0], te))
    return folds


LADDER = [                                  # (رقم الدرجة، الوسم، حذف التسريب، البروتوكول)
    (1, "flight", False, "random"),
    (2, "flight", False, "temporal"),
    (3, "flight", False, "loao"),
    (4, "window", False, "loao"),
    (5, "window", True,  "loao"),
]
STEP_NAME = {1: "as published (folder labels, all feats, random split)",
             2: "+ temporal split", 3: "+ leave-one-airframe-out",
             4: "+ window-level labels", 5: "+ drop label-defining features"}

def main():
    #  الاستئناف يقرأ **كل** ملفات الشظايا: عمل أُنجز في تشغيل سابق لا يُعاد
    import glob
    done = set()
    for f in glob.glob(os.path.join(CFG.RESULTS_DIR, "p3_published_ladder*.csv")):
        if f.endswith("_agg.csv") or f.endswith("_combined.csv"):
            continue
        try:
            prev = pd.read_csv(f)
        except Exception:
            continue
        done |= {(r.model, int(r.step), r.fold, int(r.seed)) for r in prev.itertuples()}
    print(f">> استئناف: {len(done)} صفّاً منجزاً مسبقاً (كل الشظايا) | "
          f"models={MODELS} threads={torch.get_num_threads()} → {os.path.basename(OUT)}", flush=True)

    for step, lm, ex, proto in LADDER:
        X, y, groups, flights, sigs = build_seq_dataset(lm, ex, verbose=False)
        print(f"\n=== الدرجة {step} [{proto}/{lm}] X={X.shape} ===", flush=True)
        for model in MODELS:
            for seed in SEEDS:
                for fold_name, tr, te in make_folds(y, groups, flights, proto, seed):
                    if (model, step, fold_name, seed) in done:
                        continue
                    if len(te) == 0 or len(np.unique(y[tr])) < 2:
                        continue
                    t0 = time.time()
                    p = fit_predict(model, X[tr], y[tr], X[te], seed)
                    sr = (recall_score(y[te], p, labels=[SPOOF], average="macro",
                                       zero_division=0) if SPOOF in set(y[te]) else np.nan)
                    row = {"model": model, "step": step, "step_name": STEP_NAME[step],
                           "label_mode": lm, "noleak": ex, "protocol": proto,
                           "fold": fold_name, "seed": seed,
                           "n_train": len(tr), "n_test": len(te),
                           "raw_acc": accuracy_score(y[te], p),
                           "bal_acc": balanced_accuracy_score(y[te], p),
                           "spoof_recall": sr, "secs": round(time.time() - t0, 1)}
                    pd.DataFrame([row]).to_csv(OUT, mode="a", index=False,
                                               header=not os.path.exists(OUT))
                    done.add((model, step, fold_name, seed))
                    print(f"   {model:9s} s{step} {fold_name:16s} seed{seed}  "
                          f"BalAcc={row['bal_acc']:.3f}  spoof={sr:.3f}  ({row['secs']}s)",
                          flush=True)

    print(f"\n✓ {os.path.relpath(OUT)}")


#  الحارس ضروري: بدونه يُشغّل مجرّد الاستيراد السُّلّم كاملاً
if __name__ == "__main__":
    main()
