# -*- coding: utf-8 -*-
# =====================================================================
#  بناء مجموعة **تسلسلية** (نافذة × زمن × إشارة) للنماذج العميقة
# ---------------------------------------------------------------------
#  poison_core يُلخّص كل نافذة بخمس إحصاءات (85 ميزة) — وهذا مناسب لـ RF.
#  أمّا LSTM/GRU فتحتاج التسلسل الخام: (n, WINDOW, n_signals).
#
#  المطلوب حاسم: أن تكون **نفس النوافذ ونفس الأوسمة ونفس الموازنة** التي
#  استعملها سُلّم RF — وإلّا صار الفارق بين النموذجين لا بين البروتوكولين.
#  لذلك نُكرّر منطق build_dataset حرفياً (نفس RandomState(0)، نفس الترتيب)
#  ولا نُغيّر إلّا خطوة التلخيص.
# =====================================================================
import os, sys, json, hashlib, numpy as np, pandas as pd
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "experiments"))
import config as CFG
import poison_core as P

NC = len(CFG.CLASSES)


def _key(label_mode, exclude_leak):
    payload = json.dumps({"W": CFG.WINDOW, "S": CFG.STRIDE, "innov": CFG.INNOV_SIGNALS,
                          "telem": CFG.TELEM_SIGNALS, "gps": CFG.USE_GPS,
                          "mode": label_mode, "noleak": exclude_leak,
                          "leaksig": getattr(CFG, "LABEL_LEAK_SIGNALS", []),
                          "thr": [CFG.SPOOF_INNOV_THR, CFG.DOS_TX_MULT, CFG.ACTIVE_FRAC],
                          "maskmode": getattr(CFG, "SPOOF_MASK_MODE", "raw"),
                          "ratio": CFG.NORMAL_RATIO, "log": CFG.LOG_SCALE_INNOV,
                          "seq": 1}, sort_keys=True)
    return hashlib.md5(payload.encode()).hexdigest()[:16]


def build_seq_dataset(label_mode, exclude_leak, verbose=True):
    """يُرجع (Xseq[n,W,C], y, groups, flights, sig_names) — مطابق نافذةً بنافذة لسُلّم RF."""
    fp = os.path.join(CFG.CACHE_DIR, f"seq_{_key(label_mode, exclude_leak)}.npz")
    if os.path.exists(fp):
        try:
            d = np.load(fp, allow_pickle=True)
            if verbose:
                print(f">> [seq] من الـ cache: X={d['X'].shape}", flush=True)
            return d["X"], d["y"], d["groups"], d["flights"], list(d["sigs"])
        except Exception as e:
            #  عمليات متوازية قد تقرأ ملفاً نصف مكتوب ⇒ نُعيد البناء بدل السقوط
            print(f">> [seq] cache غير صالح ({type(e).__name__}) — إعادة بناء", flush=True)

    tx_base = P._tx_baselines()
    Xs, ys, gs, fs = [], [], [], []
    sigs = None
    for af, cls, paths in P.discover_flights():
        df, sigs = P._fuse(paths)
        active = P._active_mask(df, cls, af, tx_base)     # قبل الحذف — فهو تعريف الوسم
        if exclude_leak:
            sigs = [x for x in sigs if x not in set(CFG.LABEL_LEAK_SIGNALS)]
        A = df[sigs].apply(pd.to_numeric, errors="coerce")
        A = A.replace([np.inf, -np.inf], np.nan).ffill().fillna(0.0).values.astype("float32")
        n = len(A)
        if n < CFG.WINDOW:
            continue
        cls_id = CFG.CLASSES.index(cls)
        nw = na = 0
        for st in range(0, n - CFG.WINDOW + 1, CFG.STRIDE):
            if label_mode == "window":
                frac = active[st:st + CFG.WINDOW].mean()
                lab = cls_id if frac >= CFG.ACTIVE_FRAC else 0
                if 0 < frac < CFG.ACTIVE_FRAC and cls_id != 0:
                    continue                               # نافذة انتقالية: وسمها غامض
            else:
                lab = cls_id
            Xs.append(A[st:st + CFG.WINDOW])
            ys.append(lab); gs.append(af); fs.append(f"{af}|{cls}")
            nw += 1; na += (lab != 0)
        if verbose:
            print(f"   {af:16s} {cls:13s} rows={n:6d} → windows={nw:5d} (attack {na})", flush=True)

    X = np.asarray(Xs, dtype="float32")
    X = np.clip(np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0), -1e6, 1e6)
    y = np.asarray(ys, dtype="int64")
    groups = np.asarray(gs); flights = np.asarray(fs)

    #  موازنة مطابقة لـ poison_core (نفس البذرة ⇒ نفس النوافذ المختارة بالضبط)
    if label_mode == "window" and CFG.NORMAL_RATIO:
        counts = np.bincount(y, minlength=NC)
        atk = [c for c in counts[1:] if c > 0]
        if atk:
            cap = int(CFG.NORMAL_RATIO * min(atk))
            rng = np.random.RandomState(0)
            keep = []
            for c in range(NC):
                idx_c = np.where(y == c)[0]
                if len(idx_c) <= cap:
                    keep += list(idx_c); continue
                per_flight = {}
                for i in idx_c:
                    per_flight.setdefault(flights[i], []).append(i)
                share = max(1, cap // len(per_flight))
                sub = []
                for fl, idxs in per_flight.items():
                    sub += list(rng.permutation(np.array(idxs))[:share])
                keep += sub
            sel = np.sort(np.array(keep))
            X, y, groups, flights = X[sel], y[sel], groups[sel], flights[sel]

    #  كتابة ذرّية: ملف مؤقّت خاص بالعملية ثم استبدال — وإلّا قرأت عمليةٌ أخرى نصفَ ملف
    #  ⚠️ np.savez يُلحق ".npz" إن لم ينتهِ الاسم بها ⇒ يجب أن ينتهي المؤقّت بـ .npz
    tmp = f"{fp}.{os.getpid()}.tmp.npz"
    np.savez(tmp, X=X, y=y, groups=groups, flights=flights, sigs=np.array(sigs))
    os.replace(tmp, fp)
    if verbose:
        print(f">> [seq] X={X.shape}  y={np.bincount(y)}", flush=True)
    return X, y, groups, flights, sigs


if __name__ == "__main__":
    for lm, ex in [("flight", False), ("window", False), ("window", True)]:
        X, y, g, f, s = build_seq_dataset(lm, ex)
        print(f"{lm:7s} noleak={ex}  X={X.shape}  y={np.bincount(y)}\n")
