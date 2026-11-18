# -*- coding: utf-8 -*-
# =====================================================================
#  مُحسِّنات ميتاهيوريستية بميزانية تقييمات موحّدة (عدالة المقارنة)
#  المتاح ستة ؛ المفعَّل في config.py خمسة (random متاح للتفعيل عند الحاجة)
#  Six metaheuristic optimizers under an EQUAL function-evaluation budget
# ---------------------------------------------------------------------
#  الواجهة: run_optimizer(name, fn, dim, pop, budget, opt_seed)
#     fn(vec)->float يُراد تعظيمه ; vec في الفضاء [0,1]^dim
#  كل مُحسِّن يتوقّف عند استنفاد الميزانية (نفس عدد التقييمات للجميع).
#  المُحسِّنات: random, pso, gwo, lgwo (GWO+Lévy), hho, cs
# =====================================================================
import math, numpy as np

class Budgeted:
    """يلفّ دالة الهدف: يعدّ التقييمات، يتتبّع الأفضل، ويوقف عند الميزانية."""
    def __init__(self, fn, budget):
        self.fn = fn; self.budget = budget; self.n = 0
        self.best_val = -np.inf; self.best_vec = None; self.history = []
    def __call__(self, vec):
        vec = np.clip(np.asarray(vec, float), 0.0, 1.0)
        if self.n >= self.budget:
            return self.best_val
        val = self.fn(vec); self.n += 1
        if val > self.best_val:
            self.best_val = val; self.best_vec = vec.copy()
        self.history.append(self.best_val)
        return val
    @property
    def exhausted(self): return self.n >= self.budget

def levy(dim, rng, beta=1.5):
    sig = (math.gamma(1 + beta) * math.sin(math.pi * beta / 2) /
           (math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)
    u = rng.normal(0, sig, dim); v = rng.normal(0, 1, dim)
    return u / (np.abs(v) ** (1 / beta))

# ---------------- (1) Random Search ----------------
def _random(obj, dim, pop, rng):
    while not obj.exhausted:
        obj(rng.uniform(0, 1, dim))

# ---------------- (2) PSO ----------------
def _pso(obj, dim, pop, rng, w=0.7, c1=1.5, c2=1.5):
    X = rng.uniform(0, 1, (pop, dim)); V = rng.uniform(-0.1, 0.1, (pop, dim))
    pbest = X.copy(); pbest_val = np.full(pop, -np.inf)
    for i in range(pop):
        if obj.exhausted: break
        pbest_val[i] = obj(X[i])
    while not obj.exhausted:
        g = obj.best_vec
        for i in range(pop):
            if obj.exhausted: break
            r1 = rng.uniform(0, 1, dim); r2 = rng.uniform(0, 1, dim)
            V[i] = w * V[i] + c1 * r1 * (pbest[i] - X[i]) + c2 * r2 * (g - X[i])
            X[i] = np.clip(X[i] + V[i], 0, 1)
            val = obj(X[i])
            if val > pbest_val[i]: pbest_val[i] = val; pbest[i] = X[i].copy()

# ---------------- (3,4) GWO / LGWO (Grey Wolf, optional Lévy) ----------------
def _gwo(obj, dim, pop, rng, use_levy):
    max_iter = max(1, obj.budget // max(1, pop))
    X = rng.uniform(0, 1, (pop, dim))
    lead = []                                          # قادة نخبويون: [(fit,pos)] أعلى 3
    cur = []
    for i in range(pop):
        if obj.exhausted: break
        cur.append((obj(X[i]), X[i].copy()))
    lead = sorted(cur, key=lambda t: t[0], reverse=True)[:3]
    while len(lead) < 3 and lead: lead.append(lead[-1])
    t = 0
    while not obj.exhausted:
        a = max(0.0, 2 - 2 * t / max_iter)
        Xa, Xb, Xd = lead[0][1], lead[1][1], lead[2][1]
        cur = []
        for i in range(pop):
            if obj.exhausted: break
            X1 = np.zeros(dim)
            for L in (Xa, Xb, Xd):
                A = 2 * a * rng.uniform(0, 1, dim) - a; C = 2 * rng.uniform(0, 1, dim)
                X1 = X1 + (L - A * np.abs(C * L - X[i]))
            X1 = X1 / 3.0
            if use_levy: X1 = X1 + 0.01 * levy(dim, rng) * (X1 - Xa)   # ← طيران ليفي
            X[i] = np.clip(X1, 0, 1); cur.append((obj(X[i]), X[i].copy()))
        lead = sorted(lead + cur, key=lambda tup: tup[0], reverse=True)[:3]   # نخبوية
        t += 1

# ---------------- (5) HHO — Harris Hawks Optimization ----------------
def _hho(obj, dim, pop, rng):
    max_iter = max(1, obj.budget // max(1, pop))
    X = rng.uniform(0, 1, (pop, dim)); fit = np.full(pop, -np.inf)
    for i in range(pop):
        if obj.exhausted: break
        fit[i] = obj(X[i])
    t = 0
    while not obj.exhausted:
        E1 = 2 * (1 - t / max_iter)                    # طاقة الهروب المتناقصة
        Xmean = X.mean(0)
        for i in range(pop):
            if obj.exhausted: break
            rabbit = obj.best_vec                       # الفريسة = أفضل حل عالمي
            E = E1 * (2 * rng.uniform() - 1)
            if abs(E) >= 1:                             # (أ) الاستكشاف
                if rng.uniform() >= 0.5:
                    Xr = X[rng.randint(pop)]
                    X[i] = Xr - rng.uniform() * np.abs(Xr - 2 * rng.uniform() * X[i])
                else:
                    X[i] = (rabbit - Xmean) - rng.uniform() * rng.uniform(0, 1, dim)
                X[i] = np.clip(X[i], 0, 1); fit[i] = obj(X[i]); continue
            r = rng.uniform(); J = 2 * (1 - rng.uniform())
            if r >= 0.5 and abs(E) >= 0.5:              # حصار ناعم
                X[i] = (rabbit - X[i]) - E * np.abs(J * rabbit - X[i])
                X[i] = np.clip(X[i], 0, 1); fit[i] = obj(X[i])
            elif r >= 0.5 and abs(E) < 0.5:             # حصار قوي
                X[i] = rabbit - E * np.abs(rabbit - X[i])
                X[i] = np.clip(X[i], 0, 1); fit[i] = obj(X[i])
            else:                                       # حصار مع غطسات سريعة (+ ليفي)
                base = X[i] if abs(E) >= 0.5 else Xmean
                Y = np.clip(rabbit - E * np.abs(J * rabbit - base), 0, 1)
                fY = obj(Y)
                if fY > fit[i]:
                    X[i] = Y; fit[i] = fY
                elif not obj.exhausted:
                    Z = np.clip(Y + rng.uniform(0, 1, dim) * levy(dim, rng), 0, 1)
                    fZ = obj(Z)
                    if fZ > fit[i]: X[i] = Z; fit[i] = fZ
        t += 1

# ---------------- (6) CS — Cuckoo Search ----------------
def _cs(obj, dim, pop, rng, pa=0.25):
    X = rng.uniform(0, 1, (pop, dim)); fit = np.full(pop, -np.inf)
    for i in range(pop):
        if obj.exhausted: break
        fit[i] = obj(X[i])
    while not obj.exhausted:
        best = obj.best_vec
        for i in range(pop):                           # طيران ليفي لكل عش
            if obj.exhausted: break
            Xnew = np.clip(X[i] + 0.01 * levy(dim, rng) * (X[i] - best), 0, 1)
            fnew = obj(Xnew); j = rng.randint(pop)
            if fnew > fit[j]: X[j] = Xnew; fit[j] = fnew
        if obj.exhausted: break
        nab = max(1, int(pa * pop))                    # هجر أسوأ الأعشاش
        for i in np.argsort(fit)[:nab]:
            if obj.exhausted: break
            d1, d2 = rng.randint(pop), rng.randint(pop)
            Xnew = np.clip(X[i] + rng.uniform() * (X[d1] - X[d2]), 0, 1)
            fit[i] = obj(Xnew); X[i] = Xnew

_ALGOS = {"random": _random, "pso": _pso,
          "gwo": lambda o, d, p, r: _gwo(o, d, p, r, False),
          "lgwo": lambda o, d, p, r: _gwo(o, d, p, r, True),
          "hho": _hho, "cs": _cs}

def run_optimizer(name, fn, dim, pop, budget, opt_seed):
    """يشغّل مُحسِّناً واحداً بميزانية محدّدة ويُرجع (أفضل متجه, أفضل قيمة, تاريخ التقارب)."""
    rng = np.random.RandomState(opt_seed)
    obj = Budgeted(fn, budget)
    _ALGOS[name](obj, dim, pop, rng)
    return obj.best_vec, obj.best_val, obj.history
