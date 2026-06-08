"""Runtime sanity-check of the *novel* numpy logic in the notebooks (the parts not covered by
the motionmapperpy/slowmode libraries). Not a substitute for a real Colab run — it exercises the
generators + analysis math to catch shape/logic bugs. Run: python tools/test_logic.py"""
import sys, traceback
import numpy as np
from sklearn.cluster import MiniBatchKMeans  # noqa
from scipy.stats import mannwhitneyu

OK = "  [ok]"

# Use the real Morlet transform from the local slowmode clone if available, else a tiny stand-in.
try:
    sys.path.insert(0, "/Users/gordonberman/Documents/GitHub/slowmode")
    import pipeline as _pp
    def wavelets(x, fs, fmin, fmax, nf):
        a = _pp.morlet_wavelet_amplitudes(x, fs, fmin, fmax, nf)
        return np.asarray(a[0] if isinstance(a, tuple) else a)
    WSRC = "slowmode.morlet"
except Exception:
    def wavelets(x, fs, fmin, fmax, nf):
        f = np.linspace(fmin, fmax, nf)
        return np.column_stack([np.abs(np.sin(2*np.pi*fi*np.arange(len(x))/fs)) * np.abs(x) for fi in f])
    WSRC = "stub"


def test_05_concept():
    # the live concept demo: a hidden slow state switches a fast oscillation's frequency
    rng = np.random.default_rng(0); fs, T = 30.0, 12000
    hidden = np.zeros(T, int)
    for t in range(1, T):
        hidden[t] = hidden[t-1] if rng.random() > 1/600 else 1 - hidden[t-1]
    freq = np.where(hidden == 0, 2.0, 6.0)
    x = np.sin(np.cumsum(2*np.pi*freq/fs)) + 0.3*rng.standard_normal(T)
    w = wavelets(x, fs, 0.5, 12.0, 25)
    r_raw = abs(np.corrcoef(x, hidden)[0, 1])
    r_wav = max(abs(np.corrcoef(w[:, k], hidden)[0, 1]) for k in range(w.shape[1]))
    assert r_wav > 0.7 and r_raw < 0.3, (r_raw, r_wav)
    print("%s nb05 concept: raw |r|=%.2f  wavelet band |r|=%.2f  (wavelets=%s)" % (OK, r_raw, r_wav, WSRC))


def test_03_aib():
    def make(n_flies=6, T=1500, n_super=4, per_super=8, dwell=120, seed=0):
        r = np.random.default_rng(seed); N = n_super * per_super
        super_of = np.repeat(np.arange(n_super), per_super); out = []
        for _ in range(n_flies):
            mood = r.integers(n_super); s = int(np.where(super_of == mood)[0][0]); seq = []
            for t in range(T):
                if r.random() < 1/dwell: mood = r.integers(n_super)
                w = np.where(super_of == mood, 8., 1.); w[s] = 0; w /= w.sum()
                s = int(r.choice(N, p=w)); seq.append(s)
            out.append(np.array(seq))
        return out, N

    def getTransitions(s): return s[np.r_[True, np.diff(s) != 0]]

    def future_distributions(states, lag, n):
        Pj = np.zeros((n, n))
        for a, b in zip(states[:-lag], states[lag:]): Pj[b, a] += 1
        p = Pj.sum(0); p = p / p.sum(); cond = Pj / np.clip(Pj.sum(0), 1, None)
        return cond.T, p

    def js(p, q, wp, wq):
        m = wp * p + wq * q
        def kl(a, b):
            k = a > 0; return np.sum(a[k] * np.log2(a[k] / np.clip(b[k], 1e-12, None)))
        return wp * kl(p, m) + wq * kl(q, m)

    def aib(states):
        n = int(states.max()) + 1; cond, p = future_distributions(states, 5, n)
        clusters = {i: [i] for i in range(n)}; cdist = {i: cond[i].copy() for i in range(n)}
        cp = {i: p[i] for i in range(n)}; members = {i: i for i in range(n)}; merges = []
        while len(clusters) > 1:
            ids = list(clusters); best, bc = None, np.inf
            for ai in range(len(ids)):
                for bi in range(ai+1, len(ids)):
                    a, b = ids[ai], ids[bi]; w = cp[a] + cp[b]
                    cost = 0. if w == 0 else w * js(cdist[a], cdist[b], cp[a]/w, cp[b]/w)
                    if cost < bc: bc, best = cost, (a, b)
            a, b = best; w = cp[a] + cp[b] or 1.
            cdist[a] = (cp[a]*cdist[a] + cp[b]*cdist[b]) / w; cp[a] += cp[b]; clusters[a] += clusters[b]
            for s in clusters[b]: members[s] = a
            del clusters[b], cdist[b], cp[b]; merges.append(dict(members))
        return merges[::-1]

    sl, N = make()
    states = np.concatenate([getTransitions(s) for s in sl])
    merges = aib(states)
    ks = sorted({len(set(m.values())) for m in merges})
    assert ks[0] >= 1 and max(ks) <= N
    print("%s nb03 aIB: built %d->%d clusters, %d snapshots" % (OK, max(ks), min(ks), len(merges)))


def test_04_opto():
    def make(n_exp=8, n_ctrl=8, fps=30, n_trials=8, on_s=5, off_s=10, trig=4, seed=0):
        r = np.random.default_rng(seed)
        blobs = np.array([[-30,20],[0,30],[28,18],[30,-15],[0,-32],[-30,-12]], float)
        flies, leds, ctrl = [], [], []
        for f in range(n_exp + n_ctrl):
            c = f >= n_exp; led = np.tile(np.r_[np.ones(on_s*fps), np.zeros(off_s*fps)], n_trials).astype(bool)
            base = np.ones(len(blobs)) / len(blobs); z = np.zeros((len(led), 2)); b = r.integers(len(blobs))
            for t in range(len(led)):
                w = base.copy()
                if led[t] and not c: w[trig] += 1.5
                if r.random() < .06: b = r.choice(len(blobs), p=w/w.sum())
                z[t] = blobs[b] + r.normal(scale=3.5, size=2)
            flies.append(z); leds.append(led); ctrl.append(c)
        return flies, leds, np.array(ctrl)

    flies, leds, is_ctrl = make(); R = max(np.abs(np.concatenate(flies)).max() + 5, 1); NP = 51
    def occ(z):
        H, _, _ = np.histogram2d(z[:, 1], z[:, 0], bins=NP, range=[[-R, R], [-R, R]])
        from scipy.ndimage import gaussian_filter
        return gaussian_filter(H / max(H.sum(), 1), 2)
    diffs = np.array([occ(z[l]) - occ(z[~l]) for z, l in zip(flies, leds)])
    exp, ctl = diffs[~is_ctrl], diffs[is_ctrl]
    inside = np.abs(diffs).sum(0) > 0; pmap = np.ones((NP, NP)); ii, jj = np.where(inside)
    for i, j in zip(ii, jj):
        e, c = exp[:, i, j], ctl[:, i, j]
        if np.ptp(e) + np.ptp(c) > 0: pmap[i, j] = mannwhitneyu(e, c, alternative="two-sided").pvalue
    def bh(pvals, tested, q=0.05):
        p = pvals[tested]; ranked = np.sort(p); thr = q * (np.arange(1, len(p)+1)/len(p))
        passed = np.where(ranked <= thr)[0]; crit = ranked[passed.max()] if len(passed) else -1.
        m = np.zeros_like(pvals, bool); m[tested] = pvals[tested] <= crit; return m
    sig = bh(pmap, inside)
    assert sig.sum() > 0, "no significant pixels — effect/power too weak"
    print("%s nb04 opto: %d significant pixels (BH-FDR, 8v8)" % (OK, int(sig.sum())))


def test_02_social():
    CANON = np.array([[4,0,1.],[3,0,1.1],[2,0,1.],[0,0,1.],[-2,0,.9],[-1.8,.8,.7],[-1.8,-.8,.7],[-3.5,0,.6]])
    def poses(T, behav, head, center, rear, groom, fps):
        P = np.tile(CANON, (T, 1, 1)).astype(float); front = [0,1,2,3]; lift = np.array([1.,.8,.5,.2])
        P[:, front, 2] += rear[:, None]*lift; g = (behav == 3)
        P[:, 0, 2] += np.where(g, .6*np.sin(groom), 0); P[:, 1, 2] += np.where(g, .4*np.sin(groom), 0)
        c, s = np.cos(head), np.sin(head); x, y = P[:, :, 0].copy(), P[:, :, 1].copy()
        P[:, :, 0] = x*c[:, None]-y*s[:, None]+center[:, None, 0]; P[:, :, 1] = x*s[:, None]+y*c[:, None]+center[:, None, 1]
        return P
    def walk(T, fps, amph, seed):
        r = np.random.default_rng(seed)
        Tm = np.array([[.6,.2,.1,.1],[.15,.7,.1,.05],[.2,.2,.55,.05],[.2,.15,.05,.6]])
        if amph: Tm[:, 1] += .15; Tm /= Tm.sum(1, keepdims=True)
        b = np.zeros(T, int)
        for t in range(1, T): b[t] = r.choice(4, p=Tm[b[t-1]])
        head = np.cumsum(r.normal(0, .15, T)); center = np.clip(np.cumsum(r.normal(0, .3, (T, 2)), 0), -18, 18)
        rear = np.clip((b == 2).astype(float), 0, 1.2); groom = np.cumsum(np.full(T, 2*np.pi*5/fps))
        return b, head, center, rear, groom
    fps, T = 30, 800
    A = poses(T, *walk(T, fps, 0, 1), fps); B = poses(T, *walk(T, fps, 0, 2), fps)
    assert A.shape == (T, 8, 3)
    def egocenter(P):
        Q = P - P[:, 4:5, :]; th = np.arctan2(Q[:, 0, 1], Q[:, 0, 0]); c, s = np.cos(-th), np.sin(-th)
        x, y = Q[:, :, 0].copy(), Q[:, :, 1].copy()
        Q[:, :, 0], Q[:, :, 1] = x*c[:, None]-y*s[:, None], x*s[:, None]+y*c[:, None]
        return Q.reshape(len(P), -1)
    assert egocenter(A).shape == (T, 24)
    cAxy, cBxy = A[:, 4, :2], B[:, 4, :2]; d = np.linalg.norm(cAxy - cBxy, axis=1)
    headA = np.arctan2(*(A[:, 0, :2] - A[:, 4, :2]).T[::-1]); to_B = np.arctan2(*(cBxy - cAxy).T[::-1])
    rel = np.arctan2(np.sin(to_B - headA), np.cos(to_B - headA)); assert rel.shape == (T,)
    tact = np.zeros((8, 8))
    for t in range(0, T, 10):
        tact += np.linalg.norm(A[t][:, None] - B[t][None], axis=2) < 1.2
    assert tact.shape == (8, 8) and d.shape == (T,)
    print("%s nb02 social: poses %s, ego ok, social feats + tactogram ok" % (OK, A.shape))


for name, fn in [("05", test_05_concept), ("03", test_03_aib), ("04", test_04_opto), ("02", test_02_social)]:
    try:
        fn()
    except Exception:
        print("  [FAIL] nb%s" % name); traceback.print_exc()
print("\ndone")
