"""Runtime sanity-check of the *novel* numpy logic in the notebooks (the parts not covered by
the motionmapperpy/slowmode libraries). Not a substitute for a real Colab run — it exercises the
generators + analysis math to catch shape/logic bugs. Run: python tools/test_logic.py"""
import sys, traceback
import numpy as np
from sklearn.cluster import MiniBatchKMeans  # noqa
from sklearn.metrics import adjusted_rand_score
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


def test_03_dib():
    # The Deterministic Information Bottleneck (Strouse & Schwab 2017) used in nb03 -- same code
    # the notebook runs, checked here for shape/logic bugs and for recovering known structure.
    def make(n_flies=6, T=1500, n_super=4, per_super=8, dwell=120, seed=0):
        r = np.random.default_rng(seed); N = n_super * per_super
        super_of = np.repeat(np.arange(n_super), per_super); out = []
        for _ in range(n_flies):
            mood = r.integers(n_super); s = int(np.where(super_of == mood)[0][0]); seq = []
            for t in range(T):
                if r.random() < 1/dwell: mood = r.integers(n_super)
                w = np.where(super_of == mood, 8., 1.); w[s] = 0; w /= w.sum()
                s = int(r.choice(N, p=w)); seq.append(s)
            out.append(np.array(seq) + 1)                 # 1-based states (state v <-> region v)
        return out, N, super_of

    def getTransitions(s): return s[np.r_[True, np.diff(s) != 0]]

    def _safe_log2(A):
        o = np.zeros_like(A, dtype=float); np.log2(A, out=o, where=A > 0); return o

    def dib_single(pXY, pX, pY_X, Hx, K, beta, rng, tol=1e-6, max_iter=200):
        Nx, Ny = pXY.shape; f = rng.integers(0, K, size=Nx)
        def stats(f):
            oh = np.zeros((Nx, K)); oh[np.arange(Nx), f] = 1.; pT = oh.T @ pX; pYT = oh.T @ pXY
            return pT, np.divide(pYT, pT[:, None], out=np.zeros_like(pYT), where=pT[:, None] > 0)
        def cost(pT, pY_T):
            idx = pT > 0; H_T = -np.sum(pT[idx]*np.log2(pT[idx]))
            pYT = pY_T*pT[:, None]; pY = pYT.sum(0); den = pT[:, None]*pY[None, :]
            ratio = np.divide(pYT, den, out=np.zeros_like(pYT), where=den > 0)
            return H_T, (pYT*_safe_log2(ratio)).sum()
        pT, pY_T = stats(f); H_T, I_YT = cost(pT, pY_T); prev = H_T - beta*I_YT
        for _ in range(max_iter):
            DKL = (-pY_X @ _safe_log2(pY_T).T) - Hx[:, None]
            f = np.argmax(np.where(pT > 0, _safe_log2(pT), -np.inf)[None, :] - beta*DKL, axis=1)
            pT, pY_T = stats(f); H_T, I_YT = cost(pT, pY_T); J = H_T - beta*I_YT
            if abs(J - prev) < tol: break
            prev = J
        used = np.unique(f); return np.searchsorted(used, f), I_YT, H_T

    def build_joint(trans_list, state_vals, lag):
        n = len(state_vals); F = np.zeros((n, n))
        for s in trans_list:
            s = np.searchsorted(state_vals, s)
            if len(s) > lag: np.add.at(F, (s[:-lag], s[lag:]), 1.)
        return F

    def pareto_front(pts):
        keep = np.ones(len(pts), bool)
        for i in range(len(pts)): keep[i] = not np.any(np.all(pts > pts[i], axis=1))
        return keep

    def run_dib(trans_list, state_vals, lag, n_restarts=300, min_clusters=2, max_clusters=12, seed=0):
        rng = np.random.default_rng(seed)
        pXY = build_joint(trans_list, state_vals, lag); pXY /= pXY.sum(); pX = pXY.sum(1)
        pY_X = np.divide(pXY, pX[:, None], out=np.zeros_like(pXY), where=pX[:, None] > 0)
        Hx = -np.sum(pY_X*_safe_log2(pY_X), axis=1)
        HT = np.zeros(n_restarts); IYT = np.zeros(n_restarts); ncl = np.zeros(n_restarts, int); clus = [None]*n_restarts
        for i in range(n_restarts):
            beta = 10.**(-1 + 5*rng.random()); K = int(rng.integers(min_clusters, max_clusters+1))
            clus[i], IYT[i], HT[i] = dib_single(pXY, pX, pY_X, Hx, K, beta, rng); ncl[i] = len(np.unique(clus[i]))
        on = pareto_front(np.c_[-HT, IYT]); best = {}
        for j in np.where(on)[0]:
            if ncl[j] not in best or IYT[j] > IYT[best[ncl[j]]]: best[ncl[j]] = j
        return dict(HT=HT, IYT=IYT, ncl=ncl, on=on, chosen=[best[k] for k in sorted(best)], clus=clus)

    sl, N, super_of = make()
    trans_list = [getTransitions(s) for s in sl]
    state_vals = np.unique(np.concatenate(trans_list))
    dib = run_dib(trans_list, state_vals, lag=5)
    assert (dib["IYT"] >= -1e-9).all() and (dib["HT"] >= -1e-9).all(), "negative H[T] or I[Y;T]"
    I_front = dib["IYT"][dib["on"]][np.argsort(dib["HT"][dib["on"]])]
    assert np.all(np.diff(I_front) >= -1e-9), "I[Y;T] not monotone along the Pareto front"
    j = min(dib["chosen"], key=lambda j: abs(dib["ncl"][j] - 4))      # level closest to 4 superclusters
    ari = adjusted_rand_score(super_of[state_vals - 1], dib["clus"][j])
    assert ari > 0.8, "DIB failed to recover the planted superclusters (ARI=%.2f)" % ari
    ks = [int(dib["ncl"][k]) for k in dib["chosen"]]
    print("%s nb02 DIB: front %d..%d clusters; recovers %d superclusters at k=%d (ARI=%.2f)"
          % (OK, min(ks), max(ks), len(np.unique(super_of)), dib["ncl"][j], ari))


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
    print("%s nb06 opto: %d significant pixels (BH-FDR, 8v8)" % (OK, int(sig.sum())))


def test_02_social():
    # nb04 social: synchrony enrichment, joint-occupancy JS phenotype, and the region tactogram --
    # the novel numpy of the real-data social notebook, checked here on synthetic stand-ins.
    rng = np.random.default_rng(0)
    # (1) synchrony: partners with a planted tendency to share coarse behavior -> positive diagonal
    NI, T = 6, 20000
    a = rng.integers(1, NI + 1, T)
    b = np.where(rng.random(T) < 0.6, a, rng.integers(1, NI + 1, T))     # 60% match the partner
    co = np.zeros((NI, NI))
    for x, y in zip(a, b): co[x - 1, y - 1] += 1
    P = co / co.sum(); E = P.sum(1, keepdims=True) @ P.sum(0, keepdims=True)
    enr = np.log2((P + 1e-9) / (E + 1e-9))
    assert np.mean(np.diag(enr)) > 0.3 > np.mean(enr[~np.eye(NI, dtype=bool)]), "synchrony not enriched"
    # (2) phenotype: amph dyads' joint-occupancy sits further from the control profile (JS + MWU)
    def js(p, q):
        p = p + 1e-12; q = q + 1e-12; p /= p.sum(); q /= q.sum(); m = .5 * (p + q)
        kl = lambda u, v: np.sum(u * np.log2(u / v)); return .5 * kl(p, m) + .5 * kl(q, m)
    K = 8
    ctrl = rng.dirichlet(np.ones(K) * 5, 16)
    amph = rng.dirichlet(np.r_[np.ones(K - 2) * 5, [15, 15]], 16)        # shifted toward 2 classes
    ref = ctrl.mean(0)
    jsc = [js(o, ref) for o in ctrl]; jsa = [js(o, ref) for o in amph]
    assert np.mean(jsa) > np.mean(jsc) and mannwhitneyu(jsc, jsa).pvalue < 0.05, "phenotype not separated"
    # (3) region tactogram: contact when two animals' keypoints fall within threshold
    A = rng.normal(0, 50, (23, 3)); B = A + np.r_[5.0, 0, 0]            # B almost on top of A
    assert (np.linalg.norm(A[:, None] - B[None], axis=2) < 40).any(), "tactogram contact broken"
    print("%s nb04 social: synchrony diag=+%.2f bits, amph vs ctrl p<0.05, tactogram ok"
          % (OK, np.mean(np.diag(enr))))


for name, fn in [("05", test_05_concept), ("02", test_03_dib), ("06", test_04_opto), ("04", test_02_social)]:
    try:
        fn()
    except Exception:
        print("  [FAIL] nb%s" % name); traceback.print_exc()
print("\ndone")
