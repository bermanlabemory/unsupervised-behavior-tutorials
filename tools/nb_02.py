"""02 — Transitions & hierarchy (Berman et al. 2016). Python port of Gordon's
behavioral-transitions tutorial, built on motionmapperpy's demoutils."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nb_builder import md, code, badge, write_nb, setup_code, carry_from_core

REPO = "bermanlabemory/unsupervised-behavior-tutorials/blob/main"
cells = []

cells.append(badge("%s/02_transitions_and_hierarchy.ipynb" % REPO))

cells.append(md(r"""
# Transitions & hierarchy

**The question:** a behavioral map tells you *what* an animal does. This notebook asks how those
behaviors are **organized in time**. Is behavior just a roll of the dice from one moment to the
next (*Markovian*), or is there long memory and structure? We'll find that fly behavior has
**time scales far longer than any single movement**, and a **hierarchy** of behaviors &mdash;
following Berman, Bialek & Shaevitz, *PNAS* 2016.

This is a standalone track &mdash; you don't need to have finished notebook 01. It loads its own data:
sequences of behavioral states for **59 flies, 117 behaviors each**, one hour of recording apiece.

**Run time:** ~10 min.
"""))
cells.append(md(carry_from_core()))

# ---------------------------------------------------------------- setup
cells.append(md(r"""
# 1.&nbsp; Setup

The standard opening cell, the same one that opens every notebook in this series: clone
motionmapperpy, install the few packages Colab doesn't ship, and import what we need.
"""))
cells.append(code(setup_code(
    imports="import numpy as np, matplotlib.pyplot as plt\n"
            "import motionmapperpy as mmpy\n"
            "from motionmapperpy import demoutils\n"
            "%matplotlib inline")))

# ---------------------------------------------------------------- data
cells.append(md("# 2.&nbsp; Load the behavioral state sequences"))
cells.append(md(r"""
Each fly's behavior is an integer time series: which of the 117 behaviors it was doing at each
step. We also load the 2-D **region map** (so we can draw results on the behavior space) and the
behavioral **density**.

This loads the **real 59-fly dataset** (`data/transition_data.mat`, 117 behaviors) straight from this
tutorial's GitHub repo into your `/content/` folder. If the download ever fails, the cell quietly falls
back to a stand-in generator with the same structure, so the notebook always runs &mdash; set
`USE_SYNTHETIC_DATA = True` to force that yourself.
"""))
cells.append(code(r"""
USE_SYNTHETIC_DATA = False      # set True to force the stand-in generator below
# The real 59-fly transition data ships in this tutorial's repo (data/transition_data.mat).
DATA_URL = "https://raw.githubusercontent.com/bermanlabemory/unsupervised-behavior-tutorials/main/data/transition_data.mat"

def load_real():
    from scipy.io import loadmat
    if not os.path.exists("transition_data.mat"):
        !wget -q "$DATA_URL" -O transition_data.mat
    if not os.path.exists("transition_data.mat") or os.path.getsize("transition_data.mat") < 100000:
        raise RuntimeError("download failed -- check your connection, or set USE_SYNTHETIC_DATA = True above.")
    d = loadmat("transition_data.mat")           # v7 .mat -> scipy reads it directly
    states_list = [np.asarray(s).flatten().astype(int) for s in d["transition_states"].flatten()]
    return states_list, d["regionMap"], d["density"], d["xx"].flatten(), d["peakPoints"]

def make_synthetic(n_flies=12, T=4000, n_super=6, per_super=12, dwell=150, seed=0):
    # Flies whose behavior is biased by a slowly-switching hidden 'mood'. The slow mood is what
    # creates long time scales -- exactly the hidden-state structure the 2016 paper infers.
    rng = np.random.default_rng(seed)
    N = n_super * per_super
    super_of = np.repeat(np.arange(n_super), per_super)
    # 2-D layout: superclusters on a ring, states scattered around their center (so nearby
    # region numbers = similar behaviors, as in real maps).
    ang = 2 * np.pi * super_of / n_super
    centers = np.c_[np.cos(ang), np.sin(ang)] * 6
    pos = centers + rng.normal(scale=0.8, size=(N, 2))
    states_list = []
    for _ in range(n_flies):
        seq, mood = [], rng.integers(n_super)
        s = int(np.where(super_of == mood)[0][rng.integers(per_super)])
        for t in range(T):
            if rng.random() < 1 / dwell:
                mood = rng.integers(n_super)            # slow mood switch
            # prefer states in the current mood's supercluster; sometimes wander
            w = np.where(super_of == mood, 8.0, 1.0)
            w[s] = 0                                    # no self-transition (count transitions)
            w /= w.sum()
            s = int(rng.choice(N, p=w)); seq.append(s)
        # emit 1-based state values so state v <-> regionMap region v (matches the real data,
        # where regionMap==r is behavior r and 0 is background)
        states_list.append(np.array(seq) + 1)
    # a density + region map for plotting
    gx = np.linspace(-9, 9, 200)
    XX, YY = np.meshgrid(gx, gx)
    grid = np.c_[XX.ravel(), YY.ravel()]
    d2 = ((grid[:, None, :] - pos[None]) ** 2).sum(-1)
    regionMap = (d2.argmin(1) + 1).reshape(XX.shape)
    density = np.exp(-d2.min(1) / 2).reshape(XX.shape)
    regionMap[density < 0.05] = 0
    return states_list, regionMap, density, gx, pos

if not USE_SYNTHETIC_DATA:
    try:
        states_list, regionMap, density, xx, peakPoints = load_real()
        print("REAL data: %d flies, %d behaviors" % (len(states_list), int(regionMap.max())))
    except Exception as e:
        print("Could not load the real data (%s) -- falling back to synthetic." % e)
        USE_SYNTHETIC_DATA = True
if USE_SYNTHETIC_DATA:
    states_list, regionMap, density, xx, peakPoints = make_synthetic()
    print("SYNTHETIC data:", len(states_list), "flies,", int(max(s.max() for s in states_list)), "states")
"""))
cells.append(md("The region map &mdash; note that nearby region numbers tend to be similar behaviors:"))
cells.append(code(r"""
cmap = plt.cm.tab20.copy(); cmap.set_bad("white")             # paint the background (region 0) white
fig, ax = plt.subplots(figsize=(5, 5))
ax.imshow(np.ma.masked_where(regionMap == 0, regionMap), cmap=cmap, origin="lower")
ax.axis("equal"); ax.axis("off")
ax.set_title("behavioral regions (colored by region number)"); plt.show()
"""))

# ---------------------------------------------------------------- T(1)
cells.append(md("# 3.&nbsp; The one-step transition matrix"))
cells.append(md(r"""
Let's start with the simplest question we can ask: if the fly is doing behavior $j$ now, what is it
likely to be doing a moment later? That's a **transition matrix**.
$T(\tau)_{ij}=P(S(n+\tau)=i \mid S(n)=j)$ is the probability of being in behavior $i$ a time $\tau$
later, given behavior $j$ now. We start with $\tau=1$ and pool all 59 flies. (We use `getTransitions`
to count transitions *between distinct behaviors*, rather than counting every frame a behavior
persists, exactly as in the paper.)
"""))
cells.append(code(r"""
states = np.concatenate([demoutils.getTransitions(s) for s in states_list])
T1 = demoutils.makeTransitionMatrix(states, 1)

fig, ax = plt.subplots(figsize=(6, 5))
im = ax.imshow(T1, cmap="PuRd", vmax=np.percentile(T1, 99))
ax.set_xlabel("behavior now  (j)"); ax.set_ylabel("behavior next  (i)")
ax.set_title("T(1)"); plt.colorbar(im, fraction=0.046); plt.show()
"""))
cells.append(md(r"""
**Does it look random?** No &mdash; there's block structure along the diagonal. Because nearby
region numbers are similar behaviors, this says the fly mostly transitions *between similar
behaviors* (grooming → grooming, one gait → a neighboring gait). Behavior is locally smooth.
"""))

# ---------------------------------------------------------------- transitions on the map
cells.append(md(r"""
## 3.1&nbsp; The same matrix, drawn on the map

A matrix is hard to *feel*. Those same transition probabilities are really a **graph**: each behavior is
a node sitting at its place on the behavioral map, and each arrow is a transition. Below, a dot's size
is how often the animal visits that region and an arc's width is how often it makes that transition (we
hide the faintest ones, and bow each i&rarr;j arc to its own side so the two directions don't overlap).
Watch how the arrows hug the map's local neighborhoods &mdash; behavior flows mostly between *nearby*
regions, the spatial echo of the block structure in the matrix above.
"""))
cells.append(code(r"""
uniq = np.unique(states[states > 0])                          # region labels, in T1's row/col order
cents = np.full((len(uniq), 2), np.nan)                       # each region's centroid, in image pixels
freq = np.zeros(len(uniq))                                    # how often each region is visited
for k, r in enumerate(uniq):
    ys, xs = np.where(regionMap == r)
    if len(xs):
        cents[k] = [xs.mean(), ys.mean()]
    freq[k] = np.sum(states == r)

def curved(p, q, bend=0.18, n=24):                            # quadratic Bezier p->q, bowed sideways
    p, q = np.asarray(p, float), np.asarray(q, float)
    d = q - p
    ctrl = (p + q) / 2 + bend * np.array([-d[1], d[0]])       # control point, offset perpendicular to p->q
    t = np.linspace(0, 1, n)[:, None]
    return (1 - t) ** 2 * p + 2 * (1 - t) * t * ctrl + t ** 2 * q

maxT = T1.max()
hide = 0.05 * maxT                                            # don't draw the faintest transitions
fig, ax = plt.subplots(figsize=(7, 7))
if density.shape == regionMap.shape:                          # faint behavioral density behind the graph
    ax.imshow(density, origin="lower", cmap="Purples")
ax.set_xlim(0, regionMap.shape[1]); ax.set_ylim(0, regionMap.shape[0])
for i in range(len(uniq)):
    for j in range(len(uniq)):
        if i != j and T1[i, j] >= hide and not np.isnan(cents[i, 0]) and not np.isnan(cents[j, 0]):
            c = curved(cents[i], cents[j])
            ax.plot(c[:, 0], c[:, 1], "-", color="k", lw=5 * T1[i, j] / maxT, alpha=0.5)
ax.scatter(cents[:, 0], cents[:, 1], s=400 * freq / freq.max(), c="firebrick",
           edgecolor="k", linewidth=0.4, zorder=3)
ax.set_aspect("equal"); ax.axis("off")
ax.set_title("transitions on the map  (dot = how often, arc = transition rate)"); plt.show()
"""))

# ---------------------------------------------------------------- lags + Markov
cells.append(md("# 4.&nbsp; Longer lags: does behavior remember?"))
cells.append(md(r"""
Now the question that motivated this whole line of work. Is a fly's next move set entirely by what it's
doing *right now* &mdash; **Markovian**, no memory &mdash; or does its deeper past still leak through
(it's hungry, it's been grooming a while, it's wound up)? This matters: if behavior were memoryless, a
simple model would capture it and there would be little left to explain; if it isn't, that leftover
memory is a clue to internal states we never measured.

There's a clean test. If behavior were Markovian, all that matters is the current state, so the
$\tau$-step matrix would just be the one-step matrix raised to the power $\tau$:
$T_\text{Markov}(\tau)=T(1)^\tau$. Let's compare the *real* $T(\tau)$ to that prediction.
"""))
cells.append(code(r"""
lags = [1, 10, 100, 1000]
fig, axes = plt.subplots(2, len(lags), figsize=(15, 7))
for k, L in enumerate(lags):
    Tdata = demoutils.makeTransitionMatrix(states, L)
    Tmark = np.linalg.matrix_power(T1, L)
    for row, (M, lab) in enumerate([(Tdata, "data"), (Tmark, "Markov  T(1)^%d" % L)]):
        axes[row, k].imshow(M, cmap="PuRd", vmax=np.percentile(Tdata, 99))
        axes[row, k].set_xticks([]); axes[row, k].set_yticks([])
        axes[row, k].set_title(r"$\tau$=%d  (%s)" % (L, lab), fontsize=9)
plt.tight_layout(); plt.show()
"""))
cells.append(md(r"""
The Markov prediction (bottom) washes out into vertical stripes &mdash; it forgets where it
started. The real data (top) **keeps structure far longer**. The fly remembers.
"""))

# ---------------------------------------------------------------- eigenvalues
cells.append(md("# 5.&nbsp; Measuring the time scales: eigenvalues"))
cells.append(md(r"""
The pictures already hint that memory outlives the Markov prediction; now let's put a number on *how
long*. A clean way to read the longest time scale of $T(\tau)$ is its **second eigenvalue**
$|\lambda_2|$ (the first is always 1): the closer to 1, the longer-lived the structure. If behavior
were Markovian, the eigenvalues would decay as $|\lambda_2(\tau)| = |\lambda_2(1)|^\tau$ (red below).
Let's see what the data actually does (blue). *(The sweep over lags below takes only a few seconds.)*
"""))
cells.append(code(r"""
def lam(M, k):  # k-th largest eigenvalue magnitude
    return np.sort(np.abs(np.linalg.eigvals(M)))[::-1][k - 1]

taus = np.unique(np.r_[np.arange(1, 20), np.arange(20, 200, 5), np.arange(200, 1001, 50)])
data2  = [lam(demoutils.makeTransitionMatrix(states, L), 2) for L in taus]
markov2 = [lam(np.linalg.matrix_power(T1, L), 2) for L in taus]

# null floor: shuffle the sequence to destroy temporal order
shuf = demoutils.doTheShannonShuffle(states)
null2 = [lam(demoutils.makeTransitionMatrix(shuf, L), 2) for L in taus]

fig, ax = plt.subplots(figsize=(8, 5))
ax.semilogx(taus, data2, "b.-", label=r"data  $|\lambda_2|$")
ax.semilogx(taus, markov2, "r--", label=r"Markov prediction $|\lambda_2(1)|^\tau$")
ax.semilogx(taus, null2, color="grey", ls=":", label="shuffled (null)")
ax.set_xlabel(r"lag $\tau$ (# transitions)"); ax.set_ylabel(r"$|\lambda_2|$")
ax.legend(); ax.set_title("behavior carries memory far beyond the Markov prediction"); plt.show()
"""))
cells.append(md(r"""
The blue curve sits **far above** the red Markov prediction: there are time scales of hundreds of
transitions, *orders of magnitude* longer than any single movement (which lasts a handful of
frames). Where do they come from? The animal must have **hidden internal states** &mdash; things
we don't measure (hunger, arousal, ...) &mdash; that persist and bias behavior over long times.

> 🔧 **Your turn:** rerun §5 using `lam(..., 3)` and `lam(..., 4)` (the 3rd and 4th eigenvalues).
> Multiple slow modes, multiple time scales.
"""))

# ---------------------------------------------------------------- hierarchy / DIB
cells.append(md("# 6.&nbsp; A hierarchy of behaviors: the deterministic information bottleneck"))
cells.append(md(r"""
§5 told us behavior carries memory far longer than any single movement &mdash; but *where in the
repertoire* does that structure live? Here we coarse-grain the 117 behaviors into a few
**clusters**, keeping the grouping that best predicts what the fly does **next**. This is the
**information bottleneck** (Tishby, Pereira & Bialek 1999); we use the **deterministic** variant
(DIB; Strouse & Schwab 2017), which is exactly the method Berman, Bialek & Shaevitz (2016) used to
pull out the fly's behavioral hierarchy.

In plain terms: we want to bundle the 117 behaviors into a handful of groups that are simple to name but
still tell us what comes next. More groups always predict a little better; fewer are simpler &mdash; and
the whole trade-off between them *is* the hierarchy. Concretely, we summarize the current behavior $X$ by
a cluster label $T$, then score that summary by how much it tells us about the future behavior $Y$. Two
quantities pull against each other:

- $H[T]$ &mdash; the **size** of the summary (bits to name the cluster). Smaller = more compressed.
- $I[Y;T]$ &mdash; how much the summary **predicts the future**. Larger = more useful.

One knob $\beta$ trades them off: we minimize $H[T]-\beta\,I[Y;T]$. Sweep $\beta$ and the number of
clusters and you trace a **Pareto front** &mdash; the most prediction achievable at each level of
compression. Each point on that front is one level of the hierarchy.
"""))
cells.append(md(r"""
**The DIB algorithm** is a hard-clustering loop. Fix a number of clusters $K$ and a trade-off
$\beta$, start from a random assignment of behaviors to clusters, then repeat to convergence:

1. each cluster's **predictive signature** is $p(Y\mid T)$ &mdash; the future-behavior distribution
   averaged over its members;
2. **reassign** every behavior $x$ to the cluster whose signature best matches its own future,
   $\;f(x)=\arg\max_t\,\big[\log p(t)-\beta\,D_\mathrm{KL}(p(Y\mid x)\,\|\,p(Y\mid t))\big].$

This is a direct port of the MATLAB `deterministicInformationBottleneck` / `run_DIB` from Gordon's
behavioral-transitions tutorial.
"""))
cells.append(code(r"""
def _safe_log2(A):
    out = np.zeros_like(A, dtype=float)              # define log2(0):=0  (so 0*log0 -> 0)
    np.log2(A, out=out, where=A > 0)
    return out

def dib_single(pXY, pX, pY_X, Hx, K, beta, rng, tol=1e-6, max_iter=200):
    # One DIB run at fixed (K, beta). pXY = p(current, future); pX, pY_X, Hx are precomputed once
    # (they don't change across runs). Returns (assignment f, I[Y;T], H[T]).
    Nx, Ny = pXY.shape
    f = rng.integers(0, K, size=Nx)                  # random hard assignment behavior -> cluster

    def cluster_stats(f):
        onehot = np.zeros((Nx, K)); onehot[np.arange(Nx), f] = 1.0
        pT = onehot.T @ pX                           # p(T): cluster occupancy
        pYT = onehot.T @ pXY                          # unnormalized p(future, T)
        pY_T = np.divide(pYT, pT[:, None], out=np.zeros_like(pYT), where=pT[:, None] > 0)
        return pT, pY_T                              # p(future | T): each cluster's signature

    def cost(pT, pY_T):
        idx = pT > 0
        H_T = -np.sum(pT[idx] * np.log2(pT[idx]))    # H[T]  (compression)
        pYT = pY_T * pT[:, None]; pY = pYT.sum(0)
        denom = pT[:, None] * pY[None, :]
        ratio = np.divide(pYT, denom, out=np.zeros_like(pYT), where=denom > 0)
        I_YT = (pYT * _safe_log2(ratio)).sum()       # I[Y;T]  (prediction)
        return H_T, I_YT

    pT, pY_T = cluster_stats(f)
    H_T, I_YT = cost(pT, pY_T)
    prev = H_T - beta * I_YT
    for _ in range(max_iter):
        DKL = (-pY_X @ _safe_log2(pY_T).T) - Hx[:, None]      # D_KL(p(Y|x) || p(Y|t)), (Nx, K)
        logpT = np.where(pT > 0, _safe_log2(pT), -np.inf)
        f = np.argmax(logpT[None, :] - beta * DKL, axis=1)    # reassign every behavior
        pT, pY_T = cluster_stats(f)
        H_T, I_YT = cost(pT, pY_T)
        J = H_T - beta * I_YT                                  # the DIB objective
        if abs(J - prev) < tol:
            break
        prev = J
    used = np.unique(f)
    return np.searchsorted(used, f), I_YT, H_T        # drop empty clusters, relabel 0..k-1
"""))
cells.append(md(r"""
One DIB run finds *a* clustering. To map out the whole trade-off we do many runs from random
starts, each with a random number of clusters $K$ and random $\beta$, and keep the **Pareto-optimal**
ones (you can't predict more without spending more bits). It's a Monte-Carlo search, and more restarts
give a smoother front. Since the restarts are independent, we run a generous **10,000** of them in
parallel across whatever CPU cores the runtime has. If parallelism isn't available (`joblib` missing, or
only one core), `run_dib` quietly drops to **1,000** serial restarts &mdash; a coarser but perfectly
usable front that still finishes quickly. **Time taken:** tens of seconds for the parallel 10,000;
~2&ndash;3 s for the 1,000-restart fallback.
"""))
cells.append(code(r"""
def build_joint(trans_list, state_vals, lag):
    # p(current, future) at this lag, counted *within* each fly only (no cross-fly transitions).
    n = len(state_vals)
    F = np.zeros((n, n))
    for s in trans_list:
        s = np.searchsorted(state_vals, s)            # state value -> 0..n-1 index
        if len(s) > lag:
            np.add.at(F, (s[:-lag], s[lag:]), 1.0)
    return F

def pareto_front(pts):
    # rows not strictly dominated in *every* column (here: higher I[Y;T] AND lower H[T])
    keep = np.ones(len(pts), bool)
    for i in range(len(pts)):
        keep[i] = not np.any(np.all(pts > pts[i], axis=1))
    return keep

def run_dib(trans_list, state_vals, lag, n_restarts=None, min_clusters=2, max_clusters=30,
            min_log_beta=-1, max_log_beta=4, seed=0, n_jobs=-1):
    # Precompute the joint distribution and the per-state quantities ONCE -- every restart reuses them.
    pXY = build_joint(trans_list, state_vals, lag); pXY = pXY / pXY.sum()
    pX = pXY.sum(1)                                                  # p(current behavior)
    pY_X = np.divide(pXY, pX[:, None], out=np.zeros_like(pXY), where=pX[:, None] > 0)
    Hx = -np.sum(pY_X * _safe_log2(pY_X), axis=1)                   # entropy of each p(Y|x), reused

    # The restarts are independent, so we run them in parallel across CPU cores when we can. That lets us
    # afford a generous, front-smoothing 10,000 restarts by default; if joblib (or a second core) isn't
    # available we fall back to 1,000 serial restarts so the cell still finishes quickly.
    try:
        from joblib import Parallel, delayed, effective_n_jobs      # ships with scikit-learn on Colab
        parallel = effective_n_jobs(n_jobs) > 1
    except Exception:                                               # no joblib -> serial
        parallel = False
    if n_restarts is None:
        n_restarts = 10000 if parallel else 1000

    # Each restart gets its own reproducible random stream (SeedSequence.spawn), so the result doesn't
    # depend on how many cores we happened to use.
    seeds = np.random.SeedSequence(seed).spawn(n_restarts)
    def one_restart(ss):
        rng = np.random.default_rng(ss)
        beta = 10.0 ** (min_log_beta + (max_log_beta - min_log_beta) * rng.random())
        K = int(rng.integers(min_clusters, max_clusters + 1))
        return dib_single(pXY, pX, pY_X, Hx, K, beta, rng)          # -> (assignment f, I[Y;T], H[T])

    if parallel:
        out = Parallel(n_jobs=n_jobs)(delayed(one_restart)(ss) for ss in seeds)
    else:
        out = [one_restart(ss) for ss in seeds]

    clus = [f for f, _, _ in out]
    IYT = np.array([iyt for _, iyt, _ in out])
    HT  = np.array([ht for _, _, ht in out])
    ncl = np.array([len(np.unique(f)) for f in clus])
    on = pareto_front(np.c_[-HT, IYT])                              # Pareto-optimal trade-offs
    best = {}                                                       # per cluster-count, the max-I[Y;T] solution
    for j in np.where(on)[0]:
        if ncl[j] not in best or IYT[j] > IYT[best[ncl[j]]]:
            best[ncl[j]] = j
    chosen = [best[k] for k in sorted(best)]
    return dict(HT=HT, IYT=IYT, ncl=ncl, on=on, chosen=chosen, clus=clus,
                parallel=parallel, n_restarts=n_restarts)

trans_list = [demoutils.getTransitions(s) for s in states_list]   # per-fly transition sequences
state_vals = np.unique(np.concatenate(trans_list))                # behaviors that actually occur
print("%d behaviors, %d transitions pooled over %d flies"
      % (len(state_vals), sum(len(t) for t in trans_list), len(trans_list)))

dib = run_dib(trans_list, state_vals, lag=5, seed=0)
print("%d restarts (%s); Pareto front: %d optimal clusterings; cluster counts %s"
      % (dib["n_restarts"], "parallel" if dib["parallel"] else "serial",
         len(dib["chosen"]), [int(dib["ncl"][j]) for j in dib["chosen"]]))
"""))
cells.append(md("The trade-off curve. Each red point is an optimal clustering; grey points are runs it beats. Labels mark the number of clusters."))
cells.append(code(r"""
fig, ax = plt.subplots(figsize=(6.5, 5))
off = ~dib["on"]
ax.plot(dib["HT"][off], dib["IYT"][off], "x", color="0.7", ms=4, label="sub-optimal runs")
front = sorted(np.where(dib["on"])[0], key=lambda j: dib["HT"][j])
ax.plot(dib["HT"][front], dib["IYT"][front], "s-", color="crimson", ms=4, label="Pareto front")
for j in dib["chosen"]:
    if dib["ncl"][j] <= 8:
        ax.annotate(str(int(dib["ncl"][j])), (dib["HT"][j], dib["IYT"][j]),
                    textcoords="offset points", xytext=(5, -10), fontsize=9, color="crimson")
ax.set_xlabel(r"$H[T]$  (bits to name the cluster)  $\to$  more compression")
ax.set_ylabel(r"$I[Y;T]$  (bits about the future)  $\to$  more prediction")
ax.legend(loc="lower right"); ax.set_title("DIB trade-off: prediction vs. compression (lag = 5)")
plt.show()
"""))
cells.append(md(r"""
The curve climbs steeply, then saturates: the **first few clusters buy almost all the predictive
information**, and past a handful you pay more and more bits of $H[T]$ for less and less $I[Y;T]$.
Those first, cheap splits are the coarse behavioral categories &mdash; the top of the hierarchy.
"""))
cells.append(md("Now draw the optimal clusterings themselves on the behavior map &mdash; one panel per labelled point on the front, coarse to fine:"))
cells.append(code(r"""
def partition_image(f, state_vals):
    img = np.full(regionMap.shape, np.nan)            # NaN = background / watershed boundaries
    for i, v in enumerate(state_vals):
        img[regionMap == v] = f[i]                    # color region v by its cluster (state v <-> region v)
    return img

levels = [j for j in dib["chosen"] if 2 <= dib["ncl"][j] <= 7]
fig, axes = plt.subplots(1, len(levels), figsize=(3.1 * len(levels), 3.7))
for ax, j in zip(np.atleast_1d(axes), levels):
    ax.imshow(partition_image(dib["clus"][j], state_vals), cmap="tab10",
              origin="lower", interpolation="nearest")
    ax.axis("off")
    ax.set_title("%d clusters\nH[T]=%.2f  I[Y;T]=%.2f"
                 % (dib["ncl"][j], dib["HT"][j], dib["IYT"][j]), fontsize=10)
fig.suptitle("levels of the behavioral hierarchy (DIB partitions)", fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.85]); plt.show()
"""))
cells.append(md(r"""
Two things to notice. The clusters are **spatially contiguous** &mdash; each is a connected
territory on the map, even though the DIB never sees the 2-D layout: it groups behaviors purely by
*shared future*. And the levels are **approximately nested** &mdash; more clusters mostly
*subdivide* existing ones rather than reshuffling. That nesting **is** the hierarchy: coarse
categories (idle / groom / locomote) splitting into finer and finer actions, recovered from
temporal statistics alone. Fly behavior is organized like a tree.
"""))

# ---------------------------------------------------------------- bridge
cells.append(md(r"""
# 7.&nbsp; Where next

Those slow eigenvalues you found in §5 are exactly the spectrum of a *transfer operator*. There's
a principled, modern way to pull the **slow collective modes** straight out of them &mdash; that's
notebook **`05_slow_modes.ipynb`** (and it's what Greg Stephens will go deeper on tomorrow).

🔧 **Your turn:** re-run with `dib = run_dib(trans_list, state_vals, lag=50)` to compress for the
*distant* future, then redraw the two plots above. Do you get *coarser* groups &mdash; fewer cheap
splits on the front &mdash; than predicting the near future? You can also widen the search with
`max_clusters=40`, or pass `n_jobs=1` to force the slow serial path and feel what the parallelism buys.
"""))

write_nb(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                      "02_transitions_and_hierarchy.ipynb"), cells)
