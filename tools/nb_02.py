"""03 — Transitions & hierarchy (Berman et al. 2016). Python port of Gordon's
behavioral-transitions tutorial, built on motionmapperpy's demoutils."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nb_builder import md, code, badge, write_nb

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

You don't need to have finished notebook 01 &mdash; this loads its own data: sequences of
behavioral states for **59 flies, 117 behaviors each**, from one hour of recording apiece.

**Run time:** ~10 min.
"""))

# ---------------------------------------------------------------- setup
cells.append(md("# 1.&nbsp; Setup"))
cells.append(code(r"""
import os
if not os.path.exists("motionmapperpy"):
    !git clone -q https://github.com/bermanlabemory/motionmapperpy
# This notebook uses matplotlib (not moviepy) for visuals. The released package still imports
# moviepy at load time, so we stub it out -- sidestepping the whole moviepy/ffmpeg mess on Colab.
import sys, types
def _stub(name, **attrs):
    m = sys.modules.get(name) or types.ModuleType(name)
    for k, v in attrs.items():
        setattr(m, k, v)
    sys.modules[name] = m
    return m
_stub("moviepy"); _stub("moviepy.editor", VideoClip=object, VideoFileClip=object)
_stub("moviepy.video"); _stub("moviepy.video.io")
_stub("moviepy.video.io.bindings", mplfig_to_npimage=lambda *a, **k: None)

# Import straight from the clone -- avoids the "setup.py install" empty-namespace trap; no restart.
sys.path.insert(0, os.path.abspath("motionmapperpy"))
for _m in [k for k in list(sys.modules) if k.startswith("motionmapperpy")]:
    del sys.modules[_m]
!pip install -q hdf5storage easydict 2>/dev/null

import numpy as np, matplotlib.pyplot as plt
import motionmapperpy as mmpy
from motionmapperpy import demoutils
%matplotlib inline
print("ready")
"""))

# ---------------------------------------------------------------- data
cells.append(md("# 2.&nbsp; Load the behavioral state sequences"))
cells.append(md(r"""
Each fly's behavior is an integer time series: which of the 117 behaviors it was doing at each
step. We also load the 2-D **region map** (so we can draw results on the behavior space) and the
behavioral **density**.

This loads the **real 59-fly dataset** (`data/transition_data.mat`, 117 behaviors) straight from
this tutorial's GitHub repo. If the download fails (e.g. the repo isn't public yet), it falls back
to a stand-in generator with the same structure &mdash; set `USE_SYNTHETIC_DATA = True` to force that.
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
        raise RuntimeError("download failed -- is the repo public yet?")
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
fig, ax = plt.subplots(figsize=(5, 5))
ax.imshow(regionMap, cmap="tab20", origin="lower"); ax.axis("equal"); ax.axis("off")
ax.set_title("behavioral regions (colored by region number)"); plt.show()
"""))

# ---------------------------------------------------------------- T(1)
cells.append(md("# 3.&nbsp; The one-step transition matrix"))
cells.append(md(r"""
$T(\tau)_{ij}=P(S(n+\tau)=i \mid S(n)=j)$ is the probability of being in behavior $i$ a time
$\tau$ later, given behavior $j$ now. Start with $\tau=1$. We pool all flies. (We use
`getTransitions` to count *transitions* between distinct behaviors, as in the paper.)
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

# ---------------------------------------------------------------- lags + Markov
cells.append(md("# 4.&nbsp; Longer lags: is behavior Markovian?"))
cells.append(md(r"""
If behavior were **Markovian** (no memory), the only thing that matters is the current state, and
the $\tau$-step matrix would just be the one-step matrix raised to the power $\tau$:
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
A clean way to read the longest time scale of $T(\tau)$ is its **second eigenvalue** $|\lambda_2|$
(the first is always 1). Closer to 1 = longer-lived structure. If behavior were Markovian, the
eigenvalues would decay as $|\lambda_2(\tau)| = |\lambda_2(1)|^\tau$ (red below). Let's see what
the data does (blue).
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

Summarize the current behavior $X$ by a cluster label $T$, then score that summary by how much it
tells us about the future behavior $Y$. Two quantities pull against each other:

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
ones (you can't predict more without spending more bits). It's a Monte-Carlo search &mdash; more
restarts give a smoother front; ~600 takes a few seconds.
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

def run_dib(trans_list, state_vals, lag, n_restarts=600, min_clusters=2, max_clusters=30,
            min_log_beta=-1, max_log_beta=4, seed=0):
    rng = np.random.default_rng(seed)
    pXY = build_joint(trans_list, state_vals, lag); pXY = pXY / pXY.sum()
    pX = pXY.sum(1)                                                  # p(current behavior)
    pY_X = np.divide(pXY, pX[:, None], out=np.zeros_like(pXY), where=pX[:, None] > 0)
    Hx = -np.sum(pY_X * _safe_log2(pY_X), axis=1)                   # entropy of each p(Y|x), reused
    HT = np.zeros(n_restarts); IYT = np.zeros(n_restarts); ncl = np.zeros(n_restarts, int)
    clus = [None] * n_restarts
    for i in range(n_restarts):
        beta = 10.0 ** (min_log_beta + (max_log_beta - min_log_beta) * rng.random())
        K = int(rng.integers(min_clusters, max_clusters + 1))
        clus[i], IYT[i], HT[i] = dib_single(pXY, pX, pY_X, Hx, K, beta, rng)
        ncl[i] = len(np.unique(clus[i]))
    on = pareto_front(np.c_[-HT, IYT])                              # Pareto-optimal trade-offs
    best = {}                                                       # per cluster-count, the max-I[Y;T] solution
    for j in np.where(on)[0]:
        if ncl[j] not in best or IYT[j] > IYT[best[ncl[j]]]:
            best[ncl[j]] = j
    chosen = [best[k] for k in sorted(best)]
    return dict(HT=HT, IYT=IYT, ncl=ncl, on=on, chosen=chosen, clus=clus)

trans_list = [demoutils.getTransitions(s) for s in states_list]   # per-fly transition sequences
state_vals = np.unique(np.concatenate(trans_list))                # behaviors that actually occur
print("%d behaviors, %d transitions pooled over %d flies"
      % (len(state_vals), sum(len(t) for t in trans_list), len(trans_list)))

dib = run_dib(trans_list, state_vals, lag=5, n_restarts=600, seed=0)
print("Pareto front: %d optimal clusterings; cluster counts %s"
      % (len(dib["chosen"]), [int(dib["ncl"][j]) for j in dib["chosen"]]))
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
splits on the front &mdash; than predicting the near future? Then try widening the search with
`max_clusters=40` or `n_restarts=2000` for a cleaner front.
"""))

write_nb(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                      "02_transitions_and_hierarchy.ipynb"), cells)
