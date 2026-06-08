"""03 — Transitions & hierarchy (Berman et al. 2016). Python port of Gordon's
behavioral-transitions tutorial, built on motionmapperpy's demoutils."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nb_builder import md, code, badge, write_nb

REPO = "bermanlabemory/unsupervised_behavior_tutorial/blob/main"
cells = []

cells.append(badge("%s/03_transitions_and_hierarchy.ipynb" % REPO))

cells.append(md(r"""
# Transitions & hierarchy &nbsp;·&nbsp; 🟠 medium–hard

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
%cd motionmapperpy
!python setup.py install -q 2>/dev/null
%cd ..
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

> **Instructors:** host `transition_data.mat` and set `USE_SYNTHETIC_DATA = False` to use the
> real 59-fly dataset. Until then this generates stand-in data with the same structure.
"""))
cells.append(code(r"""
USE_SYNTHETIC_DATA = True
DATA_URL = "https://PLACEHOLDER-HOST/transition_data.mat"   # TODO: real link

def load_real():
    import hdf5storage
    if not os.path.exists("transition_data.mat"):
        !wget -q "$DATA_URL" -O transition_data.mat
    d = hdf5storage.loadmat("transition_data.mat")
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
        states_list.append(np.array(seq))
    # a density + region map for plotting
    gx = np.linspace(-9, 9, 200)
    XX, YY = np.meshgrid(gx, gx)
    grid = np.c_[XX.ravel(), YY.ravel()]
    d2 = ((grid[:, None, :] - pos[None]) ** 2).sum(-1)
    regionMap = (d2.argmin(1) + 1).reshape(XX.shape)
    density = np.exp(-d2.min(1) / 2).reshape(XX.shape)
    regionMap[density < 0.05] = 0
    return states_list, regionMap, density, gx, pos

if USE_SYNTHETIC_DATA:
    states_list, regionMap, density, xx, peakPoints = make_synthetic()
    print("SYNTHETIC data:", len(states_list), "flies,", int(max(s.max() for s in states_list)) + 1, "states")
else:
    states_list, regionMap, density, xx, peakPoints = load_real()
    print("REAL data:", len(states_list), "flies")
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
nstate = int(states.max()) + 1
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
cells.append(md("# 6.&nbsp; Where do the time scales live? A hierarchy of behaviors"))
cells.append(md(r"""
If we had to *coarse-grain* the 117 behaviors into a few groups while preserving as much
information about the **future** as possible, what groups would we get? This is the
**information bottleneck** idea. We do it agglomeratively: repeatedly merge the two behaviors
whose *futures* look most alike (lose the least predictive information), building a tree from 117
behaviors down to 2.
"""))
cells.append(code(r"""
def future_distributions(states, lag, n):
    # P(future state | current cluster), plus each cluster's occupancy p.
    Pj = np.zeros((n, n))               # rows: future, cols: current
    for a, b in zip(states[:-lag], states[lag:]):
        Pj[b, a] += 1
    p = Pj.sum(0); p = p / p.sum()
    cond = Pj / np.clip(Pj.sum(0), 1, None)     # columns normalized -> p(future|current)
    return cond.T, p                            # cond[i] = future dist of cluster i

def js(p, q, wp, wq):                  # information lost by merging (weighted Jensen-Shannon)
    m = wp * p + wq * q
    def kl(a, b):
        mask = a > 0
        return np.sum(a[mask] * np.log2(a[mask] / np.clip(b[mask], 1e-12, None)))
    return wp * kl(p, m) + wq * kl(q, m)

def agglomerative_ib(states, lag):
    n = int(states.max()) + 1
    cond, p = future_distributions(states, lag, n)
    clusters = {i: [i] for i in range(n)}
    cdist = {i: cond[i].copy() for i in range(n)}
    cp = {i: p[i] for i in range(n)}
    merges, info = [], []
    members = {i: i for i in range(n)}        # state -> current cluster id
    while len(clusters) > 1:
        ids = list(clusters)
        best, bc = None, np.inf
        for a_i in range(len(ids)):
            for b_i in range(a_i + 1, len(ids)):
                a, b = ids[a_i], ids[b_i]
                w = cp[a] + cp[b]
                if w == 0:
                    cost = 0.0
                else:
                    cost = w * js(cdist[a], cdist[b], cp[a] / w, cp[b] / w)
                if cost < bc:
                    bc, best = cost, (a, b)
        a, b = best
        w = cp[a] + cp[b] or 1.0
        cdist[a] = (cp[a] * cdist[a] + cp[b] * cdist[b]) / w
        cp[a] += cp[b]; clusters[a] += clusters[b]
        for s in clusters[b]:
            members[s] = a
        del clusters[b], cdist[b], cp[b]
        merges.append(dict(members)); info.append(bc)
    return merges[::-1]                         # coarse -> fine snapshots

merges = agglomerative_ib(states, lag=5)
print("built hierarchy from 2 up to %d clusters" % nstate)
"""))
cells.append(md("Draw a few levels of the hierarchy on the behavior map &mdash; watch clusters *subdivide* as we allow more of them:"))
cells.append(code(r"""
def relabel(members, K):
    ids = sorted(set(members.values()))
    if len(ids) != K:                # pick the snapshot with exactly K clusters
        return None
    remap = {c: i for i, c in enumerate(ids)}
    return np.array([remap[members[s]] for s in range(nstate)])

Ks = [2, 3, 4, 6]
fig, axes = plt.subplots(1, len(Ks), figsize=(16, 4))
for ax, K in zip(axes, Ks):
    snap = next((m for m in merges if len(set(m.values())) == K), merges[min(K, len(merges)) - 1])
    lab = relabel(snap, len(set(snap.values())))
    img = np.zeros_like(regionMap, dtype=float) - 1
    for r in range(1, nstate + 1):
        if r - 1 < len(lab):
            img[regionMap == r] = lab[r - 1]
    img[regionMap == 0] = np.nan
    ax.imshow(img, cmap="tab10", origin="lower"); ax.axis("off")
    ax.set_title("%d clusters" % len(set(snap.values())))
plt.suptitle("a hierarchy: more clusters subdivide existing ones"); plt.show()
"""))
cells.append(md(r"""
The partitions are **spatially contiguous** and **nested** &mdash; new clusters carve up old ones
rather than reshuffling. That's the signature of a **hierarchy**: coarse behavioral categories
(idle / groom / locomote) that split into finer and finer actions. Fly behavior is organized like
a tree.
"""))

# ---------------------------------------------------------------- bridge
cells.append(md(r"""
# 7.&nbsp; Where next

Those slow eigenvalues you found in §5 are exactly the spectrum of a *transfer operator*. There's
a principled, modern way to pull the **slow collective modes** straight out of them &mdash; that's
notebook **`05_slow_modes.ipynb`** (and it's what Greg Stephens will go deeper on tomorrow).

🔧 **Your turn:** change the lag in `agglomerative_ib(states, lag=5)` to 50. Does coarse-graining
to predict the *distant* future give you *coarser* groups than predicting the near future?
"""))

write_nb(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                      "03_transitions_and_hierarchy.ipynb"), cells)
