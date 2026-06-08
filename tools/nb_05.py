"""05 — Slow modes & metastable states (Kaur, Jain & Berman 2026).
A robust live 'concept' demo (a hidden slow variable made visible by wavelets),
then the REAL published results loaded from the slowmode repo's cached data."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nb_builder import md, code, badge, write_nb

REPO = "bermanlabemory/unsupervised_behavior_tutorial/blob/main"
cells = []

cells.append(badge("%s/05_slow_modes.ipynb" % REPO))

cells.append(md(r"""
# Slow modes & metastable states &nbsp;·&nbsp; 🔴 hard (and worth it)

**The question:** notebook 03 showed behavior has time scales *far* longer than any single movement
&mdash; evidence of **hidden internal states** (hunger, arousal, ...) that we never measure but that
bias what the animal does. Can we **pull those slow states out of the data**?

Following Kaur, Jain & Berman 2026 ("Using timescale as a state coordinate"), the trick is to build
the state space in the **time-frequency (wavelet) domain**, so fast movements and slow modulations
are visible *at the same time*. The slow modes then fall out of the **transfer operator's** spectrum
&mdash; the same eigenvalues you met in notebook 03.

We'll (1) prove the core idea on a toy signal where we *know* the hidden variable, then (2) look at
the **real published results** &mdash; worms and flies &mdash; loaded from the `slowmode` repo. This
is exactly where **Greg Stephens picks up tomorrow.**

**Run time:** ~5 min.
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
# slowmode: the real pipeline + cached results for the worm & fly figures.
if not os.path.exists("slowmode"):
    !git clone -q https://github.com/bermanlabemory/slowmode
!pip install -q pygpcca powerlaw umap-learn easydict 2>/dev/null

import numpy as np, matplotlib.pyplot as plt
import motionmapperpy as mmpy
%matplotlib inline
rng = np.random.default_rng(0)
print("ready")
"""))

# ---------------------------------------------------------------- concept
cells.append(md("# 2.&nbsp; The core idea: make *timescale* a coordinate"))
cells.append(md(r"""
Here's a signal you might record: a fast oscillation. Secretly, a **slow hidden variable** flips its
*frequency* between two values every so often (think: an internal state that speeds the animal up).
You only get to see the fast wiggle. **Can you recover the hidden state?**
"""))
cells.append(code(r"""
fs, T = 30.0, 12000
hidden = np.zeros(T, int)                       # the slow hidden state (0 or 1)
for t in range(1, T):
    hidden[t] = hidden[t-1] if rng.random() > 1/600 else 1 - hidden[t-1]
freq = np.where(hidden == 0, 2.0, 6.0)          # hidden state secretly sets the frequency
x = np.sin(np.cumsum(2*np.pi*freq/fs)) + 0.3*rng.standard_normal(T)   # all you observe

fig, ax = plt.subplots(2, 1, figsize=(13, 4), sharex=True)
ax[0].plot(x[:3000], lw=.6, color="0.3"); ax[0].set_ylabel("observed signal x(t)")
ax[1].plot(hidden, color="seagreen"); ax[1].set_ylabel("HIDDEN state"); ax[1].set_xlabel("frame")
ax[0].set_title("you see the top; the bottom is hidden"); plt.show()
"""))
cells.append(md("**Instantaneously**, x tells you almost nothing about the hidden state. But take its **wavelet spectrogram** &mdash; make timescale a coordinate &mdash; and the hidden state is obvious:"))
cells.append(code(r"""
w, freqs = mmpy.findWavelets(x[:, None], 1, 5, 25, fs, 12.0, 0.5, -1, -1)   # (T, 25 frequencies)

fig, ax = plt.subplots(2, 1, figsize=(13, 5), sharex=True)
ax[0].imshow(w[:3000].T, aspect="auto", origin="lower", cmap="PuRd",
             extent=(0, 3000, freqs[-1], freqs[0]))
ax[0].set_ylabel("frequency (Hz)"); ax[0].set_title("wavelet spectrogram — the band visibly switches")
band = w[:, freqs > 4].mean(1) - w[:, freqs < 4].mean(1)     # high-band minus low-band power
ax[1].plot(band / np.abs(band).max(), color="firebrick", label="wavelet band power")
ax[1].plot(hidden, color="seagreen", alpha=.6, label="true hidden state"); ax[1].legend()
ax[1].set_xlabel("frame"); ax[1].set_xlim(0, 3000); plt.show()

r_raw = abs(np.corrcoef(x, hidden)[0, 1])
r_wav = max(abs(np.corrcoef(w[:, k], hidden)[0, 1]) for k in range(w.shape[1]))
print("correlation with the hidden state:   raw signal |r| = %.2f    wavelet band |r| = %.2f" % (r_raw, r_wav))
"""))
cells.append(md(r"""
The raw signal hides the slow variable ($|r|\approx0$); the time-frequency view **exposes** it
($|r|\approx1$). That's the whole idea. The full method clusters this wavelet representation, builds
a **transfer operator**, and reads its **slow eigenvectors** &mdash; and for $>2$ basins uses
**G-PCCA** to extract them. Below we look at what that recovers on real animals.
"""))

# ---------------------------------------------------------------- worms
cells.append(md("# 3.&nbsp; Worms: two metastable basins = run vs pirouette"))
cells.append(md(r"""
*C. elegans* posture is summarized by 5 "eigenworms". Running the multi-timescale operator on 12
worms, the eigenvalue spectrum shows a clear gap after **M = 2** basins, which turn out to be the
canonical **run** and **pirouette** states (Kaur 2026, Fig 3; data cached in the `slowmode` repo).
"""))
cells.append(code(r"""
W = "slowmode/data/worms"
try:
    ev = np.abs(np.load(W + "/worm_eigs_tau3s.npz")["evals_mt"])
    um = np.load(W + "/worms_umap_canonical_full.npz")
    dev, xe, ye = um["dev"], um["x_edges"], um["y_edges"]   # per-basin enrichment on the UMAP

    fig = plt.figure(figsize=(14, 4))
    ax0 = fig.add_subplot(1, 3, 1)
    ax0.plot(range(1, len(ev) + 1), ev, "ko-"); ax0.axvline(2.5, ls="--", color="orange")
    ax0.set_xlabel("eigenvalue index"); ax0.set_ylabel(r"$|\lambda|$"); ax0.set_title("spectral gap -> M=2")
    for k, name in [(0, "Pirouette"), (1, "Run")]:
        ax = fig.add_subplot(1, 3, k + 2)
        v = np.nanpercentile(np.abs(dev), 99)
        ax.pcolormesh(xe, ye, dev[k], cmap="RdBu_r", vmin=-v, vmax=v, shading="auto")
        ax.set_title("Basin %d: %s" % (k + 1, name)); ax.set_aspect("equal"); ax.axis("off")
    plt.tight_layout(); plt.show()
except Exception as e:
    print("Could not load cached worm data (%s)." % e)
    print("See slowmode/worms.ipynb — the canonical version.")
"""))

# ---------------------------------------------------------------- flies
cells.append(md("# 4.&nbsp; Flies: four basins, painted onto the behavior map"))
cells.append(md(r"""
Flies have a bigger fast/slow separation, and the method finds **M = 4** metastable basins. The
beautiful part: each basin maps onto a coherent region of the **same MotionMapper behavior map you
built in notebook 01** (Kaur 2026, Fig 5A). The slow internal states recover coarse behavioral
organization &mdash; *idle/slow, anterior movements, posterior & wing movements, locomotion* &mdash;
straight from posture.
"""))
cells.append(code(r"""
F = "slowmode/data/flies"
try:
    bd = np.load(F + "/behavior_density_chi_tau2s.npz")
    cond, bg, xe, ye = bd["cond"], bd["bg"], bd["x_edges"], bd["y_edges"]
    names = ["basin 1", "basin 2", "basin 3", "basin 4"]   # ~ idle/slow, anterior, posterior&wing, locomotion
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    v = np.nanpercentile(np.abs(cond), 99)
    for k, ax in enumerate(axes):
        ax.pcolormesh(xe, ye, cond[k], cmap="RdBu_r", vmin=-v, vmax=v, shading="auto")
        ax.contour(0.5 * (xe[:-1] + xe[1:]), 0.5 * (ye[:-1] + ye[1:]), bg, [0.5], colors="0.5", linewidths=.5)
        ax.set_title(names[k]); ax.set_aspect("equal"); ax.axis("off")
    plt.suptitle("fly metastable basins on the behavior map"); plt.tight_layout(); plt.show()
except Exception as e:
    print("Could not load cached fly maps (%s). See slowmode/flies.ipynb." % e)
"""))
cells.append(md("The **arms-and-hub** geometry the theory predicts: cluster centroids form M arms radiating from a hub, one per basin (Kaur 2026, Fig 4):"))
cells.append(code(r"""
try:
    gp = np.load(F + "/gpcca_flies_M4_tau2s.npz")
    cent, hub, chi = gp["arm_centroids"], gp["hub"], gp["chi"]    # centroids in (phi2,phi3,phi4)
    phi = np.load(F + "/fly_eigs_tau2s.npz")["phi_mt"]            # per-cluster slow eigenvectors
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
    ax[0].scatter(phi[:, 1], phi[:, 2], c=chi.argmax(1), cmap="tab10", s=12, alpha=.6)
    for k in range(len(cent)):
        ax[0].plot([hub[0], cent[k, 0]], [hub[1], cent[k, 1]], "k-", lw=2)
    ax[0].plot(*hub[:2], "k*", ms=15); ax[0].set_xlabel(r"$\phi_2$"); ax[0].set_ylabel(r"$\phi_3$")
    ax[0].set_title("four arms from a hub")
    fe = np.load(F + "/fly_eigs_tau2s.npz")
    ax[1].plot(np.abs(fe["evals_mt"]), "o-", label="multi-timescale")
    ax[1].plot(np.abs(fe["evals_fix"]), "s--", color="firebrick", label="fixed-timescale")
    ax[1].set_xlabel("eigenvalue index"); ax[1].set_ylabel(r"$|\lambda|$"); ax[1].legend()
    ax[1].set_title("multi-timescale keeps slow modes; fixed doesn't"); plt.tight_layout(); plt.show()
except Exception as e:
    print("Could not load cached fly G-PCCA (%s). See slowmode/flies.ipynb." % e)
"""))

# ---------------------------------------------------------------- diagnostics
cells.append(md("# 5.&nbsp; Don't trust it blindly: the diagnostics"))
cells.append(md(r"""
The most useful idea in the paper for *practitioners*: four **falsifiable** checks that tell you
whether slow structure is really there &mdash; (i) a clear spectral gap at M; (ii) **collective**
modes (high participation ratio), not a few lone clusters; (iii) the arms geometry; (iv) held-out
prediction beating a memoryless null. `slowmode/diagnostics.py` returns a pass/fail table. *A method
that tells you when it does **not** apply is rare and precious.*
"""))
cells.append(code(r"""
try:
    phi = np.load(F + "/fly_eigs_tau2s.npz")["phi_mt"]; ev = np.abs(np.load(F + "/fly_eigs_tau2s.npz")["evals_mt"])
    gap = ev[1] / ev[2]
    pr = (phi[:, 1] ** 2).sum() ** 2 / (phi[:, 1] ** 4).sum()
    print("flies, multi-timescale operator:")
    print("  spectral gap |lambda_2|/|lambda_3| = %.2f   (a clear gap supports a slow mode)" % gap)
    print("  participation ratio of phi_2       = %.0f   (>>1 => collective, not a few clusters)" % pr)
    print("\nFull 4-criterion pass/fail table: slowmode/diagnostics.run_diagnostics (see flies.ipynb)")
except Exception as e:
    print("see slowmode/flies.ipynb (%s)" % e)
"""))

# ---------------------------------------------------------------- your own
cells.append(md("# 6.&nbsp; Your own data &mdash; and a caution"))
cells.append(md(r"""
The `slowmode` repo ships `user_data.ipynb`, which runs this whole pipeline on any multivariate time
series &mdash; including the fly PCA projections from notebook 01, or your own. The honest caveat:
this finds **slow** structure, so it needs **long** recordings (or many pooled individuals) for the
slow process to switch enough to be seen &mdash; minutes-to-hours, not a 30-second clip. For your
projects, use the long provided datasets.

There's also a beautiful synthetic **positive control** &mdash; a chaotic Lorenz system with a hidden
bistable driver that the method recovers across nearly three decades of dwell time &mdash; in
`slowmode/lorenz.ipynb`. Worth a look.

🔧 **Your turn:** open `slowmode/user_data.ipynb` and feed it `Fly_mmpy/Projections/*_pcaModes.mat`
from notebook 01.

➡️ **Tomorrow with Greg Stephens:** maximally predictive states, operators, and the dynamics of
behavior. This notebook is your running start.
"""))

write_nb(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                      "05_slow_modes.ipynb"), cells)
