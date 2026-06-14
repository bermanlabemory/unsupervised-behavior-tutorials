"""03 — Rat individual behavior: control vs amphetamine (Ugne Klibaite's rat data).
Applies the notebook-01 mapping engine to a rat, then asks how amphetamine reshapes the behavioral
repertoire. Works from precomputed MotionMapper embeddings + watershed labels (analysis only)."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nb_builder import md, code, badge, write_nb

REPO = "bermanlabemory/unsupervised-behavior-tutorials/blob/main"
cells = []

cells.append(badge("%s/03_rat_individual_behavior.ipynb" % REPO))

cells.append(md(r"""
# 3.&nbsp; Rat individual behavior: control vs amphetamine

Same engine, new animal. In notebook 01 you built a behavioral map for a fly; here we use one for a
**rat**, built exactly the same way (3-D keypoints &rarr; egocentric pose &rarr; wavelets &rarr; map
&rarr; watershed), and put it to work on a real pharmacology question:

> **How does amphetamine reshape an individual rat's behavioral repertoire &mdash; and is that change
> bigger than the normal day-to-day variation?**

The data: **6 rats**, each recorded on **3 days**. Days 2 and 3 are **baseline**; on **day 4** the
animal received **amphetamine**. Because every animal is its own control across days, day2&rarr;day3
gives a built-in *no-drug* yardstick to compare the day3&rarr;day4 drug effect against.

We work from the **precomputed map** (embeddings + behavior labels), so this notebook is all analysis
and runs in a couple of minutes. **Run time:** ~5 min.
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
!pip install -q hdf5storage easydict umap-learn 2>/dev/null

import numpy as np, matplotlib.pyplot as plt, urllib.request
from mpl_toolkits.mplot3d import Axes3D  # noqa
from scipy.stats import wilcoxon
import motionmapperpy as mmpy
%matplotlib inline
print("ready")
"""))

# ---------------------------------------------------------------- data
cells.append(md("# 2.&nbsp; Get the rat data"))
cells.append(md(r"""
Two small files ship in this repo: a short clip of **raw 3-D keypoints** (one example session, just
to see the data modality) and the **precomputed amphetamine maps** (6 rats &times; 3 days).
"""))
cells.append(code(r"""
RAT_BASE = ("https://raw.githubusercontent.com/bermanlabemory/"
            "unsupervised-behavior-tutorials/main/data/rat_data/")
def fetch(fn):
    if not os.path.exists(fn):
        urllib.request.urlretrieve(RAT_BASE + fn, fn)
    return fn

kp = np.load(fetch("rat_keypoints_session1.npz"), allow_pickle=True)
amph = np.load(fetch("amph.npz"))
emb, clust, cclust = amph["emb"], amph["clust"], amph["cclust"]   # (3 days, 6 rats, T, 2) + labels
days = list(amph["days"]); AMPH_DAY = int(amph["amph_day"]); n_coarse = int(amph["n_coarse"])
print("amph maps:", emb.shape, "| days", days, "(day %d = amphetamine)" % AMPH_DAY)
print("keypoint clip:", kp["kp_clip"].shape, "| %d joints @ %d Hz" % (kp["kp_clip"].shape[1], int(kp["fps"])))
"""))

cells.append(md("## 2.1&nbsp; Look at the raw data first"))
cells.append(md(r"""
The front-end for a rat is **3-D keypoints** (here 20 body landmarks tracked with DANNCE), not a
fly's joint angles &mdash; but everything after that is identical. Here are a few frames of the
tracked skeleton, and the behavioral map this example session lives on (peaks = common behaviors).
"""))
cells.append(code(r"""
clip, edges = kp["kp_clip"], kp["edges"]
idxs = np.linspace(0, len(clip) - 1, 4).astype(int)
fig = plt.figure(figsize=(15, 4))
for j, fi in enumerate(idxs):
    ax = fig.add_subplot(1, 4, j + 1, projection="3d")
    P = clip[fi]
    ax.scatter(P[:, 0], P[:, 1], P[:, 2], c="firebrick", s=12)
    for a, b in edges:
        ax.plot(*P[[a, b]].T, color="0.3", lw=1.2)
    ax.set_title("frame %d" % fi); ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    ax.view_init(elev=15, azim=-70)
plt.suptitle("raw 3-D rat keypoints (one example session)"); plt.show()
"""))
cells.append(code(r"""
ze = kp["emb"]; R0 = np.abs(ze).max() + 10
_, _, d0 = mmpy.findPointDensity(ze, 1.0, 501, [-R0, R0])
fig, ax = plt.subplots(figsize=(5, 5))
ax.imshow(d0, extent=(-R0, R0, -R0, R0), origin="lower", cmap=mmpy.gencmap())
ax.set_title("one rat session as a behavioral map"); ax.axis("off"); plt.show()
"""))

# ---------------------------------------------------------------- experiment
cells.append(md("# 3.&nbsp; The amphetamine experiment"))
cells.append(md(r"""
Now the 6-rat &times; 3-day dataset. All 18 (animal, day) recordings were embedded into **one shared
map**, so their densities are directly comparable. Here's the shared map, then one animal across its
three days &mdash; watch day 4 (amphetamine) redistribute where it spends time.
"""))
cells.append(code(r"""
allz = emb.reshape(-1, 2); R = np.abs(allz).max() + 10; ext = (-R, R, -R, R)
def density(z, sigma=1.5):
    return mmpy.findPointDensity(z, sigma, 501, [-R, R])[2]
D_all = density(allz); inside = D_all > D_all.max() * 1e-3

a = 0    # 🔧 which animal (0..5)
fig, ax = plt.subplots(1, 4, figsize=(18, 4.6))
ax[0].imshow(D_all, extent=ext, origin="lower", cmap=mmpy.gencmap())
ax[0].set_title("shared map (all rats, all days)"); ax[0].axis("off")
for k, dd in enumerate(days):
    ax[k + 1].imshow(density(emb[k, a]), extent=ext, origin="lower", cmap=mmpy.gencmap())
    ax[k + 1].set_title("rat %d, day %d%s" % (a + 1, dd, "  (AMPH)" if dd == AMPH_DAY else ""))
    ax[k + 1].axis("off")
plt.show()
"""))

# ---------------------------------------------------------------- diff maps
cells.append(md("# 4.&nbsp; The drug effect vs the day-to-day drift"))
cells.append(md(r"""
A change on the amphetamine day only means something if it's bigger than how much the animal's
behavior wanders **on its own**. So we compare two difference maps, averaged over all 6 rats: the
baseline drift (**day3 &minus; day2**, no drug) and the amphetamine effect (**day4 &minus; day3**).
"""))
cells.append(code(r"""
def mean_density(k):
    return np.mean([density(emb[k, a]) for a in range(emb.shape[1])], 0)
D2, D3, D4 = mean_density(0), mean_density(1), mean_density(2)

fig, ax = plt.subplots(1, 2, figsize=(11, 5))
for a_, (Da, Db, t) in zip(ax, [(D3, D2, "baseline drift (day3 - day2)"),
                                 (D4, D3, "AMPH effect (day4 - day3)")]):
    diff = Da - Db; diff[~inside] = 0; v = np.abs(diff).max()
    im = a_.imshow(diff, extent=ext, origin="lower", cmap="RdBu_r", vmin=-v, vmax=v)
    a_.set_title(t); a_.axis("off")
plt.colorbar(im, ax=ax, fraction=0.025); plt.suptitle("red = more time after; blue = less"); plt.show()
"""))

# ---------------------------------------------------------------- quantify
cells.append(md("# 5.&nbsp; Quantify it: how much did the repertoire change?"))
cells.append(md(r"""
Summarize each (animal, day) by its **occupancy fingerprint** &mdash; the fraction of time spent in
each of the 9 coarse behavior classes &mdash; and measure the change between days with the
**Jensen-Shannon divergence** (0 = identical repertoire, larger = more different). For each rat we
compare the no-drug change (day2&rarr;3) against the amphetamine change (day3&rarr;4).
"""))
cells.append(code(r"""
def occupancy(labels):
    h = np.bincount(labels, minlength=n_coarse + 1)[1:].astype(float)
    return h / h.sum()
def js(p, q):                               # Jensen-Shannon divergence in bits
    p = p + 1e-12; q = q + 1e-12; p /= p.sum(); q /= q.sum(); m = 0.5 * (p + q)
    kl = lambda x, y: np.sum(x * np.log2(x / y))
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)

nrat = emb.shape[1]
js_baseline = np.array([js(occupancy(cclust[0, a]), occupancy(cclust[1, a])) for a in range(nrat)])
js_amph     = np.array([js(occupancy(cclust[1, a]), occupancy(cclust[2, a])) for a in range(nrat)])
pval = wilcoxon(js_baseline, js_amph).pvalue

fig, ax = plt.subplots(figsize=(5, 5))
ax.bar([0, 1], [js_baseline.mean(), js_amph.mean()], color=["royalblue", "firebrick"], alpha=.85)
ax.plot([np.zeros(nrat), np.ones(nrat)], [js_baseline, js_amph], "-", color="0.4", lw=1)
ax.plot(np.zeros(nrat), js_baseline, "ko"); ax.plot(np.ones(nrat), js_amph, "ko")
ax.set_xticks([0, 1]); ax.set_xticklabels(["baseline\n(day2->3)", "amphetamine\n(day3->4)"])
ax.set_ylabel("JS divergence (bits)")
ax.set_title("amphetamine changes the repertoire more than\nday-to-day drift  (%d/%d rats, p=%.3f)"
             % (int((js_amph > js_baseline).sum()), nrat, pval)); plt.show()
"""))
cells.append(md(r"""
Every rat moves up: the amphetamine day reshapes the behavioral repertoire **more than the animal's
own day-to-day variation**, and the paired (Wilcoxon signed-rank) test says that's unlikely to be
chance. This is the unsupervised version of a drug-effect readout &mdash; no behavior was ever
hand-defined.
"""))

# ---------------------------------------------------------------- which behaviors
cells.append(md("# 6.&nbsp; Which behaviors does amphetamine drive?"))
cells.append(md("Break the change down by coarse behavior class: which does the drug push *up*, and which *down*?"))
cells.append(code(r"""
occ_base = np.array([occupancy(cclust[1, a]) for a in range(nrat)]).mean(0)   # day3 baseline
occ_amph = np.array([occupancy(cclust[2, a]) for a in range(nrat)]).mean(0)   # day4 amphetamine
fold = np.log2((occ_amph + 1e-3) / (occ_base + 1e-3))

fig, ax = plt.subplots(figsize=(8, 4))
ax.bar(range(1, n_coarse + 1), fold, color=["firebrick" if f > 0 else "royalblue" for f in fold])
ax.axhline(0, color="k", lw=.5)
ax.set_xlabel("coarse behavior class"); ax.set_ylabel("log2 fold change (AMPH / baseline)")
ax.set_title("amphetamine up-regulates some behaviors, suppresses others"); plt.show()
print("To NAME these classes, watch example clips of each region (as in notebook 01 section 9).")
"""))

# ---------------------------------------------------------------- exercises
cells.append(md(r"""
# 7.&nbsp; 🔧 Your turn

1. **Pick a different rat.** Change `a` in section 3 (0&ndash;5) and rerun &mdash; is the
   amphetamine redistribution in the same part of the map for every animal?
2. **Fine vs coarse.** Redo section 5 with the **fine** 163-class labels (use `clust` instead of
   `cclust`, and set the bin count to `int(amph["n_fine"])`). Does the drug effect look bigger or
   smaller at fine grain &mdash; and why might that be?
3. **How steady is baseline?** We used day2&rarr;3 as the no-drug yardstick. How variable is that
   drift across animals? Is a single baseline day enough to trust the day4 effect?

Next, with the same machinery: **`04_rat_social_behavior.ipynb`** asks what new questions appear when
you put *two* animals together.
"""))

write_nb(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                      "03_rat_individual_behavior.ipynb"), cells)
