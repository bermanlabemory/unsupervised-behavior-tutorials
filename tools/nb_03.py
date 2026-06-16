"""03 — Rat individual behavior: control vs amphetamine (Ugne Klibaite's rat data).
Applies the notebook-01 mapping engine to a rat, then asks how amphetamine reshapes the behavioral
repertoire. Works from precomputed MotionMapper embeddings + watershed labels (analysis only)."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nb_builder import md, code, badge, write_nb, setup_code, carry_from_core

REPO = "bermanlabemory/unsupervised-behavior-tutorials/blob/main"
cells = []

cells.append(badge("%s/03_rat_individual_behavior.ipynb" % REPO))

cells.append(md(r"""
# Rat individual behavior &mdash; Quantifying rats on speed
In notebook 01 you built a behavioral map for a fly; here we use one for a
**rat**, built in essentially the same way (now with 3-D keypoints instead of 2-D keypoints &rarr; egocentric pose &rarr; wavelets &rarr; map
&rarr; watershed).  

Naturally, the obvious next step is to give the animals speed.  Absurdity aside, we will ask how amphetamine reshape an individual rat's behavioral repertoire and compare that change to normal day-to-day behavioral variation?

The data (from Ugne Klibaite (Harvard), paper here: https://www.cell.com/cell/fulltext/S0092-8674(25)00154-0): **6 rats**, each recorded on **3 days** (really, it was 5 days, but we're restricting it here for the sake of minimizing runtime). Days 2 and 3 are **baseline**; on **day 4** the
animal received **amphetamine**. Because every animal is its own control across days, day2&rarr;day3 gives a built-in *no-drug* yardstick to compare the day3&rarr;day4 drug effect against.

We work from the **precomputed map** (embeddings + behavior labels), so this notebook is all analysis and runs in a couple of minutes.
"""))
cells.append(md(carry_from_core()))

# ---------------------------------------------------------------- setup
cells.append(md(r"""
# 1.&nbsp; Setup

The usual opening cell &mdash; clone motionmapperpy, install the few packages Colab lacks, import what
we need.
"""))
cells.append(code(setup_code(
    imports="import numpy as np, matplotlib.pyplot as plt, urllib.request\n"
            "from mpl_toolkits.mplot3d import Axes3D  # noqa\n"
            "from scipy.stats import wilcoxon\n"
            "import motionmapperpy as mmpy\n"
            "%matplotlib inline")))

# ---------------------------------------------------------------- data
cells.append(md("# 2.&nbsp; Get the rat data"))
cells.append(md(r"""
Two small files ship in this repo and download into your `/content/` folder: a short clip of **raw 3-D
keypoints** (one example session, just so we can see what the data modality looks like) and the
**precomputed amphetamine maps** (6 rats &times; 3 days). If a download ever hiccups, re-run this cell.
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
fly's joint angles &mdash; but everything after that is nearly identical. Two rat-specific choices
shaped the map you're about to use. First, each animal's keypoints are scaled by its own body size
(the 97.5th percentile of the snout-to-tailbase distance), so a large rat and a small rat doing the same
thing land in the same spot rather than two. Second, the pose is made egocentric &mdash; which throws
away height off the floor &mdash; so height, and each joint's speed, is *added back*, precisely so the
map can tell a rear from a crouch. Here are a few frames of the tracked skeleton, and the behavioral map
this example session lives on (peaks = common behaviors).
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
_, xx, d0 = mmpy.findPointDensity(ze, 1.0, 501, [-R0, R0])
fig, ax = plt.subplots(figsize=(5, 5))
ax.imshow(d0, extent=(-R0, R0, -R0, R0), origin="lower", cmap=mmpy.gencmap())
ax.contour(xx, xx, d0, levels=[1e-6], colors="k", linewidths=1)
ax.set_title("one rat session as a behavioral map"); ax.axis("off"); plt.show()
"""))

# ---------------------------------------------------------------- experiment
cells.append(md("# 3.&nbsp; The amphetamine experiment"))
cells.append(md(r"""
Now the full dataset: 6 rats, each on 3 days. The key move is that all 18 (animal, day) recordings were
embedded into **one shared map**. Why make a shared map? Because a shared map makes the densities directly
comparable &mdash; the same spot on the map means the same behavior for every animal and every day, so a
change is a real change and not just a different map drawn differently. Here's the shared map, then one
rat across its three days. Watch day 4 (amphetamine): where does it start spending its time?
"""))
cells.append(code(r"""
animal = 0   # 🔧 which animal (0..5)

allz = emb.reshape(-1, 2); R = np.abs(allz).max() + 10; ext = (-R, R, -R, R)
def density(z, sigma=1.5):       # sigma = how much we smooth the map; larger -> coarser, smaller -> finer
    return mmpy.findPointDensity(z, sigma, 501, [-R, R])[2]
D_all = density(allz); inside = D_all > D_all.max() * 1e-3


fig, ax = plt.subplots(1, 4, figsize=(18, 12))
ax[0].imshow(D_all, extent=ext, origin="lower", cmap=mmpy.gencmap())
ax[0].contour(xx, xx, D_all, levels=[1e-6], colors="k", linewidths=1)
ax[0].set_title("shared map (all rats, all days)"); ax[0].axis("off")
for k, dd in enumerate(days):
    ax[k + 1].imshow(density(emb[k, animal]), extent=ext, origin="lower", cmap=mmpy.gencmap())
    ax[k + 1].set_title("rat %d, day %d%s" % (animal + 1, dd, "  (AMPH)" if dd == AMPH_DAY else ""))
    ax[k + 1].contour(xx, xx, D_all, levels=[1e-5], colors="k", linewidths=1)
    ax[k + 1].axis("off")
plt.show()
"""))
cells.append(md(r"""
🔧 Change `animal` to a different rat (0&ndash;5) and rerun. Does amphetamine push every animal toward the
*same* corner of the map, or does each rat have its own response? Hold that thought &mdash; the next two
sections turn this impression into a number.
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
    diff = Da - Db; diff[~inside] = 0; v = np.abs(diff).max()*.5
    im = a_.imshow(diff, extent=ext, origin="lower", cmap="RdBu_r", vmin=-v, vmax=v)
    a_.contour(xx, xx, D_all, levels=[1e-5], colors="k", linewidths=1)
    a_.set_title(t); a_.axis("off")
plt.colorbar(im, ax=ax, fraction=0.025); plt.suptitle("red = more time after; blue = less"); plt.show()
"""))

# ---------------------------------------------------------------- quantify
cells.append(md("# 5.&nbsp; Quantify it: how much did the repertoire change?"))
cells.append(md(r"""
Summarize each (animal, day) by its **occupancy fingerprint** &mdash; the fraction of time spent in
each of the 9 coarse behavior classes &mdash; and measure the change between days with the **Jensen-Shannon (J-S) divergence**. The J-S divergence is an information theoretic measure of how similar two probability distributions are.  Technically, it's the mutual information (don't worry if you don't know what that is!) between draws randomly taken out of two distribution and the identity of the distribution the draw was taken from.  Thus, it is a measure of **distinguishability**, with 0 meaning that the distributions are functionally equivalent and 1 meaning that there is no overlap. For each rat we compare the no-drug change (day2&rarr;3) against the amphetamine change (day3&rarr;4).
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
#ax.bar([0, 1], [js_baseline.mean(), js_amph.mean()], color=["royalblue", "firebrick"], alpha=.85)
ax.plot([np.zeros(nrat), np.ones(nrat)], [js_baseline, js_amph], "-", color="0.4", lw=1)
ax.plot(np.zeros(nrat), js_baseline, "ko"); ax.plot(np.ones(nrat), js_amph, "ko")
#ax.set_xticks([0, 1]); ax.set_xticklabels(["baseline\n(day2->3)", "amphetamine\n(day3->4)"])
ax.set_ylabel("JS divergence (bits)")
ax.set_title("amphetamine changes the repertoire more than\nday-to-day drift  (%d/%d rats, p=%.3f)"
             % (int((js_amph > js_baseline).sum()), nrat, pval)); plt.show()
"""))
cells.append(md(r"""
Every rat moves up: the amphetamine reshapes the behavioral repertoire **more than the animal's own day-to-day variation**, and the paired (Wilcoxon signed-rank) test says that's unlikely to be chance.
The fact that *every* rat moves up is exactly what makes a paired test convincing. This is the unsupervised version of a drug-effect readout, and no behavior was ever hand-defined, or even really defined in terms of language. We just compared repertoires.  Now, though, we do want to find out what behaviors are upregulated.  Although, given that we're giving them speed, we might have a good guess.
"""))

# ---------------------------------------------------------------- which behaviors
cells.append(md("# 6.&nbsp; Which behaviors does amphetamine drive?"))
cells.append(md(r"""
Break the change down by coarse behavior class: which does the drug push *up*, and which *down*? These
nine coarse classes are a grouping of the 163 fine clusters the watershed actually found, and the names
&mdash; idle, sniff, groom, rear, ... &mdash; came from people: two annotators independently watched
skeleton clips from each cluster and described them, reconciling where they disagreed. Coarse is easier
to read and to do statistics on; the finer view sees structure *within* a behavior, which you can try
in &sect;7.
"""))
cells.append(code(r"""
occ_base = np.array([occupancy(cclust[1, a]) for a in range(nrat)]).mean(0)   # day3 baseline
occ_amph = np.array([occupancy(cclust[2, a]) for a in range(nrat)]).mean(0)   # day4 amphetamine
fold = np.log2((occ_amph + 1e-3) / (occ_base + 1e-3))

behavior_names = ["idle", "sniffing", "grooming", "scrunching", "active crouching", "rearing", "exploring","locomotion","fast"]

fig, ax = plt.subplots(figsize=(8, 4))
ax.bar(range(1, n_coarse + 1), fold, color=["firebrick" if f > 0 else "royalblue" for f in fold])
ax.axhline(0, color="k", lw=.5)
ax.set_xticks(range(1, n_coarse + 1))
ax.set_xticklabels([n.replace(" ", "\n") for n in behavior_names], rotation=90)
ax.set_xlabel("coarse behavior class"); ax.set_ylabel("log2 fold change (AMPH / baseline)")
ax.set_title("amphetamine up-regulates some behaviors, suppresses others")
plt.tight_layout(); plt.show()
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
