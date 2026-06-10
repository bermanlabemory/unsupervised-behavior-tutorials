"""04 — Optogenetics (Cande et al. 2018). Stimulus-triggered behavior on a map.
Python port of the Fly_Optogenetic_Analysis workflow."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nb_builder import md, code, badge, write_nb

REPO = "bermanlabemory/unsupervised-behavior-tutorials/blob/main"
cells = []

cells.append(badge("%s/04_optogenetics.ipynb" % REPO))

cells.append(md(r"""
# Optogenetics &nbsp;·&nbsp; 🟡 medium

**The question:** you flash a light that activates a specific neuron in a freely moving fly.
**Which behavior does it trigger?** Instead of a human deciding in advance what to look for, we
ask the *whole behavioral map*: which regions does the fly visit *more* when the light is on,
compared to when it's off &mdash; and compared to control flies that lack the light-sensitive
channel? This is the unbiased screen from Cande et al., *eLife* 2018.

This is exactly the analysis you'd run for **your own** optogenetic or chemogenetic experiment in
any animal &mdash; the logic is identical for a mouse.

You don't need notebook 01. **Run time:** ~5 min.
"""))

# ---------------------------------------------------------------- setup
cells.append(md("# 1.&nbsp; Setup"))
cells.append(code(r"""
import os
if not os.path.exists("motionmapperpy"):
    !git clone -q https://github.com/bermanlabemory/motionmapperpy
# Import motionmapperpy straight from the clone -- avoids the "setup.py install" empty-namespace
# trap (no restart needed). moviepy<2 because the package + notebook use the moviepy 1.x "editor"
# API; FFMPEG_BINARY points moviepy at Colab's ffmpeg so it never tries the broken auto-download.
import shutil
os.environ["FFMPEG_BINARY"] = shutil.which("ffmpeg") or "/usr/bin/ffmpeg"
!pip install -q "moviepy<2"
import sys
sys.path.insert(0, os.path.abspath("motionmapperpy"))
for _m in [k for k in list(sys.modules) if k.startswith("motionmapperpy")]:
    del sys.modules[_m]
!pip install -q hdf5storage easydict 2>/dev/null

import numpy as np, matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu
import motionmapperpy as mmpy
%matplotlib inline
print("ready")
"""))

# ---------------------------------------------------------------- data
cells.append(md("# 2.&nbsp; Load the flies, their map positions, and the light"))
cells.append(md(r"""
For each fly we need: its trajectory **in behavior space** (a 2-D point per frame, from a map like
the one you built in notebook 01) and a **light on/off** time series. Half the flies are
**experimental** (light activates the neuron); half are **controls** (genetically identical but
the light does nothing &mdash; the crucial comparison).

> **Instructors:** host a few strains of `*_projections_tsne_embedding.mat` + `*_Frames.txt` from
> the Cande example data and set `USE_SYNTHETIC_DATA = False`. The stand-in below has a neuron
> whose activation drives one particular behavior.
"""))
cells.append(code(r"""
USE_SYNTHETIC_DATA = True
DATA_URL = "https://PLACEHOLDER-HOST/cande_example_data.zip"   # TODO: real link

def make_synthetic(n_exp=8, n_ctrl=8, fps=30, n_trials=20, on_s=5, off_s=10,
                   triggered_behavior=4, seed=0):
    # 6 behavior 'blobs' in a 2-D space. In experimental flies, light activation biases the
    # fly toward blob #4 ('the triggered behavior'); controls are unaffected.
    rng = np.random.default_rng(seed)
    blobs = np.array([[-30, 20], [0, 30], [28, 18], [30, -15], [0, -32], [-30, -12]], float)
    trial = (on_s + off_s) * fps
    flies, leds, is_ctrl = [], [], []
    for f in range(n_exp + n_ctrl):
        control = f >= n_exp
        led = np.tile(np.r_[np.ones(on_s * fps), np.zeros(off_s * fps)], n_trials).astype(bool)
        base = np.ones(len(blobs)) / len(blobs)
        z = np.zeros((len(led), 2))
        b = rng.integers(len(blobs))
        for t in range(len(led)):
            w = base.copy()
            if led[t] and not control:
                w[triggered_behavior] += 1.5            # light drives this behavior
            if rng.random() < 0.06:                      # occasionally switch behavior
                b = rng.choice(len(blobs), p=w / w.sum())
            z[t] = blobs[b] + rng.normal(scale=3.5, size=2)
        flies.append(z); leds.append(led); is_ctrl.append(control)
    return flies, leds, np.array(is_ctrl), blobs, triggered_behavior

flies, leds, is_ctrl, blobs, true_trig = make_synthetic()
fps = 30
print("%d experimental + %d control flies" % ((~is_ctrl).sum(), is_ctrl.sum()))
"""))

# ---------------------------------------------------------------- map
cells.append(md("# 3.&nbsp; The shared behavior space"))
cells.append(md("All flies live on one map. Here's the overall behavioral density (peaks = common behaviors):"))
cells.append(code(r"""
allz = np.concatenate(flies)
R = np.abs(allz).max() + 5
NP = 101
_, xx, density = mmpy.findPointDensity(allz, 2.0, NP, [-R, R])
inside = density > density.max() * 0.02          # the occupied part of the space

fig, ax = plt.subplots(figsize=(5.5, 5))
ax.imshow(density, extent=(-R, R, -R, R), origin="lower", cmap=mmpy.gencmap())
ax.set_title("behavioral density (all flies)"); ax.axis("off"); plt.show()

def occupancy_map(z, sigma=2.0):
    _, _, d = mmpy.findPointDensity(z, sigma, NP, [-R, R])
    return d
"""))

# ---------------------------------------------------------------- difference
cells.append(md("# 4.&nbsp; Light ON vs OFF: the difference map"))
cells.append(md(r"""
For each fly, compute where it spends time when the **light is on** vs **off**, and subtract.
A neuron that drives a behavior should make its region light up (positive) in experimental flies
but **not** in controls.
"""))
cells.append(code(r"""
diffs = np.array([occupancy_map(z[led]) - occupancy_map(z[~led]) for z, led in zip(flies, leds)])
exp_diff, ctrl_diff = diffs[~is_ctrl], diffs[is_ctrl]

fig, ax = plt.subplots(1, 2, figsize=(11, 5))
v = np.abs([exp_diff.mean(0), ctrl_diff.mean(0)]).max()
for a, D, t in zip(ax, [exp_diff.mean(0), ctrl_diff.mean(0)], ["experimental", "control"]):
    im = a.imshow(D, extent=(-R, R, -R, R), origin="lower", cmap="RdBu_r", vmin=-v, vmax=v)
    a.set_title("mean (ON - OFF), %s" % t); a.axis("off")
plt.colorbar(im, ax=ax, fraction=0.025); plt.show()
"""))

# ---------------------------------------------------------------- significance
cells.append(md("# 5.&nbsp; Which regions are *significantly* driven?"))
cells.append(md(r"""
A blob in the experimental map isn't enough &mdash; it has to be bigger than in controls. We test,
**at each location in the map**, whether the ON−OFF change differs between experimental and control
flies (a rank-sum test), keep only the occupied locations, and **control the false-discovery rate**
(Benjamini-Hochberg) &mdash; the correction the paper uses. (Plain Bonferroni is hopelessly strict
here: with 8 vs 8 flies the smallest possible p-value already fights hundreds of tests.)
"""))
cells.append(code(r"""
pmap = np.ones((NP, NP))
ii, jj = np.where(inside)
for i, j in zip(ii, jj):
    e, c = exp_diff[:, i, j], ctrl_diff[:, i, j]
    if np.ptp(e) + np.ptp(c) > 0:
        pmap[i, j] = mannwhitneyu(e, c, alternative="two-sided").pvalue

def bh_fdr_mask(pvals, tested, q=0.05):          # Benjamini-Hochberg over the tested locations
    p = pvals[tested]; ranked = np.sort(p)
    thresh = q * (np.arange(1, len(p) + 1) / len(p))
    passed = np.where(ranked <= thresh)[0]
    crit = ranked[passed.max()] if len(passed) else -1.0
    mask = np.zeros_like(pvals, dtype=bool); mask[tested] = pvals[tested] <= crit
    return mask

sig = bh_fdr_mask(pmap, inside, q=0.05)
difference_map = np.where(sig, exp_diff.mean(0) - ctrl_diff.mean(0), 0.0)

fig, ax = plt.subplots(figsize=(5.5, 5))
v = np.abs(difference_map).max() or 1
ax.contour(np.linspace(-R, R, NP), np.linspace(-R, R, NP), inside, [0.5], colors="0.6", linewidths=0.7)
im = ax.imshow(difference_map, extent=(-R, R, -R, R), origin="lower", cmap="RdBu_r", vmin=-v, vmax=v)
ax.set_title("significant light-driven behavior\n(red = up-regulated by activation)")
ax.axis("off"); plt.colorbar(im, fraction=0.046); plt.show()
print("up-regulated region near true triggered behavior at", blobs[true_trig])
"""))
cells.append(md(r"""
The pipeline found the triggered behavior with **no human labels** &mdash; just "where does the
light reliably push the fly, beyond what we see in controls." That red region is what you'd then
make exemplar movies of (as in notebook 01 §9) to *name* the behavior.
"""))

# ---------------------------------------------------------------- PSTH
cells.append(md("# 6.&nbsp; Time course: behavior locked to the light"))
cells.append(md("How fast does the behavior come on after the light? Average occupancy of the driven region around light onset:"))
cells.append(code(r"""
# "driven region" = significantly up-regulated pixels
driven = sig & (difference_map > 0)
def in_driven(z):
    xi = np.clip(((z[:, 0] + R) / (2 * R) * (NP - 1)).astype(int), 0, NP - 1)
    yi = np.clip(((z[:, 1] + R) / (2 * R) * (NP - 1)).astype(int), 0, NP - 1)
    return driven[yi, xi].astype(float)

win = 3 * fps
def psth(group):
    out = []
    for z, led in zip([flies[k] for k in group], [leds[k] for k in group]):
        occ, onsets = in_driven(z), np.where(np.diff(led.astype(int)) == 1)[0]
        trials = [occ[o - win:o + 2 * win] for o in onsets if o - win >= 0 and o + 2 * win < len(occ)]
        out.append(np.mean(trials, 0))
    return np.array(out)

t = (np.arange(-win, 2 * win)) / fps
fig, ax = plt.subplots(figsize=(8, 4))
for grp, lab, c in [(np.where(~is_ctrl)[0], "experimental", "firebrick"),
                    (np.where(is_ctrl)[0], "control", "grey")]:
    P = psth(grp); ax.plot(t, P.mean(0), color=c, label=lab)
    ax.fill_between(t, P.mean(0) - P.std(0), P.mean(0) + P.std(0), color=c, alpha=0.2)
ax.axvspan(0, 5, color="red", alpha=0.1, label="light ON")
ax.set_xlabel("time from light onset (s)"); ax.set_ylabel("P(in driven behavior)")
ax.legend(); plt.show()
"""))

# ---------------------------------------------------------------- MI
cells.append(md("# 7.&nbsp; How much does the light *tell you* about behavior?"))
cells.append(md(r"""
One number to summarize the effect: the **mutual information** between the light state and whether
the fly is in the driven behavior. Zero = the light tells you nothing; larger = the light strongly
predicts behavior. (Cande et al. use this to rank descending neurons by how behaviorally potent
they are.)
"""))
cells.append(code(r"""
def mutual_information(led, occ):
    led = led.astype(int); occ = occ.astype(int)
    p = np.histogram2d(led, occ, bins=[2, 2])[0]; p = p / p.sum()
    px, py = p.sum(1, keepdims=True), p.sum(0, keepdims=True)
    m = p > 0
    return float(np.sum(p[m] * np.log2(p[m] / (px @ py)[m])))

for grp, lab in [(np.where(~is_ctrl)[0], "experimental"), (np.where(is_ctrl)[0], "control")]:
    mis = [mutual_information(leds[k], in_driven(flies[k])) for k in grp]
    print("%-13s  I(light; behavior) = %.4f bits" % (lab, np.mean(mis)))
"""))

# ---------------------------------------------------------------- exercises
cells.append(md(r"""
# 8.&nbsp; 🔧 Your turn

1. In `make_synthetic(...)`, change `triggered_behavior` to a different blob (0–5) and rerun. Does
   the pipeline track it to the new location?
2. Shrink the effect (`w[triggered_behavior] += 0.4`). At what point does it stop being
   significant? (This is your statistical power &mdash; very relevant for designing real screens.)
3. Narrow the analysis window to the first second after light onset. Do fast and slow behaviors
   separate?
4. **Bring your own:** this is the template for *your* opto/chemo experiment &mdash; see
   `06_bring_your_own_data.ipynb` to build the map, then drop your stimulus times in here.
"""))

write_nb(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                      "04_optogenetics.ipynb"), cells)
