"""06 — Optogenetics (Cande et al. 2018). Stimulus-triggered behavior on a map.
Python port of the Fly_Optogenetic_Analysis workflow."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nb_builder import md, code, badge, write_nb, setup_code, carry_from_core

REPO = "bermanlabemory/unsupervised-behavior-tutorials/blob/main"
cells = []

cells.append(badge("%s/06_optogenetics.ipynb" % REPO))

cells.append(md(r"""
# Optogenetics

**The question:** we flash a light that activates a specific neuron in a freely moving fly.
**Which behavior does it trigger?** Instead of a human deciding in advance what to look for, we ask
the *whole behavioral map*: which regions does the fly visit *more* when the light is on, compared to
when it's off &mdash; and compared to control flies that lack the light-sensitive channel? This is the
unbiased screen from Cande et al., *eLife* 2018.

It's exactly the analysis you'd run for **your own** optogenetic or chemogenetic experiment in any
animal &mdash; the logic is identical for a mouse.

This is a standalone track; you don't need notebook 01. **Run time:** ~5 min.
"""))
cells.append(md(carry_from_core()))

# ---------------------------------------------------------------- setup
cells.append(md(r"""
# 1.&nbsp; Setup

The usual opening cell &mdash; clone motionmapperpy, install the few packages Colab lacks, import what
we need.
"""))
cells.append(code(setup_code(
    imports="import numpy as np, matplotlib.pyplot as plt\n"
            "from scipy.stats import mannwhitneyu\n"
            "import motionmapperpy as mmpy\n"
            "%matplotlib inline")))

# ---------------------------------------------------------------- data
cells.append(md("# 2.&nbsp; Load the flies, their map positions, and the light"))
cells.append(md(r"""
For each fly we need its trajectory **in behavior space** (a 2-D point per frame, from a map like
the one in notebook 01) and a **light on/off** time series.

This loads a **real strain from Cande et al. 2018** (`ss02635`): one filming session of **12 flies**
recorded at once. The design has its control built in &mdash; **cameras 7&ndash;12 are
experimental** (fed all-*trans*-retinal, so the light-gated channel *Chrimson* is functional) and
**cameras 1&ndash;6 are controls** (no retinal, so the identical light does nothing). The light
pulses in ~15&nbsp;s cycles; the embeddings were computed exactly as in notebook 01 and
down-sampled to 25&nbsp;Hz to keep the download small (~4&nbsp;MB).

Set `USE_SYNTHETIC_DATA = True` (or if the download fails) to use the stand-in generator instead,
which has a neuron whose activation drives one particular behavior.
"""))
cells.append(code(r"""
USE_SYNTHETIC_DATA = False           # set True to force the stand-in generator below
STRAIN = "ss02635"                   # a Cande 2018 driver line (see the full menu in section 9)
OUTFILE = STRAIN + ".npz"
DATA_URL = ("https://raw.githubusercontent.com/bermanlabemory/"
            "unsupervised-behavior-tutorials/main/data/optogenetic_data/" + OUTFILE)

def load_real():
    # one filming session, 12 flies: cams 1-6 = control (no retinal), 7-12 = experimental.
    if not os.path.exists(OUTFILE):
        !wget -q "$DATA_URL" -O "$OUTFILE"
    if not os.path.exists(OUTFILE) or os.path.getsize(OUTFILE) < 100000:
        raise RuntimeError("download failed -- check your connection, or set USE_SYNTHETIC_DATA = True above.")
    d = np.load(OUTFILE)
    bounds = np.cumsum(d["fly_lengths"])[:-1]
    flies = [z.astype(float) for z in np.split(d["z"], bounds)]   # per-fly (T x 2) map trajectory
    leds = [l.astype(bool) for l in np.split(d["led"], bounds)]   # per-fly light on/off
    return flies, leds, np.array(d["is_control"]), int(d["fps"])

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

blobs = true_trig = None             # only set by the synthetic generator (used in section 5)
if not USE_SYNTHETIC_DATA:
    try:
        flies, leds, is_ctrl, fps = load_real()
        print("REAL data (%s): %d experimental + %d control flies @ %d fps"
              % (STRAIN, (~is_ctrl).sum(), is_ctrl.sum(), fps))
    except Exception as e:
        print("Could not load real data (%s) -- falling back to synthetic." % e)
        USE_SYNTHETIC_DATA = True
if USE_SYNTHETIC_DATA:
    flies, leds, is_ctrl, blobs, true_trig = make_synthetic()
    fps = 30
    print("SYNTHETIC: %d experimental + %d control flies" % ((~is_ctrl).sum(), is_ctrl.sum()))
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

# ---------------------------------------------------------------- look first
cells.append(md(r"""
## 3.1&nbsp; Look first: where does each fly sit when the light is on?

Before averaging anything, let's just look &mdash; the same instinct as notebook 01. Each point below is
one frame of one fly, placed in the behavior space, colored by whether the light was on at that moment.
Here's the prediction to hold in mind: if activating this neuron drives a behavior, the **experimental**
flies should pile their light-ON (red) points into one region, while the **control** flies &mdash; same
light, no functional channel &mdash; should scatter red and grey together. Everything after this is just
a careful way of testing that by eye is not enough.
"""))
cells.append(code(r"""
exp_ids = np.where(~is_ctrl)[0][:2]
ctrl_ids = np.where(is_ctrl)[0][:2]
picks = [(i, "experimental") for i in exp_ids] + [(i, "control") for i in ctrl_ids]

fig, axes = plt.subplots(1, 4, figsize=(16, 4.2))
for ax, (k, lab) in zip(axes, picks):
    z, led = flies[k], leds[k]
    ax.imshow(density, extent=(-R, R, -R, R), origin="lower", cmap=mmpy.gencmap(), alpha=0.45)
    s = max(1, len(z) // 4000)               # thin dense trajectories so the colors stay readable
    ax.scatter(z[~led][::s, 0], z[~led][::s, 1], s=3, color="0.45", label="light OFF")
    ax.scatter(z[led][::s, 0], z[led][::s, 1], s=3, color="firebrick", label="light ON")
    ax.set_xlim(-R, R); ax.set_ylim(-R, R); ax.set_aspect("equal"); ax.axis("off")
    ax.set_title("%s  (fly %d)" % (lab, k))
axes[0].legend(loc="upper left", markerscale=3, fontsize=8, framealpha=0.9)
plt.suptitle("each fly in behavior space — red = light ON, grey = light OFF"); plt.show()
"""))

# ---------------------------------------------------------------- difference
cells.append(md("# 4.&nbsp; Light ON vs OFF: the difference map"))
cells.append(md(r"""
For each fly, compute where it spends time when the **light is on** vs **off**, and subtract. A neuron
that drives a behavior should make its region light up (positive) in experimental flies but **not** in
controls. Before you run it, make a prediction: where on the map do you expect the experimental panel to
light up &mdash; and what exactly does the control panel let us rule out?
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
**at each location in the map**, whether the ON&minus;OFF change differs between experimental and control
flies (a rank-sum test), keep only the occupied locations, and **control the false-discovery rate**
(Benjamini-Hochberg) &mdash; the correction the paper uses. Controlling the false-discovery rate just
means we pick a p-value threshold such that we expect only ~5% of the locations we *call* significant to
be false alarms. That's looser, on purpose, than demanding zero false alarms anywhere (plain Bonferroni),
which here is hopelessly strict: with 8 vs 8 flies the smallest possible p-value is already fighting
hundreds of simultaneous tests, so Bonferroni would reject everything. *(The per-location tests below
take ~5&ndash;10 s.)*
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
if true_trig is not None:            # synthetic: we know the ground-truth triggered behavior
    print("up-regulated region near true triggered behavior at", blobs[true_trig])
print("%d significant map locations (BH-FDR, q=0.05)" % int(sig.sum()))
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

def on_duration(leds, fps):          # median light-ON length (s), measured from the data
    durs = []
    for led in leds:
        e = np.diff(np.r_[0, led.astype(int), 0])
        durs += list((np.where(e == -1)[0] - np.where(e == 1)[0]) / fps)
    return float(np.median(durs)) if durs else 1.0

on_s = on_duration(leds, fps)
pre, post = int(round(on_s * fps)), int(round(2 * on_s * fps))   # show one ON before, two after

def psth(group):
    out = []
    for z, led in zip([flies[k] for k in group], [leds[k] for k in group]):
        occ, onsets = in_driven(z), np.where(np.diff(led.astype(int)) == 1)[0]
        trials = [occ[o - pre:o + post] for o in onsets if o - pre >= 0 and o + post <= len(occ)]
        if trials:
            out.append(np.mean(trials, 0))
    return np.array(out)

t = np.arange(-pre, post) / fps
fig, ax = plt.subplots(figsize=(8, 4))
for grp, lab, c in [(np.where(~is_ctrl)[0], "experimental", "firebrick"),
                    (np.where(is_ctrl)[0], "control", "grey")]:
    P = psth(grp); ax.plot(t, P.mean(0), color=c, label=lab)
    ax.fill_between(t, P.mean(0) - P.std(0), P.mean(0) + P.std(0), color=c, alpha=0.2)
ax.axvspan(0, on_s, color="red", alpha=0.1, label="light ON")
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

1. **Swap the driver line.** Set `STRAIN` in &sect;2 to any line from the menu in &sect;9 below and
   rerun the whole notebook. A *different* neuron drives a *different* behavior &mdash; where does
   this one land on the map, and is it up- or down-regulated?
2. **Spend less data.** Keep only the first 3 experimental + 3 control flies. Does the driven region
   survive the FDR correction with half the flies? (That's your statistical power &mdash; directly
   relevant to designing a real screen.)
3. **Tighten the window.** Count only the first second after each light onset as "ON" (instead of
   the whole pulse) and rerun &sect;4&ndash;5. Do fast and slow behaviors separate?
4. **Know the ground truth.** Set `USE_SYNTHETIC_DATA = True` for a generator where *you* set the
   answer &mdash; move `triggered_behavior`, shrink the effect (`w[triggered_behavior] += 0.4`), and
   watch significance appear and vanish.
5. **Bring your own:** this is the template for *your* opto/chemo experiment &mdash; see
   `07_bring_your_own_data.ipynb` to build the map, then drop your stimulus times in here.
"""))

# ---------------------------------------------------------------- strain menu
cells.append(md("# 9.&nbsp; The full menu of driver lines"))
cells.append(md(r"""
**Seven** real driver lines from Cande et al. 2018 ship with this repo &mdash; each one a different
descending neuron, each its own 12-fly session (cameras 1&ndash;6 control, 7&ndash;12 experimental).
Set `STRAIN` in &sect;2 to any of these and rerun:

| `STRAIN` | what this simple ON&ndash;vs&ndash;OFF screen finds |
|---|---|
| `"ss02635"` | **the default** &mdash; the strongest, cleanest hit (a large up-regulated region) |
| `"ss02617_0226"` | another clear hit |
| `"ss01049"` | a clear hit elsewhere on the map, with both up- **and** down-regulation |
| `"ss01540"` | a subtler hit (mostly *down*-regulation &mdash; a behavior the light suppresses) |
| `"ss01597_1v_1022"` | no region survives the FDR test here |
| `"ss01602"` | no region survives the FDR test here |
| `"ss02393_1v_1009"` | no region survives the FDR test here |

Those last three are a deliberate reality check. Not every neuron yields a hit &mdash; and our
**simplified** screen (whole-stimulus ON vs whole-stimulus OFF) deliberately trades sensitivity for
clarity, so it can miss a *transient* response that the full windowed analysis in the paper (a tight
post-onset window vs a matched OFF window) would catch. Finding nothing is a real, informative
outcome: it's what most of a genome-scale screen looks like.

> 🔧 Loop over all seven (`for STRAIN in [...]:`) and print the number of significant locations for
> each &mdash; you've just built a miniature behavioral screen.
"""))

write_nb(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                      "06_optogenetics.ipynb"), cells)
