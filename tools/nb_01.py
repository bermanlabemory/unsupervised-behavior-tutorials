"""01 — Core: build a fly behavioral map (Act 1, everyone). Adapted from the
motionmapperpy fly demo. This is the engine every Act-2 track builds on."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nb_builder import md, code, badge, write_nb, setup_code

REPO = "bermanlabemory/unsupervised-behavior-tutorials/blob/main"
cells = []

cells.append(badge("%s/01_build_a_behavioral_map.ipynb" % REPO))

cells.append(md(r"""
# Building a Behavioral Map

Welcome! By the end of this notebook, you'll have taken raw, tracked
body-part positions and turned them into a **behavioral map**: a 2-D atlas of (nearly) everything a fly does during an experiment.

Earlier in the course, you saw **supervised** behavior classification &mdash; you hand a classifier labelled examples and it learns to find more of them. This is the **unsupervised** complement: we let the structure of the movement itself carve up the repertoire, and only afterwards go and ask what we found. 

Almost all unsupervised methods follow the same formula:

> **postures &rarr; postural dynamics &rarr; behavioral description**

The idea is the same whether the animal is a worm, a mouse, or a rat &mdash;  (although we need to think carefully about how we represent all of these things). For the method we'll talk about today, this formula will specifically use

> **postures &rarr; postural *dynamics* (wavelets) &rarr; a 2-D behavioral map &rarr; discrete behaviors (watershed transform)**

We'll do it on a fly first (although notebooks 3 and 4 run a very similar
 pipeline on 3-D tracked data from rats). 

A GPU runtime is nice but not required for this notebook (`Runtime → Change runtime type → GPU`).

| How to read the cells | |
|---|---|
| &#9654;&#65038; **Just run it** | most cells &mdash; press Shift+Enter and watch |
| 🔧 **Your turn** | optional: change one thing and see what happens |
"""))

# ---------------------------------------------------------------- setup
cells.append(md(r"""
# 1.&nbsp; Get the code and data

We start by downloading **motionmapperpy** (we'll often call it **mmpy**) from GitHub. It includes a small example dataset: two short movies of a single fly walking in a shallow dish, with **32 body
parts tracked** in every frame (tracked here with [LEAP](https://www.nature.com/articles/s41592-018-0234-5),
but SLEAP or DeepLabCut output looks similar). After this cell runs you'll see a `motionmapperpy`
folder appear in the file browser on the left &mdash; our working directory is `/content/`, in case
you ever feel lost.
"""))
cells.append(code(setup_code(ready="engine ready")))
cells.append(md(r"""
> **No restart needed.** This cell imports mmpy straight from the cloned folder. If the import ever
> fails, just re-run it &mdash; avoid *Restart session*, which would undo the `sys.path` line it adds.
"""))

# ---------------------------------------------------------------- imports
cells.append(md(r"""
# 2.&nbsp; Imports

A quick tour of the toolbox. You don't need to memorize any of this &mdash; it's here so that when a
name shows up later, you know who it belongs to.
"""))
cells.append(code(r"""
import glob, os, pickle, copy, time          # standard library: files, saving variables, timing
import numpy as np                            # arrays -- the workhorse for everything numeric
import pandas as pd                           # tidy tables; the tracking files load as one
import hdf5storage                            # reads/writes the MATLAB-style .mat files mmpy uses
import cv2                                     # reads video frames; ships with Colab, no ffmpeg hassle
import matplotlib.pyplot as plt               # plotting
from matplotlib.animation import FuncAnimation  # to animate frames
from IPython.display import HTML               # to show those animations inline
from scipy.ndimage import median_filter       # a gentle de-noiser for jittery tracking
from sklearn.decomposition import PCA         # to compress correlated features
from matplotlib import rc; rc("animation", html="jshtml")   # render animations as in-browser players

import motionmapperpy as mmpy
from motionmapperpy import demoutils
%matplotlib inline


def read_frames(path, start, n):
    # Read n consecutive RGB video frames starting at frame index `start`.
    cap = cv2.VideoCapture(path); cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    out = []
    for _ in range(n):
        ok, fr = cap.read()
        if not ok:
            break
        out.append(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))
    cap.release()
    return out

print("imports ok")
"""))

# ---------------------------------------------------------------- look first
cells.append(md(r"""
# 3.&nbsp; Look at the data *first*

Rule #1 of behavioral analysis: **never trust tracking you haven't watched.** Before any modelling,
let's load the movies and the tracked positions and just look.
"""))
cells.append(code(r"""
datasetnames = ["fly_leap_test", "fly_leap_test_2"]
moviepaths = ["motionmapperpy/data/fly/%s.mp4" % d for d in datasetnames]

# The tracking ships as a pandas table: x, y, and a confidence ("likelihood") value per body part.
h5s_pandas = [pd.read_hdf("motionmapperpy/data/fly/%s_positions.h5" % d) for d in datasetnames]
h5s_pandas[0].head()
"""))
cells.append(md(r"""
LEAP tracked **32 points** on the fly &mdash; yet the table has **96 columns**. Why? (Have a look at
the column headers above before reading on.) Each point comes with three numbers: its `x`, its `y`,
and a *likelihood* the tracker assigns to it. We only want the geometry, so we'll drop every third
(likelihood) column, leaving `x, y` per point, and reshape to `(frames, 32 points, 2)`.
"""))
cells.append(code(r"""
keep = np.mod(np.arange(1, h5s_pandas[0].shape[1] + 1), 3) != 0     # drop every 3rd (likelihood) column
h5s = [h.values[:, keep].reshape((-1, 32, 2)) for h in h5s_pandas]
for d, h in zip(datasetnames, h5s):
    print("%s: %d frames, %d body parts" % (d, h.shape[0], h.shape[1]))
"""))
cells.append(md(r"""
Tracking always has the odd glitch &mdash; a body part that leaps to a wrong spot for a frame or two and
snaps back. A light **median filter** wipes those out without blurring real movement. Below is the
vertical position of one leg tip before and after &mdash; watch the filter erase the spikes while
leaving the real motion intact:
"""))
cells.append(code(r"""
fps = 100                                                 # these movies were filmed at 100 fps
seg = slice(1400, 1800)                                   # a stretch with obvious leg-tracking glitches
raw = h5s[0][seg, 13, 1].copy()                           # y of one leg-tip keypoint (~4 s)
h5s = [median_filter(h, size=(5, 1, 1)) for h in h5s]     # 5-frame median, per coordinate
clean = h5s[0][seg, 13, 1]
t = np.arange(seg.stop - seg.start) / fps                 # seconds

fig, ax = plt.subplots(figsize=(13, 3))
ax.plot(t, raw, color="0.7", lw=2, label="raw")
ax.plot(t, clean, color="firebrick", lw=1, label="median-filtered")
ax.set_xlabel("time (s)"); ax.set_ylabel("y position (px)"); ax.legend(); plt.show()
"""))
cells.append(md("Now overlay the tracked skeleton on the video and watch the fly move (this takes ~1 min to render):"))
cells.append(code(r"""
# 'connections' just says which body parts to join with lines, so the fly looks like a fly.
connections = [np.arange(6, 10), np.arange(10, 14), np.arange(14, 18), np.arange(18, 22),
               np.arange(22, 26), np.arange(26, 30), [2, 0, 1], [0, 3, 4, 5], [31, 3, 30]]
h5ind, tstart, nframes = 0, 7000, 120
frames = read_frames(moviepaths[h5ind], tstart, nframes)
fig, ax = plt.subplots(figsize=(6, 6))

def update(i):
    ax.clear()
    ax.imshow(frames[i], origin="lower")
    for c in connections:
        ax.plot(h5s[h5ind][tstart + i, c, 0], h5s[h5ind][tstart + i, c, 1], "-", color="firebrick", lw=1)
    ax.axis("off"); ax.set_aspect("equal")

anim = FuncAnimation(fig, update, frames=len(frames), interval=50)
plt.close()
HTML(anim.to_jshtml())
"""))

# ---------------------------------------------------------------- representation
cells.append(md(r"""
# 4.&nbsp; Choose a representation

The single most consequential decision in this whole pipeline is **what numbers we use to describe a
posture**. Raw pixel positions are a poor choice: they change whenever the *whole fly* moves or turns,
even if its pose is identical. We want a representation that is **egocentric** &mdash; one that only
changes when the animal changes its *shape*.

For a fly, a natural egocentric description is a set of **joint angles** (leg joints, wing angles,
the neck). Below we compute 22 angles from the 32 tracked points. *(For a mouse or rat we'd instead
center and rotate the keypoints; for a worm we'd use "eigenworms." Same idea, different front-end
&mdash; you'll see this in the other notebooks.)*
"""))
cells.append(code(r"""
# Each entry [a, b, c, flag] is the angle a-b-c; flag picks the range (0: [-pi, pi], 1: [0, 2pi]).
angleinds = [[1,0,2,0],[0,3,4,0],[3,4,5,0],[31,3,30,1],[6,7,8,0],[7,8,9,0],[10,11,12,0],
             [11,12,13,0],[14,15,16,0],[15,16,17,0],[26,27,28,0],[27,28,29,0],[22,23,24,0],
             [23,24,25,0],[18,19,20,0],[19,20,21,0],[3,4,18,0],[3,4,22,0],[3,4,26,0],[3,4,6,0],
             [3,4,10,0],[3,4,14,0]]

def angle_between(v1, v2, small=1):
    a = np.arctan2(v1[:, 1], v1[:, 0]) - np.arctan2(v2[:, 1], v2[:, 0])
    out = np.rad2deg(a % (2 * np.pi))
    if small:
        out[out > 180] -= 360
    return out

angleh5s = []
for h in h5s:
    A = np.zeros((h.shape[0], len(angleinds)))
    for i, (a, b, c, fl) in enumerate(angleinds):
        A[:, i] = angle_between(h[:, a] - h[:, b], h[:, c] - h[:, b], fl)
    angleh5s.append(A)
print("each frame is now %d angles, instead of %d x,y positions"
      % (angleh5s[0].shape[1], h5s[0].shape[1] * 2))
"""))
cells.append(md(r"""
Some angles swing wildly and others barely move, so we rescale each one to roughly [0, 1]. That way
no single joint gets to shout over the rest just because it happens to be measured in bigger numbers.
"""))
cells.append(code(r"""
lo = np.min([A.min(0) for A in angleh5s], 0)
angleh5s = [A - lo for A in angleh5s]
hi = np.max([A.max(0) for A in angleh5s], 0)
angleh5s = [A / hi for A in angleh5s]

t = np.arange(1000) / fps
fig, ax = plt.subplots(figsize=(14, 4))
ax.plot(t, angleh5s[0][1000:2000, [3, 9, 13]])
ax.set_xlabel("time (s)"); ax.set_ylabel("normalized angle")
ax.legend(["wingtip angle", "hindleg tibia angle", "midleg femur-tibia angle"]); plt.show()
"""))
cells.append(md(r"""
**Compress with PCA.** Those 22 angles are far from independent &mdash; legs move together, wings move
together &mdash; so we rotate into a smaller set of **principal components** that keep 95% of the
variance. Fewer dimensions means faster computation and less noise downstream.
"""))
cells.append(code(r"""
X = np.concatenate(angleh5s, axis=0)
p = PCA().fit(X)
n_pca = int(np.argmax(np.cumsum(p.explained_variance_ratio_) > 0.95)) + 1

fig, ax = plt.subplots(figsize=(8, 3))
ax.plot(np.arange(1, X.shape[1] + 1), np.cumsum(p.explained_variance_ratio_), "firebrick", marker=".")
ax.axvline(n_pca, ls="--", color="royalblue"); ax.axhline(0.95, ls=":", color="grey")
ax.set_xlabel("# PCA components"); ax.set_ylabel("cumulative variance explained"); plt.show()

y = p.transform(X)[:, :n_pca]
projs_list = np.split(y, np.cumsum([A.shape[0] for A in angleh5s])[:-1])
print("keeping %d PCA modes (95%% of the variance) -- does it surprise you it's so few?" % n_pca)
"""))
cells.append(md(r"""
It's worth a peek at *what* those leading components are. Each bar below shows how much each original
angle contributes to a principal component &mdash; i.e. the coordinated patterns of joints the fly
actually uses. PCA didn't invent these; it found them in the fly's movement.
"""))
cells.append(code(r"""
fig, ax = plt.subplots(figsize=(13, 3))
show = min(n_pca, 4)
width = 0.8 / show
for i in range(show):
    ax.bar(np.arange(X.shape[1]) + i * width, p.components_[i], width=width, label="PC%d" % (i + 1))
ax.set_xlabel("angle # (the 22 joint angles)"); ax.set_ylabel("contribution to PC")
ax.legend(ncol=show); ax.set_title("what the leading principal components are made of"); plt.show()
"""))

# ---------------------------------------------------------------- project + params
cells.append(md(r"""
# 5.&nbsp; Set up a motionmapperpy project

mmpy keeps everything for one map in a **project folder**: it's how you stay organized once you have
many recordings, and it lets the pipeline load datasets one at a time instead of holding them all in
memory at once. We create one and drop our low-dimensional time series into its `Projections/`
subfolder (one file per movie, named `*_pcaModes.mat` &mdash; mmpy recognizes that suffix).
"""))
cells.append(code(r"""
projectPath = "Fly_mmpy"
mmpy.createProjectDirectory(projectPath)
for d, projs in zip(datasetnames, projs_list):
    hdf5storage.savemat("%s/Projections/%s_pcaModes.mat" % (projectPath, d),
                        {"projections": projs})
print("saved %d projection files into %s/Projections/" % (len(projs_list), projectPath))
"""))
cells.append(md(r"""
Now the **parameters**. There are only a handful of real choices in this pipeline, and they fall into
two groups: the ones tied to your camera and your question (change these for every new dataset), and
the machine knobs you can usually leave alone. We've grouped and commented them so the cell reads like
a checklist rather than a wall of settings.
"""))
cells.append(code(r"""
parameters = mmpy.setRunParameters()        # sensible defaults for everything, which we now adjust

# ---- choices tied to YOUR camera and question (revisit these for every new dataset) ----
parameters.projectPath  = projectPath
parameters.method       = "UMAP"     # "UMAP" (fast -- our default) or "TSNE" (the classic; slower)
parameters.samplingFreq = 100        # frames per second of the movies (these flies were filmed at 100)
parameters.minF         = 1          # slowest movement frequency the map should care about (Hz)
parameters.maxF         = 50         # fastest; keep this <= samplingFreq / 2 (the Nyquist limit) -> 50
parameters.numPeriods   = 25         # how many frequency channels to place between minF and maxF
parameters.pcaModes     = n_pca      # number of low-d features we feed in (PCA chose this above)

# ---- machine knobs (leave as-is unless you switched on a GPU, or hit a memory error) ----
parameters.useGPU         = -1       # set to 0 if you enabled a GPU runtime
parameters.numProcessors  = -1       # -1 = use all CPU cores for the wavelet transform
parameters.training_numPoints = 3000 # points per mini-embedding while building the training set
parameters.trainingSetSize    = 5000 # total training points (raise it with more RAM; 36k is good at 64 GB)
print("parameters set")
"""))

# ---------------------------------------------------------------- wavelets
cells.append(md(r"""
# 6.&nbsp; From postures to postural *dynamics*

A posture at a single instant doesn't pin down the *behavior* &mdash; a walking fly and a standing fly
can share the very same pose. The behavior is in the **dynamics**: how the pose is changing. We capture
that with a **wavelet transform**, which asks, for each PCA mode, "how much wiggling is there at each
frequency, at each moment?" The result is a spectrogram. Watch how grooming (fast, high-frequency) and
resting (almost nothing) look completely different:
"""))
cells.append(code(r"""
wlets, freqs = mmpy.findWavelets(projs_list[0], n_pca, parameters.omega0,
                                 parameters.numPeriods, parameters.samplingFreq,
                                 parameters.maxF, parameters.minF,
                                 parameters.numProcessors, parameters.useGPU)
# On a GPU runtime findWavelets hands back CuPy (GPU) arrays; pull them to NumPy so matplotlib can plot.
wlets = wlets.get() if hasattr(wlets, "get") else np.asarray(wlets)
freqs = freqs.get() if hasattr(freqs, "get") else np.asarray(freqs)
nshow = 600
fig, axes = plt.subplots(n_pca, 1, figsize=(14, 1.3 * n_pca), sharex=True)
for i, ax in enumerate(np.atleast_1d(axes)):
    ax.imshow(wlets[:nshow, 25 * i:25 * (i + 1)].T, cmap="PuRd", aspect="auto", origin="lower",
              extent=(0, nshow / fps, freqs[0], freqs[-1]))
    ax.set_ylabel("PC%d freq (Hz)" % (i + 1), fontsize=8); ax.set_yticks([1, 50])
axes[-1].set_xlabel("time (s)"); plt.tight_layout(); plt.show()
print("each frame is now described by %d numbers (%d modes x %d frequencies)"
      % (n_pca * parameters.numPeriods, n_pca, parameters.numPeriods))
"""))
cells.append(md(r"""
Notice how the bottom (slow) frequencies carry the lazy movements and the top (fast) ones light up
during quick leg-flicks &mdash; bouts of grooming or fast walking. That's the whole point of wavelets:
they encode *dynamics*, not just shape. And notice the bill we just ran up &mdash; a tidy few PCA modes
became hundreds of numbers per frame. That's exactly why we spent effort compressing first.
"""))

# ---------------------------------------------------------------- embed
cells.append(md(r"""
# 7.&nbsp; Compress into 2-D: the behavioral map

Every frame is now a point in a high-dimensional space of "postural dynamics," where similar movements
sit near one another. We use **UMAP** (or t-SNE) to flatten that space down to 2-D so we can actually
*see* it, while doing its best to keep neighbors as neighbors.

Doing this on every frame at once would melt your instance's RAM, so ```mmpy``` first builds a representative **training
set** &mdash; it runs many small embeddings and samples broadly, which is what lets rare behaviors earn
their own spot &mdash; embeds that, and later re-embeds everything onto it.

**Run time:** ~1&ndash;3 min.
"""))
cells.append(code(r"""
t0 = time.time()
mmpy.subsampled_tsne_from_projections(parameters, parameters.projectPath)
print("training embedding built in %d s" % (time.time() - t0))
"""))
cells.append(md(r"""
Here's the training map as a smooth density. **Peaks are behaviors the fly does often.** Try changing
`sigma` (the smoothing): small values give many little peaks, large values blur them into a few blobs.
There's no single "correct" sigma &mdash; it depends on how finely you want to look.
"""))
cells.append(code(r"""
ty = hdf5storage.loadmat("%s/%s/training_embedding.mat" % (projectPath, parameters.method))["trainingEmbedding"]
m = np.abs(ty).max()
sigma = 3.0    # 🔧 your turn: try 1 or 5.0
_, xx, dens = mmpy.findPointDensity(ty, sigma, 511, [-m - 15, m + 15])

fig, ax = plt.subplots(1, 2, figsize=(12, 6))
ax[0].scatter(ty[:, 0], ty[:, 1], s=1, c=np.arange(len(ty)))
ax[0].set_xlim(xx[0], xx[-1]); ax[0].set_ylim(xx[0], xx[-1]); ax[0].set_aspect("equal")
ax[0].set_title("training points")
ax[1].imshow(dens, extent=(xx[0], xx[-1], xx[0], xx[-1]), cmap=mmpy.gencmap(), origin="lower")
ax[1].set_aspect("equal"); ax[1].set_title("behavioral density (sigma=%.1f)" % sigma); plt.show()
"""))
cells.append(md(r"""
Now **re-embed all the data** onto that map and save it, so every frame of every movie gets a 2-D
coordinate.

**Run time:** ~2&ndash;3 min (UMAP).
"""))
cells.append(code(r"""
import h5py
tfolder = "%s/%s/" % (projectPath, parameters.method)
with h5py.File(tfolder + "training_data.mat", "r") as f: trainingSetData = f["trainingSetData"][:].T
with h5py.File(tfolder + "training_embedding.mat", "r") as f: trainingEmbedding = f["trainingEmbedding"][:].T
zstr = "uVals" if parameters.method == "UMAP" else "zVals"

for pf in glob.glob(projectPath + "/Projections/*_pcaModes.mat"):
    if os.path.exists(pf[:-4] + "_%s.mat" % zstr):
        continue
    projections = hdf5storage.loadmat(pf)["projections"]
    z, _ = mmpy.findEmbeddings(projections, trainingSetData, trainingEmbedding, parameters)
    z = z.get() if hasattr(z, "get") else np.asarray(z)            # GPU (CuPy) -> NumPy before saving
    hdf5storage.write(data={"zValues": z}, path="/", filename=pf[:-4] + "_%s.mat" % zstr,
                      store_python_metadata=False, matlab_compatible=True, truncate_existing=True)
print("all data embedded")
"""))

# ---------------------------------------------------------------- watershed
cells.append(md(r"""
# 8.&nbsp; Carve the map into behaviors (watershed)

The peaks in that density are distinct, stereotyped behaviors; the valleys between them are natural
borders. A **watershed transform** &mdash; picture flooding the landscape until walls form where the
pools meet &mdash; turns the smooth map into a labelled patchwork of behavioral regions. We ask for a
target number of regions with `num_regions` (here 12, to keep the map readable); the function tries
each amount of smoothing and keeps the one that lands closest. *(Prefer "at least N regions"? Drop
`num_regions` and pass `minimum_regions=N` instead.)*
"""))
cells.append(code(r"""
startsigma = .5 if parameters.method == "UMAP" else 4.2
mmpy.findWatershedRegions(parameters, minimum_regions=12, startsigma=startsigma,
                          pThreshold=[0.33, 0.67], saveplot=True, endident="*_pcaModes.mat")
from IPython.display import Image
Image(glob.glob("%s/%s/zWshed*.png" % (projectPath, parameters.method))[0])
"""))

# ---------------------------------------------------------------- ethogram + map
cells.append(md(r"""
# 9.&nbsp; Read out behavior over time: the ethogram

Everything the pipeline produced lives in one file, `zVals_wShed_groups.mat`. Its `watershedRegions`
field is the headline result: **one behavior label per frame.** Plot that against time and you get an
*ethogram* &mdash; the same kind of behavior-versus-time chart you'd otherwise score by hand, except
nobody scored it.
"""))
cells.append(code(r"""
wfile = hdf5storage.loadmat("%s/%s/zVals_wShed_groups.mat" % (projectPath, parameters.method))
wregs = wfile["watershedRegions"].flatten()
etho = np.zeros((wregs.max() + 1, len(wregs)))
for r in range(1, wregs.max() + 1):
    etho[r, wregs == r] = 1
etho = np.split(etho.T, np.cumsum(wfile["zValLens"][0].flatten())[:-1])

fig, axes = plt.subplots(len(etho), 1, figsize=(16, 6))
for e, nm, ax in zip(etho, wfile["zValNames"][0], np.atleast_1d(axes)):
    ax.imshow(e.T, aspect="auto", cmap=mmpy.gencmap(), extent=(0, e.shape[0] / fps, 0, e.shape[1]))
    ax.set_title(str(nm[0][0]), fontsize=9); ax.set_ylabel("region")
axes[-1].set_xlabel("time (s)"); plt.tight_layout(); plt.show()
"""))
cells.append(md("## 9.1&nbsp; Watch the fly move across its own map"))
cells.append(md("The dot is where the fly is *in behavior space* as the movie plays (~1&ndash;2 min to render):"))
cells.append(code(r"""
zValues = wfile["zValues"]; m = np.abs(zValues).max()
_, xx, dens = mmpy.findPointDensity(zValues, 3, 511, [-m - 10, m + 10])
h5ind, tstart, nframes = 0, 50, 120
frames = read_frames(moviepaths[h5ind], tstart, nframes)

fig, ax = plt.subplots(1, 2, figsize=(11, 5.5))
ax[0].imshow(dens, extent=(xx[0], xx[-1], xx[0], xx[-1]), cmap=mmpy.gencmap(), origin="lower")
ax[0].axis("off"); ax[0].set_title("behavior space")
dot = ax[0].scatter([], [], s=300, color="c")

def update(i):
    f = tstart + i
    ax[1].clear(); ax[1].imshow(frames[i], origin="lower")
    for c in connections:
        ax[1].plot(h5s[h5ind][f, c, 0], h5s[h5ind][f, c, 1], "m-", lw=1)
    ax[1].axis("off"); dot.set_offsets(zValues[f])

anim = FuncAnimation(fig, update, frames=len(frames), interval=66)
plt.close()
HTML(anim.to_jshtml())
"""))

# ---------------------------------------------------------------- region videos + naming
cells.append(md(r"""
# 10.&nbsp; What *is* each region? Watch example bouts, then name them

A region is just a number until you've *watched* it. The helper below finds the **longest few bouts** of
a chosen region across the recordings and plays them **together in a grid**, each clipped to the shortest
so they stay in sync (video + skeleton), with a **colored dot** tracing each fly's path across the
behavioral map on the left (its color matches the box around its movie), so you can see what they share
and decide what it is.
"""))
cells.append(code(r"""
def show_region(region, n_show=4, max_len=150, downsample=2):
    # Find every contiguous bout of `region` across all recordings, take the longest few, clip them all
    # to the shortest one's length (so they stay in sync), and play them in a grid -- with a coloured dot
    # tracing each fly's path through the behavioral map on the left (one colour per fly, matched by the
    # coloured box around its movie). Frames are spatially subsampled by `downsample` to keep it light.
    zValues = wfile["zValues"]
    offsets = np.r_[0, np.cumsum(wfile["zValLens"][0].flatten())]   # row in zValues where each recording starts
    wregs = np.split(wfile["watershedRegions"].flatten(), offsets[1:-1])
    bouts = []                                            # (length, dataset, start) for every bout
    for di in range(len(datasetnames)):
        on = (wregs[di] == region).astype(int)
        starts = np.where(np.diff(np.r_[0, on]) == 1)[0]
        ends = np.where(np.diff(np.r_[on, 0]) == -1)[0] + 1
        bouts += [(int(e - s), di, int(s)) for s, e in zip(starts, ends)]
    if not bouts:
        print("region %d never occurs in these recordings." % region); return None
    bouts = sorted(bouts, reverse=True)[:n_show]          # the longest few
    nframes = min(min(b[0] for b in bouts), max_len)      # clip all to the shortest bout (capped for speed)

    clips = []                                            # (dataset, start, [subsampled frames]) per bout
    for _, di, s in bouts:
        fr = read_frames(moviepaths[di], s, nframes)
        clips.append((di, s, [f[::downsample, ::downsample] for f in fr]))
    nframes = min([nframes] + [len(fr) for _, _, fr in clips])    # guard against a short read
    n = len(clips)
    colors = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3"][:n]      # one colour per fly

    # behavioral map on the left, the n movies in a grid on the right (smaller grid if < 4 bouts)
    ncol = 1 if n == 1 else 2
    nrow = int(np.ceil(n / ncol))
    grid = [["map"] + ["m%d" % (r * ncol + cc) if r * ncol + cc < n else "." for cc in range(ncol)]
            for r in range(nrow)]
    fig, axd = plt.subplot_mosaic(grid, figsize=(3.2 * (1.4 + ncol), 3.2 * nrow),
                                  gridspec_kw={"width_ratios": [1.4] + [1] * ncol})

    # draw the map once (only the dots move); each dot is one fly, coloured to match its box
    axm = axd["map"]; m = np.abs(zValues).max()
    _, xx, dens = mmpy.findPointDensity(zValues, 1.0, 511, [-m - 10, m + 10])
    axm.imshow(dens, extent=(xx[0], xx[-1], xx[0], xx[-1]), origin="lower", cmap=mmpy.gencmap())
    axm.set_aspect("equal"); axm.axis("off")
    def dot_xy(i):
        return np.array([zValues[offsets[di] + s + i] for di, s, _ in clips])
    dots = axm.scatter(*dot_xy(0).T, c=colors, s=160, edgecolor="k", linewidth=0.7, zorder=3)

    def update(i):
        dots.set_offsets(dot_xy(i))                        # move each fly's dot to its place on the map
        for k, (di, s, fr) in enumerate(clips):
            ax = axd["m%d" % k]; ax.clear(); ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():                  # coloured box matching this fly's dot
                sp.set_color(colors[k]); sp.set_linewidth(3)
            ax.imshow(fr[i], origin="lower")
            for c in connections:                          # skeleton, scaled to the subsampled frame
                ax.plot(h5s[di][s + i, c, 0] / downsample, h5s[di][s + i, c, 1] / downsample,
                        "-", color="firebrick", lw=1)
            ax.set_title("%s  (frame %d)" % (datasetnames[di], s + i), fontsize=8)
    anim = FuncAnimation(fig, update, frames=nframes, interval=50); plt.close()
    return HTML(anim.to_jshtml())

wmax = int(wfile["watershedRegions"].max())
show_region(wmax // 2)        # 🔧 your turn: try other region numbers, 1 .. wmax
"""))
cells.append(md(r"""
This is the step where unsupervised analysis hands the science back to *you*. Scroll through a few
regions with `show_region(r)`, and as you recognize what the fly is doing, write it down. The
dictionary below is yours to fill &mdash; it turns anonymous region numbers into a behavioral
vocabulary you can use in every notebook that follows.
"""))
cells.append(code(r"""
# 🔧 Your turn: watch show_region(r) for a few r in 1..wmax, then label what you saw.
behavior_dict = {
    # region number : "what you saw"   (these are just examples -- replace with your own!)
    # wmax // 2: "slow walking",
    # 1: "grooming",
}
for r, name in behavior_dict.items():
    print("region %2d = %s" % (r, name))
if not behavior_dict:
    print("(empty for now -- fill it in by watching a few regions above)")
"""))

# ---------------------------------------------------------------- exercises
cells.append(md(r"""
# 11.&nbsp; 🔧 Your turn

Pick one or two &mdash; each is a one-line change; rerun from that cell downward:

1. **Frequency band (&sect;5):** set `parameters.maxF = 20`. Which behaviors blur together once you
   stop "listening" to the fast leg movements?
2. **Granularity (&sect;8):** set `num_regions = 25`. Do the new regions *subdivide* the old ones,
   or reshuffle them? (A sneak peek at hierarchy &mdash; see notebook 02.)
3. **Smoothing (&sect;7):** sweep `sigma` over 0.5, 1, 3. What's the "right" number of behaviors?
   (A bit of a trick question &mdash; it depends on what you want to measure.)
4. **t-SNE vs UMAP (&sect;5):** set `parameters.method = "TSNE"` and rebuild from &sect;7. Same biology,
   different-looking map? (t-SNE is slower: ~15 min to re-embed.)
"""))

# ---------------------------------------------------------------- checkpoint
cells.append(md("# 12.&nbsp; Save your map, then on to the rest"))
cells.append(code(r"""
!zip -qr Fly_mmpy.zip Fly_mmpy
print("checkpoint saved to Fly_mmpy.zip -- download it from the file browser on the left if you like")
"""))
cells.append(md(r"""
**You built a behavioral map.** That's the engine the rest of the afternoon runs on. We'll go through
the next few **together** &mdash; each is a standalone notebook that loads its own data, so you can run
any of them even if your map here didn't come out perfect:

| Notebook | You'll answer |
|---|---|
| `02_transitions_and_hierarchy.ipynb` | Is behavior **Markovian**? How is it organized in time? |
| `03_rat_individual_behavior.ipynb` | How does **amphetamine** reshape a rat's repertoire? |
| `04_rat_social_behavior.ipynb` | How do you **quantify what two animals do together**? |

And for the end of the session (or your own time):

| Notebook | You'll answer |
|---|---|
| `05_slow_modes.ipynb` | What **slow internal states** bias the fast actions? |
| `06_optogenetics.ipynb` | Which behaviors does **activating a neuron** trigger? |
| `07_bring_your_own_data.ipynb` | Run all of this on **your own** animal. |
"""))

write_nb(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                      "01_build_a_behavioral_map.ipynb"), cells)
