"""01 — Core: build a fly behavioral map (Act 1, everyone). Adapted from the
motionmapperpy fly demo. This is the engine every Act-2 track builds on."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nb_builder import md, code, badge, write_nb

REPO = "bermanlabemory/unsupervised-behavior-tutorials/blob/main"
cells = []

cells.append(badge("%s/01_build_a_behavioral_map.ipynb" % REPO))

cells.append(md(r"""
# 1.&nbsp; Build a behavioral map &mdash; the Core

Welcome! This is the notebook **everyone** does. By the end you'll have taken raw tracked
body-part positions and turned them into a **behavioral map**: a 2-D atlas of everything a
fly does, discovered *without telling the computer a single behavior name*.

You've already seen **supervised** behavior classification this week (train a classifier on
labels a human provides). This is the **unsupervised** complement: we let the structure of the
movement itself define the behaviors, then go look at what we found.

The recipe (it's the same for a worm, a mouse, or a rat &mdash; only the first step changes):

> **postures → postural *dynamics* (wavelets) → a 2-D map → discrete behaviors (watershed)**

**Total run time:** ~20-30 min. A GPU runtime is nice but not required
(`Runtime → Change runtime type → GPU`).

| How to read the cells | |
|---|---|
| ▶︎ **Just run it** | most cells &mdash; press Shift+Enter and watch |
| 🔧 **Your turn** | optional: change something and see what happens |
"""))

# ---------------------------------------------------------------- setup
cells.append(md("# 1.1&nbsp; Get the code and data"))
cells.append(md(r"""
We download **motionmapperpy** from GitHub. It comes with a small example dataset: two short
movies of a single fly walking in a shallow dish, with **32 body parts tracked** in each frame
(tracked here with [LEAP](https://www.nature.com/articles/s41592-018-0234-5), but SLEAP /
DeepLabCut output looks the same).
"""))
cells.append(code(r"""
import os, sys, types
if not os.path.exists("motionmapperpy"):
    !git clone -q https://github.com/bermanlabemory/motionmapperpy
!pip install -q hdf5storage easydict umap-learn

# We read video with OpenCV and animate with matplotlib, so this notebook needs no moviepy at all.
# The released package still imports moviepy at load time, so we stub it out -- which sidesteps the
# whole moviepy/ffmpeg mess on modern Colab.
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
print("ready")
"""))
cells.append(md(r"""
> **No restart needed.** The setup cell imports motionmapperpy straight from the cloned folder. If
> the import ever fails, re-run the setup cell above — avoid *Restart session*, which would undo the
> `sys.path` line it adds.
"""))

# ---------------------------------------------------------------- imports
cells.append(md("# 1.2&nbsp; Imports"))
cells.append(code(r"""
import glob, os, pickle, copy, time
import numpy as np
import pandas as pd
import hdf5storage
import cv2                                    # reads video frames; ships with Colab, no ffmpeg hassle
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from IPython.display import HTML
from scipy.ndimage import median_filter
from sklearn.decomposition import PCA
from matplotlib import rc; rc("animation", html="jshtml")

import motionmapperpy as mmpy
from motionmapperpy import demoutils
%matplotlib inline

def read_frames(path, start, n):
    # read n consecutive RGB video frames starting at frame index `start`
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

# ---------------------------------------------------------------- load
cells.append(md("# 2.&nbsp; Look at the data *first*"))
cells.append(md(r"""
Rule #1 of behavioral analysis: **never trust tracking you haven't looked at.** Let's load the
movies and the tracked positions and watch them together.
"""))
cells.append(code(r"""
datasetnames = ["fly_leap_test", "fly_leap_test_2"]
moviepaths = ["motionmapperpy/data/fly/%s.mp4" % d for d in datasetnames]

# The tracking is stored as a pandas table with x, y, and a confidence ("likelihood") column
# per body part. Drop the confidence columns and keep x,y -> shape (frames, 32 parts, 2).
h5s_pandas = [pd.read_hdf("motionmapperpy/data/fly/%s_positions.h5" % d) for d in datasetnames]
keep = np.mod(np.arange(1, h5s_pandas[0].shape[1] + 1), 3) != 0     # drop every 3rd col
h5s = [median_filter(h.values[:, keep], size=(5, 1)) for h in h5s_pandas]   # light de-noise
h5s = [h.reshape((-1, h.shape[1] // 2, 2)) for h in h5s]

for d, h in zip(datasetnames, h5s):
    print("%s: %d frames, %d body parts" % (d, h.shape[0], h.shape[1]))
"""))
cells.append(md("Overlay the skeleton on the video (this takes ~1 min to render):"))
cells.append(code(r"""
# 'connections' just says which body parts to draw lines between, so the fly looks like a fly.
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

# ---------------------------------------------------------------- features
cells.append(md("# 3.&nbsp; Choose a representation (this is where the biology lives)"))
cells.append(md(r"""
The single most important decision in this whole pipeline is **what numbers describe the
posture**. Raw pixel positions are a bad choice: they move when the *whole fly* moves or turns,
even if its pose is identical. We want a representation that is **egocentric** &mdash; it only
changes when the animal changes its *shape*.

For a fly, a natural egocentric representation is a set of **joint angles** (leg joints, wing
angles, neck angle). Below we compute 22 angles from the 32 tracked points. *(For a mouse or
rat we'd instead center & rotate the keypoints; for a worm we'd use "eigenworms". Same idea,
different front-end &mdash; you'll see this in the other notebooks.)*
"""))
cells.append(code(r"""
# Each entry: [a, b, c, flag] = angle a-b-c; flag picks the angle range (0: [-pi,pi], 1: [0,2pi]).
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
print("each frame is now %d angles instead of %d x,y positions" %
      (angleh5s[0].shape[1], h5s[0].shape[1] * 2))
"""))
cells.append(md("Normalize each angle to roughly [0, 1] so no single joint dominates:"))
cells.append(code(r"""
lo = np.min([A.min(0) for A in angleh5s], 0)
angleh5s = [A - lo for A in angleh5s]
hi = np.max([A.max(0) for A in angleh5s], 0)
angleh5s = [A / hi for A in angleh5s]

fig, ax = plt.subplots(figsize=(14, 4))
ax.plot(angleh5s[0][1000:2000, [3, 1, 13]])
ax.set_xlabel("frame"); ax.set_ylabel("normalized angle")
ax.legend(["wingtip angle", "neck angle", "midleg femur-tibia angle"]); plt.show()
"""))
cells.append(md(r"""
**Compress with PCA.** The 22 angles are correlated, so we rotate into a smaller set of
principal components that keep 95% of the variance. Fewer dimensions = faster, less noise.
"""))
cells.append(code(r"""
X = np.concatenate(angleh5s, axis=0)
p = PCA().fit(X)
n_pca = int(np.argmax(np.cumsum(p.explained_variance_ratio_) > 0.95)) + 1

fig, ax = plt.subplots(figsize=(8, 3))
ax.plot(np.arange(1, X.shape[1] + 1), np.cumsum(p.explained_variance_ratio_), "firebrick")
ax.axvline(n_pca, ls="--", color="royalblue"); ax.axhline(0.95, ls=":", color="grey")
ax.set_xlabel("# PCA components"); ax.set_ylabel("cumulative variance explained"); plt.show()

y = p.transform(X)[:, :n_pca]
projs_list = np.split(y, np.cumsum([A.shape[0] for A in angleh5s])[:-1])
print("keeping %d PCA modes" % n_pca)
"""))

# ---------------------------------------------------------------- project + params
cells.append(md("# 4.&nbsp; Set up a motionmapperpy project"))
cells.append(md(r"""
motionmapperpy keeps everything for one map in a **project folder**. We create one and drop our
low-dimensional time series into its `Projections/` subfolder (one file per movie, named
`*_pcaModes.mat`).
"""))
cells.append(code(r"""
projectPath = "Fly_mmpy"
mmpy.createProjectDirectory(projectPath)
for d, projs in zip(datasetnames, projs_list):
    hdf5storage.savemat("%s/Projections/%s_pcaModes.mat" % (projectPath, d),
                        {"projections": projs})
print("saved %d projection files" % len(projs_list))
"""))
cells.append(md(r"""
Now the **parameters**. These are the few real choices in the pipeline. The most important ones
are tied to your camera: `samplingFreq` is your frame rate, and `minF`/`maxF` are the slowest
and fastest movements (in Hz) you want the map to care about (`maxF` ≤ half your frame rate).
"""))
cells.append(code(r"""
parameters = mmpy.setRunParameters()
parameters.projectPath = projectPath
parameters.method      = "UMAP"     # 'UMAP' (fast) or 'TSNE' (classic, slower)
parameters.pcaModes    = n_pca
parameters.samplingFreq = 100       # the fly movies are 100 fps
parameters.minF        = 1          # slowest movement frequency to track (Hz)
parameters.maxF        = 50         # fastest (here = Nyquist = 100/2)
parameters.numPeriods  = 25         # number of frequency channels between minF and maxF
parameters.useGPU      = -1         # set to 0 if you enabled a GPU runtime
parameters.numProcessors = -1       # use all available CPU cores for the wavelet transform
parameters.training_numPoints = 3000
parameters.trainingSetSize    = 5000
print("parameters set")
"""))

# ---------------------------------------------------------------- wavelets
cells.append(md("# 5.&nbsp; From postures to postural *dynamics*"))
cells.append(md(r"""
A posture at a single instant doesn't tell you the *behavior* &mdash; walking and standing can
share a pose. Behavior is in the **dynamics**. We capture them with a **wavelet transform**: for
each PCA mode we ask "how much wiggling is there at each frequency, at each moment?" The result
is a spectrogram. Notice how grooming (fast, high-frequency) and resting (nothing) look totally
different here:
"""))
cells.append(code(r"""
wlets, freqs = mmpy.findWavelets(projs_list[0], n_pca, parameters.omega0,
                                 parameters.numPeriods, parameters.samplingFreq,
                                 parameters.maxF, parameters.minF,
                                 parameters.numProcessors, parameters.useGPU)
fig, axes = plt.subplots(n_pca, 1, figsize=(14, 1.3 * n_pca), sharex=True)
for i, ax in enumerate(np.atleast_1d(axes)):
    ax.imshow(wlets[:600, 25 * i:25 * (i + 1)].T, cmap="PuRd", aspect="auto", origin="lower")
    ax.set_ylabel("PC%d" % (i + 1), fontsize=8); ax.set_yticks([])
axes[-1].set_xlabel("frame"); plt.tight_layout(); plt.show()
print("each frame is now described by %d numbers (%d modes x %d frequencies)" %
      (n_pca * parameters.numPeriods, n_pca, parameters.numPeriods))
"""))

# ---------------------------------------------------------------- embed
cells.append(md("# 6.&nbsp; Squash it into 2-D: the behavioral map"))
cells.append(md(r"""
Every frame is now a point in a high-dimensional space of "postural dynamics". Similar movements
sit near each other there. We use **UMAP** (or t-SNE) to flatten that space to 2-D so we can
*see* it, while keeping neighbors as neighbors.

Computing this on every frame at once would melt the RAM, so motionmapperpy first builds a
representative **training set** (by running many mini-embeddings and sampling broadly &mdash;
this is what catches rare behaviors), embeds that, and later re-embeds everything onto it.

**Run time:** ~1-3 min.
"""))
cells.append(code(r"""
t0 = time.time()
mmpy.subsampled_tsne_from_projections(parameters, parameters.projectPath)
print("training embedding built in %d s" % (time.time() - t0))
"""))
cells.append(md(r"""
Let's look at the training map as a smooth density. **Peaks = behaviors the fly does often.**
Try changing `sigma` (the smoothing): small = lots of little peaks, large = a few blobs.
"""))
cells.append(code(r"""
ty = hdf5storage.loadmat("%s/%s/training_embedding.mat" % (projectPath, parameters.method))["trainingEmbedding"]
m = np.abs(ty).max()
sigma = 1.0    # 🔧 your turn: try 0.5 or 3.0
_, xx, dens = mmpy.findPointDensity(ty, sigma, 511, [-m - 15, m + 15])

fig, ax = plt.subplots(1, 2, figsize=(12, 6))
ax[0].scatter(ty[:, 0], ty[:, 1], s=1, c=np.arange(len(ty))); ax[0].set_title("training points")
ax[1].imshow(dens, extent=(xx[0], xx[-1], xx[0], xx[-1]), cmap=mmpy.gencmap(), origin="lower")
ax[1].set_title("behavioral density (sigma=%.1f)" % sigma); plt.show()
"""))
cells.append(md(r"""
Now **re-embed all the data** onto that map and save it. Each frame gets a 2-D coordinate.

**Run time:** ~2-3 min (UMAP).
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
    hdf5storage.write({"zValues": z}, "/", pf[:-4] + "_%s.mat" % zstr,
                      matlab_compatible=True, truncate_existing=True)
print("all data embedded")
"""))

# ---------------------------------------------------------------- watershed
cells.append(md("# 7.&nbsp; Carve the map into behaviors (watershed)"))
cells.append(md(r"""
Peaks in the density are distinct, stereotyped behaviors; the valleys between them are natural
borders. A **watershed transform** (think: flood the landscape, walls form where pools meet)
turns the smooth map into a labelled patchwork of behavioral regions. `minimum_regions` controls
how finely we carve.
"""))
cells.append(code(r"""
startsigma = 1.0 if parameters.method == "UMAP" else 4.2
mmpy.findWatershedRegions(parameters, minimum_regions=12, startsigma=startsigma,
                          pThreshold=[0.33, 0.67], saveplot=True, endident="*_pcaModes.mat")
from IPython.display import Image
Image(glob.glob("%s/%s/zWshed*.png" % (projectPath, parameters.method))[0])
"""))

# ---------------------------------------------------------------- ethogram + map
cells.append(md("# 8.&nbsp; Read out behavior over time: the ethogram"))
cells.append(md(r"""
Everything the pipeline produced lives in one file, `zVals_wShed_groups.mat`. Its
`watershedRegions` field is the headline result: **one behavior label per frame**. Plotting that
as a function of time gives an *ethogram*.
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
    ax.imshow(e.T, aspect="auto", cmap=mmpy.gencmap()); ax.set_title(str(nm[0][0]), fontsize=9)
    ax.set_ylabel("region")
axes[-1].set_xlabel("frame"); plt.tight_layout(); plt.show()
"""))
cells.append(md("## 8.1&nbsp; Watch the fly move across its own map"))
cells.append(md("The dot is where the fly is *in behavior space* as the movie plays (~1-2 min to render):"))
cells.append(code(r"""
zValues = wfile["zValues"]; m = np.abs(zValues).max()
_, xx, dens = mmpy.findPointDensity(zValues, 1.0, 511, [-m - 10, m + 10])
h5ind, tstart, nframes = 0, 1500, 120
frames = read_frames(moviepaths[h5ind], tstart, nframes)

fig, ax = plt.subplots(1, 2, figsize=(11, 5.5))
ax[0].imshow(dens, extent=(xx[0], xx[-1], xx[0], xx[-1]), cmap=mmpy.gencmap(), origin="lower")
ax[0].axis("off"); ax[0].set_title("behavior space")
dot = ax[0].scatter([], [], s=300, color="k")

def update(i):
    f = tstart + i
    ax[1].clear(); ax[1].imshow(frames[i], origin="lower")
    for c in connections:
        ax[1].plot(h5s[h5ind][f, c, 0], h5s[h5ind][f, c, 1], "k-", lw=1)
    ax[1].axis("off"); dot.set_offsets(zValues[f])

anim = FuncAnimation(fig, update, frames=len(frames), interval=66)
plt.close()
HTML(anim.to_jshtml())
"""))

# ---------------------------------------------------------------- region videos
cells.append(md("# 9.&nbsp; What *is* each region? Watch example bouts"))
cells.append(md(r"""
A region is only meaningful once you've *watched* it. The helper below finds the longest bout of a
chosen region and plays it back (video + skeleton) right here &mdash; so you can name it
("region 5 = grooming", ...).
"""))
cells.append(code(r"""
def show_region(region, dataset=0, max_len=150):
    # find contiguous bouts of `region` in one recording, play back the longest one
    wr = np.split(wfile["watershedRegions"].flatten(),
                  np.cumsum(wfile["zValLens"][0].flatten())[:-1])[dataset]
    on = (wr == region).astype(int)
    starts = np.where(np.diff(np.r_[0, on]) == 1)[0]
    ends = np.where(np.diff(np.r_[on, 0]) == -1)[0] + 1
    if len(starts) == 0:
        print("region %d not found in %s" % (region, datasetnames[dataset])); return None
    k = (ends - starts).argmax()
    s, e = int(starts[k]), int(min(ends[k], starts[k] + max_len))
    frames = read_frames(moviepaths[dataset], s, e - s)
    fig, ax = plt.subplots(figsize=(5, 5))
    def update(i):
        ax.clear(); ax.imshow(frames[i], origin="lower")
        for c in connections:
            ax.plot(h5s[dataset][s + i, c, 0], h5s[dataset][s + i, c, 1], "-", color="firebrick", lw=1)
        ax.axis("off"); ax.set_title("region %d  (frame %d)" % (region, s + i))
    anim = FuncAnimation(fig, update, frames=len(frames), interval=50); plt.close()
    return HTML(anim.to_jshtml())

wmax = int(wfile["watershedRegions"].max())
show_region(wmax // 2)        # 🔧 your turn: try other region numbers, 1 .. wmax
"""))

# ---------------------------------------------------------------- exercises
cells.append(md(r"""
# 10.&nbsp; 🔧 Your turn

Pick one or two &mdash; each is a one-line change, rerun from that cell down:

1. **Frequency band (§4):** set `parameters.maxF = 20`. Which behaviors blur together when you
   stop "listening" to fast leg movements?
2. **Granularity (§7):** set `minimum_regions = 25`. Do the new regions *subdivide* old ones, or
   reshuffle them? (This is a sneak peek at hierarchy &mdash; see notebook 03.)
3. **Smoothing (§6):** sweep `sigma` over 0.5, 1, 3. What's the "right" number of behaviors?
   (Trick question &mdash; it depends on what you want to measure.)
4. **t-SNE vs UMAP (§4):** set `parameters.method = "TSNE"` and rebuild from §6. Same biology,
   different-looking map? (t-SNE is slower: ~15 min to re-embed.)
"""))

# ---------------------------------------------------------------- checkpoint
cells.append(md("# 11.&nbsp; Save your map, then choose your adventure"))
cells.append(code(r"""
!zip -qr Fly_mmpy.zip Fly_mmpy
print("checkpoint saved to Fly_mmpy.zip — download it from the file browser on the left if you like")
"""))
cells.append(md(r"""
🎉 **You built a behavioral map.** That's the engine. Now go *do science* with it &mdash; pick a
track (each is a standalone notebook; none requires you to have finished this one perfectly):

| Notebook | You'll answer |
|---|---|
| `02_social_behavior_rats.ipynb` | What do two animals do **together**? |
| `03_transitions_and_hierarchy.ipynb` | Is behavior **Markovian**? How is it organized in time? |
| `04_optogenetics.ipynb` | Which behaviors does **activating a neuron** trigger? |
| `05_slow_modes.ipynb` | What **slow internal states** bias the fast actions? |
| `06_bring_your_own_data.ipynb` | Run all of this on **your own** animal. |
"""))

write_nb(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                      "01_build_a_behavioral_map.ipynb"), cells)
