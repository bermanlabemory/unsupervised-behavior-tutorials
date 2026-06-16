"""07 — Bring your own data. A template that walks a student from their own
tracked keypoints to a behavioral map. Lots of TODO markers."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nb_builder import md, code, badge, write_nb, setup_code, carry_from_core

REPO = "bermanlabemory/unsupervised-behavior-tutorials/blob/main"
cells = []

cells.append(badge("%s/07_bring_your_own_data.ipynb" % REPO))

cells.append(md(r"""
# Bring your own data &mdash; your own behavioral map

This is the template for running the whole pipeline on *your own* tracked animal &mdash; the fly,
zebrafish, or mouse you tracked this week, or data waiting back in your home lab. It's notebook 01 with
the species-specific bits swapped out for **`# TODO`** markers; your job is to fill those in.

The good news: the pipeline is the same for any animal, and only the first two steps actually depend on
yours.
> **your keypoints → egocenter → features → wavelets → map → watershed → ethogram**

**Run time:** 10&ndash;20 min, depending on how much data you bring.
"""))
cells.append(md(carry_from_core()))

# ---------------------------------------------------------------- setup
cells.append(md(r"""
# 1.&nbsp; Setup

The usual opening cell &mdash; clone motionmapperpy, install the few packages Colab lacks, import what
we need.
"""))
cells.append(code(setup_code(
    imports="import glob, numpy as np, matplotlib.pyplot as plt, hdf5storage\n"
            "from scipy.ndimage import median_filter\n"
            "from sklearn.decomposition import PCA\n"
            "import motionmapperpy as mmpy\n"
            "%matplotlib inline")))

# ---------------------------------------------------------------- load
cells.append(md("# 2.&nbsp; Load your tracking"))
cells.append(md(r"""
Drag your tracked files into Colab's file browser on the left, then load them into a **list of arrays**,
one per recording, each shaped **`(frames, n_keypoints, 2 or 3)`**. We give loaders for the common
formats below &mdash; keep the one that matches your data and delete the rest.
"""))
cells.append(code(r"""
# ---- TODO: choose the loader that matches your data, delete the others ----

# (a) SLEAP / DeepLabCut .h5  (pandas table: x, y, likelihood per keypoint)
# import pandas as pd
# files = sorted(glob.glob("/content/*.h5"))
# def load_dlc(f):
#     df = pd.read_hdf(f); keep = np.mod(np.arange(1, df.shape[1] + 1), 3) != 0   # drop likelihood
#     a = df.values[:, keep]; return a.reshape(len(a), -1, 2)
# h5s = [load_dlc(f) for f in files]

# (b) numpy .npy  already shaped (frames, keypoints, 2 or 3)
# files = sorted(glob.glob("/content/*.npy"))
# h5s = [np.load(f) for f in files]

# (c) SLEAP analysis .h5 ('tracks' dataset) or .csv — adapt as needed.

# TODO: set these:
h5s = []                 # list of (frames, n_keypoints, dims) arrays  <-- FILL IN
FPS = 30                 # your camera frame rate (Hz)                 <-- FILL IN
assert h5s, "Load your data into h5s first!"
for i, h in enumerate(h5s):
    print("recording %d: %d frames, %d keypoints, %dD" % (i, h.shape[0], h.shape[1], h.shape[2]))
"""))
cells.append(md(r"""
**Look at it first.** Plot a few keypoint trajectories and ask: jumps, NaNs, swaps? Clean tracking is
everything here &mdash; garbage in, garbage out. Whatever survives into your features is what the map
will treat as "behavior," tracking errors included.
"""))
cells.append(code(r"""
plt.figure(figsize=(13, 4))
plt.plot(h5s[0][:1000, :, 0])      # x of every keypoint, first 1000 frames
plt.xlabel("frame"); plt.ylabel("x position"); plt.title("recording 0 — sanity check"); plt.show()
# Optional light de-noise (median filter). Skip if your tracking is already clean.
h5s = [median_filter(h, size=(5, 1, 1)) for h in h5s]
"""))

# ---------------------------------------------------------------- features
cells.append(md("# 3.&nbsp; Make an egocentric representation"))
cells.append(md(r"""
Remove **where** the animal is and **which way** it faces, so only its *pose* remains. Pick two
keypoints that define the body axis (e.g. tail-base → snout) and one to center on.

**Tips by animal:**
- *fly / insect:* joint **angles** work beautifully (see notebook 01 §3).
- *mouse / rat:* center on the body, rotate by the spine axis, keep x,y(,z).
- *worm:* use **eigenworms** (PCA of the body-centerline tangent angles) — already egocentric.
- *zebrafish:* tail-segment angles relative to the heading.
- *human:* joint angles or keypoints in a pelvis-centered, hip-aligned frame.
"""))
cells.append(code(r"""
# TODO: set the two keypoints that define the body axis, and the one to center on.
HEAD_KP, TAIL_KP, CENTER_KP = 0, 1, 1     # <-- FILL IN (indices into your keypoints)

def egocenter(P):                         # P: (T, K, D) -> centered & rotated in the xy-plane
    Q = P - P[:, CENTER_KP:CENTER_KP + 1, :]
    v = Q[:, HEAD_KP, :2] - Q[:, TAIL_KP, :2]
    th = np.arctan2(v[:, 1], v[:, 0])
    c, s = np.cos(-th), np.sin(-th)
    x, y = Q[:, :, 0].copy(), Q[:, :, 1].copy()
    Q[:, :, 0], Q[:, :, 1] = x * c[:, None] - y * s[:, None], x * s[:, None] + y * c[:, None]
    return Q.reshape(len(P), -1)

ego = [egocenter(h) for h in h5s]
print("egocentric feature dim:", ego[0].shape[1])
"""))
cells.append(md("Compress with PCA, keeping ~95% of the variance. The curve shows how many modes that takes:"))
cells.append(code(r"""
X = np.concatenate(ego, 0)
p = PCA().fit(X)
n_pca = int(np.argmax(np.cumsum(p.explained_variance_ratio_) > 0.95)) + 1

fig, ax = plt.subplots(figsize=(8, 3))
ax.plot(np.arange(1, X.shape[1] + 1), np.cumsum(p.explained_variance_ratio_), "firebrick", marker=".")
ax.axvline(n_pca, ls="--", color="royalblue"); ax.axhline(0.95, ls=":", color="grey")
ax.set_xlabel("# PCA components"); ax.set_ylabel("cumulative variance explained"); plt.show()

projs_list = np.split(p.transform(X)[:, :n_pca], np.cumsum([len(e) for e in ego])[:-1])
print("keeping %d PCA modes (95%% of the variance) out of %d features" % (n_pca, X.shape[1]))
"""))
cells.append(md(r"""
How many features did you start with, and how many modes did 95% take? A big drop means your keypoints
were highly correlated &mdash; the pose lives on a low-dimensional manifold, which is exactly what makes
this approach work. Barely any drop means either a genuinely high-dimensional animal or noisy features
worth revisiting.
"""))

# ---------------------------------------------------------------- params + map
cells.append(md("# 4.&nbsp; Parameters (the part that needs your judgment)"))
cells.append(code(r"""
parameters = mmpy.setRunParameters()
parameters.projectPath = "MyData_mmpy"
parameters.method      = "UMAP"
parameters.pcaModes    = n_pca
parameters.samplingFreq = FPS
parameters.minF        = 0.5            # TODO: slowest movement frequency you care about (Hz)
parameters.maxF        = min(FPS / 2, 25)   # TODO: fastest; must be <= FPS/2 (Nyquist)
parameters.numPeriods  = 25
parameters.useGPU      = -1
parameters.numProcessors = -1           # use all available CPU cores for the wavelet transform
mmpy.createProjectDirectory(parameters.projectPath)
for i, projs in enumerate(projs_list):
    hdf5storage.savemat("%s/Projections/rec_%d_pcaModes.mat" % (parameters.projectPath, i),
                        {"projections": projs})
print("project ready with %d recordings" % len(projs_list))
"""))
cells.append(md(r"""
This creates a **`MyData_mmpy`** folder in the file browser on the left. Inside, `Projections/` holds
your compressed time series; after the next steps, a `UMAP/` (or `TSNE/`) folder will hold the embedding,
the watershed regions, and the all-important `zVals_wShed_groups.mat`.
"""))
cells.append(md(r"""
# 5.&nbsp; Build the map (same engine as the Core notebook)

This is the compute-heavy part: building the training embedding and then re-embedding all your data.
**Run time:** ~2&ndash;5 min for a few short recordings, longer for big datasets.
"""))
cells.append(code(r"""
import time, h5py
t0 = time.time(); mmpy.subsampled_tsne_from_projections(parameters, parameters.projectPath)
tf = "%s/%s/" % (parameters.projectPath, parameters.method)
with h5py.File(tf + "training_data.mat", "r") as f: tD = f["trainingSetData"][:].T
with h5py.File(tf + "training_embedding.mat", "r") as f: tE = f["trainingEmbedding"][:].T
zstr = "uVals" if parameters.method == "UMAP" else "zVals"
for pf in glob.glob(parameters.projectPath + "/Projections/*_pcaModes.mat"):
    z, _ = mmpy.findEmbeddings(hdf5storage.loadmat(pf)["projections"], tD, tE, parameters)
    z = z.get() if hasattr(z, "get") else np.asarray(z)            # GPU (CuPy) -> NumPy before saving
    hdf5storage.write(data={"zValues": z}, path="/", filename=pf[:-4] + "_%s.mat" % zstr,
                      store_python_metadata=False, matlab_compatible=True, truncate_existing=True)
print("map built in %d s" % (time.time() - t0))
"""))
cells.append(code(r"""
startsigma = 1.0 if parameters.method == "UMAP" else 4.2
mmpy.findWatershedRegions(parameters, minimum_regions=12, startsigma=startsigma,
                          pThreshold=[0.33, 0.67], saveplot=True, endident="*_pcaModes.mat")
from IPython.display import Image
Image(glob.glob("%s/%s/zWshed*.png" % (parameters.projectPath, parameters.method))[0])
"""))
cells.append(md("# 6.&nbsp; Your ethogram"))
cells.append(code(r"""
w = hdf5storage.loadmat("%s/%s/zVals_wShed_groups.mat" % (parameters.projectPath, parameters.method))
wr = w["watershedRegions"].flatten()
e = np.zeros((wr.max() + 1, len(wr)))
for r in range(1, wr.max() + 1): e[r, wr == r] = 1
plt.figure(figsize=(15, 4)); plt.imshow(e, aspect="auto", cmap=mmpy.gencmap())
plt.xlabel("frame"); plt.ylabel("behavior region"); plt.title("your ethogram"); plt.show()
"""))

# ---------------------------------------------------------------- next
cells.append(md(r"""
# 7.&nbsp; Now go analyze it

You have a map and an ethogram for *your* animal. Take it into any track:
- compare conditions / dynamics → **`02_transitions_and_hierarchy.ipynb`**
- stimulus-triggered behavior → **`06_optogenetics.ipynb`**
- slow internal states (needs long recordings) → **`05_slow_modes.ipynb`**
- two animals interacting → **`04_rat_social_behavior.ipynb`**

**Checklist before you trust a result:**
1. Did you *watch* region videos to confirm the regions are real behaviors?
2. Is `maxF` ≤ half your frame rate, and `minF` slow enough for your slowest behavior?
3. Is your tracking clean (no swaps/jumps leaking into "behaviors")?
4. Enough data? Rare behaviors need enough frames to form their own region.
"""))

write_nb(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                      "07_bring_your_own_data.ipynb"), cells)
