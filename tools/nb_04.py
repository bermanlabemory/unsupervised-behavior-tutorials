"""04 — Rat social behavior: the choices you make to quantify what two animals do together.
Real Long-Evans dyads (Klibaite/Berman): each session ships precomputed individual + social/joint
maps (cz_action/sz_joint + coarse labels) and both animals' 23-joint 3-D keypoints. We walk the
modeling decisions behind Klibaite et al. 2025 on this real data."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nb_builder import md, code, badge, write_nb

REPO = "bermanlabemory/unsupervised-behavior-tutorials/blob/main"
cells = []

cells.append(badge("%s/04_rat_social_behavior.ipynb" % REPO))

cells.append(md(r"""
# 4.&nbsp; Rat social behavior: quantifying what two animals do *together*

In notebook 03 a rat's behavior was its own pose over time. Put **two** animals in an arena and
"behavior" stops being a property of one body &mdash; it's about the *relationship* between them.
There's no single right way to measure that; you face a sequence of **choices**, and the answers
depend on them. We walk those choices on **real Long-Evans dyads** (Klibaite/Berman), following
Klibaite et al., *Cell* 2025 ("Mapping the landscape of social behavior"):

> the unit of analysis &middot; which social variables &middot; the *joint* behavior map &middot;
> synchrony &middot; touch &middot; and a control-vs-amphetamine phenotype.

Each session ships the **maps Ugne already computed** &mdash; an individual action space
(`cz_action`) and a **social/joint space** (`sz_joint`), each with coarse behavior labels &mdash;
plus both animals' raw 3-D keypoints. So we analyze the real spaces directly (no re-embedding) and
use the keypoints for the geometric choices.

**Run time:** ~5 min.
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
from scipy.stats import mannwhitneyu
import motionmapperpy as mmpy
%matplotlib inline
print("ready")
"""))

# ---------------------------------------------------------------- data / choice 0
cells.append(md("# 2.&nbsp; Choice 0 &mdash; work from pose, not video"))
cells.append(md(r"""
The first choice is forced by what's tractable: the 3-D tracking (s-DANNCE) is too heavy for Colab,
so we start from **pre-tracked 3-D poses** &mdash; here **23 body landmarks** per rat. We load two
files: the **precomputed maps + labels** for all 45 dyads and 30 lone sessions (small), and the
**raw keypoints** for two example dyads (one control, one amphetamine).

Each dyad has a drug code per animal: **0 = control (saline), 1 = amphetamine, 2 = the (saline)
cage-mate of an amph rat**. A *control dyad* is two saline animals; an *amph dyad* pairs a 1 with a 2.
"""))
cells.append(code(r"""
RAT_BASE = ("https://raw.githubusercontent.com/bermanlabemory/"
            "unsupervised-behavior-tutorials/main/data/rat_data/")
def fetch(fn):
    if not os.path.exists(fn):
        urllib.request.urlretrieve(RAT_BASE + fn, fn)
    return fn

d = np.load(fetch("rat_social.npz"), allow_pickle=True)
kp = np.load(fetch("rat_social_keypoints.npz"), allow_pickle=True)

soc_cz, soc_sz = d["soc_cz"], d["soc_sz"]                       # individual + joint embeddings (per dyad)
hlac, part_hlac, hljc = d["soc_hlac"], d["soc_part_hlac"], d["soc_hljc"]   # coarse labels
amph, amphP = d["soc_amph"], d["soc_amphP"]
is_amph_dyad = (amph > 0) | (amphP > 0)                         # control dyad = both saline
print("%d dyads (%d control + %d amph) + %d lone sessions, %d frames each @ %d Hz"
      % (len(soc_cz), (~is_amph_dyad).sum(), is_amph_dyad.sum(), len(d["lone_cz"]),
         soc_cz.shape[1], int(d["fps"])))

# the 23-keypoint sDANNCE rat skeleton (read off the data; standard sDANNCE order)
JOINTS = ["snout", "earL", "earR", "spineF", "spineM", "spineL", "tailbase",
          "shoulderL", "elbowL", "wristL", "forepawL", "shoulderR", "elbowR", "wristR", "forepawR",
          "hipL", "kneeL", "ankleL", "hindpawL", "hipR", "kneeR", "ankleR", "hindpawR"]
EDGES = [(0, 3), (1, 3), (2, 3), (3, 4), (4, 5), (5, 6),
         (3, 7), (7, 8), (8, 9), (9, 10), (3, 11), (11, 12), (12, 13), (13, 14),
         (5, 15), (15, 16), (16, 17), (17, 18), (5, 19), (19, 20), (20, 21), (21, 22)]
SNOUT, SPINE_MID, TAILBASE = 0, 4, 6
"""))

cells.append(md("## 2.1&nbsp; Always look first"))
cells.append(md("A few frames of the two tracked rats (control dyad), chosen when they're far apart vs close together:"))
cells.append(code(r"""
A, B = kp["clip_m1"], kp["clip_m2"]              # (Tclip, 23, 3) full-rate clip of one dyad
sep = np.linalg.norm(A.mean(1) - B.mean(1), axis=1)         # inter-animal distance per frame
picks = [int(np.argmax(sep)), int(np.argsort(sep)[len(sep)//2]), int(np.argmin(sep))]
fig = plt.figure(figsize=(15, 5))
for j, fr in enumerate(picks):
    ax = fig.add_subplot(1, 3, j + 1, projection="3d")
    for P, col in [(A[fr], "firebrick"), (B[fr], "royalblue")]:
        ax.scatter(P[:, 0], P[:, 1], P[:, 2], color=col, s=12)
        for a, b in EDGES: ax.plot(*P[[a, b]].T, color=col, lw=1.2)
    ax.set_title("frame %d  (apart=%.0fmm)" % (fr, sep[fr])); ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    ax.view_init(elev=20, azim=-70)
plt.suptitle("two rats, real 3-D keypoints"); plt.show()
"""))

# ---------------------------------------------------------------- choice 1
cells.append(md("# 3.&nbsp; Choice 1 &mdash; is the individual enough?"))
cells.append(md(r"""
Before building anything social, ask whether you need to. Ugne's **individual** action map
(`cz_action`) places each animal's own posture-dynamics in 2-D. Compare where animals spend time
when **alone** vs **in company**: if the map shifts, a one-animal description is missing something
&mdash; and that something is what the rest of the notebook has to capture.
"""))
cells.append(code(r"""
def density(z, R, sigma=1.5):
    return mmpy.findPointDensity(z.reshape(-1, 2), sigma, 301, [-R, R])[2]

lone_cz, soc_cz_ = d["lone_cz"], soc_cz
Ri = float(np.abs(np.concatenate([lone_cz.reshape(-1, 2), soc_cz_.reshape(-1, 2)])).max() + 5)
Dl, Ds = density(lone_cz, Ri), density(soc_cz_, Ri)
ext = (-Ri, Ri, -Ri, Ri)
fig, ax = plt.subplots(1, 3, figsize=(15, 5))
ax[0].imshow(Dl, extent=ext, origin="lower", cmap=mmpy.gencmap()); ax[0].set_title("alone"); ax[0].axis("off")
ax[1].imshow(Ds, extent=ext, origin="lower", cmap=mmpy.gencmap()); ax[1].set_title("in company"); ax[1].axis("off")
diff = Ds - Dl; v = np.abs(diff).max()
im = ax[2].imshow(diff, extent=ext, origin="lower", cmap="RdBu_r", vmin=-v, vmax=v)
ax[2].set_title("social - alone"); ax[2].axis("off"); plt.colorbar(im, ax=ax[2], fraction=0.046)
plt.suptitle("an individual's own behavior shifts when a partner is present"); plt.show()
"""))
cells.append(md(r"""
The individual map **moves** between alone and social &mdash; context matters &mdash; but a one-animal
map still can't say *what the pair is doing*. That motivates everything below.
"""))

# ---------------------------------------------------------------- choice 2
cells.append(md("# 4.&nbsp; Choice 2 &mdash; which social variables?"))
cells.append(md(r"""
What makes behavior *social* is the relationship **between** the animals. You could measure many
inter-animal quantities; the two simplest and most informative (Klibaite 2025, Fig 4B) are the
**inter-animal distance** and the **relative orientation** &mdash; is one animal facing the other?
Even defining these takes choices: we use each rat's **body centroid** for position and the
**spine&rarr;snout** vector for heading.
"""))
cells.append(code(r"""
def body_center(P): return P.mean(1)                              # centroid of the 23 keypoints
def heading(P):     return P[:, SNOUT, :2] - P[:, SPINE_MID, :2]  # facing direction (xy)

A, B = kp["m1"][0], kp["m2"][0]                                   # one example dyad, downsampled
cA, cB = body_center(A)[:, :2], body_center(B)[:, :2]
dist = np.linalg.norm(cA - cB, axis=1)
hA = heading(A); to_B = cB - cA
bearing = np.arctan2(hA[:, 0]*to_B[:, 1] - hA[:, 1]*to_B[:, 0], (hA*to_B).sum(1))  # 0 = A faces B
t = np.arange(len(dist)) / int(d["fps"])
fig, ax = plt.subplots(2, 1, figsize=(13, 4), sharex=True)
ax[0].plot(t, dist, "k"); ax[0].set_ylabel("distance (mm)")
ax[0].axhline(np.median(dist), color="orange", ls="--", lw=1, label="median")
ax[1].plot(t, np.abs(np.degrees(bearing)), "purple"); ax[1].set_ylabel("|facing angle| (deg)")
ax[1].set_xlabel("time (s)"); ax[0].legend(); ax[0].set_title("inter-animal distance & orientation (one dyad)"); plt.show()
"""))

# ---------------------------------------------------------------- choice 3
cells.append(md("# 5.&nbsp; Choice 3 &mdash; the *joint* behavior map"))
cells.append(md(r"""
The key step. Instead of two separate animals, build **one** representation of the *pair* &mdash;
both animals' posture dynamics **plus** the social variables &mdash; and embed that. Ugne's
`sz_joint` is exactly this **social/joint map**: each point is a moment of *joint* behavior (mutual
rearing, inspect-from-behind, chasing, ...). Here it is, with its coarse joint-behavior classes
(`hljc`) drawn on top.
"""))
cells.append(code(r"""
Rs = float(np.abs(soc_sz.reshape(-1, 2)).max() + 5)
Dj = density(soc_sz, Rs); exts = (-Rs, Rs, -Rs, Rs)
allsz = soc_sz.reshape(-1, 2); allhljc = hljc.reshape(-1)
m = allhljc >= 1                                  # drop unassigned (-1/0)
fig, ax = plt.subplots(1, 2, figsize=(12, 5.5))
ax[0].imshow(Dj, extent=exts, origin="lower", cmap=mmpy.gencmap())
ax[0].set_title("joint (social) behavior map: sz_joint"); ax[0].axis("off")
samp = np.random.default_rng(0).choice(np.where(m)[0], 40000, replace=False)
sc = ax[1].scatter(allsz[samp, 0], allsz[samp, 1], c=allhljc[samp], cmap="tab10", s=2)
ax[1].set_xlim(-Rs, Rs); ax[1].set_ylim(-Rs, Rs); ax[1].set_aspect("equal"); ax[1].axis("off")
ax[1].set_title("its coarse joint-behavior classes (hljc)"); plt.show()
"""))

# ---------------------------------------------------------------- choice 4
cells.append(md("# 6.&nbsp; Choice 4 &mdash; how to measure synchrony"))
cells.append(md(r"""
Are the partners **coordinated**? One choice: take each animal's own coarse behavior and ask, for
every pair of behaviors, whether the two animals are in those states *together* more (or less) than
chance &mdash; the co-occurrence enrichment of Klibaite 2025 (Fig 3F). A bright **diagonal** means
they tend to do the **same** thing at the same time.
"""))
cells.append(code(r"""
NI = int(d["n_hlac"])
co = np.zeros((NI, NI))
for a, b in zip(hlac.reshape(-1), part_hlac.reshape(-1)):
    if 1 <= a <= NI and 1 <= b <= NI:
        co[a - 1, b - 1] += 1
P = co / co.sum()
enrich = np.log2((P + 1e-9) / ((P.sum(1, keepdims=True) @ P.sum(0, keepdims=True)) + 1e-9))
diag = np.mean(np.diag(enrich))
fig, ax = plt.subplots(figsize=(5.5, 5))
v = np.abs(enrich).max()
im = ax.imshow(enrich, cmap="RdBu_r", vmin=-v, vmax=v)
ax.set_xlabel("partner's behavior"); ax.set_ylabel("focal animal's behavior")
ax.set_title("synchrony: log2(observed / chance)\nmean diagonal = +%.2f bits" % diag)
plt.colorbar(im, fraction=0.046); plt.show()
"""))
cells.append(md("Partners do the **same** coarse behavior at the same time well above chance (positive diagonal) &mdash; behavioral synchrony, straight from the labels."))

# ---------------------------------------------------------------- choice 5
cells.append(md("# 7.&nbsp; Choice 5 &mdash; defining touch"))
cells.append(md(r"""
The paper fits a body **mesh** to detect contact &mdash; the most faithful choice, too heavy here. A
lighter one: call it touch when **keypoints of the two animals come within a threshold**, binned by
body region. Both the threshold and the regions are choices that change what "touch" means. We pool
the two example dyads' keypoints into a **tactogram**.
"""))
cells.append(code(r"""
REGIONS = {"head": [0, 1, 2, 3], "trunk": [4, 5, 6],
           "forelimbs": [7, 8, 9, 10, 11, 12, 13, 14], "hindlimbs": [15, 16, 17, 18, 19, 20, 21, 22]}
names = list(REGIONS); THRESH = 40.0    # mm
tact = np.zeros((len(names), len(names)))
for di in range(kp["m1"].shape[0]):                              # the example dyads
    A, B = kp["m1"][di], kp["m2"][di]
    for fr in range(0, len(A), 2):
        Dpair = np.linalg.norm(A[fr][:, None] - B[fr][None], axis=2)   # (23,23) keypoint distances
        for i, ri in enumerate(names):
            for j, rj in enumerate(names):
                if Dpair[np.ix_(REGIONS[ri], REGIONS[rj])].min() < THRESH:
                    tact[i, j] += 1
fig, ax = plt.subplots(figsize=(5.5, 5))
im = ax.imshow(np.log1p(tact), cmap="magma")
ax.set_xticks(range(len(names))); ax.set_xticklabels(names, rotation=30, ha="right")
ax.set_yticks(range(len(names))); ax.set_yticklabels(names)
ax.set_xlabel("rat B"); ax.set_ylabel("rat A"); ax.set_title("tactogram: where do they touch? (<%dmm)" % THRESH)
plt.colorbar(im, fraction=0.046); plt.show()
"""))

# ---------------------------------------------------------------- phenotype
cells.append(md("# 8.&nbsp; Putting the choices to work: control vs amphetamine"))
cells.append(md(r"""
The payoff. Summarize each dyad by how often it visits each **joint** behavior class (`hljc`) and
compare **control dyads** (both saline) with **amphetamine dyads** &mdash; a social phenotype, at the
level of the pair.
"""))
cells.append(code(r"""
def occ(h):
    v = h[h >= 1]; b = np.bincount(v, minlength=int(d["n_hljc"]) + 1)[1:int(d["n_hljc"]) + 1].astype(float)
    return b / max(b.sum(), 1)
O = np.array([occ(hljc[i]) for i in range(len(hljc))])
fold = np.log2((O[is_amph_dyad].mean(0) + 1e-3) / (O[~is_amph_dyad].mean(0) + 1e-3))

def js(p, q):
    p = p + 1e-12; q = q + 1e-12; p /= p.sum(); q /= q.sum(); m = 0.5 * (p + q)
    kl = lambda x, y: np.sum(x * np.log2(x / y)); return 0.5 * kl(p, m) + 0.5 * kl(q, m)
ref = O[~is_amph_dyad].mean(0)                                   # mean control profile
jsc = np.array([js(O[i], ref) for i in np.where(~is_amph_dyad)[0]])
jsa = np.array([js(O[i], ref) for i in np.where(is_amph_dyad)[0]])
pval = mannwhitneyu(jsc, jsa).pvalue

fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
ax[0].bar(range(1, len(fold) + 1), fold, color=["firebrick" if f > 0 else "royalblue" for f in fold])
ax[0].axhline(0, color="k", lw=.5); ax[0].set_xlabel("joint-behavior class (hljc)")
ax[0].set_ylabel("log2 fold (amph / control)"); ax[0].set_title("which joint behaviors change?")
ax[1].bar([0, 1], [jsc.mean(), jsa.mean()], color=["royalblue", "firebrick"], alpha=.85)
ax[1].plot(np.zeros(len(jsc)), jsc, "ko"); ax[1].plot(np.ones(len(jsa)), jsa, "ko")
ax[1].set_xticks([0, 1]); ax[1].set_xticklabels(["control\ndyads", "amph\ndyads"])
ax[1].set_ylabel("JS distance from mean control profile")
ax[1].set_title("amph dyads differ from controls\n(Mann-Whitney p = %.4f)" % pval); plt.show()
"""))
cells.append(md(r"""
Amphetamine measurably reshapes **joint** behavior &mdash; specific social-behavior classes go up and
others down, and amph dyads sit significantly further from the control profile. No social behavior
was ever hand-defined; it all falls out of the joint map.
"""))

# ---------------------------------------------------------------- exercises
cells.append(md(r"""
# 9.&nbsp; 🔧 Your turn

1. **Change what "touch" means (&sect;7).** Halve `THRESH`, or build the tactogram from only the
   `head` rows. Does the contact pattern still look the same?
2. **Asymmetric amphetamine.** In an amph dyad one rat got the drug (`amph==1`) and one didn't
   (`amph==2`). Split the *individual* map (&sect;3) by `amph` &mdash; does the injected rat's own
   behavior shift more than its sober partner's?
3. **Finer synchrony (&sect;6).** Swap the coarse labels (`hlac`) for the fine ones (`llac` /
   `part_llac`) &mdash; does the diagonal sharpen or wash out?
4. **Distance as a social axis.** Using the keypoints, compute each dyad's median inter-animal
   distance and ask whether amph dyads stay closer or farther apart.

The slow-modes idea (`05_slow_modes.ipynb`) extends here too: is there a *slow drift* in how engaged
a pair is across a 30-minute session?
"""))

write_nb(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                      "04_rat_social_behavior.ipynb"), cells)
