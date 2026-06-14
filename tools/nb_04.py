"""04 — Rat social behavior: the choices you make to quantify what two animals do together.
Walks through the modeling decisions behind Klibaite et al. 2025, on a synthetic two-rat stand-in
(real dyad poses swap in later -- everything downstream runs on real poses unchanged)."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nb_builder import md, code, badge, write_nb

REPO = "bermanlabemory/unsupervised-behavior-tutorials/blob/main"
cells = []

cells.append(badge("%s/04_rat_social_behavior.ipynb" % REPO))

cells.append(md(r"""
# 4.&nbsp; Rat social behavior: quantifying what two animals do *together*

In notebook 03 a rat's behavior was its own pose over time. Put **two** animals in a box and
"behavior" stops being a property of one body &mdash; it's about the *relationship* between them. There
is no single right way to measure that; you face a sequence of **choices**, and the answers you get
depend on them. This notebook walks through those choices, following Klibaite et al., *Cell* 2025
("Mapping the landscape of social behavior"):

> the unit of analysis &middot; which social variables &middot; how to build a *joint* representation
> &middot; how to measure synchrony &middot; how to define touch.

We use a **synthetic two-rat stand-in** so everything runs today and the *choices* (not the data
wrangling) stay in focus. The real CTRL+amphetamine dyad poses will drop in later &mdash; the code
downstream is written to run on real poses unchanged.

**Run time:** ~8&ndash;12 min.
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

import numpy as np, matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import umap
import motionmapperpy as mmpy
%matplotlib inline
rng = np.random.default_rng(0)
print("ready")
"""))

# ---------------------------------------------------------------- data / choice 0
cells.append(md("# 2.&nbsp; Choice 0 &mdash; work from pose, not video"))
cells.append(md(r"""
The first choice is already made for us by what's tractable: the 3-D tracking (s-DANNCE) and the
mesh-based touch detection are too heavy for Colab, so we start from **pre-tracked 3-D poses**
&mdash; exactly the data you'd have after tracking. Each **session** has two rats, each tracked as 8
3-D keypoints over time (snout, head, neck, mid-spine, tail-base, two hips, tail-tip). Some sessions
are **lone** (one rat), some **social** (a pair); half the animals are **CTRL**, half **AMPH**.

> **Swapping in real data:** set `USE_SYNTHETIC_DATA = False` and point `DATA_URL` at a real
> CTRL+AMPH dyad subset (same dict structure). Everything below is written to run unchanged.
"""))
cells.append(code(r"""
USE_SYNTHETIC_DATA = True
DATA_URL = "https://PLACEHOLDER-HOST/klibaite_rat_dyads_subset.npz"   # TODO: real link

EDGES = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 7), (4, 5), (4, 6)]
CANON = np.array([[4, 0, 1.0], [3, 0, 1.1], [2, 0, 1.0], [0, 0, 1.0],
                  [-2, 0, .9], [-1.8, .8, .7], [-1.8, -.8, .7], [-3.5, 0, .6]])

def _poses(T, behav, head, center, rear, groom, fps):
    K = len(CANON); P = np.tile(CANON, (T, 1, 1)).astype(float)
    front = [0, 1, 2, 3]; lift = np.array([1.0, .8, .5, .2])
    P[:, front, 2] += rear[:, None] * lift                      # rearing lifts the front up
    g = (behav == 3)                                            # grooming: fast head wobble
    P[:, 0, 2] += np.where(g, .6 * np.sin(groom), 0)
    P[:, 1, 2] += np.where(g, .4 * np.sin(groom), 0)
    P[:, :, 2] += .1 * np.sin(2 * np.pi * 2 * np.arange(T)[:, None] / fps) * (behav == 1)[:, None]
    c, s = np.cos(head), np.sin(head)                           # rotate by heading about z
    x, y = P[:, :, 0].copy(), P[:, :, 1].copy()
    P[:, :, 0] = x * c[:, None] - y * s[:, None] + center[:, None, 0]
    P[:, :, 1] = x * s[:, None] + y * c[:, None] + center[:, None, 1]
    return P

def _walk(T, fps, amph, seed):
    r = np.random.default_rng(seed)
    # behavior Markov chain: idle, locomote, rear, groom (amph -> more locomotion)
    Tm = np.array([[.6, .2, .1, .1], [.15, .7, .1, .05], [.2, .2, .55, .05], [.2, .15, .05, .6]])
    if amph: Tm[:, 1] += .15; Tm /= Tm.sum(1, keepdims=True)
    b = np.zeros(T, int)
    for t in range(1, T): b[t] = r.choice(4, p=Tm[b[t-1]])
    head = np.cumsum(r.normal(0, .15, T)) + r.uniform(0, 6)
    center = np.cumsum(r.normal(0, .25 + .15*amph, (T, 2)), 0)
    center = np.clip(center, -18, 18)
    rear = np.clip(np.where(b == 2, 1.0, 0.0) + r.normal(0, .05, T), 0, 1.2)
    groom = np.cumsum(np.full(T, 2*np.pi*5/fps))
    return b, head, center, rear, groom

def make_synthetic(fps=30, T=2400, n_lone=3, n_social=5, seed=0):
    sessions = []
    sid = 0
    for group, amph in [("CTRL", 0), ("AMPH", 1)]:
        for _ in range(n_lone):
            b, h, c, re, g = _walk(T, fps, amph, seed + sid); sid += 1
            sessions.append(dict(group=group, kind="lone", fps=fps,
                                 A=_poses(T, b, h, c, re, g, fps), B=None, engaged=np.zeros(T, bool)))
        for _ in range(n_social):
            bA, hA, cA, rA, gA = _walk(T, fps, amph, seed + sid); sid += 1
            bB, hB, cB, rB, gB = _walk(T, fps, amph, seed + sid); sid += 1
            r = np.random.default_rng(seed + sid)
            eng = np.zeros(T, bool); p_on = .02 + .02*amph        # amph -> more engagement
            for t in range(1, T):
                eng[t] = r.random() < (.95 if eng[t-1] else p_on) if eng[t-1] else r.random() < p_on
            for t in np.where(eng)[0]:                            # when engaged: get close + face partner
                off = r.normal(0, 1.5, 2); cB[t] = cA[t] + off + 3 * off / (np.linalg.norm(off)+1e-6)
                hA[t] = np.arctan2(*(cB[t]-cA[t])[::-1]); hB[t] = np.arctan2(*(cA[t]-cB[t])[::-1])
                if r.random() < .3: rA[t] = rB[t] = 1.0          # mutual rearing
            sessions.append(dict(group=group, kind="social", fps=fps,
                                 A=_poses(T, bA, hA, cA, rA, gA, fps),
                                 B=_poses(T, bB, hB, cB, rB, gB, fps), engaged=eng))
    return sessions

if USE_SYNTHETIC_DATA:
    sessions = make_synthetic()
else:
    d = np.load(DATA_URL.split("/")[-1], allow_pickle=True)     # expects same dict structure
    sessions = list(d["sessions"])
print("%d sessions (%d social)" % (len(sessions), sum(s["kind"] == "social" for s in sessions)))
"""))

cells.append(md("## 2.1&nbsp; Always look first"))
cells.append(code(r"""
sess = next(s for s in sessions if s["kind"] == "social")
t = int(np.where(sess["engaged"])[0][len(np.where(sess["engaged"])[0])//2])  # an engaged moment
fig = plt.figure(figsize=(6, 6)); ax = fig.add_subplot(111, projection="3d")
for P, col in [(sess["A"], "firebrick"), (sess["B"], "royalblue")]:
    pts = P[t]
    ax.scatter(*pts.T, color=col, s=20)
    for i, j in EDGES: ax.plot(*pts[[i, j]].T, color=col, lw=1.5)
ax.set_title("two rats, engaged (frame %d)" % t); ax.set_zlim(0, 6)
ax.view_init(elev=20, azim=-60); plt.show()
"""))

# ---------------------------------------------------------------- choice 1
cells.append(md("# 3.&nbsp; Choice 1 &mdash; is the individual enough?"))
cells.append(md(r"""
Before building anything social, ask whether you even need to. Map each animal **on its own** (the
same engine as notebook 01 &mdash; egocenter the pose, take wavelets, embed) and ask: does an animal
behave **differently when a partner is present**? If yes, the individual map is *not* capturing
something, and that something is what we have to model.
"""))
cells.append(code(r"""
def egocenter(P):                       # P:(T,K,3) -> egocentric (tail-base at origin, snout +x)
    Q = P - P[:, 4:5, :]
    th = np.arctan2(Q[:, 0, 1], Q[:, 0, 0])
    c, s = np.cos(-th), np.sin(-th)
    x, y = Q[:, :, 0].copy(), Q[:, :, 1].copy()
    Q[:, :, 0], Q[:, :, 1] = x*c[:, None]-y*s[:, None], x*s[:, None]+y*c[:, None]
    return Q.reshape(len(P), -1)        # (T, 24)

def wavelet_amps(stream, fps):                       # log wavelet amplitudes for one animal stream
    w, _ = mmpy.findWavelets(stream, stream.shape[1], 5, 20, fps, 10.0, 0.5, -1, -1)
    return np.log(w + 1e-3)

# collect each animal stream, tagged lone/social and by group
streams, tags = [], []
for s in sessions:
    for who in ["A", "B"]:
        if s[who] is None: continue
        streams.append(egocenter(s[who])); tags.append((s["group"], s["kind"]))
amps = [wavelet_amps(st, 30) for st in streams]
PCA_W = PCA(6).fit(np.vstack(amps))                  # ONE shared feature basis for every map below
W = [PCA_W.transform(a) for a in amps]
allW = np.vstack(W)
sub = rng.choice(len(allW), min(8000, len(allW)), replace=False)
ireducer = umap.UMAP(n_neighbors=30, min_dist=0.1, random_state=0).fit(allW[sub])
emb = [ireducer.transform(w) for w in W]
print("individual maps built for %d animal-streams" % len(W))
"""))
cells.append(code(r"""
allemb = np.vstack(emb); R = np.abs(allemb).max() + 2
def dens(e): return mmpy.findPointDensity(e, 1.0, 101, [-R, R])[2]
lone = np.vstack([e for e, t in zip(emb, tags) if t[1] == "lone"])
soc  = np.vstack([e for e, t in zip(emb, tags) if t[1] == "social"])

fig, ax = plt.subplots(1, 3, figsize=(15, 5))
for a, (e, t) in zip(ax, [(lone, "alone"), (soc, "social"), (None, "social - alone")]):
    if e is not None:
        a.imshow(dens(e), extent=(-R, R, -R, R), origin="lower", cmap=mmpy.gencmap()); a.set_title(t)
    else:
        d = dens(soc) - dens(lone); v = np.abs(d).max()
        im = a.imshow(d, extent=(-R, R, -R, R), origin="lower", cmap="RdBu_r", vmin=-v, vmax=v)
        a.set_title(t); plt.colorbar(im, ax=a, fraction=0.046)
    a.axis("off")
plt.suptitle("an individual's behavior shifts when a partner is present"); plt.show()
"""))
cells.append(md(r"""
The individual map *moves* between alone and social &mdash; so social context matters, but a one-animal
map can't say **what the pair is doing**. That motivates everything below.
"""))

# ---------------------------------------------------------------- choice 2
cells.append(md("# 4.&nbsp; Choice 2 &mdash; which social variables?"))
cells.append(md(r"""
What makes behavior *social* is the relationship **between** the animals. You could measure dozens of
inter-animal quantities; the two simplest and most informative (Klibaite 2025, Fig 4B) are the
**inter-animal distance** and the **relative orientation** &mdash; is one animal facing the other?
The choice of *which* variables, and how to define a "body position" and "heading" from keypoints, is
yours &mdash; here body = tail-base, heading = tail-base&rarr;snout.
"""))
cells.append(code(r"""
def social_features(A, B):
    cA, cB = A[:, 4, :2], B[:, 4, :2]          # tail-base xy as body position
    d = np.linalg.norm(cA - cB, axis=1)
    headA = np.arctan2(*(A[:, 0, :2] - A[:, 4, :2]).T[::-1])
    to_B = np.arctan2(*(cB - cA).T[::-1])
    rel = np.arctan2(np.sin(to_B - headA), np.cos(to_B - headA))   # 0 = A faces B
    return d, rel

s = next(s for s in sessions if s["kind"] == "social")
d, rel = social_features(s["A"], s["B"])
fig, ax = plt.subplots(2, 1, figsize=(13, 4), sharex=True)
ax[0].plot(d, "k"); ax[0].fill_between(np.arange(len(d)), 0, d.max(), where=s["engaged"], color="orange", alpha=.2)
ax[0].set_ylabel("distance"); ax[1].plot(np.abs(rel), "purple"); ax[1].set_ylabel("|facing angle|")
ax[1].set_xlabel("frame"); ax[0].set_title("inter-animal distance & orientation (orange = engaged)"); plt.show()
"""))

# ---------------------------------------------------------------- choice 3
cells.append(md("# 5.&nbsp; Choice 3 &mdash; build a *joint* representation"))
cells.append(md(r"""
Now the key choice. To get a map of **joint behaviors** &mdash; "both rearing, facing", "one inspecting
the other from behind" &mdash; build **one** feature vector per moment that contains *both* animals'
postural dynamics **plus** the social variables, and embed *that*. How you weight the social channels
against the postural ones is a real knob (we scale them up so they aren't drowned out).
"""))
cells.append(code(r"""
def dyadic_features(s):
    wA = PCA_W.transform(wavelet_amps(egocenter(s["A"]), s["fps"]))
    wB = PCA_W.transform(wavelet_amps(egocenter(s["B"]), s["fps"]))
    d, rel = social_features(s["A"], s["B"])
    soc = np.c_[np.exp(-d / 10), np.cos(rel), np.sin(rel)] * 3        # scaled social channels
    return np.c_[wA, wB, soc]

social_sessions = [s for s in sessions if s["kind"] == "social"]
F = [dyadic_features(s) for s in social_sessions]
allF = np.vstack(F)
sub = rng.choice(len(allF), min(8000, len(allF)), replace=False)
dreducer = umap.UMAP(n_neighbors=30, min_dist=0.1, random_state=1).fit(allF[sub])
demb = [dreducer.transform(f) for f in F]
Rd = np.abs(np.vstack(demb)).max() + 2

fig, ax = plt.subplots(figsize=(5.5, 5))
ax.imshow(mmpy.findPointDensity(np.vstack(demb), 1.0, 101, [-Rd, Rd])[2],
          extent=(-Rd, Rd, -Rd, Rd), origin="lower", cmap=mmpy.gencmap())
ax.set_title("joint (dyadic) behavior map"); ax.axis("off"); plt.show()
"""))
cells.append(md("Carve the joint map into joint-behavior classes (KMeans here; watershed in the full pipeline):"))
cells.append(code(r"""
NJ = 8
joint_km = KMeans(NJ, n_init=10, random_state=0).fit(np.vstack(demb))
jlabels = [joint_km.predict(e) for e in demb]
fig, ax = plt.subplots(figsize=(5.5, 5))
sc = ax.scatter(np.vstack(demb)[:, 0], np.vstack(demb)[:, 1], c=np.concatenate(jlabels), s=2, cmap="tab10")
ax.set_title("%d joint-behavior classes" % NJ); ax.axis("off"); plt.show()
print("In the real pipeline you'd now make region videos of each joint class to name them",
      "(mutual rear, inspect-from-behind, ...).")
"""))

# ---------------------------------------------------------------- choice 4
cells.append(md("# 6.&nbsp; Choice 4 &mdash; how to measure synchrony"))
cells.append(md(r"""
Are the partners' behaviors **coordinated**? One choice: give each animal a coarse individual label
and ask, for each pair of labels, whether the two animals are in those states *together* more (or
less) than chance &mdash; the co-occurrence enrichment of Klibaite 2025 (Fig 3F).
"""))
cells.append(code(r"""
NI = 6
ikm = KMeans(NI, n_init=10, random_state=0).fit(allW[sub])
def labels_of(P): return ikm.predict(PCA_W.transform(wavelet_amps(egocenter(P), 30)))

co = np.zeros((NI, NI))
for s in social_sessions:
    la, lb = labels_of(s["A"]), labels_of(s["B"])
    for a, b in zip(la, lb): co[a, b] += 1
P = co / co.sum()
exp = P.sum(1, keepdims=True) @ P.sum(0, keepdims=True)
enrich = np.log2((P + 1e-9) / (exp + 1e-9))            # >0 = co-occur above chance

fig, ax = plt.subplots(figsize=(5.5, 5))
v = np.abs(enrich).max()
im = ax.imshow(enrich, cmap="RdBu_r", vmin=-v, vmax=v)
ax.set_xlabel("rat B behavior"); ax.set_ylabel("rat A behavior")
ax.set_title("synchrony: log2(observed / chance) co-occurrence"); plt.colorbar(im, fraction=0.046); plt.show()
"""))
cells.append(md("A bright diagonal (if present) means partners tend to do the **same** thing at the same time &mdash; synchronized rearing, synchronized locomotion."))

# ---------------------------------------------------------------- choice 5
cells.append(md("# 7.&nbsp; Choice 5 &mdash; defining touch"))
cells.append(md(r"""
The paper fits a full body **mesh** to detect touch; that's the most faithful choice but too heavy
here. A lighter choice approximates contact as **keypoints of the two animals coming very close**,
binned by body region &mdash; a simple **tactogram** (Klibaite 2025, Fig 6). The threshold and the
keypoint set are choices that change what "touch" means.
"""))
cells.append(code(r"""
REGION = ["snout", "head", "neck", "spine", "tail-base", "hipL", "hipR", "tail"]
THRESH = 1.2
tact = np.zeros((8, 8))
for s in social_sessions:
    for t in range(0, len(s["A"]), 3):
        D = np.linalg.norm(s["A"][t][:, None] - s["B"][t][None], axis=2)  # (8,8) keypoint distances
        tact += D < THRESH
fig, ax = plt.subplots(figsize=(5.5, 5))
im = ax.imshow(np.log1p(tact), cmap="magma")
ax.set_xticks(range(8)); ax.set_xticklabels(REGION, rotation=90)
ax.set_yticks(range(8)); ax.set_yticklabels(REGION)
ax.set_xlabel("rat B"); ax.set_ylabel("rat A"); ax.set_title("tactogram: where do they touch?")
plt.colorbar(im, fraction=0.046); plt.show()
"""))

# ---------------------------------------------------------------- payoff
cells.append(md("# 8.&nbsp; Putting the choices to work: a social phenotype"))
cells.append(md(r"""
Once the choices are made, the payoff is the same as for individuals (notebook 03): summarize each
session by **how often it visits each joint-behavior class** and compare groups &mdash; here CTRL vs
AMPH, now at the level of the *pair*.
"""))
cells.append(code(r"""
occ, grp = [], []
for s, lab in zip(social_sessions, jlabels):
    h = np.bincount(lab, minlength=NJ).astype(float); occ.append(h / h.sum()); grp.append(s["group"])
occ = np.array(occ); grp = np.array(grp)

fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
fold = np.log2((occ[grp == "AMPH"].mean(0) + 1e-3) / (occ[grp == "CTRL"].mean(0) + 1e-3))
ax[0].bar(range(NJ), fold, color=["firebrick" if f > 0 else "royalblue" for f in fold])
ax[0].axhline(0, color="k", lw=.5); ax[0].set_xlabel("joint-behavior class")
ax[0].set_ylabel("log2 fold change (AMPH / CTRL)"); ax[0].set_title("which joint behaviors change?")

pc = PCA(2).fit_transform(occ)
for g, c in [("CTRL", "royalblue"), ("AMPH", "firebrick")]:
    ax[1].scatter(*pc[grp == g].T, color=c, label=g, s=60)
ax[1].set_xlabel("PC1"); ax[1].set_ylabel("PC2"); ax[1].legend()
ax[1].set_title("each dot = a session, in joint-repertoire space"); plt.show()
"""))

# ---------------------------------------------------------------- exercises
cells.append(md(r"""
# 9.&nbsp; 🔧 Your turn / where next

1. **Re-weight the social channels (§5).** The `* 3` on the social features sets how much the *pair's*
   geometry matters versus each animal's posture. Try `* 1` and `* 8` &mdash; how does the joint map
   reorganize? (This is the central modeling choice, laid bare.)
2. **Change what "touch" means (§7).** Halve `THRESH`, or build the tactogram only from snout/head
   rows. Does the contact pattern still separate the groups?
3. **Finer synchrony (§6).** Bump `NI` to 10. Does the co-occurrence structure sharpen or wash out?
4. **Swap in real data.** Set `USE_SYNTHETIC_DATA = False` once the real CTRL+AMPH dyad poses are
   wired in &mdash; every choice above then applies to real rats.

The slow-modes idea from `05_slow_modes.ipynb` extends here too: is there a *slow drift* in how
engaged a pair is over a session?
"""))

write_nb(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                      "04_rat_social_behavior.ipynb"), cells)
