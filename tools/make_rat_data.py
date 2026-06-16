"""Build the compact rat individual data shipped in data/rat_data/ for nb_03, from the
LE_CONTROL_AMPH 23-keypoint sDANNCE *lone* (`_L`) sessions (same cohort as the social nb_04, recorded
alone). Each session ships its precomputed individual action embedding (`cz_action`) + coarse/fine
watershed labels (`hlac`/`llac`); all sessions share ONE individual map. The amphetamine experiment is
6 rats, each recorded on consecutive days; the rats received amphetamine on 2022-10-20. We keep three
days for runtime -- two baseline (10-18, 10-19) and the amphetamine day (10-20) -- and label them
day 2, day 3, day 4 to match the notebook's "every animal is its own control across days" framing.

Two outputs:
  amph.npz  -- the amphetamine experiment: 6 rats x 3 days, precomputed 2-D individual embeddings
               (`cz_action`) + fine/coarse watershed labels. Days 2 & 3 are baseline; day 4 is
               amphetamine. All 18 (animal, day) embeddings share ONE map space.
  rat_keypoints_session1.npz -- one example lone session's 3-D keypoints (sDANNCE, 23 joints @ 50 Hz)
               + skeleton + that session's individual-map embedding & coarse labels, for the
               "look at the raw data / rat front-end" intro. This session is one of the 18 above, so
               it lives on the same shared map.

Usage:  python tools/make_rat_data.py
"""
import os, glob, numpy as np, scipy.io as sio

SRC = os.path.expanduser("~/Library/CloudStorage/Dropbox/LE_CONTROL_AMPH")
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "rat_data")
DOWNSAMPLE = 2                      # 50 Hz -> 25 Hz for the embeddings (keeps files lean)

RATS = ["M0", "M1", "M2", "M3", "M4", "M5"]
DATES = ["20221018", "20221019", "20221020"]    # -> notebook day 2, 3, 4
DAY_LABELS = [2, 3, 4]
AMPH_DATE = "20221020"             # amphetamine given this day (= day 4)

# The 23-keypoint sDANNCE rat skeleton (same order/edges as nb_04; standard sDANNCE).
JOINT_NAMES = ["snout", "earL", "earR", "spineF", "spineM", "spineL", "tailbase",
               "shoulderL", "elbowL", "wristL", "forepawL", "shoulderR", "elbowR", "wristR", "forepawR",
               "hipL", "kneeL", "ankleL", "hindpawL", "hipR", "kneeR", "ankleR", "hindpawR"]
EDGES = [(0, 3), (1, 3), (2, 3), (3, 4), (4, 5), (5, 6),
         (3, 7), (7, 8), (8, 9), (9, 10), (3, 11), (11, 12), (12, 13), (13, 14),
         (5, 15), (15, 16), (16, 17), (17, 18), (5, 19), (19, 20), (20, 21), (21, 22)]
# coarse-class names (two annotators watched skeleton clips of each class), classes 1..9
COARSE_NAMES = ["idle", "sniffing", "grooming", "scrunching", "active crouching",
                "rearing", "exploring", "locomotion", "fast"]


def load(rat, date):
    hits = glob.glob(os.path.join(SRC, "LONGEVANS_%s_%s_*_L.mat" % (rat, date)))
    assert len(hits) == 1, "expected one lone file for %s %s, got %s" % (rat, date, hits)
    return sio.loadmat(hits[0])["sdannce"][0, 0]


def build_amph():
    nday, nrat = len(DATES), len(RATS)
    s0 = load(RATS[0], DATES[0])
    T = np.asarray(s0["cz_action"])[::DOWNSAMPLE].shape[0]
    emb = np.zeros((nday, nrat, T, 2), np.float32)
    clust = np.zeros((nday, nrat, T), np.uint8)
    cclust = np.zeros((nday, nrat, T), np.uint8)
    nfine, ncoarse, nanpct = 0, 0, 0.0
    for k, date in enumerate(DATES):
        for a, rat in enumerate(RATS):
            s = load(rat, date)
            z = np.asarray(s["cz_action"], np.float32)[::DOWNSAMPLE]
            h = np.asarray(s["hlac"]).ravel()[::DOWNSAMPLE].astype(np.uint8)
            l = np.asarray(s["llac"]).ravel()[::DOWNSAMPLE].astype(np.uint8)
            emb[k, a], clust[k, a], cclust[k, a] = z[:T], l[:T], h[:T]
            nfine = max(nfine, int(l.max())); ncoarse = max(ncoarse, int(h.max()))
            nanpct = max(nanpct, 100 * np.isnan(z).mean())
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, "amph.npz")
    np.savez_compressed(path, emb=emb, clust=clust, cclust=cclust, days=np.array(DAY_LABELS),
                        amph_day=DAY_LABELS[DATES.index(AMPH_DATE)], fps=50 // DOWNSAMPLE,
                        n_fine=nfine, n_coarse=ncoarse)
    print("wrote %s  (%.2f MB)  emb%s  fine=%d coarse=%d  nan%%=%.3f  fps=%d"
          % (path, os.path.getsize(path) / 1e6, emb.shape, nfine, ncoarse, nanpct, 50 // DOWNSAMPLE))


def best_clip(P, h, n=750):
    """pick the n-frame window (stride n//3) with the most distinct coarse behaviors and least NaN."""
    valid = ~np.isnan(P).any((1, 2))
    best, bi = -1.0, 0
    for st in range(0, len(P) - n, n // 3):
        sl = slice(st, st + n)
        if valid[sl].mean() < 0.98:
            continue
        score = len(np.unique(h[sl])) + valid[sl].mean()
        if score > best:
            best, bi = score, st
    return bi


def build_keypoints(rat="M0", date="20221018"):
    s = load(rat, date)
    P = np.transpose(np.asarray(s["m1"], np.float32), (0, 2, 1))    # (T,3,23) -> (T,23,3)
    h = np.asarray(s["hlac"]).ravel()
    start = best_clip(P, h)
    clip = P[start:start + 750]                                     # 15 s @ 50 Hz
    z = np.asarray(s["cz_action"], np.float32)[::DOWNSAMPLE]        # this session on the shared map
    coarse = h[::DOWNSAMPLE].astype(np.uint8)
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, "rat_keypoints_session1.npz")
    np.savez_compressed(path, kp_clip=clip, clip_start=start,
                        joint_names=np.array(JOINT_NAMES), edges=np.array(EDGES),
                        emb=z, coarse_labels=coarse, coarse_names=np.array(COARSE_NAMES), fps=50)
    print("wrote %s  (%.2f MB)  clip%s nan%%=%.2f (start=%d)  emb%s  joints=%d"
          % (path, os.path.getsize(path) / 1e6, clip.shape,
             100 * np.isnan(clip).mean(), start, z.shape, clip.shape[1]))


if __name__ == "__main__":
    build_amph()
    build_keypoints()
