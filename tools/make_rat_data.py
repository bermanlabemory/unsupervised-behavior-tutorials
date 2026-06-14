"""Build the compact rat data shipped in data/rat_data/ for the rat notebooks, from Ugne's
tutorial-data-uk folder (single-animal DANNCE + MotionMapper outputs).

Two outputs:
  amph.npz  -- the amphetamine experiment (analyze_amph_data.m): 6 rats x 3 days, precomputed
               2-D MotionMapper embeddings + fine/coarse watershed cluster labels. Days 2 & 3 are
               baseline; day 4 is amphetamine. All 18 (animal,day) embeddings share ONE map space.
  rat_keypoints_session1.npz -- one example session's 3-D keypoints (DANNCE, 20 joints @ 50 Hz) +
               skeleton + that session's own behavioral-map embedding & coarse labels, for the
               "look at the raw data / rat front-end" intro.

Usage:  python tools/make_rat_data.py
"""
import os, numpy as np, scipy.io as sio

SRC = os.path.expanduser("~/Library/CloudStorage/Dropbox/tutorial-data-uk (1)")
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "rat_data")
DOWNSAMPLE = 2                      # 50 Hz -> 25 Hz for the amph embeddings (keeps files lean)


def build_amph():
    d = sio.loadmat(os.path.join(SRC, "amph_data.mat"))
    days = [2, 3, 4]
    def stack(prefix, cast):
        return np.stack([np.stack([cast(c) for c in d["%s_day%d" % (prefix, dd)].ravel()]) for dd in days])
    emb = stack("embed", lambda c: c.astype(np.float32)[::DOWNSAMPLE])           # (3, 6, T, 2)
    clust = stack("clust", lambda c: c.ravel().astype(np.uint8)[::DOWNSAMPLE])    # (3, 6, T) fine
    cclust = stack("cclust", lambda c: c.ravel().astype(np.uint8)[::DOWNSAMPLE])  # (3, 6, T) coarse
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, "amph.npz")
    np.savez_compressed(path, emb=emb, clust=clust, cclust=cclust, days=np.array(days),
                        amph_day=4, fps=50 // DOWNSAMPLE,
                        n_fine=int(clust.max()), n_coarse=int(cclust.max()))
    print("wrote %s  (%.2f MB)  emb%s  fine=%d coarse=%d  fps=%d"
          % (path, os.path.getsize(path) / 1e6, emb.shape, clust.max(), cclust.max(), 50 // DOWNSAMPLE))


def build_keypoints(start=12000, n=750):
    dn = sio.loadmat(os.path.join(SRC, "L1_4_20210323_dannce.mat"))
    pred = dn["pred"].astype(np.float32)                  # (frames, 3, 20) = (frames, xyz, joints)
    kp = np.transpose(pred, (0, 2, 1))                    # -> (frames, 20, 3)
    clip = kp[start:start + n]
    sk = sio.loadmat(os.path.join(SRC, "jesse_skeleton.mat"))
    names = np.array([str(x[0]) for x in sk["joint_names"].ravel()])
    edges = sk["joints_idx"].astype(int) - 1             # MATLAB 1-based -> 0-based
    emb = sio.loadmat(os.path.join(SRC, "zvals_session1.mat"))["zvalues1"].astype(np.float32)
    coarse = sio.loadmat(os.path.join(SRC, "data_for_sesssion1.mat"))["coarse_labels_1"].ravel().astype(np.uint8)
    bl = sio.loadmat(os.path.join(SRC, "new_behavior_list.mat"))["new_behavior_list"].ravel()
    coarse_names = np.array([str(x[0]) for x in bl])
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, "rat_keypoints_session1.npz")
    np.savez_compressed(path, kp_clip=clip, clip_start=start, joint_names=names, edges=edges,
                        emb=emb, coarse_labels=coarse, coarse_names=coarse_names, fps=50)
    print("wrote %s  (%.2f MB)  clip%s nan%%=%.2f  emb%s  coarse_names=%d"
          % (path, os.path.getsize(path) / 1e6, clip.shape,
             100 * np.isnan(clip).mean(), emb.shape, len(coarse_names)))


if __name__ == "__main__":
    build_amph()
    build_keypoints()
