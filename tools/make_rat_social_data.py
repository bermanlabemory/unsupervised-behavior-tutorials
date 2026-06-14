"""Build the compact rat SOCIAL data shipped in data/rat_data/ for nb_04, from the LE_CONTROL_AMPH
sDANNCE files (each file = one rat's session; m1 = self, m2 = partner; precomputed individual map
`cz_action`+`llac`/`hlac` and social/joint map `sz_joint`+`lljc`/`hljc`; isamph 0=control,
1=amphetamine, 2=saline partner of an amph rat).

Outputs (data/rat_data/):
  rat_social.npz            -- individual + joint embeddings & labels for ALL dyads + lone sessions
                               (no keypoints; drives the maps, synchrony, and CTRL-vs-AMPH phenotype)
  rat_social_keypoints.npz  -- 23-joint 3-D keypoints (m1,m2) for a few example dyads + one short
                               full-rate clip, for the geometric social variables, touch, skeleton

Usage:  python tools/make_rat_social_data.py
"""
import os, re, glob, collections, numpy as np, scipy.io as sio

SRC = os.path.expanduser("~/Library/CloudStorage/Dropbox/LE_CONTROL_AMPH")
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "rat_data")
D = 10                      # 50 Hz -> 5 Hz for the shipped per-frame arrays
FPS = 50 // D
N_KP_DYADS = 2             # how many example dyads to keep keypoints for (1 control + 1 amph)


def _f(s, k):
    return np.asarray(s[k])


def _scalar(s, k, default=-1):
    v = np.asarray(s[k]).ravel()
    return int(v[0]) if v.size else default


def _str(s, k):
    v = np.asarray(s[k]).ravel()
    return str(v[0]) if v.size else ""


def dyad_files():
    """One file per social recording (dedupe the swapped pair); pick the lexicographically smaller
    ratid so 'self' (m1) is deterministic."""
    by_rec = collections.defaultdict(list)
    for f in glob.glob(os.path.join(SRC, "*_S.mat")):
        m = re.match(r"LONGEVANS_(M\d)_(\d{8})_(\d+)_S", os.path.basename(f))
        by_rec[(m.group(2), m.group(3))].append((m.group(1), f))
    return [sorted(v)[0][1] for v in by_rec.values()]


def ds(a):                  # downsample a per-frame array in time
    return a[::D]


def build():
    os.makedirs(OUT, exist_ok=True)
    soc = collections.defaultdict(list)
    dyads = sorted(dyad_files())
    print("loading %d social dyads ..." % len(dyads), flush=True)
    kp_pick = {"control": None, "amph": None}        # remember one file per condition for keypoints
    for i, f in enumerate(dyads):
        s = sio.loadmat(f)["sdannce"][0, 0]
        amph, amphP = _scalar(s, "isamph"), _scalar(s, "isamphP")
        soc["cz"].append(ds(_f(s, "cz_action")).astype(np.float32))
        soc["part_cz"].append(ds(_f(s, "part_cz_action")).astype(np.float32))
        soc["sz"].append(ds(_f(s, "sz_joint")).astype(np.float32))
        soc["hlac"].append(ds(_f(s, "hlac").ravel()).astype(np.uint8))
        soc["part_hlac"].append(ds(_f(s, "part_hlac").ravel()).astype(np.uint8))
        soc["llac"].append(ds(_f(s, "llac").ravel()).astype(np.uint8))
        soc["part_llac"].append(ds(_f(s, "part_llac").ravel()).astype(np.uint8))
        soc["hljc"].append(ds(_f(s, "hljc").ravel()).astype(np.int16))
        soc["lljc"].append(ds(_f(s, "lljc").ravel()).astype(np.int16))
        soc["amph"].append(amph); soc["amphP"].append(amphP)
        soc["id"].append(_str(s, "ratid")); soc["pid"].append(_str(s, "ratp_id"))
        soc["date"].append(_str(s, "ratdate"))
        cond = "control" if (amph == 0 and amphP == 0) else "amph"
        if kp_pick[cond] is None:
            kp_pick[cond] = f
        if (i + 1) % 10 == 0:
            print("  %d/%d" % (i + 1, len(dyads)), flush=True)

    lone = collections.defaultdict(list)
    lones = sorted(glob.glob(os.path.join(SRC, "*_L.mat")))
    print("loading %d lone sessions ..." % len(lones), flush=True)
    for f in lones:
        s = sio.loadmat(f)["sdannce"][0, 0]
        lone["cz"].append(ds(_f(s, "cz_action")).astype(np.float32))
        lone["hlac"].append(ds(_f(s, "hlac").ravel()).astype(np.uint8))
        lone["llac"].append(ds(_f(s, "llac").ravel()).astype(np.uint8))
        lone["amph"].append(_scalar(s, "isamph"))
        lone["id"].append(_str(s, "ratid")); lone["date"].append(_str(s, "ratdate"))

    nctrl = sum(1 for a, b in zip(soc["amph"], soc["amphP"]) if a == 0 and b == 0)
    np.savez_compressed(
        os.path.join(OUT, "rat_social.npz"),
        soc_cz=np.stack(soc["cz"]), soc_part_cz=np.stack(soc["part_cz"]), soc_sz=np.stack(soc["sz"]),
        soc_hlac=np.stack(soc["hlac"]), soc_part_hlac=np.stack(soc["part_hlac"]),
        soc_llac=np.stack(soc["llac"]), soc_part_llac=np.stack(soc["part_llac"]),
        soc_hljc=np.stack(soc["hljc"]), soc_lljc=np.stack(soc["lljc"]),
        soc_amph=np.array(soc["amph"]), soc_amphP=np.array(soc["amphP"]),
        soc_id=np.array(soc["id"]), soc_pid=np.array(soc["pid"]), soc_date=np.array(soc["date"]),
        lone_cz=np.stack(lone["cz"]), lone_hlac=np.stack(lone["hlac"]), lone_llac=np.stack(lone["llac"]),
        lone_amph=np.array(lone["amph"]), lone_id=np.array(lone["id"]), lone_date=np.array(lone["date"]),
        fps=FPS, n_hlac=9, n_hljc=8)
    p = os.path.join(OUT, "rat_social.npz")
    print("wrote %s (%.2f MB): %d dyads (%d control + %d amph), %d lone, T=%d @ %d Hz"
          % (p, os.path.getsize(p) / 1e6, len(dyads), nctrl, len(dyads) - nctrl,
             len(lones), np.stack(soc["cz"]).shape[1], FPS), flush=True)

    # keypoints for one control + one amph dyad, + one short full-rate clip for smooth skeleton viz
    km1, km2, ka, kaP, kid, kpid = [], [], [], [], [], []
    clip = {}
    for cond, f in kp_pick.items():
        if f is None:
            continue
        s = sio.loadmat(f)["sdannce"][0, 0]
        m1 = np.transpose(_f(s, "m1"), (0, 2, 1)).astype(np.float32)   # (T,3,23)->(T,23,3)
        m2 = np.transpose(_f(s, "m2"), (0, 2, 1)).astype(np.float32)
        km1.append(ds(m1)); km2.append(ds(m2))
        ka.append(_scalar(s, "isamph")); kaP.append(_scalar(s, "isamphP"))
        kid.append(_str(s, "ratid")); kpid.append(_str(s, "ratp_id"))
        if not clip:                                                    # 20 s @ 50 Hz from the first
            clip = dict(m1=m1[20000:21000], m2=m2[20000:21000])
    pk = os.path.join(OUT, "rat_social_keypoints.npz")
    np.savez_compressed(pk, m1=np.stack(km1), m2=np.stack(km2),
                        amph=np.array(ka), amphP=np.array(kaP), id=np.array(kid), pid=np.array(kpid),
                        clip_m1=clip["m1"], clip_m2=clip["m2"], clip_fps=50, fps=FPS)
    print("wrote %s (%.2f MB): %d example dyads, kp%s + clip%s"
          % (pk, os.path.getsize(pk) / 1e6, len(km1), np.stack(km1).shape, clip["m1"].shape), flush=True)


if __name__ == "__main__":
    build()
