"""Build the compact optogenetics .npz files shipped in data/optogenetic_data/ for nb_04, from
the Cande et al. 2018 example data (Fly_Optogenetic_Analysis/example_data/).

Each strain is one 12-fly filming session. For every camera we take the 2-D map embedding
(`zValues`, with out-of-convex-hull points replaced by `zGuesses` exactly as the MATLAB
`loadZValuesAndLEDs.m` does) and reconstruct a per-frame LED on/off trace from the strain's
`*_Frames.txt` (column 13 = LED, aligned to each camera's own frame counter). Cameras 1-6 are
controls (no all-trans-retinal); 7-12 are experimentals. Data are down-sampled 100 Hz -> 25 Hz and
stored as float32 to keep each file ~4 MB.

Usage:  python tools/make_optogenetic_data.py [STRAIN ...]      (default: ss02635 ss01049)

Output .npz keys: z (Ntot x 2 float32, all flies concatenated), fly_lengths (12,), led (Ntot uint8),
camera (12,), is_control (12, bool), fps (int), strain (str). Reconstruct per fly with
np.split(z, np.cumsum(fly_lengths)[:-1]).
"""
import os, numpy as np, scipy.io as sio

# Source: the MATLAB tutorial's example data (edit if your copy lives elsewhere).
SRC = os.path.expanduser(
    "~/Library/CloudStorage/Dropbox/teaching/Cajal_behavior/Fly_Optogenetic_Analysis/example_data")
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "data", "optogenetic_data")
FRAME_CAP = 181000          # max frame index used by the MATLAB pipeline
DOWNSAMPLE = 4              # 100 Hz -> 25 Hz
FPS = 100 // DOWNSAMPLE


def led_for_camera(frames, led_vals, cam, nframes):
    """Per-frame LED on/off for one camera (port of loadZValuesAndLEDs.m): map each LED-on block in
    the master clock to the camera's own frame range via the camera's frame-number column."""
    t = frames[:, cam - 1]
    pos = np.where(t > 0)[0]
    if len(pos) == 0:
        return np.zeros(nframes, bool)
    q = pos[np.argmin(t[pos])]
    idx = pos[pos >= q]
    on = (led_vals[idx] > 0).astype(int)
    d = np.diff(np.r_[0, on, 0])
    starts, ends = np.where(d == 1)[0], np.where(d == -1)[0] - 1
    led = np.zeros(nframes, bool)
    for s_, e_ in zip(starts, ends):
        f1 = int(max(t[idx[s_]], 1)); f2 = int(min(t[idx[e_]], FRAME_CAP))
        led[f1 - 1:f2] = True
    return led[:nframes]


def build_strain(strain):
    frames = np.genfromtxt(f"{SRC}/{strain}_Frames.txt", skip_header=1)
    led_vals = frames[:, 12]                                     # column 13 = LED state
    zs, leds, cams = [], [], []
    for cam in range(1, 13):
        m = sio.loadmat(f"{SRC}/{strain}_Cam{cam}_projections_tsne_embedding.mat")
        z = m["zValues"].astype(np.float64).copy()
        g = m["zGuesses"].astype(np.float64)
        ih = m["inConvHull"].astype(bool).ravel()
        z[~ih] = g[~ih]                                          # use guesses outside the convex hull
        led = led_for_camera(frames, led_vals, cam, len(z))
        zs.append(z[::DOWNSAMPLE].astype(np.float32))
        leds.append(led[::DOWNSAMPLE])
        cams.append(cam)
    return zs, leds, np.array(cams)


def save_strain(strain):
    zs, leds, cams = build_strain(strain)
    os.makedirs(OUT, exist_ok=True)
    is_control = cams < 7                                        # cams 1-6 control, 7-12 experimental
    path = f"{OUT}/{strain}.npz"
    np.savez_compressed(
        path,
        z=np.concatenate(zs).astype(np.float32),
        fly_lengths=np.array([len(z) for z in zs]),
        led=np.concatenate(leds).astype(np.uint8),
        camera=cams, is_control=is_control, fps=FPS, strain=strain)
    print("wrote %s  (%.2f MB)  %d ctrl + %d exp flies @ %d fps"
          % (path, os.path.getsize(path) / 1e6, int(is_control.sum()),
             int((~is_control).sum()), FPS))


ALL_STRAINS = ["ss02635", "ss01049", "ss02617_0226", "ss01540",
               "ss01597_1v_1022", "ss01602", "ss02393_1v_1009"]

if __name__ == "__main__":
    import sys
    for s in sys.argv[1:] or ALL_STRAINS:
        save_strain(s)
