# Data

Datasets are loaded at runtime by each notebook (downloaded from this repo, or generated as a
stand-in) so the repo stays Colab-friendly. Real data small enough to ship lives here in-repo.

## What ships here

- **`transition_data.mat`** (1.9 MB) — the real 59-fly transition dataset for
  `03_transitions_and_hierarchy.ipynb` (defaults to `USE_SYNTHETIC_DATA = False`).
- **`optogenetic_data/ss02635.npz`, `optogenetic_data/ss01049.npz`** (~4 MB each) — two real
  driver lines from the Cande et al. 2018 example data, for `04_optogenetics.ipynb` (defaults to
  the real data). Each file is one 12-fly filming session: cameras 1–6 are controls (no
  all-*trans*-retinal), 7–12 are experimentals. Stored as the 2-D map embedding (out-of-hull
  points replaced by `zGuesses`, as in the MATLAB `loadZValuesAndLEDs.m`) plus a per-frame LED
  on/off trace, down-sampled from 100 Hz to 25 Hz as `float32`. Regenerate (or add more strains)
  with `python tools/make_optogenetic_data.py`; the full 7-strain source is in
  `Dropbox/.../Cajal_behavior/Fly_Optogenetic_Analysis/example_data/`.

Each notebook still has `USE_SYNTHETIC_DATA = True` as a fallback (and auto-falls-back if a
download fails), so everything runs even without network access.

## Still to wire up (before 17 June)

| Notebook | What to host | Source on disk |
|---|---|---|
| `02_social_behavior_rats.ipynb` | a small CTRL + amphetamine rat-dyad subset (3D keypoints, a few sessions per group) | Klibaite 2025 s-DANNCE release |

`01_build_a_behavioral_map.ipynb` (fly LEAP) and `05_slow_modes.ipynb` (cached `.npz`/`.pkl`)
download their data automatically by cloning `motionmapperpy` and `slowmode` — no action needed.
