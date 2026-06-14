# Data

Datasets are loaded at runtime by each notebook (downloaded from this repo, or generated as a
stand-in) so the repo stays Colab-friendly. Real data small enough to ship lives here in-repo.

## What ships here

- **`transition_data.mat`** (1.9 MB) — the real 59-fly transition dataset for
  `02_transitions_and_hierarchy.ipynb` (defaults to `USE_SYNTHETIC_DATA = False`).
- **`rat_data/`** (~7 MB) — Ugne Klibaite's rats for `03_rat_individual_behavior.ipynb`:
  `amph.npz` (6 rats × 3 days of precomputed MotionMapper embeddings + fine/coarse watershed labels;
  days 2–3 baseline, day 4 amphetamine; all 18 share one map space) and
  `rat_keypoints_session1.npz` (a 3-D DANNCE keypoint clip + skeleton + one session's behavioral map,
  for the "look at the raw data" intro). Regenerate with `python tools/make_rat_data.py`; source is
  `Dropbox/.../tutorial-data-uk/`.
- **`optogenetic_data/*.npz`** (7 files, ~4 MB each) — all seven real driver lines from the Cande
  et al. 2018 example data (`ss02635`, `ss02617_0226`, `ss01049`, `ss01540`, `ss01597_1v_1022`,
  `ss01602`, `ss02393_1v_1009`), for `06_optogenetics.ipynb` (defaults to the real data; the
  notebook's §9 is a menu of all seven). Each file is one 12-fly filming session: cameras 1–6 are
  controls (no all-*trans*-retinal), 7–12 are experimentals. Stored as the 2-D map embedding
  (out-of-hull points replaced by `zGuesses`, as in the MATLAB `loadZValuesAndLEDs.m`) plus a
  per-frame LED on/off trace, down-sampled from 100 Hz to 25 Hz as `float32`. Regenerate all seven
  with `python tools/make_optogenetic_data.py`; the source is in
  `Dropbox/.../Cajal_behavior/Fly_Optogenetic_Analysis/example_data/`.

The synthetic-fallback notebooks (`04_rat_social_behavior`, `07_bring_your_own_data`) keep
`USE_SYNTHETIC_DATA = True` so they run with no network access.

## Still to wire up (before 17 June)

| Notebook | What to host | Source |
|---|---|---|
| `04_rat_social_behavior.ipynb` | a small CTRL + amphetamine rat-**dyad** subset (3-D keypoints, a few sessions per group) — currently synthetic | Klibaite 2025 s-DANNCE release (Ugne) |

`01_build_a_behavioral_map.ipynb` (fly LEAP) and `05_slow_modes.ipynb` (cached `.npz`/`.pkl`)
download their data automatically by cloning `motionmapperpy` and `slowmode` — no action needed.
