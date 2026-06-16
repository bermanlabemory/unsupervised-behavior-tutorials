# Data

Datasets are loaded at runtime by each notebook (downloaded from this repo, or generated as a
stand-in) so the repo stays Colab-friendly. Real data small enough to ship lives here in-repo.

## What ships here

- **`transition_data.mat`** (1.9 MB) — the real 59-fly transition dataset for
  `02_transitions_and_hierarchy.ipynb` (defaults to `USE_SYNTHETIC_DATA = False`).
- **`rat_data/`** (~7 MB) — Ugne Klibaite's rats for `03_rat_individual_behavior.ipynb`:
  `amph.npz` (6 rats × 3 days of precomputed individual-map (`cz_action`) embeddings + fine/coarse
  watershed labels; days 2–3 baseline, day 4 amphetamine; all 18 share one map space) and
  `rat_keypoints_session1.npz` (a 23-joint 3-D s-DANNCE keypoint clip + skeleton + one lone session's
  behavioral map, for the "look at the raw data" intro). Regenerate with `python tools/make_rat_data.py`;
  source is `Dropbox/LE_CONTROL_AMPH/` (the `_L` lone sessions — same cohort as the social data below).
- **`rat_data/rat_social.npz` + `rat_data/rat_social_keypoints.npz`** (~20 MB) — real Long-Evans
  dyads for `04_rat_social_behavior.ipynb`: 45 dyads (24 control + 21 amph) + 30 lone sessions, each
  with the precomputed **individual** map (`cz_action` + coarse labels) and **social/joint** map
  (`sz_joint` + coarse labels) and partner labels (`rat_social.npz`), plus 23-joint 3-D keypoints for
  2 example dyads + a viz clip (`rat_social_keypoints.npz`), down-sampled to 5 Hz. `isamph`: 0 =
  control, 1 = amphetamine, 2 = saline cage-mate of an amph rat. Regenerate with
  `python tools/make_rat_social_data.py`; source is `Dropbox/LE_CONTROL_AMPH/`.
- **`optogenetic_data/*.npz`** (7 files, ~4 MB each) — all seven real driver lines from the Cande
  et al. 2018 example data (`ss02635`, `ss02617_0226`, `ss01049`, `ss01540`, `ss01597_1v_1022`,
  `ss01602`, `ss02393_1v_1009`), for `06_optogenetics.ipynb` (defaults to the real data; the
  notebook's §9 is a menu of all seven). Each file is one 12-fly filming session: cameras 1–6 are
  controls (no all-*trans*-retinal), 7–12 are experimentals. Stored as the 2-D map embedding
  (out-of-hull points replaced by `zGuesses`, as in the MATLAB `loadZValuesAndLEDs.m`) plus a
  per-frame LED on/off trace, down-sampled from 100 Hz to 25 Hz as `float32`. Regenerate all seven
  with `python tools/make_optogenetic_data.py`; the source is in
  `Dropbox/.../Cajal_behavior/Fly_Optogenetic_Analysis/example_data/`.

`07_bring_your_own_data.ipynb` is the only notebook without bundled data — it's a template you point
at your own tracking.

## Still to wire up

All Act-2 datasets are now in the repo. `01_build_a_behavioral_map.ipynb` (fly LEAP) and
`05_slow_modes.ipynb` (cached `.npz`/`.pkl`) download their data automatically by cloning
`motionmapperpy` and `slowmode` — no action needed.
