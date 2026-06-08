# Data

This folder is intentionally (almost) empty. Datasets are **downloaded at runtime** by each
notebook so the repo stays small and Colab-friendly.

## Status: stand-in data for now

Every Act-2 track currently defaults to `USE_SYNTHETIC_DATA = True` and generates plausible
stand-in data, so the notebooks run end-to-end **today** for development and for rehearsing the
teaching flow. The synthetic data is good enough to exercise every analysis and figure, but it
is *not* real biology — swap in the real datasets before the course.

## To wire up real data (before 17 June)

For each track, host the dataset (Dropbox/Zenodo/Google Drive direct-download link) and edit
the `DATA_URL` constant near the top of the notebook, then set `USE_SYNTHETIC_DATA = False`.

| Notebook | What to host | Source on disk |
|---|---|---|
| `02_social_behavior_rats.ipynb` | a small CTRL + amphetamine rat-dyad subset (3D keypoints, a few sessions per group) | Klibaite 2025 s-DANNCE release |
| `03_transitions_and_hierarchy.ipynb` | `transition_data.mat` | `Dropbox/.../Cajal_behavior/behavioral_transitions_tutorial/transition_data.mat` |
| `04_optogenetics.ipynb` | a few strains of `*_projections_tsne_embedding.mat` + `*_Frames.txt` | `Dropbox/.../Cajal_behavior/Fly_Optogenetic_Analysis/example_data/` |

`01_build_a_behavioral_map.ipynb` (fly LEAP) and `05_slow_modes.ipynb` (cached `.npz`/`.pkl`)
download their data automatically by cloning `motionmapperpy` and `slowmode` — no action needed.
