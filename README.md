# Unsupervised Behavioral Analysis Tutorials

Hands-on notebooks (originally developed for the **Cajal Quantitative Approaches to Behaviour & Virtual Reality** course in Lisbon, Portugal).

The first notebook is provides a framework for the otherones.  I would recommend going through the first 4 notebooks, and then the other ones are more optional, depending on your interests. That said, each notebook is standalone and loads its own data.

| # | Notebook | Question it answers |
|---|---|---|
| 1 | [`01_build_a_behavioral_map.ipynb`](01_build_a_behavioral_map.ipynb) | How do you turn pose into a map of behavior, with no labels? |
| 2 | [`02_transitions_and_hierarchy.ipynb`](02_transitions_and_hierarchy.ipynb) | Is behavior Markovian? How is it organized in time? |
| 3 | [`03_rat_individual_behavior.ipynb`](03_rat_individual_behavior.ipynb) | How does amphetamine reshape a rat's repertoire? |
| 4 | [`04_rat_social_behavior.ipynb`](04_rat_social_behavior.ipynb) | How do you quantify what two animals do *together*? |
| 5 | [`05_slow_modes.ipynb`](05_slow_modes.ipynb) | What slow internal states bias the fast actions? |
| 6 | [`06_optogenetics.ipynb`](06_optogenetics.ipynb) | Which behaviors does activating a neuron trigger? |
| 7 | [`07_bring_your_own_data.ipynb`](07_bring_your_own_data.ipynb) | Does this work on *my* data? |

Every notebook mixes **▶︎ "just run it"** cells with **🔧 "your turn"** cells, so it works whether
this is your first Python or your hundredth.

## Before starting
Open [`00_colab_check.ipynb`](00_colab_check.ipynb) in Colab and run it top-to-bottom (~2 min)
to confirm your runtime works. Do this the night before — it means minute 1 of the workshop
isn't spent debugging environments.

## Running in Colab
Click the **Open in Colab** badge at the top of any notebook. A free GPU runtime
(`Runtime → Change runtime type → GPU`) speeds up UMAP but is optional — everything runs on CPU too.

## The science behind each notebook
- **Berman et al. 2014**, *J. R. Soc. Interface* — MotionMapper (the Core pipeline) → **01**
- **Berman et al. 2016**, *PNAS* — Predictability & hierarchy in *Drosophila* behavior → **02**
- **Klibaite et al. 2025**, *Cell* — Mapping the landscape of social behavior → **04** (rat data also in **03**)
- **Kaur, Jain & Berman 2026** — Timescale as a state coordinate → **05**
- **Cande et al. 2018**, *eLife* — Optogenetic dissection of descending control → **06**

## Code these build on
- motionmapperpy: https://github.com/bermanlabemory/motionmapperpy
- slowmode: https://github.com/bermanlabemory/slowmode

## Data
Most data ships **in this repo** (`data/`) or inside the cloned engines, so the notebooks run as-is.
See [`data/README.md`](data/README.md); real data is regenerated from source by `tools/make_*.py`.

| Notebook | Dataset | Status |
|---|---|---|
| 01 core | fly LEAP data — ships inside motionmapperpy | ✅ |
| 02 transitions | `data/transition_data.mat` (59 flies, 117 states; Berman 2016) | ✅ in repo |
| 03 rat individual | `data/rat_data/` (amph embeddings + a keypoint sample; Klibaite/Berman rats) | ✅ in repo |
| 04 rat social | `data/rat_data/rat_social*.npz` (real Long-Evans dyads: 45 sessions, ctrl + amph) | ✅ in repo |
| 05 slow modes | ships inside the slowmode repo (cached `.npz`/`.pkl`) | ✅ |
| 06 optogenetics | `data/optogenetic_data/` (7 Cande 2018 driver lines) | ✅ in repo |
| 07 bring-your-own | you provide it | — |