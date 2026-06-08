# Unsupervised Behavioral Analysis — CAJAL 2026

Hands-on notebooks for the **Quantitative Approaches to Behaviour & Virtual Reality** course
(Champalimaud Centre for the Unknown, Lisbon). Workshop afternoon: **17 June 2026, 14:30–19:00**.

**Instructors:** Gordon Berman & Ugne Klibaite

By now in the course you have built rigs, tracked animals (SLEAP), and seen **supervised**
behavior classification (Kennedy, Branson). These notebooks are the **unsupervised**
complement: take pose data, discover the behavioral repertoire *with no labels*, and then
actually *do science* with it — compare groups, quantify dynamics, map social interaction,
read out neural perturbations, and uncover slow internal states.

## How the afternoon works

**Act 1 — Core (everyone builds this):** a fly behavioral map from scratch.
→ [`01_build_a_behavioral_map.ipynb`](01_build_a_behavioral_map.ipynb)

**Act 2 — Choose your own adventure:** pick one or two tracks that match your interests.

| Notebook | Track | Question it answers | Level |
|---|---|---|---|
| [`02_social_behavior_rats.ipynb`](02_social_behavior_rats.ipynb) | Social behavior | What do two animals do *together*? | 🟡 medium |
| [`03_transitions_and_hierarchy.ipynb`](03_transitions_and_hierarchy.ipynb) | Transitions & hierarchy | Is behavior Markovian? How is it organized in time? | 🟠 medium–hard |
| [`04_optogenetics.ipynb`](04_optogenetics.ipynb) | Optogenetics | Which behaviors does activating these neurons trigger? | 🟡 medium |
| [`05_slow_modes.ipynb`](05_slow_modes.ipynb) | Slow modes | What slow internal states bias the fast actions? | 🔴 hard |
| [`06_bring_your_own_data.ipynb`](06_bring_your_own_data.ipynb) | Bring your own data | Does this work on *my* data? | ⚪ open |

Levels: 🟢 run-and-observe · 🟡 modify-a-parameter · 🟠/🔴 conceptual + some coding · ⚪ open-ended.
Every notebook mixes **▶︎ "just run it"** cells with **🔧 "your turn"** cells, so the same
notebook works whether this is your first Python or your hundredth. Each Act-2 track loads a
**checkpoint**, so you can start any track even if you didn't finish the Core.

## Before the session
Open [`00_colab_check.ipynb`](00_colab_check.ipynb) in Colab and run it top-to-bottom (~2 min)
to confirm your runtime works. Do this the night before — it means minute 1 of the workshop
isn't spent debugging environments.

## Running in Colab
Click the **Open in Colab** badge at the top of any notebook. (Set the GitHub org in the
badge once this repo is pushed — edit `REPO` at the top of each `tools/nb_*.py`.) A free GPU
runtime (`Runtime → Change runtime type → GPU`) speeds up UMAP and the autoencoder but is
optional — everything runs on CPU too.

## The science behind each notebook
- **Berman et al. 2014**, *J. R. Soc. Interface* — MotionMapper (the Core pipeline)
- **Berman et al. 2016**, *PNAS* — Predictability & hierarchy in *Drosophila* behavior → **03**
- **Cande et al. 2018**, *eLife* — Optogenetic dissection of descending control → **04**
- **Klibaite et al. 2025**, *Cell* — Mapping the landscape of social behavior → **02**
- **Kaur, Jain & Berman 2026** — Timescale as a state coordinate → **05**

## Code these build on
- motionmapperpy: https://github.com/bermanlabemory/motionmapperpy
- slowmode: https://github.com/bermanlabemory/slowmode

## Data (TODO before the course)
Real datasets are hosted separately. **Every track defaults to `USE_SYNTHETIC_DATA = True`**,
so all notebooks run *today* on generated stand-in data. Replace the placeholder URLs and flip
the flag to `False` once the real data is hosted:

| Track | Dataset | Owner |
|---|---|---|
| 02 social | rat 3D dyads — subset of the Klibaite 2025 s-DANNCE release (CTRL + amphetamine) | Ugne |
| 03 transitions | `transition_data.mat` (59 flies, 117 states) from the Berman 2016 tutorial | Gordon |
| 04 optogenetics | split-GAL4 t-SNE embeddings + `*_Frames.txt` LED times (Cande 2018) | Gordon |
| 05 slow modes | ships **inside** the slowmode repo (cached `.npz`/`.pkl`) — no action needed | — |
| 01 core | fly LEAP data ships inside motionmapperpy — no action needed | — |

## Regenerating the notebooks
These `.ipynb` files are generated from plain-Python builders so they're easy to mass-edit:

```bash
python tools/build_all.py     # rebuilds all notebooks
python tools/nb_01.py         # or rebuild just one
```

Edit the prose/code in `tools/nb_*.py`, rerun, commit the regenerated `.ipynb`.
