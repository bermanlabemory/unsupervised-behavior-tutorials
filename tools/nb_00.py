"""00 — Colab check. Run before the workshop to confirm the runtime works."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nb_builder import md, code, badge, write_nb, setup_code

REPO = "bermanlabemory/unsupervised-behavior-tutorials/blob/main"
cells = []

cells.append(badge("%s/00_colab_check.ipynb" % REPO))

cells.append(md(r"""
# Colab check &mdash; run this *before* the workshop

**Time: about 2 minutes.** This little notebook doesn't teach anything; it just confirms that your
Google Colab runtime can do everything we'll ask of it on the afternoon of the workshop. Run every
cell top to bottom (Shift+Enter, or `Runtime → Run all`). If the last cell prints a happy message,
you're set. If something breaks, send us the error *before* the session &mdash; a couple of minutes now
is far less painful than troubleshooting once we've all started.

> **New to Colab?** It's a free Jupyter notebook that runs on Google's computers, so you don't
> install anything on your own laptop. Cells run top to bottom; a cell is "running" while it shows a
> spinner, and prints its output underneath when it's done.
"""))

# ---------------------------------------------------------------- machine
cells.append(md("# 1.&nbsp; What machine did Google give us?"))
cells.append(code(r"""
import sys, platform, subprocess
print("Python:", sys.version.split()[0], "on", platform.system())

# A GPU is optional for this workshop, but nice to have -- it speeds up UMAP and the autoencoders.
# To ask for one: Runtime -> Change runtime type -> Hardware accelerator -> GPU, then rerun this cell.
gpu = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                     capture_output=True, text=True)
print("GPU:", gpu.stdout.strip() if gpu.returncode == 0 else "none (that's perfectly fine)")
"""))

# ---------------------------------------------------------------- engine
cells.append(md(r"""
# 2.&nbsp; Grab the analysis engine

We'll lean on **motionmapperpy** all afternoon &mdash; a GPU-friendly Python implementation of
MotionMapper. The cell below clones it from GitHub (which also brings the small example datasets),
installs the handful of packages Colab doesn't already ship, and imports it. The same setup cell
opens every notebook in this series, so this is your first look at it.
"""))
cells.append(code(setup_code(
    imports="import numpy as np\nimport matplotlib.pyplot as plt\nimport motionmapperpy as mmpy",
    ready="engine ready")))

cells.append(md(r"""
> **No restart needed.** This cell imports motionmapperpy straight from the cloned folder. If
> `import motionmapperpy` ever fails, just re-run the cell &mdash; do **not** use *Restart session*,
> which would undo the `sys.path` line it adds. Your files are kept either way.
"""))

# ---------------------------------------------------------------- smoke test
cells.append(md(r"""
# 3.&nbsp; The smoke test

One real step of the pipeline on fake data. A **wavelet transform** is the engine's core move &mdash;
it turns a posture time series into a picture of *how fast things are wiggling, at each moment*
(you'll see why that matters in notebook 01). If the cell below draws a spectrogram and prints a
check mark, every piece we need is working.
"""))
cells.append(code(r"""
import numpy as np
import matplotlib.pyplot as plt
import motionmapperpy as mmpy

params = mmpy.setRunParameters()
params.samplingFreq, params.minF, params.maxF, params.numPeriods = 100, 1, 50, 25

# A tiny fake "postural" time series: 3 modes drifting over 2000 frames. If the wavelet transform
# returns an array we can plot, the whole engine runs on this machine.
x = np.cumsum(np.random.randn(2000, 3), axis=0)
wavelets, freqs = mmpy.findWavelets(x, 3, params.omega0, params.numPeriods, params.samplingFreq,
                                    params.maxF, params.minF, params.numProcessors, -1)

fig, ax = plt.subplots(figsize=(9, 3))
ax.imshow(wavelets[:500].T, aspect="auto", cmap="PuRd", origin="lower")
ax.set_title("a wavelet spectrogram of fake data — if you can see this, you're ready")
ax.set_xlabel("frame"); ax.set_ylabel("wavelet channel"); plt.show()

print("\n✅ All set. See you at the workshop!")
"""))

write_nb(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                      "00_colab_check.ipynb"), cells)
