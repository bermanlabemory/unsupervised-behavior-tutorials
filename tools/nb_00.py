"""00 — Colab check. Run before the workshop to confirm the runtime works."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nb_builder import md, code, badge, write_nb

REPO = "bermanlabemory/unsupervised-behavior-tutorials/blob/main"
cells = []

cells.append(badge("%s/00_colab_check.ipynb" % REPO))

cells.append(md(r"""
# 0.&nbsp; Colab check &mdash; run this *before* the workshop

**Time: ~2 minutes.** This notebook just makes sure your Google Colab runtime can do everything
we'll need on the afternoon of the workshop. Run every cell top-to-bottom (Shift+Enter, or
`Runtime → Run all`). If the last cell prints a happy message, you're ready. If something
breaks, send us the error before the session and we'll sort it out.

> New to Colab? It's a free Jupyter notebook that runs on Google's computers. You don't install
> anything on your laptop. Cells run top to bottom; a cell is "running" while it shows a spinner.
"""))

cells.append(md("## 0.1&nbsp; What machine did Google give us?"))
cells.append(code(r"""
import sys, platform
print("Python:", sys.version.split()[0], "on", platform.system())

# A GPU is optional for this workshop, but nice to have (it speeds up UMAP + autoencoders).
# To request one: Runtime -> Change runtime type -> Hardware accelerator -> GPU, then rerun.
import subprocess
gpu = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                     capture_output=True, text=True)
print("GPU:", gpu.stdout.strip() if gpu.returncode == 0 else "none (that's fine!)")
"""))

cells.append(md("## 0.2&nbsp; Install the few packages Colab doesn't ship"))
cells.append(code(r"""
# Quietly install what's missing. This takes ~30-60 s.
!pip install -q hdf5storage easydict umap-learn imageio==2.4.1 2>/dev/null
print("packages installed")
"""))

cells.append(md(r"""
## 0.3&nbsp; Grab the analysis engine

We'll use **motionmapperpy** &mdash; a GPU-friendly Python implementation of MotionMapper.
We download it by cloning its GitHub repository (this also brings along the small example
datasets we'll use).
"""))
cells.append(code(r"""
import os
if not os.path.exists("motionmapperpy"):
    !git clone -q https://github.com/bermanlabemory/motionmapperpy
%cd motionmapperpy
!python setup.py install -q 2>/dev/null
%cd ..
print("motionmapperpy installed")
"""))

cells.append(md(r"""
> **One-time step:** if the import in the next cell fails, do `Runtime → Restart session`
> and then run the rest of the notebook again. (Installing a package sometimes needs a restart
> before Python will see it. Restarting does **not** delete your files.)
"""))

cells.append(md("## 0.4&nbsp; The smoke test"))
cells.append(code(r"""
import numpy as np
import matplotlib.pyplot as plt
import motionmapperpy as mmpy

# Make a tiny fake postural time series and run one real step of the pipeline (a wavelet
# transform). If this returns an array, the whole engine works.
x = np.cumsum(np.random.randn(2000, 3), axis=0)      # 3 "postural modes" over 2000 frames
params = mmpy.setRunParameters()
wavelets, freqs = mmpy.findWavelets(x, 3, params.omega0, 25, 100, 50, 1,
                                    params.numProcessors, -1)

fig, ax = plt.subplots(figsize=(9, 3))
ax.imshow(wavelets[:500].T, aspect="auto", cmap="PuRd", origin="lower")
ax.set_title("a wavelet spectrogram of fake data — if you can see this, you're ready")
ax.set_xlabel("frame"); ax.set_ylabel("wavelet channel")
plt.show()

print("\n✅ All set. See you on the 17th!")
"""))

write_nb(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                      "00_colab_check.ipynb"), cells)
