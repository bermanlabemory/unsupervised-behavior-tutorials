"""Tiny helper for authoring Jupyter/Colab notebooks from plain Python.

We keep the notebooks as code so they are easy to mass-edit for tone/structure.
Each notebook builder (tools/nb_*.py) imports `md`, `code`, `write_nb` from here.

Run `python tools/build_all.py` to regenerate every notebook.
"""
import json
import os
import textwrap


def _src(s):
    """Turn a (possibly triple-quoted, indented) string into an .ipynb source list."""
    s = textwrap.dedent(s).strip("\n")
    if s == "":
        return []
    lines = s.split("\n")
    # Every line keeps its trailing newline except the last (nbformat convention).
    return [l + "\n" for l in lines[:-1]] + [lines[-1]]


def md(s):
    """A markdown cell."""
    return {"cell_type": "markdown", "metadata": {}, "source": _src(s)}


def code(s):
    """A code cell."""
    return {"cell_type": "code", "metadata": {}, "execution_count": None,
            "outputs": [], "source": _src(s)}


def badge(repo_path):
    """A Colab 'Open in Colab' badge cell. `repo_path` is e.g. 'org/repo/blob/main/x.ipynb'."""
    url = "https://colab.research.google.com/github/%s" % repo_path
    return md('<a href="%s" target="_parent">'
              '<img src="https://colab.research.google.com/assets/colab-badge.svg" '
              'alt="Open In Colab"/></a>' % url)


# --------------------------------------------------------------------------- #
# Shared Colab setup. Every notebook is self-contained (the code below is written
# into each .ipynb), but it comes from ONE source here so the boilerplate can't
# drift between notebooks. The clone + moviepy-stub + sys.path block is byte-for-byte
# identical everywhere; only the per-notebook imports / pip extras / extra clones vary.
# --------------------------------------------------------------------------- #

_CLONE = r'''import os, sys, types
if not os.path.exists("motionmapperpy"):
    !git clone -q https://github.com/bermanlabemory/motionmapperpy'''

_STUB = r'''# motionmapperpy imports moviepy the moment it loads, and Colab's moviepy/ffmpeg stack tries to
# fetch an ancient ffmpeg from a (long-dead) URL. None of these notebooks push video through moviepy,
# so we hand Python a tiny stand-in for it -- harmless here, and it sidesteps the whole mess.
def _stub(name, **attrs):
    m = sys.modules.get(name) or types.ModuleType(name)
    for k, v in attrs.items():
        setattr(m, k, v)
    sys.modules[name] = m
    return m
_stub("moviepy"); _stub("moviepy.editor", VideoClip=object, VideoFileClip=object)
_stub("moviepy.video"); _stub("moviepy.video.io")
_stub("moviepy.video.io.bindings", mplfig_to_npimage=lambda *a, **k: None)

# We import motionmapperpy straight from the cloned folder. (Running its setup.py on Colab leaves an
# importable-but-empty package -- the classic trap; this route avoids it, and needs no kernel restart.)
sys.path.insert(0, os.path.abspath("motionmapperpy"))
for _m in [k for k in list(sys.modules) if k.startswith("motionmapperpy")]:
    del sys.modules[_m]'''


def setup_code(pip_extra="", extra_clones=(), imports="", ready="ready"):
    """The standard Colab setup as a single code-cell string.

    pip_extra    : extra pip packages (space-separated) appended to the standard install.
    extra_clones : iterable of (dirname, git_url) to clone besides motionmapperpy.
    imports      : this notebook's import lines (string); placed after the install.
    ready        : the message printed at the end so a student knows the cell finished.
    """
    parts = [_CLONE]
    for name, url in extra_clones:
        parts.append('if not os.path.exists("%s"):\n    !git clone -q %s' % (name, url))
    parts.append(_STUB)
    pip = "hdf5storage easydict umap-learn" + ((" " + pip_extra) if pip_extra else "")
    parts.append("!pip install -q " + pip + " 2>/dev/null")
    body = imports.strip("\n")
    if body:
        parts.append(body)
    parts.append('print("' + ready + '")')
    return "\n\n".join(parts)


def carry_from_core():
    """A consistent reminder of the notebook-01 vocabulary, for the Act-2 tracks."""
    return (
        "> **Carried over from notebook 01.** A *behavioral map* places every frame as a point in 2-D "
        "so that similar movements land near each other; **peaks** in its density are the behaviors an "
        "animal does often; a **watershed** carves the map into discrete behavioral **regions**; and "
        "reading the region label out over time gives an **ethogram**. You don't need to have run "
        "notebook 01 to follow along &mdash; this notebook loads its own data."
    )


def write_nb(path, cells):
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)
    nb = {
        "cells": cells,
        "metadata": {
            "colab": {"provenance": [], "toc_visible": True},
            "kernelspec": {"name": "python3", "display_name": "Python 3"},
            "language_info": {"name": "python"},
            "accelerator": "GPU",
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    with open(path, "w") as f:
        json.dump(nb, f, indent=1)
    print("wrote %-48s %3d cells" % (path, len(cells)))
