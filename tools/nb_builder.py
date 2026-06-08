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
