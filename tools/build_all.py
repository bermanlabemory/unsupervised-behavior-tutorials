"""Regenerate every notebook. Run from the repo root:  python tools/build_all.py

The notebooks are generated from the plain-Python builders in this folder so they're easy to
mass-edit for tone/structure. Edit a tools/nb_*.py file, rerun this, commit the new .ipynb.

To change the GitHub org/repo in the 'Open in Colab' badges, edit REPO at the top of each
tools/nb_*.py (search-and-replace 'bermanlabemory/unsupervised_behavioral_analysis').
"""
import os
import runpy

HERE = os.path.dirname(os.path.abspath(__file__))
for nb in ["nb_00", "nb_01", "nb_02", "nb_03", "nb_04", "nb_05", "nb_06"]:
    runpy.run_path(os.path.join(HERE, nb + ".py"), run_name="__main__")
print("\nall notebooks rebuilt.")
