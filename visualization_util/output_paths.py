"""
output_paths.py — Centralised output directory definitions.

All analysis / visualisation scripts import from here so that
generated files land in the organised folder structure.
"""

import os

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

RESULTS_DIR = os.path.join(_ROOT, "results")
IMAGES_DIR = os.path.join(_ROOT, "visualization", "images")
GIFS_DIR = os.path.join(_ROOT, "visualization", "gifs")
PLOTS_DIR = os.path.join(_ROOT, "visualization", "plots")

for _d in (RESULTS_DIR, IMAGES_DIR, GIFS_DIR, PLOTS_DIR):
    os.makedirs(_d, exist_ok=True)
