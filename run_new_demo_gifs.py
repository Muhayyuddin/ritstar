#!/usr/bin/env python3
"""Generate GIFs for the newly added demo environments only.

Environments:
  - 2D Hyper-Dense (2D heatmap + 3D surface GIFs)
  - 3D Dense Labyrinth (3D tree growth GIF)

Bug Trap already has GIFs from the existing pipeline.
"""
import sys
sys.stdout.reconfigure(line_buffering=True)

import gc

from rit_star.environments import (
    env_2d_hyper_dense,
    env_3d_dense_labyrinth,
)
from visualize_riemannian import (
    animate_tree_growth,
    animate_3d_surface_tree,
    animate_3d_env,
)


if __name__ == '__main__':
    print('=' * 60)
    print('  GIF Generation — New Demo Environments')
    print('=' * 60)

    # ── 2D Hyper-Dense: heatmap tree growth ──
    print('\n[1/3] 2D Hyper-Dense — Heatmap Tree Growth')
    animate_tree_growth(
        '2D Hyper-Dense', env_2d_hyper_dense,
        'riemannian_2d_hyper-dense',
        max_iterations=80, batch_size=100, frame_every=2, fps=8, res=100,
    )
    gc.collect()

    # ── 2D Hyper-Dense: 3D surface tree growth ──
    print('\n[2/3] 2D Hyper-Dense — 3D Surface Tree Growth')
    animate_3d_surface_tree(
        '2D Hyper-Dense', env_2d_hyper_dense,
        'riemannian_2d_hyper-dense',
        max_iterations=80, batch_size=100, frame_every=2, fps=8, res=80,
    )
    gc.collect()

    # ── 3D Dense Labyrinth: 3D tree growth ──
    print('\n[3/3] 3D Dense Labyrinth — 3D Tree Growth')
    animate_3d_env(
        '3D Dense Lab', env_3d_dense_labyrinth,
        'riemannian_3d_dense_lab',
        max_iterations=80, batch_size=100, frame_every=2, fps=8,
    )
    gc.collect()

    print('\nAll new demo GIFs generated!')
