#!/usr/bin/env python
"""Generate path visualization PNGs showing how each planner navigates
around obstacles. Creates side-by-side comparison for 2D environments."""
import sys
sys.stdout.reconfigure(line_buffering=True)

import gc
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle

from visualization_util.output_paths import IMAGES_DIR

from rit_star.rit_star import RITStar
from rit_star.baselines import InformedRRTStar, BITStar, AITStar, EITStar, APTStar
from rit_star.environments import (
    env_2d_obstacle_inflated,
    env_2d_diagonal_anisotropic,
    env_2d_narrow_passage,
    env_2d_maze,
    env_2d_bug_trap,
    env_2d_random_forest,
    env_2d_terrain,
)


PLANNERS = {
    'RIT*':          ('RIT*', '#7B2FBE'),
    'Informed RRT*': ('Informed RRT*', '#2196F3'),
    'BIT*':          ('BIT*', '#4CAF50'),
    'AIT*':          ('AIT*', '#FF9800'),
    'ABIT*':         ('ABIT*', '#E91E63'),
    'FMT*':          ('FMT*', '#00BCD4'),
    'RRT-Connect':   ('RRT-Connect', '#795548'),
    'PRM*':          ('PRM*', '#607D8B'),
}

ENVS_2D = {
    '2D Obstacles': (env_2d_obstacle_inflated, 'obstacle'),
    '2D Diagonal':  (env_2d_diagonal_anisotropic, 'diagonal'),
    '2D Narrow':    (env_2d_narrow_passage, 'narrow'),
    '2D Maze':      (env_2d_maze, 'maze'),
    '2D Bug Trap':  (env_2d_bug_trap, 'bug_trap'),
    '2D Forest':    (env_2d_random_forest, 'forest'),
    '2D Terrain':   (env_2d_terrain, 'terrain'),
}

OBSTACLE_DEFS = {
    'obstacle': {
        'type': 'circles',
        'data': [
            ([0.30, 0.35], 0.08), ([0.30, 0.65], 0.08),
            ([0.50, 0.45], 0.09), ([0.50, 0.75], 0.09),
            ([0.70, 0.40], 0.08), ([0.70, 0.60], 0.08),
        ]
    },
    'diagonal': {
        'type': 'rects',
        'data': [
            ([0.35, 0.30], [0.45, 0.70]),
            ([0.55, 0.30], [0.65, 0.70]),
        ]
    },
    'narrow': {
        'type': 'rects',
        'data': [
            ([0.48, 0.00], [0.52, 0.47]),
            ([0.48, 0.53], [0.52, 1.00]),
            ([0.30, 0.15], [0.42, 0.35]),
            ([0.30, 0.65], [0.42, 0.85]),
            ([0.58, 0.15], [0.70, 0.35]),
            ([0.58, 0.65], [0.70, 0.85]),
        ]
    },
    'maze': {
        'type': 'rects',
        'data': [
            ([0.00, 0.22], [0.70, 0.30]),
            ([0.30, 0.46], [1.00, 0.54]),
            ([0.00, 0.70], [0.70, 0.78]),
        ]
    },
    'bug_trap': {
        'type': 'rects',
        'data': [
            ([0.15, 0.20], [0.60, 0.28]),
            ([0.15, 0.72], [0.60, 0.80]),
            ([0.15, 0.28], [0.23, 0.72]),
            ([0.52, 0.28], [0.60, 0.46]),
            ([0.52, 0.54], [0.60, 0.72]),
        ]
    },
    'forest': {
        'type': 'circles',
        'data': None,  # populated below
    },
    'terrain': {
        'type': 'none',
        'data': [],
    },
}

# Generate forest obstacles
def _forest_obs():
    import numpy as _np
    rng = _np.random.default_rng(12345)
    centres = []
    _xs = _np.array([0.05, 0.05])
    _xg = _np.array([0.95, 0.95])
    for _ in range(125):
        if len(centres) >= 25:
            break
        c = rng.uniform(0.1, 0.9, size=2)
        if _np.linalg.norm(c - _xs) < 0.12:
            continue
        if _np.linalg.norm(c - _xg) < 0.12:
            continue
        centres.append(c)
    return [(c.tolist(), 0.04) for c in centres[:25]]

OBSTACLE_DEFS['forest']['data'] = _forest_obs()


def draw_obstacles(ax, obs_key):
    obs = OBSTACLE_DEFS.get(obs_key)
    if obs is None or obs['type'] == 'none':
        return
    if obs['type'] == 'circles':
        for c, r in obs['data']:
            ax.add_patch(Circle(c, r, fc='#333333', ec='black', alpha=0.7))
    elif obs['type'] == 'rects':
        for lo, hi in obs['data']:
            w, h = hi[0] - lo[0], hi[1] - lo[1]
            ax.add_patch(Rectangle(lo, w, h, fc='#333333', ec='black', alpha=0.7))


def build_planner(name, xs, xg, bounds, coll, metric, seed=42):
    common = dict(
        x_start=xs, x_goal=xg, c_space_bounds=bounds,
        collision_checker=coll, metric=metric,
        batch_size=100, max_iterations=200, random_seed=seed,
    )
    if name == 'RIT*':
        return RITStar(xs, xg, bounds, coll, metric,
                       geodesic_tier='diagonal', batch_size=100,
                       max_iterations=200, random_seed=seed)
    elif name == 'Informed RRT*':
        return InformedRRTStar(**common)
    elif name == 'BIT*':
        return BITStar(**common)
    elif name == 'AIT*':
        return AITStar(**common)
    elif name == 'EIT*':
        return EITStar(**common)
    elif name == 'APT*':
        return APTStar(**common)


def extract_tree_edges(planner):
    """Extract tree edges as list of (parent_x, child_x) pairs."""
    edges = []
    for v in planner.vertices:
        if v.parent is not None:
            edges.append((v.parent.x, v.x))
    return edges


def plot_single(ax, planner, path, cost, elapsed, name, color, obs_key, bounds, xs, xg):
    """Plot single planner result on an axes."""
    ax.set_xlim(bounds[0])
    ax.set_ylim(bounds[1])
    ax.set_aspect('equal')
    draw_obstacles(ax, obs_key)

    # Tree edges
    edges = extract_tree_edges(planner)
    for p_x, c_x in edges:
        ax.plot([p_x[0], c_x[0]], [p_x[1], c_x[1]],
                'gray', lw=0.15, alpha=0.3)

    # Tree vertices
    verts = np.array([v.x for v in planner.vertices])
    ax.scatter(verts[:, 0], verts[:, 1], s=1, c='gray', alpha=0.25, zorder=2)

    # Path
    if path and len(path) > 1:
        pp = np.array(path)
        ax.plot(pp[:, 0], pp[:, 1], color=color, lw=2.5, zorder=4)

    # Start/Goal
    ax.plot(*xs, 'go', ms=8, zorder=5)
    ax.plot(*xg, 'r^', ms=8, zorder=5)

    # Title
    if np.isfinite(cost):
        ax.set_title(f'{name}\ncost={cost:.4f}  time={elapsed:.1f}s',
                     fontsize=10, color=color, fontweight='bold')
    else:
        ax.set_title(f'{name}\nno solution  time={elapsed:.1f}s',
                     fontsize=10, color='red', fontweight='bold')


def main():
    for env_name, (env_fn, obs_key) in ENVS_2D.items():
        print(f'\n=== {env_name} ===')
        coll, _, metric, xs, xg, bounds = env_fn()

        fig, axes = plt.subplots(2, 4, figsize=(24, 12))
        fig.suptitle(f'Path Comparison — {env_name}', fontsize=16, y=0.98)

        for idx, (pname, (label, color)) in enumerate(PLANNERS.items()):
            row, col = idx // 4, idx % 4
            ax = axes[row, col]

            print(f'  Running {label}...', end=' ', flush=True)
            planner = build_planner(pname, xs, xg, bounds, coll, metric, seed=42)
            path, cost = planner.plan()
            stats = planner.get_stats()
            elapsed = stats[-1]['time_elapsed'] if stats else 0.0
            print(f'cost={cost:.4f}, time={elapsed:.1f}s')

            plot_single(ax, planner, path, cost, elapsed, label, color,
                       obs_key, bounds, xs, xg)
            del planner
            gc.collect()

        fig.tight_layout(rect=[0, 0, 1, 0.96])
        safe = env_name.lower().replace(' ', '_')
        fname = os.path.join(IMAGES_DIR, f'path_comparison_{safe}.png')
        fig.savefig(fname, dpi=150, bbox_inches='tight')
        plt.close(fig)
        del fig
        gc.collect()
        print(f'  → saved {fname}')

    print('\nAll path visualizations complete.')


if __name__ == '__main__':
    main()
