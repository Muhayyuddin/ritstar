#!/usr/bin/env python3
"""
run_from_config.py — Config-driven planner benchmark runner.

Reads config/run_config.yaml (or a path given as CLI argument) and runs the
requested planners on the requested environments.

Usage:
    python run_from_config.py                   # uses config/run_config.yaml
    python run_from_config.py my_config.yaml    # uses custom config

Config format (YAML):
    planners: all                  # or [RIT, BIT, AIT, ...]
    environments: all              # or [2D, 3D, maze, narrow, ...]
    n_trials: 2
    max_iterations: 150
    batch_size: 100
    base_seed: 42
"""

from __future__ import annotations

import gc
import os
import sys
import time
import yaml
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from visualization_util.output_paths import PLOTS_DIR, GIFS_DIR, IMAGES_DIR
from rit_star.rit_star import RITStar
from rit_star.baselines import InformedRRTStar, BITStar, AITStar, EITStar, APTStar
from rit_star.metric import EuclideanMetric
from rit_star.environments import (
    env_2d_diagonal_anisotropic,
    env_2d_obstacle_inflated,
    env_2d_narrow_passage,
    env_2d_maze,
    env_2d_bug_trap,
    env_2d_random_forest,
    env_2d_terrain,
    env_2d_hyper_dense,
    env_2d_joint_arm,
    env_2d_random_world,
    env_2d_dividing_wall,
    env_3d_diagonal_anisotropic,
    env_3d_sphere_field,
    env_3d_narrow_passage,
    env_3d_dense_labyrinth,
    env_3d_anisotropic_corridor,
    env_3d_obstacle_gauntlet,
    env_3d_wall_and_gaps,
    env_3d_box_field,
    env_6d_hyper_passage,
    env_2d_obstacle_euclidean,
    env_2d_narrow_euclidean,
    env_2d_maze_euclidean,
    env_2d_forest_euclidean,
    ALL_6D_ENVS,
)

# ═══════════════════════════════════════════════════════════════════════
#  Registries
# ═══════════════════════════════════════════════════════════════════════

# Canonical planner names and aliases (lowercase) → canonical name
PLANNER_REGISTRY = {
    'rit':           'RIT*',
    'rit*':          'RIT*',
    'bit':           'BIT*',
    'bit*':          'BIT*',
    'irrt':          'Informed RRT*',
    'informed rrt':  'Informed RRT*',
    'informed rrt*': 'Informed RRT*',
    'ait':           'AIT*',
    'ait*':          'AIT*',
    'eit':           'EIT*',
    'eit*':          'EIT*',
    'apt':           'APT*',
    'apt*':          'APT*',
    'carm':          'RIT*-CARM',
    'rit-carm':      'RIT*-CARM',
    'rit*-carm':     'RIT*-CARM',
}

ALL_PLANNER_NAMES = ['RIT*', 'Informed RRT*', 'BIT*', 'AIT*', 'EIT*', 'APT*']

# Environment registry: canonical name → (factory_fn, dimension_tag)
ENV_REGISTRY = {
    '2D Diagonal':    (env_2d_diagonal_anisotropic, '2d'),
    '2D Obstacle':    (env_2d_obstacle_inflated,    '2d'),
    '2D Narrow':      (env_2d_narrow_passage,       '2d'),
    '2D Maze':        (env_2d_maze,                 '2d'),
    '2D Bug Trap':    (env_2d_bug_trap,             '2d'),
    '2D Forest':      (env_2d_random_forest,        '2d'),
    '2D Terrain':     (env_2d_terrain,              '2d'),
    '2D Hyper-Dense': (env_2d_hyper_dense,          '2d'),
    '2D Joint Arm':   (env_2d_joint_arm,            '2d'),
    '2D Random World': (env_2d_random_world,          '2d'),
    '2D Dividing Wall': (env_2d_dividing_wall,        '2d'),
    '3D Diagonal':    (env_3d_diagonal_anisotropic, '3d'),
    '3D Spheres':     (env_3d_sphere_field,         '3d'),
    '3D Narrow':      (env_3d_narrow_passage,       '3d'),
    '3D Dense Lab':   (env_3d_dense_labyrinth,      '3d'),
    '3D Corridor':    (env_3d_anisotropic_corridor, '3d'),
    '3D Gauntlet':    (env_3d_obstacle_gauntlet,    '3d'),
    '3D Wall & Gaps': (env_3d_wall_and_gaps,        '3d'),
    '3D Box Field':   (env_3d_box_field,            '3d'),
    '6D Hyper-Passage': (env_6d_hyper_passage,      '6d'),
    # Euclidean-cost variants (same obstacles, Euclidean metric)
    '2D Obstacle-E':  (env_2d_obstacle_euclidean,    '2d_euclid'),
    '2D Narrow-E':    (env_2d_narrow_euclidean,       '2d_euclid'),
    '2D Maze-E':      (env_2d_maze_euclidean,         '2d_euclid'),
    '2D Forest-E':    (env_2d_forest_euclidean,        '2d_euclid'),
}

# Aliases mapping short names (lowercase) → list of canonical env names
ENV_ALIASES = {
    'diagonal_2d': ['2D Diagonal'],
    'diagonal':    ['2D Diagonal', '3D Diagonal'],
    'obstacle':    ['2D Obstacle'],
    'narrow':      ['2D Narrow', '3D Narrow'],
    'maze':        ['2D Maze'],
    'bug_trap':    ['2D Bug Trap'],
    'forest':      ['2D Forest'],
    'terrain':     ['2D Terrain'],
    'hyper_dense': ['2D Hyper-Dense'],
    'joint_arm':   ['2D Joint Arm'],
    'random_world': ['2D Random World'],
    'dividing_wall': ['2D Dividing Wall'],
    'dw': ['2D Dividing Wall'],
    'diagonal_3d': ['3D Diagonal'],
    'spheres':     ['3D Spheres'],
    'narrow_3d':   ['3D Narrow'],
    'dense_lab':   ['3D Dense Lab'],
    'corridor':    ['3D Corridor'],
    'gauntlet':    ['3D Gauntlet'],
    'wall_and_gaps': ['3D Wall & Gaps'],
    'box_field':   ['3D Box Field'],
    'hyper_passage': ['6D Hyper-Passage'],
    'obstacle_e':  ['2D Obstacle-E'],
    'narrow_e':    ['2D Narrow-E'],
    'maze_e':      ['2D Maze-E'],
    'forest_e':    ['2D Forest-E'],
    'tabletop':    ['6D Tabletop'],
    'shelf':       ['6D Shelf'],
    'cluttered':   ['6D Cluttered'],
    'real_setup':  ['6D Real Setup'],
}

# Add 6D PyBullet environments if available
if ALL_6D_ENVS:
    from rit_star.environments import env_6d_tabletop, env_6d_shelf, env_6d_cluttered, env_6d_real_setup
    ENV_REGISTRY['6D Tabletop']  = (env_6d_tabletop,  '6d')
    ENV_REGISTRY['6D Shelf']     = (env_6d_shelf,      '6d')
    ENV_REGISTRY['6D Cluttered'] = (env_6d_cluttered,  '6d')
    ENV_REGISTRY['6D Real Setup'] = (env_6d_real_setup, '6d')


# ═══════════════════════════════════════════════════════════════════════
#  Planner builder
# ═══════════════════════════════════════════════════════════════════════

def _build_planner(name, xs, xg, bounds, coll, metric,
                   batch_size, max_iterations, seed):
    common = dict(
        x_start=xs, x_goal=xg, c_space_bounds=bounds,
        collision_checker=coll, metric=metric,
        batch_size=batch_size, max_iterations=max_iterations,
        random_seed=seed,
    )
    if name == 'RIT*':
        return RITStar(xs, xg, bounds, coll, metric,
                       geodesic_tier='diagonal', batch_size=batch_size,
                       max_iterations=max_iterations, random_seed=seed)
    elif name == 'RIT*-CARM':
        dim = len(xs)
        euclid = EuclideanMetric(dim)
        return RITStar(xs, xg, bounds, coll, euclid,
                       geodesic_tier='diagonal', batch_size=batch_size,
                       max_iterations=max_iterations, random_seed=seed,
                       adaptive_metric=True,
                       carm_sigma=0.08, carm_alpha=6.0,
                       carm_rebuild_interval=15)
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
    else:
        raise ValueError(f'Unknown planner: {name}')


# ═══════════════════════════════════════════════════════════════════════
#  Config parser
# ═══════════════════════════════════════════════════════════════════════

def _resolve_planners(spec) -> list[str]:
    """Resolve planner specification to a list of canonical planner names."""
    if isinstance(spec, str):
        spec = spec.strip()
        if spec.lower() == 'all':
            return list(ALL_PLANNER_NAMES)
        # Single planner or comma/space separated
        tokens = [t.strip() for t in spec.replace(',', ' ').split()]
        spec = tokens
    # spec is a list
    resolved = []
    for token in spec:
        token_lower = str(token).strip().lower()
        if token_lower == 'all':
            return list(ALL_PLANNER_NAMES)
        if token_lower in PLANNER_REGISTRY:
            name = PLANNER_REGISTRY[token_lower]
            if name not in resolved:
                resolved.append(name)
        else:
            # Try partial match
            matches = [v for k, v in PLANNER_REGISTRY.items()
                        if token_lower in k]
            if matches:
                for m in matches:
                    if m not in resolved:
                        resolved.append(m)
            else:
                print(f"  WARNING: Unknown planner '{token}', skipping.")
    return resolved


def _resolve_environments(spec) -> list[str]:
    """Resolve environment specification to a list of canonical env names."""
    if isinstance(spec, str):
        spec = spec.strip()
        if spec.lower() == 'all':
            return list(ENV_REGISTRY.keys())
        tokens = [t.strip() for t in spec.replace(',', ' ').split()]
        spec = tokens
    resolved = []
    for token in spec:
        token_lower = str(token).strip().lower()
        if token_lower == 'all':
            return list(ENV_REGISTRY.keys())
        # Check dimension tags: 2d, 3d, 6d, 2d_euclid, euclid
        if token_lower in ('2d', '3d', '6d', '2d_euclid', 'euclid'):
            match_tag = token_lower
            if token_lower == 'euclid':
                match_tag = '2d_euclid'
            for env_name, (_, dim_tag) in ENV_REGISTRY.items():
                if dim_tag == match_tag and env_name not in resolved:
                    resolved.append(env_name)
            continue
        # Check aliases
        if token_lower in ENV_ALIASES:
            for env_name in ENV_ALIASES[token_lower]:
                if env_name in ENV_REGISTRY and env_name not in resolved:
                    resolved.append(env_name)
            continue
        # Check canonical name (case-insensitive partial match)
        matched = False
        for env_name in ENV_REGISTRY:
            if token_lower in env_name.lower():
                if env_name not in resolved:
                    resolved.append(env_name)
                matched = True
        if not matched:
            print(f"  WARNING: Unknown environment '{token}', skipping.")
    return resolved


def load_config(path: str) -> dict:
    """Load and validate a run configuration file."""
    with open(path, 'r') as f:
        cfg = yaml.safe_load(f)

    planners = _resolve_planners(cfg.get('planners', 'all'))
    environments = _resolve_environments(cfg.get('environments', 'all'))

    return {
        'planners': planners,
        'environments': environments,
        'n_trials': int(cfg.get('n_trials', 2)),
        'max_iterations': int(cfg.get('max_iterations', 150)),
        'batch_size': int(cfg.get('batch_size', 100)),
        'base_seed': int(cfg.get('base_seed', 42)),
        # Output options
        'save_image': bool(cfg.get('save_image', False)),
        'save_gif': bool(cfg.get('save_gif', False)),
        # Benchmark plots
        'run_benchmark_plots': bool(cfg.get('run_benchmark_plots', False)),
        'generate_benchmark_plots': bool(cfg.get('generate_benchmark_plots', True)),
        'generate_benchmark_tables': bool(cfg.get('generate_benchmark_tables', True)),
        'bench_n_trials': int(cfg.get('bench_n_trials', 10)),
        'bench_max_iterations': int(cfg.get('bench_max_iterations', 150)),
        'bench_batch_size': int(cfg.get('bench_batch_size', 100)),
        'bench_base_seed': int(cfg.get('bench_base_seed', 42)),
        # Monte Carlo comparison
        'run_mc': bool(cfg.get('run_mc', False)),
        'mc_n_trials': int(cfg.get('mc_n_trials', 10)),
        'mc_max_iterations': int(cfg.get('mc_max_iterations', 150)),
        'mc_batch_size': int(cfg.get('mc_batch_size', 100)),
        'mc_base_seed': int(cfg.get('mc_base_seed', 42)),
        'mc_visualize': bool(cfg.get('mc_visualize', True)),
        # Ablation study (paper Table \ref{tab:ablation})
        'run_ablation': bool(cfg.get('run_ablation', False)),
        'ablation_n_trials': int(cfg.get('ablation_n_trials', 10)),
        'ablation_max_iterations': int(cfg.get('ablation_max_iterations', 150)),
        'ablation_batch_size': int(cfg.get('ablation_batch_size', 100)),
        'ablation_base_seed': int(cfg.get('ablation_base_seed', 42)),
        'ablation_envs': cfg.get('ablation_envs') or None,
    }


# ═══════════════════════════════════════════════════════════════════════
#  Main runner
# ═══════════════════════════════════════════════════════════════════════

# ═══════════════════════════════════════════════════════════════════════
#  Visualization helpers
# ═══════════════════════════════════════════════════════════════════════

# Planner colors for comparison figures (matches rit_star/comparison.py)
_PLANNER_COLORS = {
    'RIT*':          '#7B2FBE',   # purple
    'RIT*-CARM':     '#00695C',   # dark teal
    'Informed RRT*': '#2196F3',   # blue
    'BIT*':          '#E91E63',   # pink
    'AIT*':          '#FF9800',   # orange
    'EIT*':          '#795548',   # brown
    'APT*':          '#F44336',   # red
}

# Map canonical env names → obstacle-drawing keys used by _draw_obstacles_2d
_ENV_OBSTACLE_KEY = {
    '2D Diagonal':    '2d_diagonal',
    '2D Obstacle':    '2d_obstacle',
    '2D Narrow':      '2d_narrow_passage',
    '2D Maze':        '2d_maze',
    '2D Bug Trap':    '2d_bug_trap',
    '2D Forest':      '2d_random_forest',
    '2D Terrain':     '2d_terrain',
    '2D Hyper-Dense': '2d_hyper_dense',
    '2D Joint Arm':   '2d_arm',
    '2D Obstacle-E':  '2d_obstacle',
    '2D Narrow-E':    '2d_narrow_passage',
    '2D Maze-E':      '2d_maze',
    '2D Forest-E':    '2d_random_forest',
    '2D Random World': '2d_random_world',
    '2D Dividing Wall': '2d_dividing_wall',
}


def _obstacle_key(env_name):
    """Resolve env_name to the key expected by _draw_obstacles_2d."""
    # Try direct match first
    if env_name in _ENV_OBSTACLE_KEY:
        return _ENV_OBSTACLE_KEY[env_name]
    # Strip planner suffix (e.g. '2D Maze_RIT*' → '2D Maze')
    base = env_name.rsplit('_', 1)[0] if '_' in env_name else env_name
    return _ENV_OBSTACLE_KEY.get(base, env_name)


def _save_image_2d(env_name, planner, path, coll, metric, xs, xg, bounds):
    """Save PNGs: path-only and tree+path for a 2D environment."""
    from rit_star.visualize import _draw_obstacles_2d
    safe = env_name.lower().replace(' ', '_').replace('*', '')
    obs_key = _obstacle_key(env_name)

    # 1) Path-only image (in plots/)
    fig, ax = plt.subplots(1, 1, figsize=(7, 7))
    ax.set_xlim(bounds[0])
    ax.set_ylim(bounds[1])
    ax.set_aspect('equal')
    _draw_obstacles_2d(ax, obs_key)
    if path and len(path) > 1:
        px = [p[0] for p in path]
        py = [p[1] for p in path]
        ax.plot(px, py, color='red', lw=3.5, zorder=4)
    ax.plot(*xs, 'go', ms=10, zorder=5)
    ax.plot(*xg, 'r^', ms=10, zorder=5)
    ax.set_xlabel('x₁', fontsize=14)
    ax.set_ylabel('x₂', fontsize=14)
    ax.tick_params(axis='both', labelsize=12)
    out_path = os.path.join(PLOTS_DIR, f'config_{safe}_path.png')
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'        Saved path image: {out_path}')

    # 2) Tree + path image (in images/)
    fig, ax = plt.subplots(1, 1, figsize=(7, 7))
    ax.set_xlim(bounds[0])
    ax.set_ylim(bounds[1])
    ax.set_aspect('equal')
    _draw_obstacles_2d(ax, obs_key)
    # Tree edges (green, thick)
    for v in planner.vertices:
        if v.parent is not None:
            ax.plot([v.x[0], v.parent.x[0]], [v.x[1], v.parent.x[1]],
                    color='green', lw=0.8, alpha=0.6, zorder=2)
    # Path (red)
    if path and len(path) > 1:
        px = [p[0] for p in path]
        py = [p[1] for p in path]
        ax.plot(px, py, color='red', lw=3.5, zorder=4)
    ax.plot(*xs, 'ko', ms=10, zorder=5)
    ax.plot(*xg, 'k^', ms=10, zorder=5)
    ax.set_xlabel('x₁', fontsize=14)
    ax.set_ylabel('x₂', fontsize=14)
    ax.tick_params(axis='both', labelsize=12)
    out_tree = os.path.join(IMAGES_DIR, f'config_{safe}_tree.png')
    fig.savefig(out_tree, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'        Saved tree image: {out_tree}')


def _save_comparison_figure_2d(env_name, planner_records, xs, xg, bounds):
    """Save a side-by-side comparison figure for all planners on one 2D env."""
    from rit_star.visualize import _draw_obstacles_2d

    obs_key = _ENV_OBSTACLE_KEY.get(env_name, '')
    safe_env = env_name.lower().replace(' ', '_').replace('*', '').replace('-', '_')
    n = len(planner_records)
    if n == 0:
        return

    n_cols = n  # all planners in a single row
    n_rows = 1

    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(5 * n_cols, 5),
                             squeeze=False)

    for idx, rec in enumerate(planner_records):
        ax = axes[0][idx]
        pname = rec['planner_name']
        color = _PLANNER_COLORS.get(pname, '#333333')

        ax.set_xlim(bounds[0])
        ax.set_ylim(bounds[1])
        ax.set_aspect('equal')

        # Tree edges (green, matching single-planner demos)
        for p1, p2 in rec['tree_edges']:
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]],
                    color='green', lw=0.4, alpha=0.5, zorder=1)

        # Vertex scatter (green dots)
        vp = rec['vertex_pts']
        if vp is not None and len(vp) > 0:
            ax.scatter(vp[:, 0], vp[:, 1], s=1, c='green',
                       alpha=0.3, zorder=2, edgecolors='none')

        # Obstacles on top of tree
        _draw_obstacles_2d(ax, obs_key)

        # Path
        if rec['path'] is not None and len(rec['path']) > 1:
            pa = rec['path']
            ax.plot(pa[:, 0], pa[:, 1], color=color, lw=3.5,
                    zorder=6, solid_capstyle='round')

        # Start / Goal markers
        ax.plot(*xs, 'o', color='#2E7D32', ms=8, zorder=7, mec='white', mew=1.0)
        ax.plot(*xg, '^', color='#C62828', ms=9, zorder=7, mec='white', mew=0.8)

        cost_str = f'{rec["cost"]:.4f}' if np.isfinite(rec['cost']) else 'no soln'
        ax.set_title(
            f'{pname}\ncost={cost_str}  time={rec["time"]:.1f}s',
            color=color, fontsize=10, fontweight='bold'
        )

    fig.suptitle(f'Path Comparison — {env_name}', fontsize=13, fontweight='bold')
    fig.tight_layout()
    out = os.path.join(IMAGES_DIR, f'path_comparison_{safe_env}.png')
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'        Saved comparison figure: {out}')


def _save_comparison_figure_3d(env_name, planner_records, xs, xg, bounds):
    """Save a side-by-side comparison figure for all planners on one 3D env."""
    safe_env = env_name.lower().replace(' ', '_').replace('*', '').replace('-', '_')
    n = len(planner_records)
    if n == 0:
        return

    fig = plt.figure(figsize=(5 * n, 5))

    for idx, rec in enumerate(planner_records):
        ax = fig.add_subplot(1, n, idx + 1, projection='3d')
        pname = rec['planner_name']
        color = _PLANNER_COLORS.get(pname, '#333333')

        if bounds is not None:
            ax.set_xlim(bounds[0])
            ax.set_ylim(bounds[1])
            ax.set_zlim(bounds[2])

        # Tree edges
        for p1, p2 in rec['tree_edges']:
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]],
                    color='green', lw=0.4, alpha=0.5, zorder=1)

        # Obstacles on top of tree
        _draw_obstacles_3d_static(ax, env_name)

        # Path
        if rec['path'] is not None and len(rec['path']) > 1:
            pa = rec['path']
            ax.plot(pa[:, 0], pa[:, 1], pa[:, 2], color=color, lw=3.5,
                    zorder=6, solid_capstyle='round')

        # Start / Goal markers
        ax.scatter(*xs, c='#2E7D32', s=80, zorder=7, marker='o')
        ax.scatter(*xg, c='#C62828', s=80, zorder=7, marker='^')

        cost_str = f'{rec["cost"]:.4f}' if np.isfinite(rec['cost']) else 'no soln'
        ax.set_title(
            f'{pname}\ncost={cost_str}  time={rec["time"]:.1f}s',
            color=color, fontsize=10, fontweight='bold'
        )

    fig.suptitle(f'Path Comparison — {env_name}', fontsize=13, fontweight='bold')
    fig.tight_layout()
    out = os.path.join(IMAGES_DIR, f'path_comparison_{safe_env}.png')
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'        Saved 3D comparison figure: {out}')


def _draw_obstacles_3d_static(ax, env_name):
    """Draw solid grey 3D obstacles (boxes or spheres) onto an Axes3D."""
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    # Strip planner suffix (e.g. '3D Box Field_RIT*' → '3D Box Field')
    env_name = env_name.rsplit('_', 1)[0] if '_' in env_name else env_name

    def _box_faces(lo, hi):
        lo, hi = np.array(lo), np.array(hi)
        return [
            [lo, [hi[0],lo[1],lo[2]], [hi[0],hi[1],lo[2]], [lo[0],hi[1],lo[2]]],  # bottom
            [lo+[0,0,hi[2]-lo[2]], [hi[0],lo[1],hi[2]], [hi[0],hi[1],hi[2]], [lo[0],hi[1],hi[2]]],  # top
            [lo, [hi[0],lo[1],lo[2]], [hi[0],lo[1],hi[2]], [lo[0],lo[1],hi[2]]],  # front
            [[lo[0],hi[1],lo[2]], hi-[0,0,hi[2]-lo[2]], [hi[0],hi[1],hi[2]], [lo[0],hi[1],hi[2]]],  # back
            [lo, [lo[0],hi[1],lo[2]], [lo[0],hi[1],hi[2]], [lo[0],lo[1],hi[2]]],  # left
            [[hi[0],lo[1],lo[2]], hi-[0,hi[1]-lo[1],0], [hi[0],hi[1],hi[2]], [hi[0],lo[1],hi[2]]],  # right
        ]

    if env_name == '3D Diagonal':
        boxes = [
            ([0.25, 0.0, 0.0], [0.35, 0.6, 1.0]),
            ([0.45, 0.4, 0.0], [0.55, 1.0, 1.0]),
            ([0.65, 0.0, 0.0], [0.75, 0.6, 0.6]),
            ([0.65, 0.0, 0.7], [0.75, 0.6, 1.0]),
        ]
        for lo, hi in boxes:
            faces = _box_faces(lo, hi)
            poly = Poly3DCollection(faces, alpha=0.55,
                                    facecolor='#666666', edgecolor='#333333', linewidth=0.4)
            ax.add_collection3d(poly)

    elif env_name in ('3D Spheres', '3D Dense Lab', '3D Gauntlet'):
        if env_name == '3D Spheres':
            offsets = [-0.35, 0.35]
            centres = [[x, y, z]
                       for x in offsets for y in offsets for z in offsets]
            centres.append([0.0, 0.0, 0.0])
            r = 0.22
        elif env_name == '3D Dense Lab':
            centres = [
                [-0.5,-0.5,-0.5],[0.2,-0.6,-0.3],[0.6,-0.4,-0.6],
                [-0.3,0.0,0.0],[0.3,0.1,0.1],[0.0,0.5,0.0],
                [-0.6,0.3,0.2],[0.6,0.3,-0.1],[-0.4,0.6,0.5],
                [0.2,0.7,0.4],[0.5,0.5,0.6],[-0.1,-0.2,0.6],
                [0.0,0.0,0.5],[-0.6,-0.3,0.3],[0.5,-0.1,0.4],
            ]
            r = 0.18
        else:  # 3D Gauntlet
            centres = [
                [-0.6, 0.25, 0.0], [-0.2, 0.25, 0.0],
                [ 0.2, 0.25, 0.0], [ 0.6, 0.25, 0.0],
                [-0.4,-0.25, 0.0], [ 0.0,-0.25, 0.0],
                [ 0.4,-0.25, 0.0], [-0.1, 0.0, 0.30],
                [ 0.1, 0.0,-0.30], [-0.7,-0.15, 0.25],
                [ 0.7, 0.15,-0.25], [ 0.0, 0.0, 0.0],
            ]
            r = 0.20
        u = np.linspace(0, 2 * np.pi, 14)
        v = np.linspace(0, np.pi, 10)
        for c in centres:
            xs_ = c[0] + r * np.outer(np.cos(u), np.sin(v))
            ys_ = c[1] + r * np.outer(np.sin(u), np.sin(v))
            zs_ = c[2] + r * np.outer(np.ones_like(u), np.cos(v))
            ax.plot_surface(xs_, ys_, zs_, color='#606060',
                            alpha=0.6, shade=True, linewidth=0)

    elif env_name == '3D Narrow':
        # Wall slab at x=[0.47, 0.53] with cylindrical hole at (y,z)=(0.5,0.5), r=0.09
        wall_faces = _box_faces([0.47, 0.0, 0.0], [0.53, 1.0, 1.0])
        poly = Poly3DCollection(wall_faces, alpha=0.25,
                                facecolor='#888888', edgecolor='#555555', linewidth=0.3)
        ax.add_collection3d(poly)
        # Mark hole with translucent circle
        u = np.linspace(0, 2 * np.pi, 30)
        ys_ = 0.5 + 0.09 * np.cos(u)
        zs_ = 0.5 + 0.09 * np.sin(u)
        ax.plot([0.50]*len(u), ys_, zs_, color='#00cc00', lw=2.0, alpha=0.8)

    elif env_name == '3D Corridor':
        # 4 axis-aligned box obstacles
        corridor_boxes = [
            ([0.28, 0.15, 0.0], [0.38, 1.0, 1.0]),
            ([0.62, 0.0, 0.0],  [0.72, 0.85, 1.0]),
            ([0.38, 0.0, 0.35], [0.62, 0.15, 0.65]),
            ([0.38, 0.55, 0.35],[0.62, 0.85, 0.65]),
        ]
        for lo, hi in corridor_boxes:
            faces = _box_faces(lo, hi)
            poly = Poly3DCollection(faces, alpha=0.55,
                                    facecolor='#666666', edgecolor='#333333', linewidth=0.4)
            ax.add_collection3d(poly)

    elif env_name == '3D Wall & Gaps':
        # Wall slab
        wall_faces = _box_faces([0.47, 0.0, 0.0], [0.53, 1.0, 1.0])
        poly = Poly3DCollection(wall_faces, alpha=0.25,
                                facecolor='#888888', edgecolor='#555555', linewidth=0.3)
        ax.add_collection3d(poly)
        # Flanking box obstacles
        flanking_boxes = [
            ([0.20, 0.60, 0.00], [0.35, 0.85, 0.40]),
            ([0.65, 0.15, 0.60], [0.80, 0.40, 1.00]),
            ([0.30, 0.00, 0.60], [0.45, 0.25, 0.90]),
            ([0.55, 0.75, 0.10], [0.70, 1.00, 0.40]),
        ]
        for lo, hi in flanking_boxes:
            faces = _box_faces(lo, hi)
            poly = Poly3DCollection(faces, alpha=0.55,
                                    facecolor='#505050', edgecolor='#2a2a2a', linewidth=0.5)
            ax.add_collection3d(poly)
        # Mark hole positions with translucent green circles (visible clearance)
        u = np.linspace(0, 2 * np.pi, 30)
        for hc, hr in [([0.35, 0.35], 0.10), ([0.70, 0.70], 0.10)]:
            ys_ = hc[0] + hr * np.cos(u)
            zs_ = hc[1] + hr * np.sin(u)
            ax.plot([0.50]*len(u), ys_, zs_, color='#00cc00', lw=2.0, alpha=0.8)

    elif env_name == '3D Box Field':
        boxes = [
            ([0.15, 0.15, 0.00], [0.35, 0.35, 0.45]),
            ([0.45, 0.00, 0.00], [0.65, 0.25, 0.35]),
            ([0.70, 0.30, 0.00], [0.90, 0.55, 0.30]),
            ([0.10, 0.50, 0.30], [0.30, 0.75, 0.60]),
            ([0.40, 0.40, 0.35], [0.60, 0.65, 0.65]),
            ([0.65, 0.55, 0.25], [0.85, 0.80, 0.55]),
            ([0.20, 0.20, 0.60], [0.45, 0.45, 0.85]),
            ([0.50, 0.60, 0.65], [0.75, 0.85, 0.90]),
            ([0.10, 0.70, 0.55], [0.30, 0.95, 0.80]),
            ([0.70, 0.10, 0.50], [0.90, 0.35, 0.80]),
        ]
        for lo, hi in boxes:
            faces = _box_faces(lo, hi)
            poly = Poly3DCollection(faces, alpha=0.55,
                                    facecolor='#505050', edgecolor='#2a2a2a', linewidth=0.5)
            ax.add_collection3d(poly)


def _save_image_3d(env_name, planner, path):
    """Save PNGs: path-only and tree+path for a 3D environment."""
    safe = env_name.lower().replace(' ', '_').replace('*', '')
    tree_edges = [(v.parent.x, v.x) for v in planner.vertices if v.parent]
    xs_arr = planner.vertices[0].x
    xg_arr = path[-1] if path else planner.vertices[0].x

    # 1) Path-only image (in plots/)
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    _draw_obstacles_3d_static(ax, env_name)
    if path and len(path) > 1:
        parr = np.array(path)
        ax.plot(parr[:, 0], parr[:, 1], parr[:, 2], color='red', lw=2.5, zorder=4)
    ax.scatter(*xs_arr, c='green', s=80, zorder=5)
    ax.scatter(*xg_arr, c='red', s=80, marker='^', zorder=5)
    out_path = os.path.join(PLOTS_DIR, f'config_{safe}_path.png')
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'        Saved path image: {out_path}')

    # 2) Tree + path image (in images/)
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    # Tree edges (green, thick)
    for parent_x, child_x in tree_edges:
        ax.plot([parent_x[0], child_x[0]],
                [parent_x[1], child_x[1]],
                [parent_x[2], child_x[2]],
                color='green', lw=0.8, alpha=0.6, zorder=2)
    # Obstacles drawn on top of tree so they're clearly visible
    _draw_obstacles_3d_static(ax, env_name)
    # Path (red)
    if path and len(path) > 1:
        parr = np.array(path)
        ax.plot(parr[:, 0], parr[:, 1], parr[:, 2], color='red', lw=2.5, zorder=4)
    ax.scatter(*xs_arr, c='black', s=80, zorder=5)
    ax.scatter(*xg_arr, c='black', s=80, marker='^', zorder=5)
    out_tree = os.path.join(IMAGES_DIR, f'config_{safe}_tree.png')
    fig.savefig(out_tree, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'        Saved tree image: {out_tree}')


def _save_gif(env_name, env_fn, dim_tag):
    """Save a tree-growth GIF for a 2D or 3D environment.

    For 2D envs, additionally produce a CARM-aware GIF where the gradient
    field updates as RIT* grows the tree, plus a final summary PNG showing
    scene + tree + path + final gradient.
    """
    safe = env_name.lower().replace(' ', '_')
    if dim_tag.startswith('2d'):
        from visualization_util.visualize_riemannian import (
            animate_tree_growth, animate_tree_growth_carm,
        )
        animate_tree_growth(env_name, env_fn, f'config_{safe}',
                            max_iterations=80, batch_size=100,
                            frame_every=2, fps=8)
        animate_tree_growth_carm(env_name, env_fn, f'config_{safe}',
                                 max_iterations=80, batch_size=100,
                                 frame_every=2, fps=8)
    elif dim_tag == '3d':
        from visualization_util.visualize_riemannian import (
            animate_3d_env, animate_3d_env_carm,
        )
        animate_3d_env(env_name, env_fn, f'config_{safe}',
                       max_iterations=80, batch_size=100,
                       frame_every=2, fps=8)
        animate_3d_env_carm(env_name, env_fn, f'config_{safe}',
                            max_iterations=80, batch_size=100,
                            frame_every=2, fps=8)
    else:
        print(f'        GIF generation not supported for {dim_tag.upper()} environments.')
        return
    print(f'        Saved GIF for {env_name}')


def run(cfg: dict):
    """Execute the benchmark according to the resolved configuration."""
    planners = cfg['planners']
    environments = cfg['environments']
    n_trials = cfg['n_trials']
    max_iter = cfg['max_iterations']
    batch_size = cfg['batch_size']
    base_seed = cfg['base_seed']
    save_image = cfg.get('save_image', False)
    save_gif = cfg.get('save_gif', False)

    total_runs = len(planners) * len(environments) * n_trials
    print('=' * 60)
    print('  CONFIG-DRIVEN BENCHMARK')
    print('=' * 60)
    print(f'  Planners:      {planners}')
    print(f'  Environments:  {environments}')
    print(f'  Trials:        {n_trials}')
    print(f'  Max iters:     {max_iter}')
    print(f'  Batch size:    {batch_size}')
    print(f'  Base seed:     {base_seed}')
    print(f'  Save images:   {save_image}')
    print(f'  Save GIFs:     {save_gif}')
    print(f'  Total runs:    {total_runs}')
    print('=' * 60)

    all_results = {}
    run_count = 0

    for env_name in environments:
        env_fn, dim_tag = ENV_REGISTRY[env_name]
        print(f'\n{"─" * 60}')
        print(f'  Environment: {env_name} ({dim_tag.upper()})')
        print(f'{"─" * 60}')

        coll, _, metric, xs, xg, bounds = env_fn()
        env_results = {}
        comparison_records = []  # accumulate per-planner data for comparison figure

        gif_saved = False
        for planner_name in planners:
            print(f'\n    Planner: {planner_name}')
            trial_results = []

            for trial in range(n_trials):
                run_count += 1
                seed = base_seed + trial
                print(f'      Trial {trial + 1}/{n_trials} '
                      f'[{run_count}/{total_runs}] ...', end=' ', flush=True)

                t0 = time.time()
                planner = _build_planner(
                    planner_name, xs, xg, bounds, coll, metric,
                    batch_size, max_iter, seed)
                path, cost = planner.plan()
                elapsed = time.time() - t0
                stats = planner.get_stats()

                trial_results.append({
                    'final_cost': cost if np.isfinite(cost) else np.inf,
                    'time_elapsed': elapsed,
                    'iterations': stats[-1]['iteration'] if stats else 0,
                })
                print(f'cost={cost:.4f}  time={elapsed:.2f}s')

                # Save image on last trial only
                if save_image and trial == n_trials - 1:
                    dim = len(xs)
                    if dim == 2:
                        if path:
                            _save_image_2d(f'{env_name}_{planner_name}',
                                           planner, path, coll, metric,
                                           xs, xg, bounds)
                        comparison_records.append({
                            'planner_name': planner_name,
                            'tree_edges': [(v.parent.x.copy(), v.x.copy())
                                           for v in planner.vertices if v.parent],
                            'vertex_pts': np.array([v.x for v in planner.vertices]),
                            'path': np.array(path) if path else None,
                            'cost': cost,
                            'time': elapsed,
                        })
                    elif dim == 3:
                        if path:
                            _save_image_3d(f'{env_name}_{planner_name}',
                                           planner, path)
                        comparison_records.append({
                            'planner_name': planner_name,
                            'tree_edges': [(v.parent.x.copy(), v.x.copy())
                                           for v in planner.vertices if v.parent],
                            'vertex_pts': np.array([v.x for v in planner.vertices]),
                            'path': np.array(path) if path else None,
                            'cost': cost,
                            'time': elapsed,
                        })

                del planner

            gc.collect()
            env_results[planner_name] = trial_results

            # Save tree-growth GIF immediately after RIT*'s trials complete
            # (so the user sees artefacts even if the run is interrupted
            # before every planner finishes).
            if save_gif and not gif_saved and 'RIT' in planner_name.upper():
                try:
                    _save_gif(env_name, env_fn, dim_tag)
                except Exception as exc:
                    print(f'        [WARN] GIF save failed: {exc}')
                gif_saved = True

        # Save comparison figure for 2D environments
        if save_image and dim_tag.startswith('2d') and comparison_records:
            _save_comparison_figure_2d(env_name, comparison_records, xs, xg, bounds)

        # Save comparison figure for 3D environments
        if save_image and dim_tag == '3d' and comparison_records:
            _save_comparison_figure_3d(env_name, comparison_records, xs, xg, bounds)

        # Fallback: save GIF at end of env if RIT* wasn't in the planner list
        if save_gif and not gif_saved:
            try:
                _save_gif(env_name, env_fn, dim_tag)
            except Exception as exc:
                print(f'        [WARN] GIF save failed: {exc}')

        all_results[env_name] = env_results

    # ── Summary ───────────────────────────────────────────────────
    print('\n' + '=' * 60)
    print('  SUMMARY')
    print('=' * 60)
    print(f'  {"Environment":<22} {"Planner":<18} {"Cost (mean±std)":<22} {"Time (mean)"}')
    print(f'  {"─" * 22} {"─" * 18} {"─" * 22} {"─" * 12}')

    for env_name, env_results in all_results.items():
        for planner_name, trials in env_results.items():
            costs = [t['final_cost'] for t in trials]
            times = [t['time_elapsed'] for t in trials]
            finite = [c for c in costs if np.isfinite(c)]
            if finite:
                mean_c = np.mean(finite)
                std_c = np.std(finite)
                cost_str = f'{mean_c:.4f} ± {std_c:.4f}'
            else:
                cost_str = 'no solution'
            mean_t = np.mean(times)
            print(f'  {env_name:<22} {planner_name:<18} {cost_str:<22} {mean_t:.2f}s')

    print('\nDone.')
    return all_results


def run_mc(cfg: dict):
    """Run Monte Carlo comparison using rit_star.comparison."""
    from rit_star.comparison import run_full_comparison

    print('\n' + '=' * 60)
    print('  MONTE CARLO COMPARISON')
    print('=' * 60)
    print(f'  MC trials:     {cfg["mc_n_trials"]}')
    print(f'  MC max iters:  {cfg["mc_max_iterations"]}')
    print(f'  MC batch size: {cfg["mc_batch_size"]}')
    print(f'  MC base seed:  {cfg["mc_base_seed"]}')
    print(f'  MC visualize:  {cfg["mc_visualize"]}')
    print(f'  Environments:  {cfg["environments"]}')
    print('=' * 60)

    # Build env dict from configured environments
    mc_envs = {}
    for env_name in cfg['environments']:
        env_fn, _ = ENV_REGISTRY[env_name]
        mc_envs[env_name] = env_fn

    run_full_comparison(
        n_trials=cfg['mc_n_trials'],
        max_iterations=cfg['mc_max_iterations'],
        batch_size=cfg['mc_batch_size'],
        base_seed=cfg['mc_base_seed'],
        visualize=cfg['mc_visualize'],
        environments=mc_envs,
    )
    print('\nMonte Carlo comparison done.')


def run_benchmark(cfg: dict):
    """Run AIT*/EIT*-style anytime benchmark plots."""
    from run_benchmark_plots import (
        _collect_data, plot_benchmark, plot_combined,
        generate_table_ii, generate_table_iii, generate_table_aggregated,
    )
    from visualization_util.output_paths import PLOTS_DIR

    print('\n' + '=' * 60)
    print('  ANYTIME BENCHMARK PLOTS')
    print('=' * 60)
    print(f'  Bench trials:     {cfg["bench_n_trials"]}')
    print(f'  Bench max iters:  {cfg["bench_max_iterations"]}')
    print(f'  Bench batch size: {cfg["bench_batch_size"]}')
    print(f'  Bench base seed:  {cfg["bench_base_seed"]}')
    print(f'  Generate plots:   {cfg["generate_benchmark_plots"]}')
    print(f'  Generate tables:  {cfg["generate_benchmark_tables"]}')
    print('=' * 60)

    planners = cfg['planners']
    environments = cfg['environments']
    all_results = {}

    for env_name in environments:
        env_fn, dim_tag = ENV_REGISTRY[env_name]
        print(f'\n  Environment: {env_name} ({dim_tag.upper()})')
        results = _collect_data(
            env_name, env_fn, planners, cfg['bench_n_trials'],
            cfg['bench_max_iterations'], cfg['bench_batch_size'],
            cfg['bench_base_seed'])
        all_results[env_name] = results

    os.makedirs(PLOTS_DIR, exist_ok=True)

    # Generate benchmark plots (Fig 6 style)
    if cfg['generate_benchmark_plots']:
        print('\nGenerating benchmark plots...')
        for env_name in all_results:
            plot_benchmark(env_name, all_results[env_name], planners, PLOTS_DIR)
        if len(all_results) > 1:
            plot_combined(all_results, planners, PLOTS_DIR)

    # Generate benchmark tables (Table II / III style + aggregated)
    if cfg['generate_benchmark_tables']:
        print('\nGenerating benchmark tables...')
        generate_table_ii(all_results, planners, PLOTS_DIR,
                          n_trials=cfg['bench_n_trials'])
        generate_table_iii(all_results, planners, PLOTS_DIR,
                           n_trials=cfg['bench_n_trials'])
        generate_table_aggregated(all_results, planners, PLOTS_DIR)

    print('\nBenchmark plots done.')


# ═══════════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    config_path = sys.argv[1] if len(sys.argv) > 1 else 'config/run_config.yaml'
    if not os.path.isfile(config_path):
        print(f'Config file not found: {config_path}')
        print(f'\nCreate one or copy the example:')
        print(f'  cp config/run_config.yaml my_config.yaml')
        sys.exit(1)

    cfg = load_config(config_path)

    if not cfg['planners']:
        print('No planners matched. Check your config.')
        sys.exit(1)
    if not cfg['environments']:
        print('No environments matched. Check your config.')
        sys.exit(1)

    run(cfg)

    if cfg.get('run_benchmark_plots', False):
        run_benchmark(cfg)

    if cfg.get('run_mc', False):
        run_mc(cfg)

    if cfg.get('run_ablation', False):
        from run_ablation import run_ablation
        run_ablation(cfg)
