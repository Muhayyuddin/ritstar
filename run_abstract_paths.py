"""
run_abstract_paths.py — Run Euclidean and CARM planners on the 6-circle
env_2d_obstacle_inflated environment and save tree + path as PNGs.

Outputs:
  paper/figures/abstract_euclidean_tree.png
  paper/figures/abstract_carm_tree.png
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rit_star.rit_star import RITStar, riemannian_edge_cost
from rit_star.environments import env_2d_obstacle_inflated
from rit_star.metric import EuclideanMetric

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       'paper', 'figures')
os.makedirs(OUT_DIR, exist_ok=True)

# ── Environment ──────────────────────────────────────────────────
coll, _, oracle_metric, xs, xg, bounds = env_2d_obstacle_inflated()
dim = len(xs)

circles = [
    (np.array([0.30, 0.35]), 0.08),
    (np.array([0.30, 0.65]), 0.08),
    (np.array([0.50, 0.45]), 0.09),
    (np.array([0.50, 0.75]), 0.09),
    (np.array([0.70, 0.40]), 0.08),
    (np.array([0.70, 0.60]), 0.08),
]

# ── Colours ──────────────────────────────────────────────────────
C_OBS = '#546E7A'
C_OBS_E = '#37474F'
C_START = '#2E7D32'
C_GOAL = '#C62828'
C_BG = '#FAFAFA'


def draw_env(ax):
    ax.set_facecolor(C_BG)
    for cen, r in circles:
        ax.add_patch(Circle(cen, r, fc=C_OBS, ec=C_OBS_E,
                            lw=0.6, alpha=0.7, zorder=3))
    ax.scatter(*xs, s=80, c=C_START, marker='s', edgecolors='k',
              lw=0.5, zorder=10)
    ax.scatter(*xg, s=100, c=C_GOAL, marker='*', edgecolors='k',
              lw=0.5, zorder=10)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])


def draw_tree(ax, planner, edge_color, edge_lw=0.3, edge_alpha=0.4):
    for v in planner.vertices:
        if v.parent is not None:
            ax.plot([v.parent.x[0], v.x[0]],
                    [v.parent.x[1], v.x[1]],
                    color=edge_color, lw=edge_lw, alpha=edge_alpha,
                    zorder=2)


def draw_path(ax, path, color, lw=2.5, label=None):
    if path and len(path) > 1:
        px = [p[0] for p in path]
        py = [p[1] for p in path]
        ax.plot(px, py, '-', color=color, lw=lw, zorder=8,
                solid_capstyle='round', label=label)


def oracle_cost(path):
    if not path or len(path) < 2:
        return float('inf')
    return sum(riemannian_edge_cost(path[i], path[i+1], oracle_metric)
               for i in range(len(path)-1))


# ═════════════════════════════════════════════════════════════════
#  1. Euclidean planner (baseline)
# ═════════════════════════════════════════════════════════════════
print('Running Euclidean planner ...')
planner_euc = RITStar(
    xs, xg, bounds, coll,
    EuclideanMetric(dim),
    batch_size=100,
    max_iterations=150,
    random_seed=42,
    adaptive_metric=False,
)
path_euc, cost_euc = planner_euc.plan()
oc_euc = oracle_cost(path_euc)
print(f'  Euclidean: tree nodes={len(planner_euc.vertices)}, '
      f'eucl_cost={cost_euc:.4f}, oracle_cost={oc_euc:.4f}')

fig, ax = plt.subplots(figsize=(6, 6))
draw_env(ax)
draw_tree(ax, planner_euc, '#B0BEC5', edge_lw=0.3, edge_alpha=0.45)
draw_path(ax, path_euc, '#455A64', lw=2.5,
          label=f'Euclidean path (oracle cost={oc_euc:.3f})')
ax.legend(fontsize=8, loc='upper left', framealpha=0.9)
ax.set_title(f'Euclidean Baseline — {len(planner_euc.vertices)} nodes',
             fontsize=11, fontweight='bold')
fig.tight_layout()
out1 = os.path.join(OUT_DIR, 'abstract_euclidean_tree.png')
fig.savefig(out1, dpi=200, bbox_inches='tight', facecolor='white')
plt.close(fig)
print(f'  Saved: {out1}')

# ═════════════════════════════════════════════════════════════════
#  2. RIT* + CARM planner (our approach)
# ═════════════════════════════════════════════════════════════════
print('Running RIT* + CARM planner ...')
planner_carm = RITStar(
    xs, xg, bounds, coll,
    EuclideanMetric(dim),
    batch_size=100,
    max_iterations=150,
    random_seed=42,
    adaptive_metric=True,
    carm_sigma=0.08,
    carm_alpha=6.0,
    carm_rebuild_interval=15,
)
path_carm, cost_carm = planner_carm.plan()
# Also extract the full tree path (before shortcutting removed waypoints)
path_carm_full = planner_carm._extract_path()
oc_carm = oracle_cost(path_carm)
n_coll = planner_carm._carm.n_collision_points if planner_carm._carm else 0
print(f'  CARM: tree nodes={len(planner_carm.vertices)}, '
      f'planner_cost={cost_carm:.4f}, oracle_cost={oc_carm:.4f}, '
      f'collisions={n_coll}')
print(f'  CARM full tree path: {len(path_carm_full)} waypoints '
      f'(shortcutted: {len(path_carm)})')

fig, ax = plt.subplots(figsize=(6, 6))
draw_env(ax)
draw_tree(ax, planner_carm, '#90CAF9', edge_lw=0.3, edge_alpha=0.4)
draw_path(ax, path_carm, '#C62828', lw=2.5,
          label=f'CARM path (oracle cost={oc_carm:.3f})')
ax.legend(fontsize=8, loc='upper left', framealpha=0.9)
ax.set_title(f'RIT* + CARM — {len(planner_carm.vertices)} nodes, '
             f'{n_coll} collisions',
             fontsize=11, fontweight='bold')
fig.tight_layout()
out2 = os.path.join(OUT_DIR, 'abstract_carm_tree.png')
fig.savefig(out2, dpi=200, bbox_inches='tight', facecolor='white')
plt.close(fig)
print(f'  Saved: {out2}')

# ═════════════════════════════════════════════════════════════════
#  3. Side-by-side comparison
# ═════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

ax = axes[0]
draw_env(ax)
draw_tree(ax, planner_euc, '#B0BEC5', edge_lw=0.3, edge_alpha=0.4)
draw_path(ax, path_euc, '#455A64', lw=2.5,
          label=f'Euclidean (oracle cost={oc_euc:.3f})')
ax.legend(fontsize=8, loc='upper left', framealpha=0.9)
ax.set_title('(a) Euclidean Baseline', fontsize=11, fontweight='bold')

ax = axes[1]
draw_env(ax)
draw_tree(ax, planner_carm, '#90CAF9', edge_lw=0.3, edge_alpha=0.4)
draw_path(ax, path_carm, '#C62828', lw=2.5,
          label=f'CARM (oracle cost={oc_carm:.3f})')
ax.legend(fontsize=8, loc='upper left', framealpha=0.9)
ax.set_title(f'(b) RIT* + CARM ({n_coll} collisions)',
             fontsize=11, fontweight='bold')

fig.suptitle('Euclidean vs CARM — env_2d_obstacle_inflated',
             fontsize=13, fontweight='bold')
fig.tight_layout()
out3 = os.path.join(OUT_DIR, 'abstract_comparison.png')
fig.savefig(out3, dpi=200, bbox_inches='tight', facecolor='white')
plt.close(fig)
print(f'  Saved: {out3}')

# ═════════════════════════════════════════════════════════════════
#  4. Save paths as numpy arrays for use in abstract figure
# ═════════════════════════════════════════════════════════════════
np.savez(os.path.join(OUT_DIR, 'abstract_paths.npz'),
         path_euc=np.array(path_euc) if path_euc else np.array([]),
         path_carm=np.array(path_carm) if path_carm else np.array([]),
         path_carm_full=np.array(path_carm_full) if path_carm_full else np.array([]),
         tree_euc_edges=np.array([(v.parent.x, v.x) for v in planner_euc.vertices
                                   if v.parent is not None]),
         tree_carm_edges=np.array([(v.parent.x, v.x) for v in planner_carm.vertices
                                    if v.parent is not None]),
         oracle_cost_euc=oc_euc,
         oracle_cost_carm=oc_carm)
print(f'  Saved paths: {OUT_DIR}/abstract_paths.npz')
print('Done.')
