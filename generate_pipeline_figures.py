#!/usr/bin/env python
"""
generate_pipeline_figures.py — Step-by-step tree growth pipeline figures.

Shows HOW the planner builds its tree, one step at a time:
  Fig A: Full iteration pipeline overview (sample → nearest → extend → rewire → prune)
  Fig B: Nearest-neighbor r-ball lookup + cascading parent selection
  Fig C: Rewiring — new node improves existing neighbors
  Fig D: CARM feedback loop — collisions feed back into metric
  Fig E: Informed set shrinkage over iterations (Euclidean vs CARM)
  Fig F: Multi-iteration tree growth sequence (8 snapshots)
"""

import os
import sys
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, Arc, Wedge
from matplotlib.collections import LineCollection
import matplotlib.patheffects as pe

from output_paths import PLOTS_DIR
from rit_star.rit_star import RITStar
from rit_star.metric import EuclideanMetric
from rit_star.environments import env_2d_obstacle_inflated

# ── Style ────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.size': 11,
    'font.family': 'serif',
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'legend.fontsize': 9,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
})

# Colors
TREE_EDGE = '#B0BEC5'
NEW_SAMPLE = '#FF9800'
BEST_PARENT = '#2196F3'
REWIRE_COLOR = '#9C27B0'
COLLISION_PT = '#F44336'
FREE_PT = '#4CAF50'
PATH_COLOR = '#E63946'
OBSTACLE_COLOR = '#2B2D42'
RADIUS_COLOR = '#2196F3'
PRUNE_COLOR = '#FF5722'
CARM_HEAT = '#FF6F00'
START_COLOR = '#2D6A4F'
GOAL_COLOR = '#E63946'

CIRCLES_OBS = [
    (np.array([0.30, 0.35]), 0.08),
    (np.array([0.30, 0.65]), 0.08),
    (np.array([0.50, 0.45]), 0.09),
    (np.array([0.50, 0.75]), 0.09),
    (np.array([0.70, 0.40]), 0.08),
    (np.array([0.70, 0.60]), 0.08),
]


def _draw_obstacles(ax, alpha=0.7):
    for c, r in CIRCLES_OBS:
        ax.add_patch(Circle(c, r, fc=OBSTACLE_COLOR, ec='white', lw=0.8, alpha=alpha, zorder=2))


def _draw_sg(ax, xs, xg, ms=10):
    ax.plot(*xs, 's', color=START_COLOR, ms=ms, zorder=10,
            markeredgecolor='white', markeredgewidth=1.2, label='Start')
    ax.plot(*xg, '*', color=GOAL_COLOR, ms=ms+3, zorder=10,
            markeredgecolor='white', markeredgewidth=0.8, label='Goal')


def _set_ax(ax, title=''):
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect('equal')
    ax.set_xticks([]); ax.set_yticks([])
    if title:
        ax.set_title(title, fontsize=11, fontweight='bold')


def _draw_tree(ax, vertices, lw=0.4, alpha=0.4, color=TREE_EDGE):
    for v in vertices:
        if v.parent is not None:
            ax.plot([v.parent.x[0], v.x[0]], [v.parent.x[1], v.x[1]],
                    '-', color=color, lw=lw, alpha=alpha, zorder=1)


def _arrow(ax, start, end, color='black', lw=1.5, style='->', zorder=5):
    ax.annotate('', xy=end, xytext=start,
                arrowprops=dict(arrowstyle=style, color=color, lw=lw),
                zorder=zorder)


def _save(fig, name):
    p_pdf = os.path.join(PLOTS_DIR, f'{name}.pdf')
    p_png = os.path.join(PLOTS_DIR, f'{name}.png')
    fig.savefig(p_pdf, dpi=300, bbox_inches='tight')
    fig.savefig(p_png, dpi=200, bbox_inches='tight')
    print(f'  -> {p_pdf}')
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
#  FIG A: Full Iteration Pipeline (6 panels)
# ══════════════════════════════════════════════════════════════════════════════
def fig_a_iteration_pipeline():
    """6-panel figure showing one complete iteration of the planner."""
    print('=== Fig A: Full Iteration Pipeline ===')

    coll, _, oracle_m, xs, xg, bounds = env_2d_obstacle_inflated()
    euclid = EuclideanMetric(2)

    # Run planner for 30 iterations to get a partially-built tree
    planner = RITStar(xs, xg, bounds, coll, euclid,
                      geodesic_tier='diagonal', batch_size=60,
                      max_iterations=30, random_seed=42,
                      adaptive_metric=True, carm_sigma=0.08,
                      carm_alpha=6.0, carm_rebuild_interval=10)
    planner.plan()
    verts = planner.vertices

    # Pick a representative new sample point in open space
    x_new = np.array([0.42, 0.18])
    # Find real neighbors
    near_verts = [v for v in verts
                  if np.linalg.norm(v.x - x_new) < 0.15 and v.parent is not None]
    if len(near_verts) < 3:
        near_verts = sorted(verts, key=lambda v: np.linalg.norm(v.x - x_new))[:5]

    # Also pick a sample that would collide
    x_coll = np.array([0.48, 0.46])  # inside obstacle at (0.50, 0.45)

    fig, axes = plt.subplots(2, 3, figsize=(17, 11))

    # ── Panel 1: Sample batch from informed set ──────────────────
    ax = axes[0, 0]
    _draw_obstacles(ax)
    _draw_tree(ax, verts)
    # Draw informed ellipsoid outline
    from matplotlib.patches import Ellipse
    c_best = planner.c_best
    mid = (xs + xg) / 2
    diff = xg - xs
    c_min = np.linalg.norm(diff)
    angle = np.degrees(np.arctan2(diff[1], diff[0]))
    a_ell = c_best / 2
    b_ell = np.sqrt(max(c_best**2 - c_min**2, 0)) / 2
    ax.add_patch(Ellipse(mid, 2*a_ell, 2*b_ell, angle=angle,
                         fc='#E3F2FD', ec=RADIUS_COLOR, lw=1.5, ls='--',
                         alpha=0.3, zorder=0))
    # Random sample points
    rng = np.random.default_rng(123)
    n_samp = 40
    samps = []
    for _ in range(n_samp):
        s = rng.uniform([0,0], [1,1])
        samps.append(s)
    samps = np.array(samps)
    # Color: free vs collision
    for s in samps:
        if coll(s):
            ax.plot(s[0], s[1], 'o', color=FREE_PT, ms=4, alpha=0.5, zorder=3)
        else:
            ax.plot(s[0], s[1], 'x', color=COLLISION_PT, ms=5, alpha=0.7, zorder=3)
    # Highlight x_new
    ax.plot(*x_new, 'o', color=NEW_SAMPLE, ms=12, zorder=6,
            markeredgecolor='black', markeredgewidth=1.5)
    ax.annotate('$x_{new}$', xy=x_new, xytext=(x_new[0]+0.04, x_new[1]-0.06),
                fontsize=12, fontweight='bold', color=NEW_SAMPLE,
                arrowprops=dict(arrowstyle='->', color=NEW_SAMPLE, lw=1.5))
    _draw_sg(ax, xs, xg, ms=8)
    _set_ax(ax, '(1) Sample Batch\nDraw B points from informed set')
    ax.text(0.5, 0.96, 'Informed ellipsoid $I_E$', transform=ax.transAxes,
            ha='center', va='top', fontsize=9, color=RADIUS_COLOR,
            bbox=dict(fc='white', ec=RADIUS_COLOR, alpha=0.8, boxstyle='round,pad=0.2'))

    # ── Panel 2: Nearest-neighbor r-ball ─────────────────────────
    ax = axes[0, 1]
    _draw_obstacles(ax, alpha=0.5)
    _draw_tree(ax, verts)
    # Draw r-ball
    r = 0.15
    ax.add_patch(Circle(x_new, r, fc='#E3F2FD', ec=RADIUS_COLOR, lw=2,
                        ls='--', alpha=0.4, zorder=3))
    ax.plot(*x_new, 'o', color=NEW_SAMPLE, ms=12, zorder=6,
            markeredgecolor='black', markeredgewidth=1.5)
    # Highlight neighbors inside r-ball
    for v in near_verts:
        if np.linalg.norm(v.x - x_new) < r:
            ax.plot(v.x[0], v.x[1], 'o', color=RADIUS_COLOR, ms=8, zorder=5,
                    markeredgecolor='white', markeredgewidth=1)
            ax.plot([v.x[0], x_new[0]], [v.x[1], x_new[1]],
                    ':', color=RADIUS_COLOR, lw=1, alpha=0.5)
    _draw_sg(ax, xs, xg, ms=8)
    _set_ax(ax, '(2) Nearest Neighbors\nFind all vertices within radius $r_n$')
    ax.annotate(f'$r_n$ = {r:.2f}', xy=(x_new[0]+r*0.7, x_new[1]+r*0.7),
                fontsize=10, color=RADIUS_COLOR, fontweight='bold',
                bbox=dict(fc='white', ec=RADIUS_COLOR, alpha=0.8, boxstyle='round,pad=0.2'))

    # ── Panel 3: Cascading parent selection ──────────────────────
    ax = axes[0, 2]
    _draw_obstacles(ax, alpha=0.5)
    _draw_tree(ax, verts, alpha=0.2)
    ax.plot(*x_new, 'o', color=NEW_SAMPLE, ms=12, zorder=6,
            markeredgecolor='black', markeredgewidth=1.5)

    # Show candidates with different check levels
    cands = sorted(near_verts, key=lambda v: np.linalg.norm(v.x - x_new))[:4]
    labels_check = ['L1 pass\nExact pass\nCollision free\n= BEST', 'L1 pass\nExact pass\nCollision!',
                    'L1 pass\nExact fail', 'L1 fail\n(skipped)']
    colors_check = [BEST_PARENT, COLLISION_PT, '#FFA726', '#BDBDBD']

    for i, v in enumerate(cands):
        if i >= len(labels_check):
            break
        c = colors_check[i]
        ax.plot(v.x[0], v.x[1], 'o', color=c, ms=9, zorder=5,
                markeredgecolor='black', markeredgewidth=1)
        # Edge to x_new
        style = '-' if i == 0 else ('--' if i < 3 else ':')
        lw = 2.5 if i == 0 else 1.2
        ax.plot([v.x[0], x_new[0]], [v.x[1], x_new[1]],
                style, color=c, lw=lw, zorder=4)
        # Label
        offset = [(0.04, 0.03), (-0.02, 0.05), (0.04, -0.03), (-0.06, -0.04)]
        oxy = offset[i] if i < len(offset) else (0.03, 0.03)
        ax.text(v.x[0]+oxy[0], v.x[1]+oxy[1], labels_check[i],
                fontsize=7, color=c, fontweight='bold', ha='center',
                bbox=dict(fc='white', ec=c, alpha=0.85, boxstyle='round,pad=0.2'))

    _draw_sg(ax, xs, xg, ms=8)
    _set_ax(ax, '(3) Cascading Parent Selection\nL1 estimate → Exact cost → Collision check')

    # ── Panel 4: Insert new node + edge ──────────────────────────
    ax = axes[1, 0]
    _draw_obstacles(ax, alpha=0.5)
    _draw_tree(ax, verts)
    best = cands[0] if cands else near_verts[0]
    # Draw the new edge fat
    ax.plot([best.x[0], x_new[0]], [best.x[1], x_new[1]],
            '-', color=BEST_PARENT, lw=3, zorder=5)
    ax.plot(*x_new, 'o', color=NEW_SAMPLE, ms=12, zorder=6,
            markeredgecolor='black', markeredgewidth=1.5)
    ax.plot(best.x[0], best.x[1], 'o', color=BEST_PARENT, ms=9, zorder=5,
            markeredgecolor='white', markeredgewidth=1.5)
    # Label
    mid_e = (best.x + x_new) / 2
    ax.annotate('New edge', xy=mid_e, fontsize=10, color=BEST_PARENT,
                fontweight='bold', ha='center',
                bbox=dict(fc='white', ec=BEST_PARENT, alpha=0.8, boxstyle='round,pad=0.2'))
    _draw_sg(ax, xs, xg, ms=8)
    _set_ax(ax, '(4) Insert New Node\nConnect $x_{new}$ to best parent')

    # ── Panel 5: Rewiring ────────────────────────────────────────
    ax = axes[1, 1]
    _draw_obstacles(ax, alpha=0.5)
    _draw_tree(ax, verts, alpha=0.2)
    ax.plot(*x_new, 'o', color=NEW_SAMPLE, ms=12, zorder=6,
            markeredgecolor='black', markeredgewidth=1.5)
    ax.plot([best.x[0], x_new[0]], [best.x[1], x_new[1]],
            '-', color=BEST_PARENT, lw=2.5, zorder=4)

    # Show rewiring: some neighbors switch parent to x_new
    rewire_targets = [v for v in near_verts if v is not best
                      and np.linalg.norm(v.x - x_new) < r][:2]
    for v in rewire_targets:
        # Old edge (being removed) — dashed red
        if v.parent is not None:
            ax.plot([v.parent.x[0], v.x[0]], [v.parent.x[1], v.x[1]],
                    '--', color=PRUNE_COLOR, lw=2, alpha=0.7, zorder=3)
            # X mark on old edge
            mid_old = (v.parent.x + v.x) / 2
            ax.plot(mid_old[0], mid_old[1], 'x', color=PRUNE_COLOR,
                    ms=10, mew=2.5, zorder=4)
        # New edge (rewired through x_new) — solid purple
        ax.plot([x_new[0], v.x[0]], [x_new[1], v.x[1]],
                '-', color=REWIRE_COLOR, lw=2.5, zorder=5)
        ax.plot(v.x[0], v.x[1], 'o', color=REWIRE_COLOR, ms=8, zorder=5,
                markeredgecolor='white', markeredgewidth=1)

    # Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color=PRUNE_COLOR, ls='--', lw=2, label='Old edge (removed)'),
        Line2D([0], [0], color=REWIRE_COLOR, ls='-', lw=2.5, label='Rewired through $x_{new}$'),
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=8, framealpha=0.9)
    _draw_sg(ax, xs, xg, ms=8)
    _set_ax(ax, '(5) Rewire Neighbors\nCan $x_{new}$ be a cheaper parent?')

    # ── Panel 6: CARM collision feedback ─────────────────────────
    ax = axes[1, 2]
    # Get CARM scale field
    carm = planner._carm
    res = 120
    xx = np.linspace(0, 1, res)
    yy = np.linspace(0, 1, res)
    XX, YY = np.meshgrid(xx, yy)
    pts = np.column_stack([XX.ravel(), YY.ravel()])
    S = carm._collision_scale_batch(pts).reshape(XX.shape)
    ax.contourf(XX, YY, S, levels=20, cmap='YlOrRd', alpha=0.5)
    _draw_obstacles(ax, alpha=0.6)
    _draw_tree(ax, verts, alpha=0.2)

    # Show collision points feeding CARM
    cp = np.array(carm._collision_points) if carm._collision_points else np.zeros((0,2))
    if len(cp) > 0:
        ax.scatter(cp[:, 0], cp[:, 1], c='cyan', s=4, alpha=0.4,
                   zorder=3, rasterized=True, label=f'{len(cp)} collision pts')

    # Show the rejected sample
    ax.plot(*x_coll, 'x', color=COLLISION_PT, ms=14, mew=3, zorder=6)
    ax.annotate('Collision!\nFed to CARM', xy=x_coll,
                xytext=(x_coll[0]+0.1, x_coll[1]+0.1),
                fontsize=9, fontweight='bold', color=COLLISION_PT,
                arrowprops=dict(arrowstyle='->', color=COLLISION_PT, lw=2),
                bbox=dict(fc='white', ec=COLLISION_PT, alpha=0.9, boxstyle='round,pad=0.3'))
    _draw_sg(ax, xs, xg, ms=8)
    _set_ax(ax, '(6) CARM Feedback\nCollisions update metric field')

    fig.suptitle('One Iteration of RIT*-CARM: Sample → Find Neighbors → Select Parent → Insert → Rewire → Learn',
                 fontsize=14, fontweight='bold', y=1.01)
    fig.tight_layout()
    _save(fig, 'fig_pipeline_iteration')


# ══════════════════════════════════════════════════════════════════════════════
#  FIG B: Detailed nearest-neighbor + parent selection
# ══════════════════════════════════════════════════════════════════════════════
def fig_b_nearest_neighbor():
    """Zoomed-in view of nearest-neighbor lookup and cascading selection."""
    print('=== Fig B: Nearest Neighbor Detail ===')

    fig, axes = plt.subplots(1, 3, figsize=(17, 5.5))

    # Synthetic example for clarity
    rng = np.random.default_rng(55)
    # Existing tree vertices
    tree_pts = np.array([
        [0.30, 0.50], [0.40, 0.30], [0.55, 0.55], [0.25, 0.70],
        [0.60, 0.35], [0.45, 0.65], [0.35, 0.40], [0.50, 0.20],
        [0.20, 0.55], [0.65, 0.50], [0.40, 0.50], [0.50, 0.45],
    ])
    # Tree edges (parent indices)
    edges = [(0,1), (0,3), (0,6), (1,7), (2,5), (2,9), (6,10), (10,11), (10,2), (3,8), (1,4)]
    x_new = np.array([0.48, 0.40])
    r = 0.18

    # ── Panel 1: r-ball query ────────────────────────────────────
    ax = axes[0]
    # Draw all tree edges
    for i, j in edges:
        ax.plot([tree_pts[i][0], tree_pts[j][0]],
                [tree_pts[i][1], tree_pts[j][1]],
                '-', color=TREE_EDGE, lw=1.5, zorder=1)
    # Draw all vertices
    ax.scatter(tree_pts[:, 0], tree_pts[:, 1], c='#607D8B', s=50,
               zorder=3, edgecolors='white', linewidths=1)
    # r-ball
    ax.add_patch(Circle(x_new, r, fc='#E3F2FD', ec=RADIUS_COLOR, lw=2.5,
                        ls='--', alpha=0.35, zorder=2))
    # x_new
    ax.plot(*x_new, 'o', color=NEW_SAMPLE, ms=14, zorder=6,
            markeredgecolor='black', markeredgewidth=2)
    # Highlight neighbors inside r
    dists = np.linalg.norm(tree_pts - x_new, axis=1)
    inside = dists < r
    ax.scatter(tree_pts[inside, 0], tree_pts[inside, 1], c=RADIUS_COLOR,
               s=80, zorder=4, edgecolors='white', linewidths=1.5)
    # Distance lines
    for i in np.where(inside)[0]:
        ax.plot([x_new[0], tree_pts[i][0]], [x_new[1], tree_pts[i][1]],
                ':', color=RADIUS_COLOR, lw=1, alpha=0.5)
    # Labels
    ax.annotate('$x_{new}$', xy=x_new, xytext=(x_new[0]-0.08, x_new[1]-0.07),
                fontsize=13, fontweight='bold', color=NEW_SAMPLE)
    ax.annotate(f'$r_n = {r}$', xy=(x_new[0]+r*0.65, x_new[1]+r*0.75),
                fontsize=11, color=RADIUS_COLOR, fontweight='bold')
    n_in = inside.sum()
    ax.text(0.5, 0.97, f'KD-tree query: {n_in} neighbors within $r_n$',
            transform=ax.transAxes, ha='center', va='top', fontsize=10,
            bbox=dict(fc='white', ec=RADIUS_COLOR, alpha=0.9, boxstyle='round,pad=0.3'))
    _set_ax(ax, '(a) Step 1: r-ball Query\n$\\{v : \\|v - x_{new}\\| < r_n\\}$')
    ax.set_xlim(0.1, 0.75); ax.set_ylim(0.1, 0.80)

    # ── Panel 2: Cascading cost evaluation ───────────────────────
    ax = axes[1]
    # Show only the neighbors and x_new, with cost cascade
    candidates = np.where(inside)[0]
    # Sort by distance (proxy for L1 cost)
    sorted_c = sorted(candidates, key=lambda i: dists[i])

    # Draw background lightly
    for i, j in edges:
        ax.plot([tree_pts[i][0], tree_pts[j][0]],
                [tree_pts[i][1], tree_pts[j][1]],
                '-', color=TREE_EDGE, lw=0.8, alpha=0.2, zorder=1)
    ax.scatter(tree_pts[:, 0], tree_pts[:, 1], c='#CFD8DC', s=30,
               zorder=2, edgecolors='white', linewidths=0.5)

    ax.plot(*x_new, 'o', color=NEW_SAMPLE, ms=14, zorder=6,
            markeredgecolor='black', markeredgewidth=2)

    # Show cascade levels
    cascade_colors = [BEST_PARENT, '#FF9800', '#FFC107', '#BDBDBD', '#EEEEEE']
    cascade_labels = [
        'v1: L1 ok, Exact ok, Free  BEST',
        'v2: L1 ok, Exact ok, Coll!',
        'v3: L1 ok, Exact fail, skip',
        'v4: L1 fail, skip',
    ]
    for rank, idx in enumerate(sorted_c[:4]):
        v = tree_pts[idx]
        c = cascade_colors[min(rank, len(cascade_colors)-1)]
        ms = 12 if rank == 0 else 8
        lw_e = 3 if rank == 0 else 1.5
        ls = '-' if rank == 0 else ('--' if rank < 3 else ':')
        ax.plot(v[0], v[1], 'o', color=c, ms=ms, zorder=5,
                markeredgecolor='black', markeredgewidth=1)
        ax.plot([v[0], x_new[0]], [v[1], x_new[1]],
                ls, color=c, lw=lw_e, zorder=4)
        # Label
        if rank < len(cascade_labels):
            side = 1 if rank % 2 == 0 else -1
            ax.text(v[0]+0.02*side, v[1]+0.04+0.015*rank, cascade_labels[rank],
                    fontsize=7.5, color=c, fontweight='bold',
                    bbox=dict(fc='white', ec=c, alpha=0.85, boxstyle='round,pad=0.2'))

    # Cascade arrow on right
    ax.text(0.95, 0.5, 'L1 cheap\n    |\nExact cost\n    |\nCollision\ncheck',
            transform=ax.transAxes, fontsize=8, va='center', ha='center',
            bbox=dict(fc='#FFF3E0', ec='#FF9800', alpha=0.9, boxstyle='round,pad=0.3'))

    _set_ax(ax, '(b) Step 2: Cascading Selection\nCheap filter first, expensive check last')
    ax.set_xlim(0.1, 0.75); ax.set_ylim(0.1, 0.80)

    # ── Panel 3: Final result ────────────────────────────────────
    ax = axes[2]
    for i, j in edges:
        ax.plot([tree_pts[i][0], tree_pts[j][0]],
                [tree_pts[i][1], tree_pts[j][1]],
                '-', color=TREE_EDGE, lw=1.5, zorder=1)
    ax.scatter(tree_pts[:, 0], tree_pts[:, 1], c='#607D8B', s=50,
               zorder=3, edgecolors='white', linewidths=1)
    # New edge to best parent
    best_idx = sorted_c[0] if len(sorted_c) > 0 else 0
    best_v = tree_pts[best_idx]
    ax.plot([best_v[0], x_new[0]], [best_v[1], x_new[1]],
            '-', color=BEST_PARENT, lw=3, zorder=5)
    ax.plot(*x_new, 'o', color=NEW_SAMPLE, ms=14, zorder=6,
            markeredgecolor='black', markeredgewidth=2)
    ax.plot(best_v[0], best_v[1], 'o', color=BEST_PARENT, ms=10, zorder=5,
            markeredgecolor='white', markeredgewidth=1.5)
    # Rewire an existing neighbor
    if len(sorted_c) > 1:
        rw_idx = sorted_c[1]
        rw_v = tree_pts[rw_idx]
        # Find its current parent
        rw_parent = None
        for i, j in edges:
            if j == rw_idx:
                rw_parent = i
                break
        if rw_parent is not None:
            # Old edge dashed
            ax.plot([tree_pts[rw_parent][0], rw_v[0]],
                    [tree_pts[rw_parent][1], rw_v[1]],
                    '--', color=PRUNE_COLOR, lw=1.5, alpha=0.6, zorder=3)
        # New rewired edge
        ax.plot([x_new[0], rw_v[0]], [x_new[1], rw_v[1]],
                '-', color=REWIRE_COLOR, lw=2.5, zorder=5)
        ax.plot(rw_v[0], rw_v[1], 'o', color=REWIRE_COLOR, ms=9, zorder=5,
                markeredgecolor='white', markeredgewidth=1)

    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=NEW_SAMPLE,
               ms=10, markeredgecolor='black', label='$x_{new}$ (new sample)'),
        Line2D([0], [0], color=BEST_PARENT, lw=3, label='Best parent edge'),
        Line2D([0], [0], color=REWIRE_COLOR, lw=2.5, label='Rewired edge'),
        Line2D([0], [0], color=PRUNE_COLOR, ls='--', lw=1.5, label='Old edge (removed)'),
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=8, framealpha=0.9)
    _set_ax(ax, '(c) Result: Node Inserted + Rewired\nTree locally improved')
    ax.set_xlim(0.1, 0.75); ax.set_ylim(0.1, 0.80)

    fig.suptitle('How the Tree Grows: r-ball Lookup, Cascading Parent Selection, and Rewiring',
                 fontsize=14, fontweight='bold', y=1.02)
    fig.tight_layout()
    _save(fig, 'fig_pipeline_nearest')


# ══════════════════════════════════════════════════════════════════════════════
#  FIG C: Tree Growth Sequence (8 snapshots)
# ══════════════════════════════════════════════════════════════════════════════
def fig_c_tree_growth():
    """Show tree at 8 different iteration counts."""
    print('=== Fig C: Tree Growth Sequence ===')

    coll, _, oracle_m, xs, xg, bounds = env_2d_obstacle_inflated()
    euclid = EuclideanMetric(2)

    snapshot_iters = [1, 5, 10, 20, 40, 60, 80, 100]
    snapshots = {}

    planner = RITStar(xs, xg, bounds, coll, euclid,
                      geodesic_tier='diagonal', batch_size=80,
                      max_iterations=max(snapshot_iters)+1, random_seed=42,
                      adaptive_metric=True, carm_sigma=0.08,
                      carm_alpha=6.0, carm_rebuild_interval=10)

    t0 = time.time()
    planner._t0 = t0
    for it in range(planner.max_iterations):
        samples = planner._sample_batch(it)
        planner._extend_tree(samples, it)
        if planner.c_best < np.inf:
            planner._prune()
            planner._update_informed_set()
            planner._update_stall_counter()
        if planner._adaptive_mode:
            planner._maybe_rebuild_carm_cache(it)
        elapsed = time.time() - t0
        planner._record_stats(it, elapsed)

        if it + 1 in snapshot_iters:
            # Extract path
            cur_path = []
            if planner.c_best < np.inf and planner.goal_node:
                v = planner.goal_node
                while v is not None:
                    cur_path.append(v.x.copy())
                    v = v.parent
                cur_path.reverse()

            snapshots[it+1] = {
                'n_verts': len(planner.vertices),
                'c_best': planner.c_best,
                'path': cur_path,
                'verts': [(v.x.copy(), v.parent.x.copy() if v.parent else None)
                          for v in planner.vertices],
            }

    fig, axes = plt.subplots(2, 4, figsize=(20, 10))

    for idx, it in enumerate(snapshot_iters):
        ax = axes[idx // 4, idx % 4]
        snap = snapshots[it]
        _draw_obstacles(ax, alpha=0.6)

        # Tree edges
        for vx, px in snap['verts']:
            if px is not None:
                ax.plot([px[0], vx[0]], [px[1], vx[1]],
                        '-', color=TREE_EDGE, lw=0.5, alpha=0.5, zorder=1)

        # Vertices
        vx_all = np.array([vx for vx, _ in snap['verts']])
        ax.scatter(vx_all[:, 0], vx_all[:, 1], c='#78909C', s=3,
                   alpha=0.6, zorder=2)

        # Path
        if snap['path'] and len(snap['path']) > 1:
            pp = np.array(snap['path'])
            ax.plot(pp[:, 0], pp[:, 1], '-', color=PATH_COLOR, lw=2.5, zorder=5,
                    path_effects=[pe.Stroke(linewidth=4, foreground='white'), pe.Normal()])

        _draw_sg(ax, xs, xg, ms=7)
        c_str = f'{snap["c_best"]:.3f}' if np.isfinite(snap['c_best']) else 'no path'
        ax.set_title(f'Iter {it}\n{snap["n_verts"]} vertices | {c_str}',
                     fontsize=10, fontweight='bold')
        _set_ax(ax)

    fig.suptitle('Tree Growth Over Time: From Empty Space to Optimized Path',
                 fontsize=14, fontweight='bold', y=1.01)
    fig.tight_layout()
    _save(fig, 'fig_pipeline_tree_growth')


# ══════════════════════════════════════════════════════════════════════════════
#  FIG D: CARM feedback loop diagram
# ══════════════════════════════════════════════════════════════════════════════
def fig_d_carm_feedback_loop():
    """Show the 3 places where collisions feed into CARM during one iteration."""
    print('=== Fig D: CARM Feedback Loop ===')

    fig, axes = plt.subplots(1, 3, figsize=(17, 5.5))

    # Synthetic mini-scene
    obs_center = np.array([0.50, 0.50])
    obs_r = 0.15

    # ── Panel 1: Point collision ─────────────────────────────────
    ax = axes[0]
    ax.add_patch(Circle(obs_center, obs_r, fc=OBSTACLE_COLOR, ec='white', lw=1.5, alpha=0.8))
    # Sample point inside obstacle
    p_coll = np.array([0.55, 0.48])
    ax.plot(*p_coll, 'x', color=COLLISION_PT, ms=16, mew=3, zorder=6)
    ax.annotate('Sample lands\ninside obstacle', xy=p_coll,
                xytext=(0.75, 0.35), fontsize=10, color=COLLISION_PT, fontweight='bold',
                arrowprops=dict(arrowstyle='->', color=COLLISION_PT, lw=2),
                bbox=dict(fc='white', ec=COLLISION_PT, alpha=0.9, boxstyle='round,pad=0.3'))
    # Free sample
    p_free = np.array([0.30, 0.30])
    ax.plot(*p_free, 'o', color=FREE_PT, ms=10, zorder=5,
            markeredgecolor='white', markeredgewidth=1)
    ax.text(0.30, 0.22, 'Free sample\n(accepted)', fontsize=8, ha='center',
            color=FREE_PT, fontweight='bold')
    # Feed arrow to CARM
    ax.annotate('', xy=(0.55, 0.70), xytext=(0.55, 0.52),
                arrowprops=dict(arrowstyle='->', color=CARM_HEAT, lw=2.5))
    ax.text(0.55, 0.75, 'CARM.add_collision_point(x)',
            fontsize=9, ha='center', color=CARM_HEAT, fontweight='bold',
            bbox=dict(fc='#FFF3E0', ec=CARM_HEAT, alpha=0.9, boxstyle='round,pad=0.3'))
    _set_ax(ax, '(a) Source 1: Point Collision\nSample rejected by collision check')
    ax.set_xlim(0.1, 0.9); ax.set_ylim(0.1, 0.9)

    # ── Panel 2: Edge collision during parent selection ──────────
    ax = axes[1]
    ax.add_patch(Circle(obs_center, obs_r, fc=OBSTACLE_COLOR, ec='white', lw=1.5, alpha=0.8))
    # Parent vertex
    v_parent = np.array([0.25, 0.45])
    # New sample
    x_new = np.array([0.75, 0.55])
    ax.plot(*v_parent, 'o', color='#607D8B', ms=10, zorder=5,
            markeredgecolor='white', markeredgewidth=1.5)
    ax.plot(*x_new, 'o', color=NEW_SAMPLE, ms=12, zorder=6,
            markeredgecolor='black', markeredgewidth=1.5)
    ax.text(v_parent[0]-0.02, v_parent[1]-0.06, 'parent $v$', fontsize=9,
            ha='center', color='#607D8B', fontweight='bold')
    ax.text(x_new[0]+0.02, x_new[1]-0.06, '$x_{new}$', fontsize=10,
            ha='center', color=NEW_SAMPLE, fontweight='bold')
    # Edge that collides
    ax.plot([v_parent[0], x_new[0]], [v_parent[1], x_new[1]],
            '--', color=COLLISION_PT, lw=2, zorder=3)
    # Collision point on edge
    t_coll = 0.45  # parameterize along edge
    coll_on_edge = v_parent + t_coll * (x_new - v_parent)
    ax.plot(*coll_on_edge, 'x', color=COLLISION_PT, ms=14, mew=3, zorder=6)
    ax.annotate('Edge hits obstacle\nat interpolation point', xy=coll_on_edge,
                xytext=(0.50, 0.82), fontsize=9, color=COLLISION_PT, fontweight='bold',
                arrowprops=dict(arrowstyle='->', color=COLLISION_PT, lw=2),
                bbox=dict(fc='white', ec=COLLISION_PT, alpha=0.9, boxstyle='round,pad=0.3'))
    # Feed arrow
    ax.annotate('', xy=(0.60, 0.18), xytext=(coll_on_edge[0], coll_on_edge[1]-0.05),
                arrowprops=dict(arrowstyle='->', color=CARM_HEAT, lw=2.5))
    ax.text(0.60, 0.13, 'CARM.add_collision_point(p)',
            fontsize=9, ha='center', color=CARM_HEAT, fontweight='bold',
            bbox=dict(fc='#FFF3E0', ec=CARM_HEAT, alpha=0.9, boxstyle='round,pad=0.3'))
    _set_ax(ax, '(b) Source 2: Edge Collision\nParent selection edge check fails')
    ax.set_xlim(0.1, 0.9); ax.set_ylim(0.05, 0.95)

    # ── Panel 3: Edge collision during rewiring ──────────────────
    ax = axes[2]
    ax.add_patch(Circle(obs_center, obs_r, fc=OBSTACLE_COLOR, ec='white', lw=1.5, alpha=0.8))
    # x_new already inserted
    x_new2 = np.array([0.30, 0.70])
    # Existing neighbor to rewire
    v_neigh = np.array([0.65, 0.60])
    ax.plot(*x_new2, 'o', color=NEW_SAMPLE, ms=12, zorder=6,
            markeredgecolor='black', markeredgewidth=1.5)
    ax.plot(*v_neigh, 'o', color=REWIRE_COLOR, ms=10, zorder=5,
            markeredgecolor='white', markeredgewidth=1.5)
    ax.text(x_new2[0]-0.02, x_new2[1]+0.05, '$x_{new}$', fontsize=10,
            ha='center', color=NEW_SAMPLE, fontweight='bold')
    ax.text(v_neigh[0]+0.05, v_neigh[1]+0.03, 'neighbor $v$', fontsize=9,
            color=REWIRE_COLOR, fontweight='bold')
    # Rewire edge that collides
    ax.plot([x_new2[0], v_neigh[0]], [x_new2[1], v_neigh[1]],
            '--', color=COLLISION_PT, lw=2, zorder=3)
    # Collision pt
    t_coll2 = 0.55
    coll_rw = x_new2 + t_coll2 * (v_neigh - x_new2)
    ax.plot(*coll_rw, 'x', color=COLLISION_PT, ms=14, mew=3, zorder=6)
    ax.annotate('Rewire edge blocked!', xy=coll_rw,
                xytext=(0.55, 0.25), fontsize=9, color=COLLISION_PT, fontweight='bold',
                arrowprops=dict(arrowstyle='->', color=COLLISION_PT, lw=2),
                bbox=dict(fc='white', ec=COLLISION_PT, alpha=0.9, boxstyle='round,pad=0.3'))
    # Feed arrow
    ax.annotate('', xy=(0.30, 0.20), xytext=(coll_rw[0]-0.10, coll_rw[1]-0.15),
                arrowprops=dict(arrowstyle='->', color=CARM_HEAT, lw=2.5))
    ax.text(0.30, 0.14, 'CARM.add_collision_point(p)',
            fontsize=9, ha='center', color=CARM_HEAT, fontweight='bold',
            bbox=dict(fc='#FFF3E0', ec=CARM_HEAT, alpha=0.9, boxstyle='round,pad=0.3'))
    _set_ax(ax, '(c) Source 3: Rewire Collision\nRewire edge check fails')
    ax.set_xlim(0.1, 0.9); ax.set_ylim(0.05, 0.95)

    fig.suptitle('Three Sources of CARM Collision Feedback (All Free Data)',
                 fontsize=14, fontweight='bold', y=1.03)
    fig.tight_layout()
    _save(fig, 'fig_pipeline_carm_feedback')


# ══════════════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    import gc

    print('=' * 60)
    print('  PIPELINE FIGURES FOR CARM RAL PAPER')
    print('=' * 60)

    fig_a_iteration_pipeline()
    gc.collect()

    fig_b_nearest_neighbor()
    gc.collect()

    fig_c_tree_growth()
    gc.collect()

    fig_d_carm_feedback_loop()
    gc.collect()

    print('\n' + '=' * 60)
    print('  ALL PIPELINE FIGURES DONE!')
    print('=' * 60)
    print(f'\nOutput: {PLOTS_DIR}')
