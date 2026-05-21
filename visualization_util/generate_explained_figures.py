#!/usr/bin/env python3
"""Generate conceptual figures for the comprehensive explained document."""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Circle, Ellipse
from matplotlib.collections import LineCollection
from matplotlib import patheffects
import os

OUTDIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'paper', 'figures')
os.makedirs(OUTDIR, exist_ok=True)

# ── Common styling ───────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.size': 11,
    'font.family': 'serif',
    'axes.linewidth': 1.2,
    'figure.dpi': 200,
})

COLORS = {
    'start': '#2ecc71',
    'goal': '#e74c3c',
    'tree': '#3498db',
    'path': '#e67e22',
    'obstacle': '#95a5a6',
    'collision': '#e74c3c',
    'free': '#2ecc71',
    'informed_euclid': '#aed6f1',
    'informed_riem': '#f9e79f',
    'rewire_old': '#e74c3c',
    'rewire_new': '#2ecc71',
    'carm': '#e74c3c',
    'euclid_path': '#7f8c8d',
    'riem_path': '#2980b9',
}


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 1: The Problem – What is Motion Planning?
# ═══════════════════════════════════════════════════════════════════════════════
def fig_problem_overview():
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    # Panel (a): Simple problem
    ax = axes[0]
    ax.set_xlim(-0.1, 1.1); ax.set_ylim(-0.1, 1.1)
    ax.set_aspect('equal')
    ax.set_title('(a) The Motion Planning Problem', fontweight='bold', fontsize=10)

    # Obstacles
    obs = [Circle((0.3, 0.5), 0.12, fc='#bdc3c7', ec='#7f8c8d', lw=2),
           Circle((0.6, 0.3), 0.1, fc='#bdc3c7', ec='#7f8c8d', lw=2),
           Circle((0.7, 0.7), 0.11, fc='#bdc3c7', ec='#7f8c8d', lw=2)]
    for o in obs:
        ax.add_patch(o)

    ax.plot(0.05, 0.05, 'o', color=COLORS['start'], ms=14, zorder=5)
    ax.text(0.05, -0.05, 'Start', ha='center', fontweight='bold', color=COLORS['start'])
    ax.plot(0.95, 0.95, '*', color=COLORS['goal'], ms=16, zorder=5)
    ax.text(0.95, 1.05, 'Goal', ha='center', fontweight='bold', color=COLORS['goal'])

    # Question mark path
    ax.annotate('', xy=(0.85, 0.85), xytext=(0.15, 0.15),
                arrowprops=dict(arrowstyle='->', color='#2c3e50', lw=2, ls='--'))
    ax.text(0.5, 0.15, '?', fontsize=28, ha='center', color='#e74c3c', fontweight='bold')
    ax.text(0.5, -0.08, 'Find safe path from Start to Goal', ha='center', fontsize=9, style='italic')
    ax.set_xticks([]); ax.set_yticks([])

    # Panel (b): Why it's hard
    ax = axes[1]
    ax.set_xlim(-0.1, 1.1); ax.set_ylim(-0.1, 1.1)
    ax.set_aspect('equal')
    ax.set_title('(b) Why It\'s Hard', fontweight='bold', fontsize=10)

    obs2 = [Circle((0.3, 0.5), 0.12, fc='#e74c3c', ec='#c0392b', lw=2, alpha=0.4),
            Circle((0.6, 0.3), 0.1, fc='#e74c3c', ec='#c0392b', lw=2, alpha=0.4),
            Circle((0.7, 0.7), 0.11, fc='#e74c3c', ec='#c0392b', lw=2, alpha=0.4)]
    for o in obs2:
        ax.add_patch(o)

    # Bad path that hits obstacle
    bad_x = [0.05, 0.25, 0.4, 0.6, 0.95]
    bad_y = [0.05, 0.3, 0.5, 0.7, 0.95]
    ax.plot(bad_x, bad_y, '-', color='#e74c3c', lw=3, alpha=0.7)
    ax.plot(0.3, 0.42, 'X', color='#e74c3c', ms=18, mew=3, zorder=5)
    ax.text(0.38, 0.42, 'COLLISION!', fontsize=9, color='#e74c3c', fontweight='bold')

    ax.plot(0.05, 0.05, 'o', color=COLORS['start'], ms=14, zorder=5)
    ax.plot(0.95, 0.95, '*', color=COLORS['goal'], ms=16, zorder=5)
    ax.text(0.5, -0.08, 'Straight line hits obstacles', ha='center', fontsize=9, style='italic')
    ax.set_xticks([]); ax.set_yticks([])

    # Panel (c): Solution
    ax = axes[2]
    ax.set_xlim(-0.1, 1.1); ax.set_ylim(-0.1, 1.1)
    ax.set_aspect('equal')
    ax.set_title('(c) The Solution: Smart Planning', fontweight='bold', fontsize=10)

    obs3 = [Circle((0.3, 0.5), 0.12, fc='#bdc3c7', ec='#7f8c8d', lw=2),
            Circle((0.6, 0.3), 0.1, fc='#bdc3c7', ec='#7f8c8d', lw=2),
            Circle((0.7, 0.7), 0.11, fc='#bdc3c7', ec='#7f8c8d', lw=2)]
    for o in obs3:
        ax.add_patch(o)

    good_x = [0.05, 0.15, 0.15, 0.45, 0.55, 0.85, 0.95]
    good_y = [0.05, 0.15, 0.35, 0.75, 0.88, 0.9, 0.95]
    ax.plot(good_x, good_y, '-', color=COLORS['path'], lw=3.5, zorder=4)
    for gx, gy in zip(good_x, good_y):
        ax.plot(gx, gy, 'o', color=COLORS['path'], ms=5, zorder=5)

    ax.plot(0.05, 0.05, 'o', color=COLORS['start'], ms=14, zorder=5)
    ax.plot(0.95, 0.95, '*', color=COLORS['goal'], ms=16, zorder=5)
    ax.text(0.5, -0.08, 'Smart path avoids all obstacles', ha='center', fontsize=9, style='italic')
    ax.set_xticks([]); ax.set_yticks([])

    plt.tight_layout()
    plt.savefig(f'{OUTDIR}/fig_explained_problem.pdf', bbox_inches='tight')
    plt.savefig(f'{OUTDIR}/fig_explained_problem.png', bbox_inches='tight')
    plt.close()
    print("  → fig_explained_problem.pdf")


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 2: How Baselines Work (RRT*, BIT*, Informed RRT*)
# ═══════════════════════════════════════════════════════════════════════════════
def fig_baselines_overview():
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    np.random.seed(42)

    for idx, (ax, title, desc) in enumerate(zip(axes,
        ['(a) RRT* — Random Tree', '(b) Informed RRT* — Ellipse Focus', '(c) BIT* — Batch + Edge Queue'],
        ['Grows randomly in all directions', 'Focuses inside an ellipse', 'Processes best edges first'])):

        ax.set_xlim(-0.1, 1.1); ax.set_ylim(-0.1, 1.1)
        ax.set_aspect('equal')
        ax.set_title(title, fontweight='bold', fontsize=10)

        # Obstacles
        obs = [Circle((0.4, 0.5), 0.1, fc='#bdc3c7', ec='#7f8c8d', lw=1.5),
               Circle((0.7, 0.4), 0.08, fc='#bdc3c7', ec='#7f8c8d', lw=1.5)]
        for o in obs:
            ax.add_patch(o)

        sx, sy = 0.05, 0.05
        gx, gy = 0.95, 0.95

        if idx == 0:  # RRT*: random tree everywhere
            pts = np.random.rand(60, 2)
            for p in pts:
                parent = pts[np.random.randint(len(pts))]
                ax.plot([parent[0], p[0]], [parent[1], p[1]], '-', color='#3498db', lw=0.5, alpha=0.4)
            ax.plot(pts[:, 0], pts[:, 1], '.', color='#3498db', ms=3, alpha=0.5)

        elif idx == 1:  # Informed RRT*: ellipse
            cx, cy = (sx + gx) / 2, (sy + gy) / 2
            ell = Ellipse((cx, cy), 1.1, 0.5, angle=45, fc=COLORS['informed_euclid'], ec='#2980b9', lw=2, alpha=0.3, ls='--')
            ax.add_patch(ell)
            # Samples inside ellipse
            pts = np.random.rand(40, 2)
            for p in pts:
                parent = pts[np.random.randint(len(pts))]
                ax.plot([parent[0], p[0]], [parent[1], p[1]], '-', color='#3498db', lw=0.5, alpha=0.4)
            ax.plot(pts[:, 0], pts[:, 1], '.', color='#3498db', ms=3, alpha=0.5)
            ax.text(0.5, 0.8, 'Euclidean\nEllipse', ha='center', fontsize=8, color='#2980b9', fontweight='bold')

        else:  # BIT*: edge queue
            pts = np.random.rand(30, 2)
            for p in pts:
                parent = pts[np.random.randint(len(pts))]
                ax.plot([parent[0], p[0]], [parent[1], p[1]], '-', color='#3498db', lw=0.5, alpha=0.4)
            # Highlight "queue" edges
            for i in range(5):
                p1, p2 = pts[i], pts[i + 5]
                ax.annotate('', xy=p2, xytext=p1,
                            arrowprops=dict(arrowstyle='->', color='#e67e22', lw=2))
            ax.text(0.7, 0.15, 'Edge Queue\n(priority order)', fontsize=8,
                    ha='center', color='#e67e22', fontweight='bold',
                    bbox=dict(boxstyle='round', fc='#fdebd0', ec='#e67e22'))

        ax.plot(sx, sy, 'o', color=COLORS['start'], ms=12, zorder=5)
        ax.plot(gx, gy, '*', color=COLORS['goal'], ms=14, zorder=5)
        ax.text(0.5, -0.08, desc, ha='center', fontsize=9, style='italic')
        ax.set_xticks([]); ax.set_yticks([])

    plt.tight_layout()
    plt.savefig(f'{OUTDIR}/fig_explained_baselines.pdf', bbox_inches='tight')
    plt.savefig(f'{OUTDIR}/fig_explained_baselines.png', bbox_inches='tight')
    plt.close()
    print("  → fig_explained_baselines.pdf")


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 3: Complete RIT* Pipeline Step-by-Step
# ═══════════════════════════════════════════════════════════════════════════════
def fig_rit_pipeline_steps():
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    np.random.seed(123)

    titles = [
        'Step 1: Sample Batch\n(Riemannian Informed Set)',
        'Step 2: Find Nearest Neighbors\n(Ellipsoidal Search)',
        'Step 3: Connect to Best Parent\n(Cascading Edge Eval)',
        'Step 4: Rewire Existing Nodes\n(Find Cheaper Paths)',
        'Step 5: Prune Bad Nodes\n(Remove Useless Branches)',
        'Step 6: Update & Repeat\n(Shrink Informed Set)'
    ]

    for i, (ax, title) in enumerate(zip(axes.flat, titles)):
        ax.set_xlim(-0.1, 1.1); ax.set_ylim(-0.1, 1.1)
        ax.set_aspect('equal')
        ax.set_title(title, fontweight='bold', fontsize=10, pad=8)
        ax.set_xticks([]); ax.set_yticks([])

        # Common: obstacles
        obs = [Circle((0.4, 0.5), 0.1, fc='#ecf0f1', ec='#95a5a6', lw=1.5, zorder=1),
               Circle((0.7, 0.35), 0.08, fc='#ecf0f1', ec='#95a5a6', lw=1.5, zorder=1)]
        for o in obs:
            ax.add_patch(o)

        sx, sy = 0.05, 0.05
        gx, gy = 0.95, 0.95
        ax.plot(sx, sy, 'o', color=COLORS['start'], ms=10, zorder=10)
        ax.plot(gx, gy, '*', color=COLORS['goal'], ms=12, zorder=10)

        # Existing tree nodes
        tree_pts = np.array([(0.05, 0.05), (0.15, 0.2), (0.2, 0.4), (0.25, 0.15),
                             (0.5, 0.75), (0.55, 0.6), (0.8, 0.8), (0.85, 0.6)])
        tree_edges = [(0, 1), (1, 2), (1, 3), (2, 4), (4, 5), (4, 6), (6, 7)]

        if i == 0:  # SAMPLING
            # Draw informed set
            cx, cy = (sx + gx) / 2, (sy + gy) / 2
            ell = Ellipse((cx, cy), 0.9, 0.4, angle=45, fc='#f9e79f', ec='#f39c12', lw=2, alpha=0.3, ls='--', zorder=0)
            ax.add_patch(ell)
            # New samples
            samples = np.random.rand(15, 2) * 0.7 + 0.15
            ax.plot(samples[:, 0], samples[:, 1], 'D', color='#e67e22', ms=6, zorder=5, alpha=0.8)
            ax.text(0.7, 0.12, 'New\nsamples', fontsize=8, color='#e67e22', fontweight='bold')
            # Draw tree faintly
            for e in tree_edges:
                ax.plot([tree_pts[e[0], 0], tree_pts[e[1], 0]], [tree_pts[e[0], 1], tree_pts[e[1], 1]],
                        '-', color='#3498db', lw=1, alpha=0.3)

        elif i == 1:  # NEAREST NEIGHBORS
            for e in tree_edges:
                ax.plot([tree_pts[e[0], 0], tree_pts[e[1], 0]], [tree_pts[e[0], 1], tree_pts[e[1], 1]],
                        '-', color='#3498db', lw=1, alpha=0.4)
            ax.plot(tree_pts[:, 0], tree_pts[:, 1], 'o', color='#3498db', ms=5, zorder=5)
            # A new sample with ellipsoidal search
            ns = np.array([0.4, 0.7])
            ax.plot(ns[0], ns[1], 'D', color='#e67e22', ms=8, zorder=6)
            ell2 = Ellipse(ns, 0.35, 0.2, angle=30, fc='none', ec='#e67e22', lw=2, ls='--', zorder=3)
            ax.add_patch(ell2)
            # Highlight neighbors
            for pt_idx in [2, 4, 5]:
                ax.plot([ns[0], tree_pts[pt_idx, 0]], [ns[1], tree_pts[pt_idx, 1]],
                        '--', color='#e67e22', lw=1.5, alpha=0.6)
                ax.plot(tree_pts[pt_idx, 0], tree_pts[pt_idx, 1], 'o', color='#e67e22', ms=7, zorder=6)
            ax.text(0.08, 0.9, 'Riemannian\nneighbor ball', fontsize=8, color='#e67e22', fontweight='bold')

        elif i == 2:  # CONNECT TO BEST PARENT
            for e in tree_edges:
                ax.plot([tree_pts[e[0], 0], tree_pts[e[1], 0]], [tree_pts[e[0], 1], tree_pts[e[1], 1]],
                        '-', color='#3498db', lw=1, alpha=0.4)
            ax.plot(tree_pts[:, 0], tree_pts[:, 1], 'o', color='#3498db', ms=5, zorder=5)
            ns = np.array([0.4, 0.7])
            ax.plot(ns[0], ns[1], 'D', color='#e67e22', ms=8, zorder=6)
            # Show candidate connections with costs
            candidates = [(2, 0.45, '#e74c3c'), (4, 0.22, '#2ecc71'), (5, 0.38, '#e74c3c')]
            for pt_idx, cost, col in candidates:
                style = '-' if col == '#2ecc71' else '--'
                lw = 3 if col == '#2ecc71' else 1.5
                ax.plot([ns[0], tree_pts[pt_idx, 0]], [ns[1], tree_pts[pt_idx, 1]],
                        style, color=col, lw=lw, alpha=0.8)
                mid = (ns + tree_pts[pt_idx]) / 2
                ax.text(mid[0] + 0.03, mid[1], f'{cost}', fontsize=8, color=col, fontweight='bold')
            ax.text(0.08, 0.9, 'Best parent\n(lowest cost)', fontsize=8, color='#2ecc71', fontweight='bold')

        elif i == 3:  # REWIRING
            for e in tree_edges:
                col = '#e74c3c' if e == (4, 5) else '#3498db'
                ls = '--' if e == (4, 5) else '-'
                lw = 2 if e == (4, 5) else 1
                ax.plot([tree_pts[e[0], 0], tree_pts[e[1], 0]], [tree_pts[e[0], 1], tree_pts[e[1], 1]],
                        ls, color=col, lw=lw, alpha=0.6)
            ax.plot(tree_pts[:, 0], tree_pts[:, 1], 'o', color='#3498db', ms=5, zorder=5)
            # New node and rewire
            ns = np.array([0.4, 0.7])
            ax.plot(ns[0], ns[1], 'D', color='#e67e22', ms=8, zorder=6)
            ax.plot([ns[0], tree_pts[4, 0]], [ns[1], tree_pts[4, 1]], '-', color='#2ecc71', lw=2)
            # Rewire: 5 now connects through new node instead of 4
            ax.plot([ns[0], tree_pts[5, 0]], [ns[1], tree_pts[5, 1]], '-', color='#2ecc71', lw=3, zorder=4)
            ax.text(0.62, 0.52, 'Rewired!\n(cheaper)', fontsize=8, color='#2ecc71', fontweight='bold')
            ax.text(0.42, 0.58, 'Old link\n(removed)', fontsize=7, color='#e74c3c', style='italic')

        elif i == 4:  # PRUNING
            for e in tree_edges:
                ax.plot([tree_pts[e[0], 0], tree_pts[e[1], 0]], [tree_pts[e[0], 1], tree_pts[e[1], 1]],
                        '-', color='#3498db', lw=1, alpha=0.4)
            # Mark pruned nodes
            for pt_idx in [3, 7]:
                ax.plot(tree_pts[pt_idx, 0], tree_pts[pt_idx, 1], 'X', color='#e74c3c', ms=12, mew=2.5, zorder=6)
            for pt_idx in [0, 1, 2, 4, 5, 6]:
                ax.plot(tree_pts[pt_idx, 0], tree_pts[pt_idx, 1], 'o', color='#3498db', ms=5, zorder=5)
            ax.text(0.08, 0.9, 'Nodes that can\'t\nimprove solution\n→ PRUNED', fontsize=8,
                    color='#e74c3c', fontweight='bold')

        else:  # UPDATE & REPEAT
            # Smaller informed set
            cx, cy = (sx + gx) / 2, (sy + gy) / 2
            ell_big = Ellipse((cx, cy), 0.9, 0.4, angle=45, fc='#aed6f1', ec='#3498db', lw=1.5, alpha=0.15, ls='--', zorder=0)
            ell_small = Ellipse((cx, cy), 0.6, 0.25, angle=45, fc='#f9e79f', ec='#f39c12', lw=2, alpha=0.3, zorder=0)
            ax.add_patch(ell_big)
            ax.add_patch(ell_small)
            # Path
            path_pts = [(0.05, 0.05), (0.15, 0.2), (0.2, 0.4), (0.5, 0.75), (0.8, 0.8), (0.95, 0.95)]
            px, py = zip(*path_pts)
            ax.plot(px, py, '-', color=COLORS['path'], lw=3.5, zorder=4)
            ax.text(0.12, 0.85, 'Informed set\nshrinks!', fontsize=8, color='#f39c12', fontweight='bold')
            ax.annotate('Before', xy=(0.75, 0.25), fontsize=8, color='#3498db', style='italic')
            ax.annotate('After', xy=(0.62, 0.4), fontsize=8, color='#f39c12', fontweight='bold')

    plt.tight_layout()
    plt.savefig(f'{OUTDIR}/fig_explained_pipeline_steps.pdf', bbox_inches='tight')
    plt.savefig(f'{OUTDIR}/fig_explained_pipeline_steps.png', bbox_inches='tight')
    plt.close()
    print("  → fig_explained_pipeline_steps.pdf")


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 4: CARM Collision Learning
# ═══════════════════════════════════════════════════════════════════════════════
def fig_carm_learning():
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    np.random.seed(77)

    # Panel (a): Collision sampling
    ax = axes[0]
    ax.set_xlim(-0.1, 1.1); ax.set_ylim(-0.1, 1.1); ax.set_aspect('equal')
    ax.set_title('(a) Collisions = Free Data', fontweight='bold', fontsize=10)
    obs_centers = [(0.35, 0.5), (0.65, 0.35), (0.55, 0.7)]
    obs_radii = [0.12, 0.09, 0.1]
    for c, r in zip(obs_centers, obs_radii):
        ax.add_patch(Circle(c, r, fc='#d5dbdb', ec='#7f8c8d', lw=1.5))

    # Free samples
    free = np.random.rand(25, 2)
    coll = []
    free_list = []
    for p in free:
        in_obs = False
        for c, r in zip(obs_centers, obs_radii):
            if np.linalg.norm(p - np.array(c)) < r + 0.02:
                in_obs = True; break
        if in_obs:
            coll.append(p)
        else:
            free_list.append(p)
    free_arr = np.array(free_list) if free_list else np.zeros((0, 2))
    coll_arr = np.array(coll) if coll else np.zeros((0, 2))
    # Add more collision points near obstacles
    for c, r in zip(obs_centers, obs_radii):
        angles = np.random.rand(6) * 2 * np.pi
        radii_pts = r * (0.8 + 0.4 * np.random.rand(6))
        cx = c[0] + radii_pts * np.cos(angles)
        cy = c[1] + radii_pts * np.sin(angles)
        coll_arr = np.vstack([coll_arr, np.column_stack([cx, cy])])

    if len(free_arr) > 0:
        ax.plot(free_arr[:, 0], free_arr[:, 1], 'o', color=COLORS['free'], ms=5, zorder=4, label='Free')
    ax.plot(coll_arr[:, 0], coll_arr[:, 1], 'x', color=COLORS['collision'], ms=7, mew=2, zorder=4, label='Collision')
    ax.legend(loc='lower right', fontsize=8)
    ax.set_xticks([]); ax.set_yticks([])

    # Panel (b): KDE density
    ax = axes[1]
    ax.set_xlim(-0.1, 1.1); ax.set_ylim(-0.1, 1.1); ax.set_aspect('equal')
    ax.set_title('(b) Density from Collisions', fontweight='bold', fontsize=10)
    xx, yy = np.meshgrid(np.linspace(0, 1, 80), np.linspace(0, 1, 80))
    pts_grid = np.column_stack([xx.ravel(), yy.ravel()])
    sigma = 0.08
    density = np.zeros(len(pts_grid))
    for cp in coll_arr:
        density += np.exp(-np.sum((pts_grid - cp) ** 2, axis=1) / (2 * sigma ** 2))
    density = density.reshape(xx.shape) / len(coll_arr)
    ax.contourf(xx, yy, density, levels=20, cmap='YlOrRd', alpha=0.8)
    ax.contour(xx, yy, density, levels=5, colors='#c0392b', linewidths=0.5, alpha=0.5)
    ax.plot(coll_arr[:, 0], coll_arr[:, 1], '.', color='cyan', ms=2, alpha=0.5)
    ax.set_xticks([]); ax.set_yticks([])

    # Panel (c): Scale factor
    ax = axes[2]
    ax.set_xlim(-0.1, 1.1); ax.set_ylim(-0.1, 1.1); ax.set_aspect('equal')
    ax.set_title(r'(c) Cost Scale $s(x) = 1 + \alpha \hat{f}$', fontweight='bold', fontsize=10)
    alpha_carm = 6.0
    scale = 1.0 + alpha_carm * density
    ax.contourf(xx, yy, scale, levels=20, cmap='hot_r', alpha=0.8)
    ax.contour(xx, yy, scale, levels=8, colors='k', linewidths=0.3, alpha=0.3)
    for c, r in zip(obs_centers, obs_radii):
        ax.add_patch(Circle(c, r, fc='none', ec='white', lw=2, ls='--'))
    ax.set_xticks([]); ax.set_yticks([])

    # Panel (d): Result
    ax = axes[3]
    ax.set_xlim(-0.1, 1.1); ax.set_ylim(-0.1, 1.1); ax.set_aspect('equal')
    ax.set_title('(d) Better Path!', fontweight='bold', fontsize=10)
    ax.contourf(xx, yy, scale, levels=20, cmap='hot_r', alpha=0.3)
    for c, r in zip(obs_centers, obs_radii):
        ax.add_patch(Circle(c, r, fc='#bdc3c7', ec='#7f8c8d', lw=1.5))

    # Euclidean path (hugs obstacles)
    euclid_path = [(0.05, 0.05), (0.2, 0.3), (0.3, 0.38), (0.5, 0.5), (0.7, 0.55), (0.9, 0.9), (0.95, 0.95)]
    ex, ey = zip(*euclid_path)
    ax.plot(ex, ey, '--', color=COLORS['euclid_path'], lw=2, label='Euclidean', zorder=3)

    # CARM path (avoids obstacles)
    carm_path = [(0.05, 0.05), (0.12, 0.2), (0.1, 0.4), (0.15, 0.7), (0.35, 0.85), (0.75, 0.85), (0.9, 0.9), (0.95, 0.95)]
    cx_p, cy_p = zip(*carm_path)
    ax.plot(cx_p, cy_p, '-', color=COLORS['carm'], lw=3, label='CARM', zorder=4)

    ax.plot(0.05, 0.05, 'o', color=COLORS['start'], ms=10, zorder=5)
    ax.plot(0.95, 0.95, '*', color=COLORS['goal'], ms=12, zorder=5)
    ax.legend(loc='lower right', fontsize=8)
    ax.set_xticks([]); ax.set_yticks([])

    plt.tight_layout()
    plt.savefig(f'{OUTDIR}/fig_explained_carm_learning.pdf', bbox_inches='tight')
    plt.savefig(f'{OUTDIR}/fig_explained_carm_learning.png', bbox_inches='tight')
    plt.close()
    print("  → fig_explained_carm_learning.pdf")


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 5: Novelty Comparison (RIT* vs baselines)
# ═══════════════════════════════════════════════════════════════════════════════
def fig_novelty_comparison():
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Panel (a): Baseline limitations
    ax = axes[0]
    ax.set_xlim(-0.3, 1.3); ax.set_ylim(-0.1, 1.1)
    ax.set_aspect('equal')
    ax.set_title('(a) Existing Methods: Key Limitations', fontweight='bold', fontsize=11)
    ax.axis('off')

    boxes = [
        (0.0, 0.85, 'RRT*', 'Explores uniformly\n→ many wasted samples', '#e74c3c'),
        (0.0, 0.55, 'Informed RRT*', 'Euclidean ellipse only\n→ loose focus region', '#e67e22'),
        (0.0, 0.25, 'BIT*', 'Euclidean heuristics\n→ mis-ranked edges', '#f39c12'),
        (0.0, 0.0, 'All of the above', 'Ignore collision data\n→ repeat mistakes', '#c0392b'),
    ]
    for x, y, name, issue, col in boxes:
        bbox = FancyBboxPatch((x, y), 0.35, 0.18, boxstyle="round,pad=0.02",
                              fc=col, ec=col, alpha=0.15)
        ax.add_patch(bbox)
        ax.text(x + 0.175, y + 0.14, name, ha='center', va='center', fontweight='bold',
                fontsize=10, color=col)
        ax.text(x + 0.175, y + 0.05, issue, ha='center', va='center', fontsize=8, color='#2c3e50')
        # Arrow to RIT* side
        ax.annotate('', xy=(0.65, y + 0.09), xytext=(0.38, y + 0.09),
                    arrowprops=dict(arrowstyle='->', color='#2c3e50', lw=1.5))

    # Panel (b): RIT* solutions
    ax = axes[1]
    ax.set_xlim(-0.3, 1.3); ax.set_ylim(-0.1, 1.1)
    ax.set_aspect('equal')
    ax.set_title('(b) RIT* + CARM: Our Solutions', fontweight='bold', fontsize=11)
    ax.axis('off')

    solutions = [
        (0.65, 0.85, 'Riemannian\nInformed Set', 'Tighter focus\n→ 60% fewer wasted samples', '#27ae60'),
        (0.65, 0.55, 'Whitened\nSampling', 'Handles anisotropy\n→ rejection-free sampling', '#2980b9'),
        (0.65, 0.25, 'Cascading Edge\nEvaluation', 'Riemannian heuristics\n→ 3-level lazy filtering', '#8e44ad'),
        (0.65, 0.0, 'CARM', 'Collisions → cost map\n→ learns obstacle layout', '#e74c3c'),
    ]
    for x, y, name, solution, col in solutions:
        bbox = FancyBboxPatch((x, y), 0.35, 0.18, boxstyle="round,pad=0.02",
                              fc=col, ec=col, alpha=0.15)
        ax.add_patch(bbox)
        ax.text(x + 0.175, y + 0.14, name, ha='center', va='center', fontweight='bold',
                fontsize=9, color=col)
        ax.text(x + 0.175, y + 0.04, solution, ha='center', va='center', fontsize=7.5, color='#2c3e50')

    plt.tight_layout()
    plt.savefig(f'{OUTDIR}/fig_explained_novelty.pdf', bbox_inches='tight')
    plt.savefig(f'{OUTDIR}/fig_explained_novelty.png', bbox_inches='tight')
    plt.close()
    print("  → fig_explained_novelty.pdf")


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 6: Cascading Edge Evaluation Detail
# ═══════════════════════════════════════════════════════════════════════════════
def fig_cascade_detail():
    fig, ax = plt.subplots(1, 1, figsize=(12, 5))
    ax.set_xlim(-0.5, 10.5); ax.set_ylim(-1, 6)
    ax.axis('off')
    ax.set_title('Cascading Lazy Edge Evaluation — How Edges Are Checked', fontweight='bold', fontsize=13)

    # Funnel visualization
    levels = [
        (0, 5, '100 candidate edges', 'Level 1: Midpoint Check\n(1 metric lookup)', '#3498db', 100),
        (3.5, 5, '~20 pass L1', 'Level 2: Simpson\'s Rule\n(3 metric lookups)', '#e67e22', 20),
        (7, 5, '~5 pass L2', 'Level 3: Gauss-Legendre\n(10 metric lookups)', '#e74c3c', 5),
    ]

    for x, y, count_text, desc, col, bar_w in levels:
        # Bar representing edges
        bar_width = bar_w / 100 * 3
        bar = FancyBboxPatch((x, 1), bar_width, 3, boxstyle="round,pad=0.1",
                             fc=col, ec=col, alpha=0.3)
        ax.add_patch(bar)
        ax.text(x + bar_width / 2, 2.5, count_text, ha='center', va='center',
                fontsize=10, fontweight='bold', color=col)
        ax.text(x + bar_width / 2, 4.5, desc, ha='center', va='bottom',
                fontsize=9, color='#2c3e50',
                bbox=dict(boxstyle='round', fc='white', ec=col, alpha=0.8))

    # Arrows between stages
    ax.annotate('', xy=(3.3, 2.5), xytext=(3.2, 2.5),
                arrowprops=dict(arrowstyle='->', color='#2c3e50', lw=2))
    ax.annotate('', xy=(6.8, 2.5), xytext=(4.3, 2.5),
                arrowprops=dict(arrowstyle='->', color='#2c3e50', lw=2))

    # Rejected edges
    ax.text(2.0, 0.3, '~80% rejected\n(fast & cheap)', ha='center', fontsize=9,
            color='#e74c3c', style='italic')
    ax.text(5.5, 0.3, '~75% rejected\n(moderate cost)', ha='center', fontsize=9,
            color='#e74c3c', style='italic')
    ax.text(8.5, 0.3, '~5 survive\n(fully evaluated)', ha='center', fontsize=9,
            color='#27ae60', fontweight='bold')

    plt.tight_layout()
    plt.savefig(f'{OUTDIR}/fig_explained_cascade.pdf', bbox_inches='tight')
    plt.savefig(f'{OUTDIR}/fig_explained_cascade.png', bbox_inches='tight')
    plt.close()
    print("  → fig_explained_cascade.pdf")


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 7: Euclidean vs Riemannian distance concept
# ═══════════════════════════════════════════════════════════════════════════════
def fig_euclidean_vs_riemannian():
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))

    # Panel (a): Flat map (Euclidean)
    ax = axes[0]
    ax.set_xlim(-0.1, 1.1); ax.set_ylim(-0.1, 1.1)
    ax.set_aspect('equal')
    ax.set_title('(a) Euclidean: Flat Map\n(all directions equal)', fontweight='bold', fontsize=10)

    # Uniform background
    ax.fill([0, 1, 1, 0], [0, 0, 1, 1], color='#d5f5e3', alpha=0.5)
    obst = Circle((0.5, 0.5), 0.15, fc='#bdc3c7', ec='#7f8c8d', lw=2)
    ax.add_patch(obst)

    # Straight line (shortest euclidean)
    ax.plot([0.1, 0.9], [0.1, 0.9], '-', color='#3498db', lw=3, label='Shortest path')
    ax.plot([0.1, 0.9], [0.1, 0.9], 'o', color='#3498db', ms=8)

    # Equal circles at grid points
    for gx in [0.2, 0.5, 0.8]:
        for gy in [0.2, 0.5, 0.8]:
            c = Circle((gx, gy), 0.06, fc='none', ec='#3498db', lw=1, ls='--', alpha=0.5)
            ax.add_patch(c)

    ax.text(0.5, 0.08, 'Same "distance" everywhere\n→ cuts close to obstacle', ha='center', fontsize=9, style='italic')
    ax.set_xticks([]); ax.set_yticks([])

    # Panel (b): Curved landscape (Riemannian)
    ax = axes[1]
    ax.set_xlim(-0.1, 1.1); ax.set_ylim(-0.1, 1.1)
    ax.set_aspect('equal')
    ax.set_title('(b) Riemannian: Hilly Terrain\n(some directions harder)', fontweight='bold', fontsize=10)

    # Cost heatmap
    xx, yy = np.meshgrid(np.linspace(0, 1, 100), np.linspace(0, 1, 100))
    cost = 1 + 8 * np.exp(-((xx - 0.5) ** 2 + (yy - 0.5) ** 2) / (2 * 0.12 ** 2))
    ax.contourf(xx, yy, cost, levels=20, cmap='YlOrRd', alpha=0.5)
    obst2 = Circle((0.5, 0.5), 0.15, fc='#bdc3c7', ec='#7f8c8d', lw=2, zorder=3)
    ax.add_patch(obst2)

    # Curved path (avoids high-cost)
    t = np.linspace(0, 1, 50)
    px = 0.1 + 0.8 * t
    py = 0.1 + 0.8 * t + 0.25 * np.sin(np.pi * t)
    ax.plot(px, py, '-', color='#e74c3c', lw=3, label='Riemannian path', zorder=4)
    ax.plot([0.1, 0.9], [0.1, 0.9], 'o', color='#e74c3c', ms=8, zorder=5)

    # Ellipses at grid points (varying size)
    for gx in [0.2, 0.5, 0.8]:
        for gy in [0.2, 0.5, 0.8]:
            d = np.sqrt((gx - 0.5) ** 2 + (gy - 0.5) ** 2)
            s = 0.03 + 0.05 * min(d / 0.3, 1)
            e = Ellipse((gx, gy), s * 2, s, angle=45, fc='none', ec='#e74c3c', lw=1, ls='--', alpha=0.6)
            ax.add_patch(e)

    ax.text(0.5, 0.08, '"Distance" varies by location\n→ detours around expensive zones', ha='center', fontsize=9, style='italic')
    ax.set_xticks([]); ax.set_yticks([])

    plt.tight_layout()
    plt.savefig(f'{OUTDIR}/fig_explained_euclid_vs_riem.pdf', bbox_inches='tight')
    plt.savefig(f'{OUTDIR}/fig_explained_euclid_vs_riem.png', bbox_inches='tight')
    plt.close()
    print("  → fig_explained_euclid_vs_riem.pdf")


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 8: Full System Overview Flowchart
# ═══════════════════════════════════════════════════════════════════════════════
def fig_full_system_flowchart():
    fig, ax = plt.subplots(1, 1, figsize=(14, 7))
    ax.set_xlim(-1, 15); ax.set_ylim(-1, 8)
    ax.axis('off')
    ax.set_title('RIT* + CARM: Complete System Overview', fontweight='bold', fontsize=14, pad=15)

    # Flow boxes
    boxes = [
        (0.5, 6, 3, 1.2, 'START\nGet (x_start, x_goal)', '#27ae60'),
        (0.5, 4, 3, 1.2, '① Sample Batch\nfrom Riemannian\nInformed Set', '#3498db'),
        (4.5, 4, 3, 1.2, '② Find Neighbors\nRiemannian radius\n(ellipsoidal)', '#9b59b6'),
        (8.5, 4, 3, 1.2, '③ Extend Tree\nCascading edge\nevaluation (L1→L2→L3)', '#e67e22'),
        (8.5, 6, 3, 1.2, '④ Rewire Tree\nFind cheaper parents\nfor existing nodes', '#2980b9'),
        (8.5, 1.5, 3, 1.2, 'Collision?\nFeed point to\nCARM', '#e74c3c'),
        (4.5, 1.5, 3, 1.2, '⑤ Update CARM\nRebuild cost map\nevery Δ iterations', '#c0392b'),
        (0.5, 1.5, 3, 1.2, '⑥ Prune & Update\nc_best, shrink\ninformed set', '#f39c12'),
        (12.5, 4, 2, 1.2, 'DONE\nReturn\nbest path', '#27ae60'),
    ]

    for x, y, w, h, text, col in boxes:
        bbox = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.15",
                              fc=col, ec=col, alpha=0.2, lw=2)
        ax.add_patch(bbox)
        ax.text(x + w / 2, y + h / 2, text, ha='center', va='center',
                fontsize=8.5, fontweight='bold', color='#2c3e50')

    # Arrows
    arrows = [
        ((2, 5.2), (2, 5.9)),      # start → sample
        ((3.5, 4.6), (4.5, 4.6)),  # sample → neighbors
        ((7.5, 4.6), (8.5, 4.6)),  # neighbors → extend
        ((10, 5.2), (10, 5.9)),    # extend → rewire
        ((10, 4.0), (10, 2.7)),    # extend → collision check
        ((8.5, 2.1), (7.5, 2.1)),  # collision → CARM
        ((4.5, 2.1), (3.5, 2.1)),  # CARM → prune
        ((2, 2.7), (2, 3.9)),      # prune → sample (loop)
        ((11.5, 6.6), (12.5, 4.8)),  # rewire → done (converged)
    ]

    for (x1, y1), (x2, y2) in arrows:
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle='->', color='#2c3e50', lw=2))

    # Loop label
    ax.text(0.8, 3.5, 'REPEAT\nuntil\nconverged', ha='center', fontsize=8,
            color='#7f8c8d', style='italic',
            bbox=dict(boxstyle='round', fc='#ecf0f1', ec='#bdc3c7'))

    plt.tight_layout()
    plt.savefig(f'{OUTDIR}/fig_explained_system_overview.pdf', bbox_inches='tight')
    plt.savefig(f'{OUTDIR}/fig_explained_system_overview.png', bbox_inches='tight')
    plt.close()
    print("  → fig_explained_system_overview.pdf")


# ═══════════════════════════════════════════════════════════════════════════════
# Run all
# ═══════════════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    print("Generating conceptual figures for explained document...")
    fig_problem_overview()
    fig_baselines_overview()
    fig_rit_pipeline_steps()
    fig_carm_learning()
    fig_novelty_comparison()
    fig_cascade_detail()
    fig_euclidean_vs_riemannian()
    fig_full_system_flowchart()
    print("Done! All figures in paper/figures/")
