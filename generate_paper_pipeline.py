#!/usr/bin/env python3
"""Generate compact conceptual pipeline figures for the main paper.

Creates two publication-quality figures:
1. Full RIT*+CARM pipeline as a single compact figure (2-column IEEE width)
2. Compact step-by-step showing sample → neighbor → extend → rewire → CARM
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Circle, Ellipse
from matplotlib.collections import LineCollection
import os

OUTDIR = 'paper/figures'
os.makedirs(OUTDIR, exist_ok=True)

plt.rcParams.update({
    'font.size': 9,
    'font.family': 'serif',
    'axes.linewidth': 0.8,
    'figure.dpi': 300,
    'text.usetex': False,
})

COL = {
    'start': '#27ae60', 'goal': '#c0392b', 'tree': '#2980b9',
    'path': '#d35400', 'obstacle': '#bdc3c7', 'obst_edge': '#7f8c8d',
    'collision': '#e74c3c', 'free': '#27ae60', 'carm': '#c0392b',
    'new_sample': '#e67e22', 'rewire_new': '#27ae60', 'rewire_old': '#e74c3c',
    'neighbor': '#8e44ad', 'cascade_l1': '#3498db', 'cascade_l2': '#e67e22',
    'cascade_l3': '#e74c3c', 'informed': '#f39c12',
}

# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 1: Compact 6-panel step-by-step pipeline (full page width)
# ═══════════════════════════════════════════════════════════════════════════════
def fig_compact_pipeline():
    fig, axes = plt.subplots(2, 3, figsize=(7.16, 4.2))  # IEEE 2-column width

    # Common elements
    np.random.seed(42)
    obs_list = [(0.4, 0.48, 0.09), (0.68, 0.32, 0.07), (0.55, 0.72, 0.08)]
    tree_pts = np.array([
        (0.06, 0.06), (0.16, 0.20), (0.22, 0.38), (0.24, 0.14),
        (0.48, 0.62), (0.52, 0.50), (0.78, 0.80), (0.82, 0.58)
    ])
    tree_edges = [(0,1),(1,2),(1,3),(2,4),(4,5),(4,6),(6,7)]

    def draw_base(ax, title, draw_tree=True, alpha_tree=0.35):
        ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)
        ax.set_aspect('equal')
        ax.set_title(title, fontsize=8.5, fontweight='bold', pad=4)
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_linewidth(0.5)
        for cx, cy, r in obs_list:
            ax.add_patch(Circle((cx, cy), r, fc='#ecf0f1', ec='#95a5a6', lw=0.8, zorder=1))
        ax.plot(0.06, 0.06, 'o', color=COL['start'], ms=7, zorder=10)
        ax.plot(0.94, 0.94, '*', color=COL['goal'], ms=8, zorder=10)
        if draw_tree:
            for e in tree_edges:
                ax.plot([tree_pts[e[0],0], tree_pts[e[1],0]],
                        [tree_pts[e[0],1], tree_pts[e[1],1]],
                        '-', color=COL['tree'], lw=0.7, alpha=alpha_tree, zorder=2)
            ax.plot(tree_pts[:,0], tree_pts[:,1], 'o', color=COL['tree'], ms=2.5, alpha=alpha_tree, zorder=3)

    # ─── Panel (a): Sample from Riemannian Informed Set ───────────────────────
    ax = axes[0, 0]
    draw_base(ax, r'(a) Sample from $\mathcal{I}_R$')
    # Informed set ellipse
    cx, cy = (0.06+0.94)/2, (0.06+0.94)/2
    ell = Ellipse((cx, cy), 0.82, 0.36, angle=45, fc=COL['informed'], ec=COL['informed'],
                  lw=1.5, alpha=0.12, ls='--', zorder=0)
    ax.add_patch(ell)
    # New samples
    samples = []
    for _ in range(18):
        ang = np.random.uniform(0, 2*np.pi)
        r_s = np.random.uniform(0, 0.35)
        sx = cx + r_s * np.cos(ang + np.pi/4)
        sy = cy + r_s * np.sin(ang + np.pi/4)
        if 0 < sx < 1 and 0 < sy < 1:
            samples.append((sx, sy))
    samples = np.array(samples)
    ax.plot(samples[:,0], samples[:,1], 'D', color=COL['new_sample'], ms=3, zorder=5, alpha=0.8)
    ax.text(0.72, 0.12, r'$\mathcal{I}_R$', fontsize=9, color=COL['informed'],
            fontweight='bold', zorder=6)

    # ─── Panel (b): Find Riemannian Neighbors ─────────────────────────────────
    ax = axes[0, 1]
    draw_base(ax, r'(b) Riemannian neighbors ($r_n^R$)')
    ns = np.array([0.38, 0.65])
    ax.plot(ns[0], ns[1], 'D', color=COL['new_sample'], ms=5, zorder=8)
    # Ellipsoidal neighbor ball
    ell2 = Ellipse(ns, 0.30, 0.18, angle=25, fc='none', ec=COL['neighbor'],
                   lw=1.5, ls='--', zorder=4)
    ax.add_patch(ell2)
    # Neighbors
    for idx in [2, 4, 5]:
        ax.plot([ns[0], tree_pts[idx,0]], [ns[1], tree_pts[idx,1]],
                ':', color=COL['neighbor'], lw=1, alpha=0.7, zorder=3)
        ax.plot(tree_pts[idx,0], tree_pts[idx,1], 'o', color=COL['neighbor'], ms=4, zorder=6)
    ax.text(0.58, 0.78, 'ellipsoidal\nball', fontsize=6, color=COL['neighbor'],
            fontstyle='italic', ha='center')

    # ─── Panel (c): Cascading Edge Evaluation ─────────────────────────────────
    ax = axes[0, 2]
    draw_base(ax, '(c) Cascading edge check', draw_tree=True, alpha_tree=0.25)
    ns = np.array([0.38, 0.65])
    ax.plot(ns[0], ns[1], 'D', color=COL['new_sample'], ms=5, zorder=8)
    # Show 3 candidate edges with L1/L2/L3 labels
    cands = [
        (2, 'L1 reject', COL['cascade_l1'], '--', 0.5),
        (5, 'L2 reject', COL['cascade_l2'], '--', 0.6),
        (4, 'L3 accept', COL['cascade_l3'], '-', 1.0),
    ]
    for idx, label, col, ls, alpha in cands:
        lw = 2.0 if ls == '-' else 1.0
        ax.plot([ns[0], tree_pts[idx,0]], [ns[1], tree_pts[idx,1]],
                ls, color=col, lw=lw, alpha=alpha, zorder=4)
        mid = (ns + tree_pts[idx]) / 2
        ax.text(mid[0]+0.04, mid[1]-0.02, label, fontsize=5.5, color=col,
                fontweight='bold', zorder=6)
    # Best parent highlight
    ax.plot([ns[0], tree_pts[4,0]], [ns[1], tree_pts[4,1]], '-',
            color=COL['rewire_new'], lw=2, zorder=5)

    # ─── Panel (d): Rewire ────────────────────────────────────────────────────
    ax = axes[1, 0]
    draw_base(ax, '(d) Rewire through new node')
    ns = np.array([0.38, 0.65])
    # Add new node to tree
    ax.plot(ns[0], ns[1], 'D', color=COL['new_sample'], ms=5, zorder=8)
    ax.plot([ns[0], tree_pts[4,0]], [ns[1], tree_pts[4,1]], '-',
            color=COL['rewire_new'], lw=1.5, zorder=5)
    # Old edge 4->5 removed
    ax.plot([tree_pts[4,0], tree_pts[5,0]], [tree_pts[4,1], tree_pts[5,1]],
            '--', color=COL['rewire_old'], lw=1.5, alpha=0.6, zorder=4)
    ax.plot(tree_pts[4,0], tree_pts[4,1], 'X', color=COL['rewire_old'], ms=6, mew=1.5, zorder=6)
    # New edge: new_node -> 5
    ax.plot([ns[0], tree_pts[5,0]], [ns[1], tree_pts[5,1]], '-',
            color=COL['rewire_new'], lw=2, zorder=5)
    ax.text(0.48, 0.42, 'old (removed)', fontsize=5.5, color=COL['rewire_old'], fontstyle='italic')
    ax.text(0.28, 0.52, 'new\n(cheaper)', fontsize=5.5, color=COL['rewire_new'], fontweight='bold')

    # ─── Panel (e): CARM Collision Feedback ───────────────────────────────────
    ax = axes[1, 1]
    draw_base(ax, '(e) CARM: collisions build cost map', draw_tree=False)
    # Show collision points around obstacles
    coll_pts = []
    for cx_o, cy_o, r_o in obs_list:
        for _ in range(10):
            ang = np.random.uniform(0, 2*np.pi)
            rr = r_o * (0.85 + 0.3*np.random.rand())
            coll_pts.append((cx_o + rr*np.cos(ang), cy_o + rr*np.sin(ang)))
    coll_pts = np.array(coll_pts)
    ax.plot(coll_pts[:,0], coll_pts[:,1], 'x', color=COL['collision'], ms=4, mew=1.2, zorder=5)
    # Density heatmap (lightweight)
    xx, yy = np.meshgrid(np.linspace(0, 1, 60), np.linspace(0, 1, 60))
    pts_g = np.column_stack([xx.ravel(), yy.ravel()])
    sigma = 0.08
    density = np.zeros(len(pts_g))
    for cp in coll_pts:
        density += np.exp(-np.sum((pts_g - cp)**2, axis=1) / (2*sigma**2))
    density = density.reshape(xx.shape) / len(coll_pts)
    scale = 1.0 + 6.0 * density
    ax.contourf(xx, yy, scale, levels=15, cmap='YlOrRd', alpha=0.4, zorder=0)
    # Arrow showing feedback
    ax.annotate('collisions\n' + r'$\rightarrow$ cost map', xy=(0.12, 0.85),
                fontsize=6, color=COL['carm'], fontweight='bold', ha='center')

    # ─── Panel (f): Prune & Shrink → Better Path ─────────────────────────────
    ax = axes[1, 2]
    draw_base(ax, '(f) Prune, shrink, converge', draw_tree=True, alpha_tree=0.2)
    # Small informed set
    ell_big = Ellipse((cx, cy), 0.82, 0.36, angle=45, fc='none', ec=COL['tree'],
                      lw=0.8, alpha=0.3, ls=':', zorder=0)
    ell_small = Ellipse((cx, cy), 0.52, 0.2, angle=45, fc=COL['informed'],
                        ec=COL['informed'], lw=1.5, alpha=0.1, zorder=0)
    ax.add_patch(ell_big)
    ax.add_patch(ell_small)
    # Final path
    path_pts = [(0.06,0.06), (0.16,0.20), (0.22,0.38), (0.35,0.62),
                (0.62,0.85), (0.78,0.88), (0.94,0.94)]
    px, py = zip(*path_pts)
    ax.plot(px, py, '-', color=COL['path'], lw=2.5, zorder=7)
    for ppx, ppy in path_pts:
        ax.plot(ppx, ppy, 'o', color=COL['path'], ms=3, zorder=8)
    # Pruned nodes
    for idx in [3, 7]:
        ax.plot(tree_pts[idx,0], tree_pts[idx,1], 'X', color='#95a5a6', ms=5, mew=1.5, zorder=6)
    ax.text(0.72, 0.15, 'shrinks!', fontsize=6, color=COL['informed'],
            fontweight='bold', fontstyle='italic')

    plt.tight_layout(pad=0.5)
    plt.savefig(f'{OUTDIR}/fig_compact_pipeline.pdf', bbox_inches='tight', dpi=300)
    plt.savefig(f'{OUTDIR}/fig_compact_pipeline.png', bbox_inches='tight', dpi=300)
    plt.close()
    print("  -> fig_compact_pipeline.pdf")


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 2: CARM Feedback Loop (single column)
# ═══════════════════════════════════════════════════════════════════════════════
def fig_carm_feedback_loop():
    fig, axes = plt.subplots(1, 3, figsize=(7.16, 2.3))  # IEEE 2-col width, short

    np.random.seed(99)
    obs_list = [(0.35, 0.50, 0.12), (0.65, 0.35, 0.09), (0.55, 0.72, 0.10)]

    def draw_obs(ax):
        ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02); ax.set_aspect('equal')
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_linewidth(0.5)
        for cx, cy, r in obs_list:
            ax.add_patch(Circle((cx, cy), r, fc='#d5dbdb', ec='#7f8c8d', lw=0.8))

    # Panel (a): Collisions collected
    ax = axes[0]
    draw_obs(ax)
    ax.set_title('(a) Collect collisions', fontsize=8, fontweight='bold', pad=3)
    coll_pts = []
    for cx, cy, r in obs_list:
        for _ in range(12):
            ang = np.random.uniform(0, 2*np.pi)
            rr = r * (0.8 + 0.4*np.random.rand())
            coll_pts.append((cx + rr*np.cos(ang), cy + rr*np.sin(ang)))
    coll_pts = np.array(coll_pts)
    # Some free samples
    free_pts = np.random.rand(15, 2)
    ax.plot(free_pts[:,0], free_pts[:,1], 'o', color=COL['free'], ms=2.5, alpha=0.5, zorder=4)
    ax.plot(coll_pts[:,0], coll_pts[:,1], 'x', color=COL['collision'], ms=4, mew=1.3, zorder=5)
    ax.plot(0.06, 0.06, 'o', color=COL['start'], ms=6, zorder=10)
    ax.plot(0.94, 0.94, '*', color=COL['goal'], ms=7, zorder=10)

    # Panel (b): Cost field
    ax = axes[1]
    draw_obs(ax)
    ax.set_title(r'(b) Build $s(x) = 1 + \alpha\hat{f}(x)$', fontsize=8, fontweight='bold', pad=3)
    xx, yy = np.meshgrid(np.linspace(0, 1, 70), np.linspace(0, 1, 70))
    pts_g = np.column_stack([xx.ravel(), yy.ravel()])
    sigma = 0.08
    density = np.zeros(len(pts_g))
    for cp in coll_pts:
        density += np.exp(-np.sum((pts_g - cp)**2, axis=1) / (2*sigma**2))
    density = density.reshape(xx.shape) / len(coll_pts)
    scale = 1.0 + 6.0 * density
    ax.contourf(xx, yy, scale, levels=15, cmap='hot_r', alpha=0.6, zorder=0)
    ax.contour(xx, yy, scale, levels=5, colors='k', linewidths=0.3, alpha=0.3, zorder=1)
    ax.plot(coll_pts[:,0], coll_pts[:,1], '.', color='cyan', ms=1.5, alpha=0.6, zorder=4)
    ax.plot(0.06, 0.06, 'o', color=COL['start'], ms=6, zorder=10)
    ax.plot(0.94, 0.94, '*', color=COL['goal'], ms=7, zorder=10)

    # Panel (c): Tighter sampling → better path
    ax = axes[2]
    draw_obs(ax)
    ax.set_title(r'(c) Tighter $\mathcal{I}_R^{\mathrm{CARM}}$ + better path', fontsize=8, fontweight='bold', pad=3)
    ax.contourf(xx, yy, scale, levels=15, cmap='hot_r', alpha=0.2, zorder=0)
    # Euclidean ellipse (large)
    cx, cy = 0.5, 0.5
    ell_e = Ellipse((cx, cy), 0.88, 0.38, angle=45, fc='none', ec=COL['tree'],
                    lw=1, ls=':', alpha=0.5, zorder=1)
    ax.add_patch(ell_e)
    # CARM ellipse (smaller)
    ell_c = Ellipse((cx, cy), 0.6, 0.22, angle=45, fc=COL['carm'],
                    ec=COL['carm'], lw=1.5, ls='--', alpha=0.1, zorder=1)
    ax.add_patch(ell_c)
    # Euclidean path
    ep = [(0.06,0.06), (0.20,0.25), (0.40,0.42), (0.60,0.55), (0.80,0.78), (0.94,0.94)]
    epx, epy = zip(*ep)
    ax.plot(epx, epy, '--', color='#7f8c8d', lw=1.2, alpha=0.6, zorder=3, label='Euclidean')
    # CARM path
    cp = [(0.06,0.06), (0.12,0.20), (0.10,0.42), (0.18,0.68), (0.40,0.86),
          (0.70,0.88), (0.88,0.90), (0.94,0.94)]
    cpx, cpy = zip(*cp)
    ax.plot(cpx, cpy, '-', color=COL['carm'], lw=2, zorder=5, label='CARM')
    ax.plot(0.06, 0.06, 'o', color=COL['start'], ms=6, zorder=10)
    ax.plot(0.94, 0.94, '*', color=COL['goal'], ms=7, zorder=10)
    ax.legend(fontsize=5.5, loc='lower right', framealpha=0.8)
    ax.text(0.75, 0.18, r'$\mathcal{I}_E$', fontsize=7, color=COL['tree'], alpha=0.6)
    ax.text(0.58, 0.30, r'$\mathcal{I}_R^{\mathrm{CARM}}$', fontsize=7, color=COL['carm'],
            fontweight='bold')

    plt.tight_layout(pad=0.3)
    plt.savefig(f'{OUTDIR}/fig_carm_pipeline_compact.pdf', bbox_inches='tight', dpi=300)
    plt.savefig(f'{OUTDIR}/fig_carm_pipeline_compact.png', bbox_inches='tight', dpi=300)
    plt.close()
    print("  -> fig_carm_pipeline_compact.pdf")


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 3: Tree growth sequence (4 snapshots)
# ═══════════════════════════════════════════════════════════════════════════════
def fig_tree_growth_sequence():
    """Four snapshots of tree growth: early exploration, first path, refinement, converged."""
    fig, axes = plt.subplots(1, 4, figsize=(7.16, 1.9))  # IEEE 2-col width
    np.random.seed(55)

    obs_list = [(0.4, 0.48, 0.10), (0.7, 0.30, 0.08)]
    sx, sy, gx, gy = 0.06, 0.06, 0.94, 0.94

    titles = [r'$t=0$: explore', r'$t=30$: first path',
              r'$t=80$: refine', r'$t=150$: converged']
    n_pts_list = [15, 40, 60, 40]  # tree sizes at each stage
    has_path = [False, True, True, True]
    informed_size = [None, 0.9, 0.65, 0.4]

    for i, (ax, title, n_pts) in enumerate(zip(axes, titles, n_pts_list)):
        ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02); ax.set_aspect('equal')
        ax.set_title(title, fontsize=7, fontweight='bold', pad=3)
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_linewidth(0.4)

        for cx, cy, r in obs_list:
            ax.add_patch(Circle((cx, cy), r, fc='#ecf0f1', ec='#95a5a6', lw=0.6))

        # Generate tree points
        pts = [(sx, sy)]
        for j in range(n_pts):
            if informed_size[i] and j > 5:
                # Bias towards informed set
                ang = np.random.uniform(0, 2*np.pi)
                r_s = np.random.uniform(0, informed_size[i]/2.5)
                px = 0.5 + r_s * np.cos(ang + np.pi/4)
                py = 0.5 + r_s * np.sin(ang + np.pi/4)
            else:
                px, py = np.random.uniform(0.02, 0.98), np.random.uniform(0.02, 0.98)
            pts.append((px, py))
        pts = np.array(pts)

        # Draw random tree edges
        for j in range(1, len(pts)):
            parent = np.random.randint(0, j)
            ax.plot([pts[parent,0], pts[j,0]], [pts[parent,1], pts[j,1]],
                    '-', color=COL['tree'], lw=0.4, alpha=0.3)
        ax.plot(pts[:,0], pts[:,1], '.', color=COL['tree'], ms=1.5, alpha=0.4)

        # Informed set
        if informed_size[i]:
            ell = Ellipse((0.5, 0.5), informed_size[i], informed_size[i]*0.4,
                          angle=45, fc=COL['informed'], ec=COL['informed'],
                          lw=1, ls='--', alpha=0.08, zorder=0)
            ax.add_patch(ell)

        # Path
        if has_path[i]:
            path = [(sx, sy), (0.15, 0.20), (0.20, 0.42)]
            if i >= 2:
                path += [(0.30, 0.62), (0.55, 0.85), (0.80, 0.88), (gx, gy)]
            else:
                path += [(0.48, 0.60), (0.60, 0.70), (0.80, 0.82), (gx, gy)]
            ppx, ppy = zip(*path)
            lw = 1.5 if i < 3 else 2.0
            ax.plot(ppx, ppy, '-', color=COL['path'], lw=lw, zorder=6)

        ax.plot(sx, sy, 'o', color=COL['start'], ms=5, zorder=10)
        ax.plot(gx, gy, '*', color=COL['goal'], ms=6, zorder=10)

    plt.tight_layout(pad=0.3)
    plt.savefig(f'{OUTDIR}/fig_tree_growth_compact.pdf', bbox_inches='tight', dpi=300)
    plt.savefig(f'{OUTDIR}/fig_tree_growth_compact.png', bbox_inches='tight', dpi=300)
    plt.close()
    print("  -> fig_tree_growth_compact.pdf")


if __name__ == '__main__':
    print("Generating compact paper figures...")
    fig_compact_pipeline()
    fig_carm_feedback_loop()
    fig_tree_growth_sequence()
    print("Done!")
