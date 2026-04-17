#!/usr/bin/env python3
"""
generate_paper_figures.py — Conceptual figures for the RIT* T-RO paper.

Generates publication-quality figures with IEEE-compatible formatting:
  Fig 1: Euclidean vs Riemannian informed set comparison
  Fig 2: Riemannian metric tensor field visualization
  Fig 3: Sampling comparison (Euclidean uniform vs Riemannian informed)
  Fig 4: Nearest-neighbor selection (Euclidean ball vs Riemannian ball)
  Fig 5: Rewiring step illustration
  Fig 6: Full algorithm pipeline (batch processing overview)
  Fig 7: High-D improvements conceptual illustrations
  Fig 8: Convergence behavior (cost vs iteration)

All figures saved to paper/figures/ as PDF for crisp vector output.
"""

import sys
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Ellipse, FancyArrowPatch, Circle
from matplotlib.collections import LineCollection
from matplotlib import patheffects
import matplotlib.gridspec as gridspec

# ── Setup ──
OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'paper', 'figures')
os.makedirs(OUT_DIR, exist_ok=True)

# IEEE-compatible styling
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 9,
    'axes.labelsize': 10,
    'axes.titlesize': 10,
    'legend.fontsize': 8,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.02,
    'lines.linewidth': 1.0,
    'lines.markersize': 4,
    'text.usetex': False,
})

# Color palette
C_RIT = '#2563EB'       # blue
C_EUCL = '#DC2626'      # red
C_TREE = '#059669'      # green
C_OBS = '#6B7280'       # gray
C_PATH = '#F59E0B'      # amber
C_REWIRE = '#7C3AED'    # purple
C_SAMPLE = '#0EA5E9'    # sky
C_GOAL = '#EF4444'      # red
C_START = '#22C55E'     # green
C_LIGHT = '#E5E7EB'     # light gray


def _draw_obstacles(ax, circles, color=C_OBS, alpha=0.35):
    """Draw circular obstacles."""
    for (cx, cy), r in circles:
        c = Circle((cx, cy), r, fc=color, ec='#374151', lw=0.5, alpha=alpha, zorder=2)
        ax.add_patch(c)


def _draw_ellipse(ax, cx, cy, w, h, angle=0, color='blue', ls='-', lw=1.5, label=None, fill=False, alpha=0.15):
    """Draw an ellipse."""
    e = Ellipse((cx, cy), w, h, angle=angle,
                fc=color if fill else 'none',
                ec=color, ls=ls, lw=lw, alpha=alpha if fill else 1.0,
                label=label, zorder=3)
    ax.add_patch(e)
    return e


def _star_marker(ax, x, y, color, label, size=80, zorder=10):
    ax.scatter(x, y, s=size, c=color, marker='*', edgecolors='k',
               linewidths=0.4, zorder=zorder, label=label)


def _save(fig, name):
    path = os.path.join(OUT_DIR, name)
    fig.savefig(path, dpi=300)
    plt.close(fig)
    print(f'  -> {path}')


# ═══════════════════════════════════════════════════════════════════════
#  Fig 1: Euclidean vs Riemannian Informed Set
# ═══════════════════════════════════════════════════════════════════════

def fig1_informed_sets():
    """Side-by-side comparison of Euclidean and Riemannian informed sets."""
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 2.8), sharey=True)

    xs, ys = 0.15, 0.5
    xg, yg = 0.85, 0.5
    # Obstacles
    circles = [
        ((0.35, 0.35), 0.07),
        ((0.35, 0.65), 0.07),
        ((0.50, 0.45), 0.08),
        ((0.50, 0.75), 0.08),
        ((0.65, 0.40), 0.07),
        ((0.65, 0.60), 0.07),
    ]

    for i, (ax, title) in enumerate(zip(axes, ['(a) Euclidean informed set $I_E$',
                                                '(b) Riemannian informed set $I_R$'])):
        _draw_obstacles(ax, circles)

        cx = (xs + xg) / 2
        cy = (ys + yg) / 2

        if i == 0:
            # Euclidean: large symmetric ellipse
            _draw_ellipse(ax, cx, cy, 0.82, 0.60, angle=0,
                         color=C_EUCL, lw=2.0, fill=True, alpha=0.12,
                         label='$I_E$ (Euclidean)')
            _draw_ellipse(ax, cx, cy, 0.82, 0.60, angle=0,
                         color=C_EUCL, lw=2.0)
            # Wasted samples
            rng = np.random.default_rng(42)
            n_waste = 40
            for _ in range(n_waste):
                theta = rng.uniform(0, 2*np.pi)
                r = rng.uniform(0.5, 1.0)
                px = cx + r * 0.41 * np.cos(theta)
                py = cy + r * 0.30 * np.sin(theta)
                # Check if in Eucl ellipse but outside Riemannian
                in_ellipse = ((px - cx)/(0.41))**2 + ((py - cy)/(0.30))**2 <= 1
                dx_r = (px - cx) * 1.5  # anisotropic stretch
                in_riem = (dx_r/(0.41))**2 + ((py - cy)/(0.30))**2 <= 0.6
                if in_ellipse and not in_riem:
                    ax.plot(px, py, 'x', color='#EF4444', ms=3, mew=0.5,
                            alpha=0.5, zorder=4)
            # Label
            ax.annotate('wasted\nsamples', xy=(0.30, 0.72), fontsize=7,
                        color='#EF4444', ha='center', style='italic')
        else:
            # Riemannian: smaller, anisotropic ellipse
            # Show Euclidean for reference (dashed)
            _draw_ellipse(ax, cx, cy, 0.82, 0.60, angle=0,
                         color=C_EUCL, ls='--', lw=1.0, alpha=0.4,
                         label='$I_E$ (Euclidean)')
            # Riemannian: narrower along high-cost direction, rotated
            _draw_ellipse(ax, cx, cy, 0.75, 0.38, angle=15,
                         color=C_RIT, lw=2.0, fill=True, alpha=0.15,
                         label='$I_R$ (Riemannian)')
            _draw_ellipse(ax, cx, cy, 0.75, 0.38, angle=15,
                         color=C_RIT, lw=2.0)
            # Useful samples inside I_R
            rng = np.random.default_rng(42)
            n_good = 30
            cos_a = np.cos(np.radians(15))
            sin_a = np.sin(np.radians(15))
            for _ in range(n_good):
                theta = rng.uniform(0, 2*np.pi)
                r = rng.uniform(0, 0.9)
                lx = r * 0.375 * np.cos(theta)
                ly = r * 0.19 * np.sin(theta)
                px = cx + lx * cos_a - ly * sin_a
                py = cy + lx * sin_a + ly * cos_a
                if 0.05 < px < 0.95 and 0.05 < py < 0.95:
                    ax.plot(px, py, '.', color=C_RIT, ms=2.5,
                            alpha=0.6, zorder=4)
            # Volume ratio annotation
            ax.annotate(r'Vol$(I_R)$/Vol$(I_E) = \prod_i\sqrt{\lambda_{min}/\lambda_i}$',
                        xy=(0.50, 0.08), fontsize=7, ha='center',
                        color=C_RIT, bbox=dict(fc='white', ec=C_RIT, lw=0.5, pad=2))

        _star_marker(ax, xs, ys, C_START, '$x_s$')
        _star_marker(ax, xg, yg, C_GOAL, '$x_g$')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect('equal')
        ax.set_title(title, fontsize=9, pad=4)
        ax.set_xlabel('$q_1$')
        if i == 0:
            ax.set_ylabel('$q_2$')
        ax.legend(loc='lower right', fontsize=7, framealpha=0.9)

    fig.tight_layout(w_pad=0.3)
    _save(fig, 'fig_informed_sets.pdf')


# ═══════════════════════════════════════════════════════════════════════
#  Fig 2: Metric Tensor Field Visualization
# ═══════════════════════════════════════════════════════════════════════

def fig2_metric_field():
    """Metric tensor field shown as ellipses + scalar heatmap."""
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 2.8))

    circles = [
        (np.array([0.30, 0.35]), 0.08),
        (np.array([0.30, 0.65]), 0.08),
        (np.array([0.50, 0.45]), 0.09),
        (np.array([0.50, 0.75]), 0.09),
        (np.array([0.70, 0.40]), 0.08),
        (np.array([0.70, 0.60]), 0.08),
    ]
    centres = [c for c, _ in circles]
    sigma = 0.12
    alpha_m = 8.0

    def metric_scale(x, y):
        s = 1.0
        for c in centres:
            d2 = (x - c[0])**2 + (y - c[1])**2
            s += alpha_m * np.exp(-d2 / sigma**2)
        return s

    # (a) Scalar field heatmap
    ax = axes[0]
    res = 200
    xx = np.linspace(0, 1, res)
    yy = np.linspace(0, 1, res)
    X, Y = np.meshgrid(xx, yy)
    Z = np.vectorize(metric_scale)(X, Y)

    im = ax.pcolormesh(X, Y, Z, cmap='YlOrRd', shading='gouraud', zorder=1)
    cb = fig.colorbar(im, ax=ax, shrink=0.85, pad=0.02)
    cb.set_label('$g(x) = 1 + \\alpha\\sum_i e^{-\\|x-o_i\\|^2/\\sigma^2}$', fontsize=7)
    for (c, r) in circles:
        ax.add_patch(Circle(c, r, fc='none', ec='k', lw=0.8, ls='--', zorder=5))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect('equal')
    ax.set_title('(a) Metric scalar field $g(x)$', fontsize=9)
    ax.set_xlabel('$q_1$'); ax.set_ylabel('$q_2$')

    # (b) Metric ellipsoids (unit balls of G(x))
    ax = axes[1]
    ax.set_facecolor('#F9FAFB')
    for (c, r) in circles:
        ax.add_patch(Circle(c, r, fc=C_OBS, ec='#374151', lw=0.5, alpha=0.3, zorder=2))

    grid = np.linspace(0.08, 0.92, 10)
    for gx in grid:
        for gy in grid:
            s = metric_scale(gx, gy)
            # Metric is conformal: G = s * I, so unit ball is circle of radius 1/sqrt(s)
            rad = 0.035 / np.sqrt(s)
            e = Ellipse((gx, gy), 2*rad, 2*rad, fc=C_RIT, ec=C_RIT,
                        alpha=0.3, lw=0.3, zorder=3)
            ax.add_patch(e)

    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect('equal')
    ax.set_title('(b) Unit metric balls $\\{v : v^T G(x) v \\leq 1\\}$', fontsize=9)
    ax.set_xlabel('$q_1$'); ax.set_ylabel('$q_2$')
    ax.annotate('Small ball = high cost\n(near obstacles)',
                xy=(0.50, 0.45), xytext=(0.15, 0.12), fontsize=7,
                arrowprops=dict(arrowstyle='->', color='k', lw=0.8),
                bbox=dict(fc='white', ec='k', lw=0.5, pad=2))
    ax.annotate('Large ball = low cost\n(open space)',
                xy=(0.15, 0.50), xytext=(0.60, 0.88), fontsize=7,
                arrowprops=dict(arrowstyle='->', color='k', lw=0.8),
                bbox=dict(fc='white', ec='k', lw=0.5, pad=2))

    fig.tight_layout(w_pad=0.5)
    _save(fig, 'fig_metric_field.pdf')


# ═══════════════════════════════════════════════════════════════════════
#  Fig 3: Sampling Comparison
# ═══════════════════════════════════════════════════════════════════════

def fig3_sampling():
    """Euclidean uniform sampling vs Riemannian informed sampling."""
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 2.8), sharey=True)

    xs, ys = 0.15, 0.50
    xg, yg = 0.85, 0.50
    cx, cy = 0.50, 0.50

    circles = [
        ((0.35, 0.35), 0.07),
        ((0.50, 0.50), 0.08),
        ((0.65, 0.65), 0.07),
    ]
    rng = np.random.default_rng(12)

    for i, (ax, title) in enumerate(zip(axes, [
            '(a) Euclidean informed sampling',
            '(b) Riemannian informed sampling'])):
        _draw_obstacles(ax, circles)

        if i == 0:
            # Euclidean ellipse
            _draw_ellipse(ax, cx, cy, 0.82, 0.55, color=C_EUCL, lw=1.5, fill=True, alpha=0.08)
            _draw_ellipse(ax, cx, cy, 0.82, 0.55, color=C_EUCL, lw=1.5, ls='-')
            # Uniform samples in ellipse
            n_samp = 120
            for _ in range(n_samp):
                theta = rng.uniform(0, 2*np.pi)
                r = np.sqrt(rng.uniform(0, 1))
                px = cx + r * 0.41 * np.cos(theta)
                py = cy + r * 0.275 * np.sin(theta)
                if 0.02 < px < 0.98 and 0.02 < py < 0.98:
                    ax.plot(px, py, '.', color=C_EUCL, ms=2, alpha=0.5, zorder=4)
            ax.annotate('Uniform density\n(many wasted)', xy=(0.78, 0.25),
                        fontsize=7, color=C_EUCL, ha='center', style='italic')
        else:
            # Euclidean for reference
            _draw_ellipse(ax, cx, cy, 0.82, 0.55, color=C_EUCL, lw=0.8, ls='--', alpha=0.3)
            # Riemannian ellipse
            _draw_ellipse(ax, cx, cy, 0.72, 0.36, angle=10,
                         color=C_RIT, lw=1.5, fill=True, alpha=0.10)
            _draw_ellipse(ax, cx, cy, 0.72, 0.36, angle=10,
                         color=C_RIT, lw=1.5)
            # Samples concentrated in useful region
            cos_a = np.cos(np.radians(10))
            sin_a = np.sin(np.radians(10))
            n_samp = 120
            for _ in range(n_samp):
                theta = rng.uniform(0, 2*np.pi)
                r = np.sqrt(rng.uniform(0, 1))
                lx = r * 0.36 * np.cos(theta)
                ly = r * 0.18 * np.sin(theta)
                px = cx + lx * cos_a - ly * sin_a
                py = cy + lx * sin_a + ly * cos_a
                if 0.02 < px < 0.98 and 0.02 < py < 0.98:
                    ax.plot(px, py, '.', color=C_RIT, ms=2, alpha=0.6, zorder=4)
            ax.annotate('Focused density\n(metric-aware)', xy=(0.78, 0.25),
                        fontsize=7, color=C_RIT, ha='center', style='italic')

        _star_marker(ax, xs, ys, C_START, '$x_s$')
        _star_marker(ax, xg, yg, C_GOAL, '$x_g$')
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect('equal')
        ax.set_title(title, fontsize=9, pad=4)
        ax.set_xlabel('$q_1$')
        if i == 0:
            ax.set_ylabel('$q_2$')

    fig.tight_layout(w_pad=0.3)
    _save(fig, 'fig_sampling.pdf')


# ═══════════════════════════════════════════════════════════════════════
#  Fig 4: Nearest-Neighbor Selection
# ═══════════════════════════════════════════════════════════════════════

def fig4_nearest_neighbor():
    """Euclidean vs Riemannian nearest-neighbor balls."""
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 2.8), sharey=True)

    rng = np.random.default_rng(7)
    # Tree vertices
    verts = np.array([
        [0.20, 0.50], [0.35, 0.65], [0.30, 0.35],
        [0.45, 0.55], [0.55, 0.40], [0.60, 0.70],
        [0.50, 0.25], [0.70, 0.50], [0.40, 0.80],
        [0.25, 0.70], [0.65, 0.30], [0.75, 0.65],
    ])
    # Query point (new sample)
    qx, qy = 0.48, 0.52

    # Diagonal anisotropic metric: λ1=1, λ2=4 (vertical is expensive)
    lam1, lam2 = 1.0, 4.0

    for i, (ax, title) in enumerate(zip(axes, [
            '(a) Euclidean neighbors $\\|x_i - x_{new}\\| \\leq r$',
            '(b) Riemannian neighbors $d_R(x_i, x_{new}) \\leq r$'])):

        # Draw tree edges (simple MST-like)
        edges = [(0,1),(0,2),(1,3),(2,4),(3,5),(4,6),(3,7),(1,8),(0,9),(4,10),(5,11)]
        for a, b in edges:
            ax.plot([verts[a,0], verts[b,0]], [verts[a,1], verts[b,1]],
                    '-', color=C_TREE, lw=0.8, alpha=0.5, zorder=3)

        r = 0.22  # connection radius

        if i == 0:
            # Euclidean ball
            circ = Circle((qx, qy), r, fc=C_EUCL, ec=C_EUCL, alpha=0.10, lw=1.5, ls='-', zorder=2)
            ax.add_patch(circ)
            ax.add_patch(Circle((qx, qy), r, fc='none', ec=C_EUCL, lw=1.5, zorder=5))
            # Find neighbors
            for vi, v in enumerate(verts):
                d = np.sqrt((v[0]-qx)**2 + (v[1]-qy)**2)
                if d <= r:
                    ax.plot([qx, v[0]], [qy, v[1]], '--', color=C_EUCL, lw=1.0, alpha=0.7, zorder=4)
                    ax.scatter(v[0], v[1], s=30, c=C_EUCL, marker='o', edgecolors='k', lw=0.3, zorder=6)
                else:
                    ax.scatter(v[0], v[1], s=20, c=C_TREE, marker='o', edgecolors='k', lw=0.3, zorder=6)
            ax.annotate('Isotropic:\nall directions\nequal cost', xy=(0.72, 0.18),
                        fontsize=7, ha='center', color=C_EUCL, style='italic')
        else:
            # Riemannian ball (ellipsoidal due to diagonal metric)
            # d_R^2 = lam1*(dx)^2 + lam2*(dy)^2 <= r^2
            # => (dx/(r/sqrt(lam1)))^2 + (dy/(r/sqrt(lam2)))^2 <= 1
            rx_r = r / np.sqrt(lam1)
            ry_r = r / np.sqrt(lam2)
            e = Ellipse((qx, qy), 2*rx_r, 2*ry_r, fc=C_RIT, ec=C_RIT,
                        alpha=0.10, lw=1.5, zorder=2)
            ax.add_patch(e)
            ax.add_patch(Ellipse((qx, qy), 2*rx_r, 2*ry_r, fc='none', ec=C_RIT, lw=1.5, zorder=5))
            # Find Riemannian neighbors
            for vi, v in enumerate(verts):
                dr = np.sqrt(lam1*(v[0]-qx)**2 + lam2*(v[1]-qy)**2)
                if dr <= r:
                    ax.plot([qx, v[0]], [qy, v[1]], '--', color=C_RIT, lw=1.0, alpha=0.7, zorder=4)
                    ax.scatter(v[0], v[1], s=30, c=C_RIT, marker='o', edgecolors='k', lw=0.3, zorder=6)
                else:
                    ax.scatter(v[0], v[1], s=20, c=C_TREE, marker='o', edgecolors='k', lw=0.3, zorder=6)
            ax.annotate('Anisotropic:\nvertical moves\ncost more', xy=(0.72, 0.18),
                        fontsize=7, ha='center', color=C_RIT, style='italic')

        # Query point
        ax.scatter(qx, qy, s=60, c=C_PATH, marker='D', edgecolors='k',
                   lw=0.5, zorder=8, label='$x_{new}$')
        ax.set_xlim(0.05, 0.90); ax.set_ylim(0.10, 0.90); ax.set_aspect('equal')
        ax.set_title(title, fontsize=9, pad=4)
        ax.set_xlabel('$q_1$')
        if i == 0:
            ax.set_ylabel('$q_2$')
        ax.legend(loc='upper right', fontsize=7, framealpha=0.9)

    fig.tight_layout(w_pad=0.3)
    _save(fig, 'fig_nearest_neighbor.pdf')


# ═══════════════════════════════════════════════════════════════════════
#  Fig 5: Rewiring Illustration
# ═══════════════════════════════════════════════════════════════════════

def fig5_rewiring():
    """Three-panel figure: before rewire, checking, after rewire."""
    fig, axes = plt.subplots(1, 3, figsize=(6.5, 2.2), sharey=True)

    # Vertices
    nodes = {
        'A': (0.15, 0.50),  # start
        'B': (0.35, 0.70),
        'C': (0.35, 0.30),
        'D': (0.55, 0.55),  # key node
        'E': (0.75, 0.65),
        'F': (0.75, 0.35),
        'G': (0.90, 0.50),  # near goal
    }

    # Old tree edges (before rewire)
    old_edges = [('A','B'), ('A','C'), ('B','D'), ('D','E'), ('C','F'), ('E','G')]
    # Rewire: D's parent changes from B to C (cheaper via Riemannian metric)
    new_edges = [('A','B'), ('A','C'), ('C','D'), ('D','E'), ('C','F'), ('E','G')]

    titles = ['(a) Before rewiring', '(b) Cost comparison', '(c) After rewiring']

    for idx, (ax, title) in enumerate(zip(axes, titles)):
        ax.set_facecolor('#FAFAFA')

        if idx == 0:
            # Draw old tree
            for a, b in old_edges:
                xa, ya = nodes[a]
                xb, yb = nodes[b]
                lw = 2.0 if (a,b) == ('B','D') else 1.2
                color = '#EF4444' if (a,b) == ('B','D') else C_TREE
                ax.plot([xa, xb], [ya, yb], '-', color=color, lw=lw, zorder=3)
            # Highlight the expensive edge B->D
            xb, yb = nodes['B']
            xd, yd = nodes['D']
            ax.annotate('', xy=(xd, yd), xytext=(xb, yb),
                        arrowprops=dict(arrowstyle='->', color='#EF4444', lw=2.0))
            mid = ((xb+xd)/2 + 0.03, (yb+yd)/2 + 0.05)
            ax.annotate('cost = 0.42', xy=mid, fontsize=6.5, color='#EF4444',
                        ha='center')

        elif idx == 1:
            # Show both possible edges
            for a, b in old_edges:
                if (a,b) == ('B','D'):
                    continue
                xa, ya = nodes[a]
                xb, yb = nodes[b]
                ax.plot([xa, xb], [ya, yb], '-', color=C_TREE, lw=0.8, alpha=0.4, zorder=2)
            # Old edge (expensive)
            xb, yb = nodes['B']
            xd, yd = nodes['D']
            ax.plot([xb, xd], [yb, yd], '--', color='#EF4444', lw=1.5, zorder=3)
            ax.annotate('$g(B)+c_R(B,D)=0.42$', xy=((xb+xd)/2+0.04, (yb+yd)/2+0.05),
                        fontsize=6, color='#EF4444', ha='center')
            # New edge (cheaper)
            xc, yc = nodes['C']
            ax.plot([xc, xd], [yc, yd], '-', color=C_RIT, lw=2.0, zorder=4)
            ax.annotate('$g(C)+c_R(C,D)=0.35$', xy=((xc+xd)/2+0.04, (yc+yd)/2-0.08),
                        fontsize=6, color=C_RIT, ha='center',
                        bbox=dict(fc='white', ec=C_RIT, lw=0.4, pad=1))
            ax.annotate('cheaper!', xy=((xc+xd)/2+0.12, (yc+yd)/2-0.15),
                        fontsize=6.5, color=C_RIT, style='italic', weight='bold')

        else:
            # Draw new tree (after rewire)
            for a, b in new_edges:
                xa, ya = nodes[a]
                xb, yb = nodes[b]
                lw = 2.0 if (a,b) == ('C','D') else 1.2
                color = C_RIT if (a,b) == ('C','D') else C_TREE
                ax.plot([xa, xb], [ya, yb], '-', color=color, lw=lw, zorder=3)
            # Show removed edge (ghosted)
            xb, yb = nodes['B']
            xd, yd = nodes['D']
            ax.plot([xb, xd], [yb, yd], ':', color='#D1D5DB', lw=1.0, zorder=2)
            ax.annotate('removed', xy=((xb+xd)/2+0.03, (yb+yd)/2+0.05),
                        fontsize=6, color='#9CA3AF', ha='center', style='italic')
            # Highlight new edge
            xc, yc = nodes['C']
            ax.annotate('', xy=(xd, yd), xytext=(xc, yc),
                        arrowprops=dict(arrowstyle='->', color=C_RIT, lw=2.0))

        # Draw nodes
        for name, (nx, ny) in nodes.items():
            color = C_START if name == 'A' else (C_GOAL if name == 'G' else '#374151')
            marker = '*' if name in ('A', 'G') else 'o'
            size = 70 if name in ('A', 'G') else 35
            ec = 'k'
            if name == 'D':
                color = C_PATH
                size = 45
            ax.scatter(nx, ny, s=size, c=color, marker=marker, edgecolors=ec,
                       lw=0.4, zorder=8)
            offset = (0.0, 0.04)
            if name == 'C':
                offset = (0.0, -0.06)
            ax.annotate(name, xy=(nx+offset[0], ny+offset[1]), fontsize=7,
                        ha='center', weight='bold', zorder=9)

        ax.set_xlim(0.05, 0.98); ax.set_ylim(0.15, 0.85)
        ax.set_aspect('equal')
        ax.set_title(title, fontsize=9, pad=4)
        ax.set_xlabel('$q_1$')
        if idx == 0:
            ax.set_ylabel('$q_2$')

    fig.tight_layout(w_pad=0.2)
    _save(fig, 'fig_rewiring.pdf')


# ═══════════════════════════════════════════════════════════════════════
#  Fig 6: Algorithm Pipeline Overview
# ═══════════════════════════════════════════════════════════════════════

def fig6_algorithm_pipeline():
    """Four-panel figure showing one iteration of RIT*."""
    fig, axes = plt.subplots(1, 4, figsize=(6.5, 2.0))

    xs, ys = 0.12, 0.50
    xg, yg = 0.88, 0.50
    cx, cy = 0.50, 0.50

    circles = [((0.40, 0.55), 0.08), ((0.60, 0.40), 0.08)]

    rng = np.random.default_rng(99)

    # Existing tree
    tree_nodes = np.array([
        [0.12, 0.50], [0.22, 0.60], [0.22, 0.40],
        [0.32, 0.70], [0.32, 0.30],
    ])
    tree_edges = [(0,1),(0,2),(1,3),(2,4)]

    titles = [
        '(a) Sample batch\nfrom $I_R$',
        '(b) Find neighbors\nwithin $r_n^R$',
        '(c) Extend tree\n& rewire',
        '(d) Update\n$c_{best}$, prune'
    ]

    for idx, (ax, title) in enumerate(zip(axes, titles)):
        ax.set_facecolor('#FAFAFA')
        _draw_obstacles(ax, circles, alpha=0.25)

        # Draw existing tree
        for a, b in tree_edges:
            ax.plot([tree_nodes[a,0], tree_nodes[b,0]],
                    [tree_nodes[a,1], tree_nodes[b,1]],
                    '-', color=C_TREE, lw=0.8, zorder=3)
        for ni, n in enumerate(tree_nodes):
            c = C_START if ni == 0 else C_TREE
            ax.scatter(n[0], n[1], s=15, c=c, edgecolors='k', lw=0.2, zorder=5)

        if idx == 0:
            # Riemannian informed set ellipse
            _draw_ellipse(ax, cx, cy, 0.80, 0.40, angle=8,
                         color=C_RIT, lw=1.0, fill=True, alpha=0.08)
            _draw_ellipse(ax, cx, cy, 0.80, 0.40, angle=8,
                         color=C_RIT, lw=1.0)
            # New samples
            for _ in range(25):
                cos_a = np.cos(np.radians(8))
                sin_a = np.sin(np.radians(8))
                t = rng.uniform(0, 2*np.pi)
                r = np.sqrt(rng.uniform(0, 1))
                lx = r * 0.40 * np.cos(t)
                ly = r * 0.20 * np.sin(t)
                px = cx + lx * cos_a - ly * sin_a
                py = cy + lx * sin_a + ly * cos_a
                ax.plot(px, py, '+', color=C_SAMPLE, ms=3, mew=0.6, zorder=4)

        elif idx == 1:
            # Show neighbor connections for a sample
            new_pt = np.array([0.45, 0.35])
            ax.scatter(new_pt[0], new_pt[1], s=30, c=C_SAMPLE, marker='D',
                       edgecolors='k', lw=0.3, zorder=6)
            # Riemannian ball
            _draw_ellipse(ax, new_pt[0], new_pt[1], 0.24, 0.16, angle=0,
                         color=C_RIT, lw=0.8, ls='--', fill=True, alpha=0.08)
            # Connect to nearby tree nodes
            for ni, n in enumerate(tree_nodes):
                d = np.linalg.norm(n - new_pt)
                if d < 0.18:
                    ax.plot([new_pt[0], n[0]], [new_pt[1], n[1]], ':',
                            color=C_RIT, lw=0.8, zorder=4)

        elif idx == 2:
            # Extended tree with new nodes
            ext_nodes = np.array([
                [0.45, 0.35], [0.55, 0.65], [0.65, 0.55],
            ])
            ext_edges_from_tree = [(4, 0), (3, 1)]  # from tree to ext
            ext_edges = [(0, 2)]
            for a, b in ext_edges_from_tree:
                ax.plot([tree_nodes[a,0], ext_nodes[b,0]],
                        [tree_nodes[a,1], ext_nodes[b,1]],
                        '-', color=C_RIT, lw=1.2, zorder=4)
            for a, b in ext_edges:
                ax.plot([ext_nodes[a,0], ext_nodes[b,0]],
                        [ext_nodes[a,1], ext_nodes[b,1]],
                        '-', color=C_RIT, lw=1.2, zorder=4)
            for en in ext_nodes:
                ax.scatter(en[0], en[1], s=20, c=C_RIT, edgecolors='k', lw=0.2, zorder=5)
            # Rewire arrow
            ax.annotate('', xy=(0.45, 0.35), xytext=(0.22, 0.40),
                        arrowprops=dict(arrowstyle='->', color=C_REWIRE, lw=1.5, ls='--'))
            ax.annotate('rewire', xy=(0.28, 0.32), fontsize=6, color=C_REWIRE, style='italic')

        elif idx == 3:
            # Full tree reaching goal, with pruned nodes grayed
            all_nodes = np.vstack([tree_nodes, [[0.45,0.35],[0.55,0.65],[0.65,0.55],[0.78,0.50],[0.88,0.50]]])
            # Path
            path_idx = [0, 2, 4+0, 4+2, 4+3, 4+4]
            for pi in range(len(path_idx)-1):
                a, b = path_idx[pi], path_idx[pi+1]
                ax.plot([all_nodes[a,0], all_nodes[b,0]],
                        [all_nodes[a,1], all_nodes[b,1]],
                        '-', color=C_PATH, lw=2.0, zorder=6)
            # Other edges (dimmed)
            other_edges = [(0,1),(1,3)]
            for a, b in other_edges:
                ax.plot([tree_nodes[a,0], tree_nodes[b,0]],
                        [tree_nodes[a,1], tree_nodes[b,1]],
                        '-', color='#D1D5DB', lw=0.5, zorder=2)
            # Pruned nodes
            ax.scatter(tree_nodes[1,0], tree_nodes[1,1], s=15, c='#D1D5DB',
                       edgecolors='#9CA3AF', lw=0.2, zorder=3)
            ax.scatter(tree_nodes[3,0], tree_nodes[3,1], s=15, c='#D1D5DB',
                       edgecolors='#9CA3AF', lw=0.2, zorder=3)
            for n in all_nodes[5:]:
                ax.scatter(n[0], n[1], s=15, c=C_PATH, edgecolors='k', lw=0.2, zorder=7)
            ax.annotate('pruned', xy=(0.27, 0.72), fontsize=6, color='#9CA3AF', style='italic')

        _star_marker(ax, xs, ys, C_START, None, size=40)
        _star_marker(ax, xg, yg, C_GOAL, None, size=40)
        ax.set_xlim(0, 1); ax.set_ylim(0.1, 0.9); ax.set_aspect('equal')
        ax.set_title(title, fontsize=8, pad=3)
        ax.set_xticks([]); ax.set_yticks([])

    fig.tight_layout(w_pad=0.1)
    _save(fig, 'fig_algorithm_pipeline.pdf')


# ═══════════════════════════════════════════════════════════════════════
#  Fig 7: Whitened Coordinate Transform Illustration
# ═══════════════════════════════════════════════════════════════════════

def fig7_whitening():
    """Illustrate the whitened coordinate transform G^{1/2} x."""
    fig, axes = plt.subplots(1, 3, figsize=(6.5, 2.2))

    xs, ys = -0.6, 0.0
    xg, yg = 0.6, 0.0

    titles = [
        '(a) Original space\n$I_R$ is ellipsoidal',
        r'(b) Whitened space $\tilde{x}=G^{1/2}x$' + '\n$I_R$ becomes spheroidal',
        '(c) Sample in whitened,\nmap back to original'
    ]

    for idx, (ax, title) in enumerate(zip(axes, titles)):
        ax.set_facecolor('#FAFAFA')

        if idx == 0:
            # Original: Riemannian ellipse (anisotropic)
            _draw_ellipse(ax, 0, 0, 1.4, 0.5, angle=0,
                         color=C_RIT, lw=1.5, fill=True, alpha=0.12)
            _draw_ellipse(ax, 0, 0, 1.4, 0.5, angle=0, color=C_RIT, lw=1.5)
            # Euclidean for reference
            _draw_ellipse(ax, 0, 0, 1.4, 0.9, angle=0,
                         color=C_EUCL, lw=1.0, ls='--', alpha=0.5)
            ax.annotate('$I_E$', xy=(0.0, 0.48), fontsize=7, color=C_EUCL, ha='center')
            ax.annotate('$I_R$', xy=(0.0, 0.22), fontsize=7, color=C_RIT, ha='center', weight='bold')
            # Axes annotation
            ax.annotate('', xy=(0.72, 0), xytext=(0, 0),
                        arrowprops=dict(arrowstyle='->', color='gray', lw=0.8))
            ax.annotate('', xy=(0, 0.50), xytext=(0, 0),
                        arrowprops=dict(arrowstyle='->', color='gray', lw=0.8))
            ax.annotate('$q_1$ (cheap)', xy=(0.75, -0.08), fontsize=6, color='gray')
            ax.annotate('$q_2$ (costly)', xy=(0.05, 0.42), fontsize=6, color='gray')

        elif idx == 1:
            # Whitened: both become standard ellipsoid shape
            _draw_ellipse(ax, 0, 0, 1.4, 1.0, angle=0,
                         color=C_RIT, lw=1.5, fill=True, alpha=0.12)
            _draw_ellipse(ax, 0, 0, 1.4, 1.0, angle=0, color=C_RIT, lw=1.5)
            ax.annotate('$\\tilde{I}_R$\n(standard prolate\nhyperspheroid)',
                        xy=(0, 0), fontsize=7, color=C_RIT, ha='center')
            # Transform arrow
            ax.annotate('$G^{1/2}$', xy=(-0.55, 0.55), fontsize=8, weight='bold',
                        bbox=dict(fc='white', ec='k', lw=0.5, boxstyle='round,pad=0.2'))
            ax.annotate('', xy=(0, 0.48), xytext=(0, 0),
                        arrowprops=dict(arrowstyle='->', color='gray', lw=0.8))
            ax.annotate('', xy=(0.72, 0), xytext=(0, 0),
                        arrowprops=dict(arrowstyle='->', color='gray', lw=0.8))
            ax.annotate('$\\tilde{q}_1$', xy=(0.75, -0.08), fontsize=6, color='gray')
            ax.annotate('$\\tilde{q}_2$', xy=(0.05, 0.42), fontsize=6, color='gray')

        else:
            # Show samples in original space mapped back
            _draw_ellipse(ax, 0, 0, 1.4, 0.5, angle=0,
                         color=C_RIT, lw=1.5, fill=True, alpha=0.08)
            _draw_ellipse(ax, 0, 0, 1.4, 0.5, angle=0, color=C_RIT, lw=1.5)
            rng = np.random.default_rng(42)
            n = 60
            for _ in range(n):
                t = rng.uniform(0, 2*np.pi)
                r = np.sqrt(rng.uniform(0, 1))
                px = r * 0.70 * np.cos(t)
                py = r * 0.25 * np.sin(t)
                ax.plot(px, py, '.', color=C_RIT, ms=2, alpha=0.6, zorder=4)
            ax.annotate('$G^{-1/2}$', xy=(-0.55, 0.40), fontsize=8, weight='bold',
                        bbox=dict(fc='white', ec='k', lw=0.5, boxstyle='round,pad=0.2'))
            ax.annotate('Uniform in $I_R$\nvia map-back', xy=(0, -0.35), fontsize=7,
                        ha='center', color=C_RIT, style='italic')

        ax.scatter(xs, ys, s=50, c=C_START, marker='*', edgecolors='k', lw=0.3, zorder=8)
        ax.scatter(xg, yg, s=50, c=C_GOAL, marker='*', edgecolors='k', lw=0.3, zorder=8)
        ax.set_xlim(-0.85, 0.85); ax.set_ylim(-0.55, 0.60)
        ax.set_aspect('equal')
        ax.set_title(title, fontsize=8, pad=3)
        ax.set_xticks([]); ax.set_yticks([])

    fig.tight_layout(w_pad=0.3)
    _save(fig, 'fig_whitening.pdf')


# ═══════════════════════════════════════════════════════════════════════
#  Fig 8: Batch Shrinking Illustration (Informed Set Over Iterations)
# ═══════════════════════════════════════════════════════════════════════

def fig8_batch_shrinking():
    """Show how the informed set shrinks over batches as c_best decreases."""
    fig, axes = plt.subplots(1, 4, figsize=(6.5, 1.8))

    xs, ys = 0.15, 0.50
    xg, yg = 0.85, 0.50
    cx, cy = 0.50, 0.50

    circles = [((0.40, 0.55), 0.06), ((0.60, 0.40), 0.06)]

    c_bests = [np.inf, 1.8, 1.4, 1.1]
    semi_major = [0.45, 0.42, 0.36, 0.28]
    semi_minor = [0.45, 0.30, 0.22, 0.14]
    titles = ['$t=0$, $c_{best}=\\infty$', '$t=50$, $c_{best}=1.8$',
              '$t=100$, $c_{best}=1.4$', '$t=150$, $c_{best}=1.1$']

    for idx, (ax, sm, sn, t) in enumerate(zip(axes, semi_major, semi_minor, titles)):
        ax.set_facecolor('#FAFAFA')
        _draw_obstacles(ax, circles, alpha=0.25)

        if idx == 0:
            # Uniform over entire space
            ax.add_patch(mpatches.Rectangle((0,0), 1, 1, fc=C_RIT, alpha=0.05, zorder=1))
            rng = np.random.default_rng(10)
            for _ in range(40):
                ax.plot(rng.uniform(0.05, 0.95), rng.uniform(0.1, 0.9),
                        '+', color=C_SAMPLE, ms=2, mew=0.4, zorder=4)
        else:
            _draw_ellipse(ax, cx, cy, 2*sm, 2*sn, angle=12,
                         color=C_RIT, lw=1.0, fill=True, alpha=0.10)
            _draw_ellipse(ax, cx, cy, 2*sm, 2*sn, angle=12, color=C_RIT, lw=1.0)
            cos_a = np.cos(np.radians(12))
            sin_a = np.sin(np.radians(12))
            rng = np.random.default_rng(10 + idx)
            for _ in range(40):
                tt = rng.uniform(0, 2*np.pi)
                r = np.sqrt(rng.uniform(0, 1))
                lx = r * sm * np.cos(tt)
                ly = r * sn * np.sin(tt)
                px = cx + lx * cos_a - ly * sin_a
                py = cy + lx * sin_a + ly * cos_a
                ax.plot(px, py, '+', color=C_SAMPLE, ms=2, mew=0.4, zorder=4)

        _star_marker(ax, xs, ys, C_START, None, size=30)
        _star_marker(ax, xg, yg, C_GOAL, None, size=30)
        ax.set_xlim(0, 1); ax.set_ylim(0.1, 0.9); ax.set_aspect('equal')
        ax.set_title(t, fontsize=7, pad=3)
        ax.set_xticks([]); ax.set_yticks([])

    fig.suptitle('Riemannian informed set shrinks as $c_{best}$ improves', fontsize=9, y=1.02)
    fig.tight_layout(w_pad=0.1)
    _save(fig, 'fig_batch_shrinking.pdf')


# ═══════════════════════════════════════════════════════════════════════
#  Fig 9: Cascading Lazy Edge Evaluation
# ═══════════════════════════════════════════════════════════════════════

def fig9_edge_evaluation():
    """Show the 3-level cascading edge evaluation."""
    fig, ax = plt.subplots(figsize=(3.2, 2.4))
    ax.set_facecolor('#FAFAFA')

    # Two nodes connected by an edge
    x1, y1 = 0.15, 0.50
    x2, y2 = 0.85, 0.60

    # Edge line
    ax.plot([x1, x2], [y1, y2], '-', color='#374151', lw=1.5, zorder=3)
    ax.scatter([x1, x2], [y1, y2], s=50, c=[C_TREE, C_SAMPLE],
               edgecolors='k', lw=0.4, zorder=6)
    ax.annotate('$u$', xy=(x1-0.04, y1+0.04), fontsize=9, weight='bold')
    ax.annotate('$v$', xy=(x2+0.02, y2+0.04), fontsize=9, weight='bold')

    # Level markers along the edge
    levels = [
        (0.50, 'L1: midpoint\n$G(m)$', '#22C55E', 'o'),
        ([0.33, 0.50, 0.67], "L2: Simpson's\n3-point", '#F59E0B', 's'),
        (np.linspace(0.1, 0.9, 10).tolist(), 'L3: Gauss-Legendre\n10-point', '#EF4444', '^'),
    ]

    y_offsets = [0.18, 0.06, -0.06]
    for li, (ts, label, color, marker) in enumerate(levels):
        if isinstance(ts, (int, float)):
            ts = [ts]
        for t in ts:
            px = x1 + t * (x2 - x1)
            py = y1 + t * (y2 - y1) + y_offsets[li]
            ax.scatter(px, py, s=18, c=color, marker=marker, edgecolors='k',
                       lw=0.2, zorder=5)
        # Label
        labx = 0.92
        laby = y1 + 0.5*(y2-y1) + y_offsets[li]
        ax.annotate(label, xy=(labx, laby), fontsize=6, color=color,
                    va='center')

    ax.annotate('Cheap checks first;\nonly proceed if promising',
                xy=(0.50, 0.17), fontsize=6.5, ha='center', style='italic',
                color='#374151',
                bbox=dict(fc='white', ec='#D1D5DB', lw=0.5, pad=2))

    ax.set_xlim(0.0, 1.30); ax.set_ylim(0.10, 0.80)
    ax.set_aspect('equal')
    ax.set_title('Cascading lazy edge evaluation', fontsize=9, pad=4)
    ax.set_xticks([]); ax.set_yticks([])

    fig.tight_layout()
    _save(fig, 'fig_edge_evaluation.pdf')


# ═══════════════════════════════════════════════════════════════════════
#  Fig 10: Path Comparison (Euclidean vs Riemannian optimal path)
# ═══════════════════════════════════════════════════════════════════════

def fig10_path_comparison():
    """Show how Riemannian metric produces different optimal paths."""
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 2.8), sharey=True)

    xs, ys = 0.10, 0.50
    xg, yg = 0.90, 0.50

    # Obstacle field
    circles = [
        ((0.30, 0.50), 0.10),
        ((0.55, 0.65), 0.08),
        ((0.55, 0.35), 0.08),
        ((0.75, 0.55), 0.07),
    ]

    # Centres for metric field
    centres = [np.array(c) for c, _ in circles]
    sigma = 0.12
    alpha_m = 6.0

    def metric_scale(x, y):
        s = 1.0
        for c in centres:
            d2 = (x - c[0])**2 + (y - c[1])**2
            s += alpha_m * np.exp(-d2 / sigma**2)
        return s

    titles = ['(a) Euclidean cost: shortest path', '(b) Riemannian cost: safest-shortest path']

    # Euclidean optimal: straight-ish, close to obstacles
    path_eucl = np.array([
        [0.10, 0.50], [0.18, 0.55], [0.25, 0.65], [0.35, 0.70],
        [0.45, 0.72], [0.55, 0.78], [0.65, 0.73], [0.75, 0.68],
        [0.82, 0.60], [0.90, 0.50]
    ])
    # Riemannian optimal: curves away from obstacles (lower metric cost)
    path_riem = np.array([
        [0.10, 0.50], [0.18, 0.42], [0.25, 0.30], [0.35, 0.22],
        [0.45, 0.18], [0.55, 0.15], [0.65, 0.18], [0.75, 0.28],
        [0.82, 0.38], [0.90, 0.50]
    ])

    for i, (ax, title, path, path_color) in enumerate(zip(
            axes, titles,
            [path_eucl, path_riem],
            [C_EUCL, C_RIT])):

        # Metric heatmap (faint)
        res = 150
        xx = np.linspace(0, 1, res)
        yy = np.linspace(0, 1, res)
        X, Y = np.meshgrid(xx, yy)
        Z = np.vectorize(metric_scale)(X, Y)
        ax.pcolormesh(X, Y, Z, cmap='YlOrRd', shading='gouraud',
                      alpha=0.25, zorder=1)

        _draw_obstacles(ax, circles, alpha=0.4)

        # Draw path
        ax.plot(path[:, 0], path[:, 1], '-', color=path_color, lw=2.5, zorder=7,
                label='Optimal path')
        ax.plot(path[:, 0], path[:, 1], '.', color=path_color, ms=4, zorder=8)

        # Reference: show other path as dashed
        other_path = path_riem if i == 0 else path_eucl
        other_color = C_RIT if i == 0 else C_EUCL
        ax.plot(other_path[:, 0], other_path[:, 1], '--', color=other_color,
                lw=1.0, alpha=0.5, zorder=5)

        _star_marker(ax, xs, ys, C_START, '$x_s$')
        _star_marker(ax, xg, yg, C_GOAL, '$x_g$')
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect('equal')
        ax.set_title(title, fontsize=9, pad=4)
        ax.set_xlabel('$q_1$')
        if i == 0:
            ax.set_ylabel('$q_2$')

    fig.tight_layout(w_pad=0.3)
    _save(fig, 'fig_path_comparison.pdf')


# ═══════════════════════════════════════════════════════════════════════
#  Fig 11: Connection Radius Comparison
# ═══════════════════════════════════════════════════════════════════════

def fig11_connection_radius():
    """Show metric-adapted connection radius vs Euclidean."""
    fig, ax = plt.subplots(figsize=(3.2, 2.8))

    dims = np.arange(2, 13)
    kappa = 10
    n = 1000

    # Euclidean radius: gamma * (log(n)/n)^{1/d} * (V_E/zeta_d)^{1/d}
    from scipy.special import gamma as gamma_fn
    zeta = lambda d: np.pi**(d/2) / gamma_fn(d/2 + 1)

    r_eucl = []
    r_riem = []
    for d in dims:
        base = (np.log(n) / n)**(1.0/d)
        vol_ratio = kappa**(-0.5)  # simplified for diagonal metric
        r_e = 2 * (1/d)**(1/d) * base  # normalized
        r_r = r_e * vol_ratio**(1/d)
        r_eucl.append(r_e)
        r_riem.append(r_r)

    ax.plot(dims, r_eucl, 'o-', color=C_EUCL, lw=1.5, ms=4, label='$r_n^E$ (Euclidean)')
    ax.plot(dims, r_riem, 's-', color=C_RIT, lw=1.5, ms=4, label='$r_n^R$ (Riemannian)')

    # Shade the savings
    ax.fill_between(dims, r_riem, r_eucl, alpha=0.12, color=C_RIT)
    ax.annotate('Fewer edges\nper vertex', xy=(7, (r_eucl[5]+r_riem[5])/2),
                fontsize=7, ha='center', color=C_RIT, style='italic')

    ax.set_xlabel('Dimension $d$')
    ax.set_ylabel('Connection radius $r_n$')
    ax.set_title(f'Metric-adapted radius ($\\kappa={kappa}$, $n={n}$)', fontsize=9)
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    _save(fig, 'fig_connection_radius.pdf')


# ═══════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    print('Generating conceptual paper figures...')
    print()

    fig1_informed_sets()
    print()
    fig2_metric_field()
    print()
    fig3_sampling()
    print()
    fig4_nearest_neighbor()
    print()
    fig5_rewiring()
    print()
    fig6_algorithm_pipeline()
    print()
    fig7_whitening()
    print()
    fig8_batch_shrinking()
    print()
    fig9_edge_evaluation()
    print()
    fig10_path_comparison()
    print()
    fig11_connection_radius()

    print()
    print(f'All figures saved to {OUT_DIR}/')
    print('Done!')
