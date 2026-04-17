#!/usr/bin/env python3
"""
generate_paper_figures_v2.py — Improved conceptual figures for the RIT* T-RO paper.

Generates 15 publication-quality figures (11 improved + 4 new 3D):
  Fig  1: Euclidean vs Riemannian informed set comparison
  Fig  2: Riemannian metric tensor field visualization
  Fig  3: Sampling comparison (Euclidean uniform vs Riemannian informed)
  Fig  4: Nearest-neighbor selection (Euclidean ball vs Riemannian ball)
  Fig  5: Rewiring step illustration
  Fig  6: Full algorithm pipeline (batch processing overview)
  Fig  7: Whitened coordinate transform
  Fig  8: Informed set shrinking over batches
  Fig  9: Cascading lazy edge evaluation
  Fig 10: Path comparison (Euclidean vs Riemannian optimal path)
  Fig 11: Connection radius vs dimension
  Fig 12: 3D Riemannian metric surface  (NEW)
  Fig 13: 3D sampling in Riemannian informed set  (NEW)
  Fig 14: 3D nearest-neighbor selection (sphere vs ellipsoid)  (NEW)
  Fig 15: 3D tree growth through metric landscape  (NEW)

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
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.special import gamma as gamma_fn

# ── Setup ──
OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'paper', 'figures')
os.makedirs(OUT_DIR, exist_ok=True)

# IEEE-compatible styling
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif', 'serif'],
    'font.size': 9,
    'axes.labelsize': 10,
    'axes.titlesize': 10,
    'legend.fontsize': 8,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.03,
    'lines.linewidth': 1.2,
    'lines.markersize': 4,
    'text.usetex': False,
    'axes.linewidth': 0.6,
    'grid.linewidth': 0.4,
    'grid.alpha': 0.3,
    'patch.linewidth': 0.6,
})

# ── Color palette (high-contrast, colorblind-friendly) ──
C_RIT     = '#1A56DB'  # strong blue
C_EUCL    = '#DC2626'  # red
C_TREE    = '#059669'  # teal-green
C_OBS     = '#3a3a3a'  # dark gray
C_OBS_FC  = '#4a4a4a'  # dark gray fill
C_PATH    = '#F59E0B'  # amber
C_REWIRE  = '#7C3AED'  # purple
C_SAMPLE  = '#0EA5E9'  # sky blue
C_GOAL    = '#EF4444'  # bright red
C_START   = '#16A34A'  # green
C_LIGHT   = '#F3F4F6'  # very light gray
C_BG      = '#FAFBFC'  # subtle background
C_ANNOT   = '#374151'  # annotation text

# ── Standard obstacles ──
OBSTACLES = [
    ((0.30, 0.35), 0.07),
    ((0.30, 0.65), 0.07),
    ((0.50, 0.45), 0.08),
    ((0.50, 0.75), 0.08),
    ((0.70, 0.40), 0.07),
    ((0.70, 0.60), 0.07),
]
OBS_CENTRES = [np.array(c) for c, _ in OBSTACLES]
SIGMA, ALPHA_M = 0.12, 8.0


def metric_scale(x, y):
    """Return scalar metric field value at (x,y)."""
    s = 1.0
    for c in OBS_CENTRES:
        d2 = (x - c[0])**2 + (y - c[1])**2
        s += ALPHA_M * np.exp(-d2 / SIGMA**2)
    return s


def _draw_obstacles(ax, circles=None, color=C_OBS_FC, ec='#4B5563',
                    alpha=0.4, lw=0.6, hatch=None):
    """Draw circular obstacles with subtle hatching."""
    circles = circles or OBSTACLES
    for (cx, cy), r in circles:
        c = Circle((cx, cy), r, fc=color, ec=ec, lw=lw, alpha=alpha,
                   hatch=hatch, zorder=2)
        ax.add_patch(c)


def _draw_ellipse(ax, cx, cy, w, h, angle=0, color='blue', ls='-',
                  lw=1.5, label=None, fill=False, alpha=0.15):
    e = Ellipse((cx, cy), w, h, angle=angle,
                fc=color if fill else 'none',
                ec=color, ls=ls, lw=lw,
                alpha=alpha if fill else 1.0,
                label=label, zorder=3)
    ax.add_patch(e)
    return e


def _endpoint_markers(ax, xs, ys, xg, yg, s=80):
    """Draw start/goal markers with clear labels."""
    ax.scatter(xs, ys, s=s, c=C_START, marker='*', edgecolors='k',
               linewidths=0.5, zorder=10)
    ax.scatter(xg, yg, s=s, c=C_GOAL, marker='*', edgecolors='k',
               linewidths=0.5, zorder=10)
    off = 0.035
    ax.annotate('$x_s$', xy=(xs - off, ys + off), fontsize=8,
                weight='bold', color=C_START, zorder=11)
    ax.annotate('$x_g$', xy=(xg + off*0.3, yg + off), fontsize=8,
                weight='bold', color=C_GOAL, zorder=11)


def _styled_axis(ax, xlim=(0, 1), ylim=(0, 1), bg=C_BG):
    ax.set_facecolor(bg)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_aspect('equal')


def _save(fig, name):
    path = os.path.join(OUT_DIR, name)
    fig.savefig(path, dpi=300)
    plt.close(fig)
    print(f'  -> {path}')


# ═══════════════════════════════════════════════════════════════════════
#  Fig 1: Euclidean vs Riemannian Informed Set
# ═══════════════════════════════════════════════════════════════════════

def fig1_informed_sets():
    print('[Fig 1] Informed sets comparison')
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 3.0))

    xs, ys = 0.12, 0.50
    xg, yg = 0.88, 0.50
    cx, cy = 0.50, 0.50

    for i, ax in enumerate(axes):
        _draw_obstacles(ax, hatch='///')
        _styled_axis(ax)

        if i == 0:
            # Euclidean informed set: large symmetric ellipse
            _draw_ellipse(ax, cx, cy, 0.84, 0.62, angle=0,
                         color=C_EUCL, lw=2.2, fill=True, alpha=0.10,
                         label='$\\mathcal{I}_E$ (Euclidean)')
            _draw_ellipse(ax, cx, cy, 0.84, 0.62, angle=0,
                         color=C_EUCL, lw=2.2)
            # Wasted samples outside useful region
            rng = np.random.default_rng(42)
            n_waste = 50
            in_e_not_r = 0
            for _ in range(n_waste * 3):
                theta = rng.uniform(0, 2*np.pi)
                r = np.sqrt(rng.uniform(0.3, 1.0))
                px = cx + r * 0.42 * np.cos(theta)
                py = cy + r * 0.31 * np.sin(theta)
                # In Euclidean but outside Riemannian
                in_e = ((px-cx)/0.42)**2 + ((py-cy)/0.31)**2 <= 1
                cos10 = np.cos(np.radians(10))
                sin10 = np.sin(np.radians(10))
                dx = (px-cx)*cos10 + (py-cy)*sin10
                dy = -(px-cx)*sin10 + (py-cy)*cos10
                in_r = (dx/0.40)**2 + (dy/0.23)**2 <= 1
                if in_e and not in_r and in_e_not_r < n_waste:
                    ax.plot(px, py, 'x', color='#DC2626', ms=3.5, mew=0.7,
                            alpha=0.55, zorder=4)
                    in_e_not_r += 1
            # Annotation with arrow
            ax.annotate('Wasted samples\n(outside $\\mathcal{I}_R$)',
                        xy=(0.32, 0.73), xytext=(0.12, 0.88),
                        fontsize=7.5, color='#DC2626', ha='center',
                        fontweight='bold',
                        arrowprops=dict(arrowstyle='->', color='#DC2626',
                                        lw=1.0, connectionstyle='arc3,rad=0.2'),
                        bbox=dict(fc='white', ec='#DC2626', lw=0.6, pad=2,
                                  boxstyle='round,pad=0.3'))
            ax.set_title('(a) Euclidean informed set $\\mathcal{I}_E$',
                         fontsize=9.5, pad=6, fontweight='bold')

        else:
            # Euclidean for reference (dashed, dimmed)
            _draw_ellipse(ax, cx, cy, 0.84, 0.62, angle=0,
                         color=C_EUCL, ls='--', lw=1.2, alpha=0.35,
                         label='$\\mathcal{I}_E$ (ref.)')
            # Riemannian: smaller, anisotropic ellipse
            _draw_ellipse(ax, cx, cy, 0.80, 0.46, angle=10,
                         color=C_RIT, lw=2.2, fill=True, alpha=0.14,
                         label='$\\mathcal{I}_R$ (Riemannian)')
            _draw_ellipse(ax, cx, cy, 0.80, 0.46, angle=10,
                         color=C_RIT, lw=2.2)
            # Good samples inside I_R
            rng = np.random.default_rng(42)
            cos10 = np.cos(np.radians(10))
            sin10 = np.sin(np.radians(10))
            for _ in range(45):
                theta = rng.uniform(0, 2*np.pi)
                r = np.sqrt(rng.uniform(0, 0.85))
                lx = r * 0.40 * np.cos(theta)
                ly = r * 0.23 * np.sin(theta)
                px = cx + lx * cos10 - ly * sin10
                py = cy + lx * sin10 + ly * cos10
                if 0.05 < px < 0.95 and 0.05 < py < 0.95:
                    ax.plot(px, py, '.', color=C_RIT, ms=3, alpha=0.6, zorder=4)

            # Volume ratio with clearer formatting
            ax.annotate(
                '$\\frac{\\mathrm{Vol}(\\mathcal{I}_R)}{\\mathrm{Vol}(\\mathcal{I}_E)}'
                ' = \\prod_i \\sqrt{\\frac{\\lambda_{\\min}}{\\lambda_i}} \\ll 1$',
                xy=(0.50, 0.06), fontsize=7.5, ha='center', color=C_RIT,
                bbox=dict(fc='white', ec=C_RIT, lw=0.7, pad=3,
                          boxstyle='round,pad=0.4'))
            ax.set_title('(b) Riemannian informed set $\\mathcal{I}_R$',
                         fontsize=9.5, pad=6, fontweight='bold')

        _endpoint_markers(ax, xs, ys, xg, yg)
        ax.set_xlabel('$q_1$', fontsize=9)
        if i == 0:
            ax.set_ylabel('$q_2$', fontsize=9)
        ax.legend(loc='lower right', fontsize=7, framealpha=0.95,
                  edgecolor='#D1D5DB')

    fig.tight_layout(w_pad=0.5)
    _save(fig, 'fig_informed_sets.pdf')


# ═══════════════════════════════════════════════════════════════════════
#  Fig 2: Metric Tensor Field Visualization
# ═══════════════════════════════════════════════════════════════════════

def fig2_metric_field():
    print('[Fig 2] Metric tensor field')
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 3.0))

    # (a) Scalar field heatmap with contours
    ax = axes[0]
    res = 250
    xx = np.linspace(0, 1, res)
    yy = np.linspace(0, 1, res)
    X, Y = np.meshgrid(xx, yy)
    Z = np.vectorize(metric_scale)(X, Y)

    im = ax.pcolormesh(X, Y, Z, cmap='inferno', shading='gouraud', zorder=1)
    # Add contour lines for clarity
    cs = ax.contour(X, Y, Z, levels=[2.0, 4.0, 6.0, 8.0],
                    colors='white', linewidths=0.5, alpha=0.6, zorder=2)
    ax.clabel(cs, inline=True, fontsize=6, fmt='%.0f')

    cb = fig.colorbar(im, ax=ax, shrink=0.85, pad=0.03, aspect=20)
    cb.set_label('$g(x) = 1 + \\alpha\\sum_i e^{-\\|x-o_i\\|^2/\\sigma^2}$',
                 fontsize=7.5)
    cb.ax.tick_params(labelsize=7)

    for (c, r) in OBSTACLES:
        ax.add_patch(Circle(c, r, fc='none', ec='white', lw=1.0,
                            ls='--', zorder=5))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect('equal')
    ax.set_title('(a) Metric cost field $g(x)$', fontsize=9.5, pad=6,
                 fontweight='bold')
    ax.set_xlabel('$q_1$'); ax.set_ylabel('$q_2$')

    # (b) Unit metric balls (ellipsoids) on lighter background
    ax = axes[1]
    ax.set_facecolor(C_BG)
    _draw_obstacles(ax, hatch='///')

    grid = np.linspace(0.06, 0.94, 12)
    for gx in grid:
        for gy in grid:
            s = metric_scale(gx, gy)
            rad = 0.032 / np.sqrt(s)
            intensity = min(1.0, (s - 1.0) / 7.0)
            color = plt.cm.inferno(0.2 + 0.6 * intensity)
            e = Ellipse((gx, gy), 2*rad, 2*rad,
                        fc=color, ec=color, alpha=0.5, lw=0.4, zorder=3)
            ax.add_patch(e)

    # Clearer annotations with styled boxes
    ax.annotate('Small ball $\\Rightarrow$ high cost\n(near obstacles)',
                xy=(0.50, 0.45), xytext=(0.08, 0.10),
                fontsize=7.5, color=C_ANNOT,
                arrowprops=dict(arrowstyle='->', color=C_ANNOT, lw=1.0,
                                connectionstyle='arc3,rad=0.3'),
                bbox=dict(fc='#FEF3C7', ec='#D97706', lw=0.6, pad=3,
                          boxstyle='round,pad=0.3'))
    ax.annotate('Large ball $\\Rightarrow$ low cost\n(free space)',
                xy=(0.10, 0.50), xytext=(0.58, 0.90),
                fontsize=7.5, color=C_ANNOT,
                arrowprops=dict(arrowstyle='->', color=C_ANNOT, lw=1.0,
                                connectionstyle='arc3,rad=-0.3'),
                bbox=dict(fc='#DBEAFE', ec='#2563EB', lw=0.6, pad=3,
                          boxstyle='round,pad=0.3'))

    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect('equal')
    ax.set_title('(b) Unit metric balls $\\{v : v^T G(x) v \\leq 1\\}$',
                 fontsize=9.5, pad=6, fontweight='bold')
    ax.set_xlabel('$q_1$'); ax.set_ylabel('$q_2$')

    fig.tight_layout(w_pad=0.6)
    _save(fig, 'fig_metric_field.pdf')


# ═══════════════════════════════════════════════════════════════════════
#  Fig 3: Sampling Comparison
# ═══════════════════════════════════════════════════════════════════════

def fig3_sampling():
    print('[Fig 3] Sampling comparison')
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 3.0))

    xs, ys = 0.12, 0.50
    xg, yg = 0.88, 0.50
    cx, cy = 0.50, 0.50
    rng = np.random.default_rng(12)

    for i, ax in enumerate(axes):
        _draw_obstacles(ax, hatch='///')
        _styled_axis(ax)

        if i == 0:
            # Euclidean informed ellipse
            _draw_ellipse(ax, cx, cy, 0.84, 0.58, color=C_EUCL, lw=2.0,
                         fill=True, alpha=0.08, label='$\\mathcal{I}_E$')
            _draw_ellipse(ax, cx, cy, 0.84, 0.58, color=C_EUCL, lw=2.0)
            # Uniform samples
            for _ in range(150):
                theta = rng.uniform(0, 2*np.pi)
                r = np.sqrt(rng.uniform(0, 1))
                px = cx + r * 0.42 * np.cos(theta)
                py = cy + r * 0.29 * np.sin(theta)
                if 0.02 < px < 0.98 and 0.02 < py < 0.98:
                    ax.plot(px, py, '.', color=C_EUCL, ms=2.2, alpha=0.45, zorder=4)
            ax.annotate('Uniform density\n(many wasted in $\\mathcal{I}_E \\setminus \\mathcal{I}_R$)',
                        xy=(0.50, 0.07), fontsize=7, ha='center', color=C_EUCL,
                        fontstyle='italic',
                        bbox=dict(fc='white', ec=C_EUCL, lw=0.5, pad=2,
                                  boxstyle='round,pad=0.3'))
            ax.set_title('(a) Euclidean informed sampling', fontsize=9.5,
                         pad=6, fontweight='bold')
        else:
            # Euclidean for reference
            _draw_ellipse(ax, cx, cy, 0.84, 0.58, color=C_EUCL, lw=1.0,
                         ls='--', alpha=0.3, label='$\\mathcal{I}_E$ (ref.)')
            # Riemannian ellipse
            _draw_ellipse(ax, cx, cy, 0.74, 0.36, angle=12,
                         color=C_RIT, lw=2.0, fill=True, alpha=0.12,
                         label='$\\mathcal{I}_R$')
            _draw_ellipse(ax, cx, cy, 0.74, 0.36, angle=12, color=C_RIT, lw=2.0)
            cos_a = np.cos(np.radians(12))
            sin_a = np.sin(np.radians(12))
            for _ in range(150):
                theta = rng.uniform(0, 2*np.pi)
                r = np.sqrt(rng.uniform(0, 1))
                lx = r * 0.37 * np.cos(theta)
                ly = r * 0.18 * np.sin(theta)
                px = cx + lx * cos_a - ly * sin_a
                py = cy + lx * sin_a + ly * cos_a
                if 0.02 < px < 0.98 and 0.02 < py < 0.98:
                    ax.plot(px, py, '.', color=C_RIT, ms=2.2, alpha=0.55, zorder=4)
            ax.annotate('Focused density\n(all samples near optimal)',
                        xy=(0.50, 0.07), fontsize=7, ha='center', color=C_RIT,
                        fontstyle='italic',
                        bbox=dict(fc='white', ec=C_RIT, lw=0.5, pad=2,
                                  boxstyle='round,pad=0.3'))
            ax.set_title('(b) Riemannian informed sampling', fontsize=9.5,
                         pad=6, fontweight='bold')

        _endpoint_markers(ax, xs, ys, xg, yg)
        ax.set_xlabel('$q_1$')
        if i == 0:
            ax.set_ylabel('$q_2$')
        ax.legend(loc='upper right', fontsize=7, framealpha=0.95,
                  edgecolor='#D1D5DB')

    fig.tight_layout(w_pad=0.5)
    _save(fig, 'fig_sampling.pdf')


# ═══════════════════════════════════════════════════════════════════════
#  Fig 4: Nearest-Neighbor Selection
# ═══════════════════════════════════════════════════════════════════════

def fig4_nearest_neighbor():
    print('[Fig 4] Nearest-neighbor selection')
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 3.0))

    rng = np.random.default_rng(7)
    # Tree vertices (denser for better illustration)
    verts = np.array([
        [0.18, 0.48], [0.30, 0.68], [0.28, 0.32],
        [0.42, 0.58], [0.52, 0.38], [0.58, 0.72],
        [0.48, 0.22], [0.68, 0.52], [0.38, 0.82],
        [0.22, 0.72], [0.62, 0.28], [0.72, 0.68],
        [0.15, 0.35], [0.78, 0.42], [0.55, 0.55],
    ])
    # Query point
    qx, qy = 0.46, 0.50
    r_conn = 0.22

    # Anisotropic metric: λ1=1, λ2=4
    lam1, lam2 = 1.0, 4.0

    # Tree edges
    edges = [(0,1),(0,2),(1,3),(2,4),(3,5),(4,6),(3,7),(1,8),(0,9),
             (4,10),(5,11),(0,12),(7,13),(3,14)]

    for i, ax in enumerate(axes):
        _styled_axis(ax, xlim=(0.02, 0.92), ylim=(0.08, 0.92))

        # Draw tree edges
        for a, b in edges:
            ax.plot([verts[a,0], verts[b,0]], [verts[a,1], verts[b,1]],
                    '-', color=C_TREE, lw=0.7, alpha=0.4, zorder=3)

        if i == 0:
            # Euclidean ball
            circ = Circle((qx, qy), r_conn, fc=C_EUCL, ec=C_EUCL,
                          alpha=0.08, lw=2.0, zorder=2)
            ax.add_patch(circ)
            ax.add_patch(Circle((qx, qy), r_conn, fc='none', ec=C_EUCL,
                                lw=2.0, zorder=5))
            # Label the ball
            ax.annotate('$r$', xy=(qx + r_conn*0.7, qy + r_conn*0.7),
                        fontsize=8, color=C_EUCL, fontweight='bold')
            # Find neighbors
            n_inside = 0
            for vi, v in enumerate(verts):
                d = np.sqrt((v[0]-qx)**2 + (v[1]-qy)**2)
                if d <= r_conn:
                    ax.plot([qx, v[0]], [qy, v[1]], '--', color=C_EUCL,
                            lw=1.2, alpha=0.7, zorder=4)
                    ax.scatter(v[0], v[1], s=35, c=C_EUCL, marker='o',
                               edgecolors='k', lw=0.4, zorder=7)
                    n_inside += 1
                else:
                    ax.scatter(v[0], v[1], s=20, c=C_TREE, marker='o',
                               edgecolors='k', lw=0.3, zorder=6)
            ax.annotate(f'{n_inside} neighbors found',
                        xy=(0.50, 0.12), fontsize=7.5, ha='center',
                        color=C_EUCL, fontweight='bold',
                        bbox=dict(fc='white', ec=C_EUCL, lw=0.5, pad=2,
                                  boxstyle='round,pad=0.3'))
            ax.set_title('(a) Euclidean: $\\|x_i - x_{\\mathrm{new}}\\|_2 \\leq r$',
                         fontsize=9.5, pad=6, fontweight='bold')
        else:
            # Riemannian ball (ellipsoidal)
            rx_r = r_conn / np.sqrt(lam1)
            ry_r = r_conn / np.sqrt(lam2)
            e = Ellipse((qx, qy), 2*rx_r, 2*ry_r, fc=C_RIT, ec=C_RIT,
                        alpha=0.08, lw=2.0, zorder=2)
            ax.add_patch(e)
            ax.add_patch(Ellipse((qx, qy), 2*rx_r, 2*ry_r, fc='none',
                                 ec=C_RIT, lw=2.0, zorder=5))
            # Dashed Euclidean for reference
            ax.add_patch(Circle((qx, qy), r_conn, fc='none', ec=C_EUCL,
                                lw=1.0, ls='--', alpha=0.3, zorder=4))
            # Find Riemannian neighbors
            n_inside = 0
            for vi, v in enumerate(verts):
                dr = np.sqrt(lam1*(v[0]-qx)**2 + lam2*(v[1]-qy)**2)
                if dr <= r_conn:
                    ax.plot([qx, v[0]], [qy, v[1]], '--', color=C_RIT,
                            lw=1.2, alpha=0.7, zorder=4)
                    ax.scatter(v[0], v[1], s=35, c=C_RIT, marker='o',
                               edgecolors='k', lw=0.4, zorder=7)
                    n_inside += 1
                else:
                    ax.scatter(v[0], v[1], s=20, c=C_TREE, marker='o',
                               edgecolors='k', lw=0.3, zorder=6)
            ax.annotate(f'{n_inside} neighbors found',
                        xy=(0.50, 0.12), fontsize=7.5, ha='center',
                        color=C_RIT, fontweight='bold',
                        bbox=dict(fc='white', ec=C_RIT, lw=0.5, pad=2,
                                  boxstyle='round,pad=0.3'))
            ax.annotate('Anisotropic: vertical\nmoves cost $2\\times$ more',
                        xy=(0.72, 0.82), fontsize=7, ha='center', color=C_RIT,
                        fontstyle='italic',
                        bbox=dict(fc='white', ec=C_RIT, lw=0.4, pad=2,
                                  boxstyle='round,pad=0.2'))
            ax.set_title('(b) Riemannian: $d_R(x_i, x_{\\mathrm{new}}) \\leq r$',
                         fontsize=9.5, pad=6, fontweight='bold')

        # Query point
        ax.scatter(qx, qy, s=80, c=C_PATH, marker='D', edgecolors='k',
                   lw=0.6, zorder=9, label='$x_{\\mathrm{new}}$')
        ax.set_xlabel('$q_1$')
        if i == 0:
            ax.set_ylabel('$q_2$')
        ax.legend(loc='upper left', fontsize=7, framealpha=0.95,
                  edgecolor='#D1D5DB')

    fig.tight_layout(w_pad=0.5)
    _save(fig, 'fig_nearest_neighbor.pdf')


# ═══════════════════════════════════════════════════════════════════════
#  Fig 5: Rewiring Illustration
# ═══════════════════════════════════════════════════════════════════════

def fig5_rewiring():
    print('[Fig 5] Rewiring illustration')
    fig, axes = plt.subplots(1, 3, figsize=(6.5, 2.4))

    nodes = {
        'A': (0.12, 0.50),   # start
        'B': (0.32, 0.72),
        'C': (0.32, 0.28),
        'D': (0.52, 0.55),   # key node
        'E': (0.72, 0.68),
        'F': (0.72, 0.32),
        'G': (0.92, 0.50),   # goal
    }

    old_edges = [('A','B'), ('A','C'), ('B','D'), ('D','E'), ('C','F'), ('E','G')]
    new_edges = [('A','B'), ('A','C'), ('C','D'), ('D','E'), ('C','F'), ('E','G')]

    titles = ['(a) Before rewiring', '(b) Cost comparison', '(c) After rewiring']

    for idx, (ax, title) in enumerate(zip(axes, titles)):
        _styled_axis(ax, xlim=(0.02, 1.0), ylim=(0.12, 0.88))

        if idx == 0:
            for a, b in old_edges:
                xa, ya = nodes[a]; xb, yb = nodes[b]
                lw = 2.5 if (a,b) == ('B','D') else 1.2
                color = '#EF4444' if (a,b) == ('B','D') else C_TREE
                ax.plot([xa, xb], [ya, yb], '-', color=color, lw=lw,
                        zorder=3, solid_capstyle='round')
            xb, yb = nodes['B']; xd, yd = nodes['D']
            ax.annotate('', xy=(xd, yd), xytext=(xb, yb),
                        arrowprops=dict(arrowstyle='->', color='#EF4444',
                                        lw=2.5, mutation_scale=12))
            ax.annotate('$c_R = 0.42$',
                        xy=((xb+xd)/2+0.04, (yb+yd)/2+0.04),
                        fontsize=7, color='#EF4444', fontweight='bold',
                        bbox=dict(fc='#FEE2E2', ec='#EF4444', lw=0.4, pad=2,
                                  boxstyle='round,pad=0.2'))

        elif idx == 1:
            for a, b in old_edges:
                if (a,b) == ('B','D'): continue
                xa, ya = nodes[a]; xb, yb = nodes[b]
                ax.plot([xa, xb], [ya, yb], '-', color=C_TREE, lw=0.6,
                        alpha=0.3, zorder=2)
            xb, yb = nodes['B']; xd, yd = nodes['D']
            ax.plot([xb, xd], [yb, yd], '--', color='#EF4444', lw=2.0, zorder=3)
            ax.annotate('$g(B)+c_R(B{\\to}D)=0.42$',
                        xy=((xb+xd)/2+0.02, (yb+yd)/2+0.06),
                        fontsize=6.5, color='#EF4444', ha='center',
                        bbox=dict(fc='#FEE2E2', ec='#EF4444', lw=0.4, pad=1.5,
                                  boxstyle='round,pad=0.2'))
            xc, yc = nodes['C']
            ax.plot([xc, xd], [yc, yd], '-', color=C_RIT, lw=2.5, zorder=4)
            ax.annotate('$g(C)+c_R(C{\\to}D)=0.35$',
                        xy=((xc+xd)/2+0.02, (yc+yd)/2-0.08),
                        fontsize=6.5, color=C_RIT, ha='center',
                        bbox=dict(fc='#DBEAFE', ec=C_RIT, lw=0.4, pad=1.5,
                                  boxstyle='round,pad=0.2'))
            ax.annotate('$\\checkmark$ cheaper!',
                        xy=((xc+xd)/2+0.14, (yc+yd)/2-0.15),
                        fontsize=7.5, color=C_RIT, fontweight='bold',
                        fontstyle='italic')

        else:
            for a, b in new_edges:
                xa, ya = nodes[a]; xb, yb = nodes[b]
                lw = 2.5 if (a,b) == ('C','D') else 1.2
                color = C_RIT if (a,b) == ('C','D') else C_TREE
                ax.plot([xa, xb], [ya, yb], '-', color=color, lw=lw,
                        zorder=3, solid_capstyle='round')
            xc, yc = nodes['C']; xd, yd = nodes['D']
            ax.annotate('', xy=(xd, yd), xytext=(xc, yc),
                        arrowprops=dict(arrowstyle='->', color=C_RIT,
                                        lw=2.5, mutation_scale=12))
            # Ghost removed edge
            xb, yb = nodes['B']
            ax.plot([xb, xd], [yb, yd], ':', color='#D1D5DB', lw=1.2, zorder=2)
            ax.annotate('removed', xy=((xb+xd)/2+0.04, (yb+yd)/2+0.05),
                        fontsize=6.5, color='#9CA3AF', fontstyle='italic')

        # Draw nodes with clear styling
        for name, (nx, ny) in nodes.items():
            if name == 'A':
                color, marker, size = C_START, '*', 90
            elif name == 'G':
                color, marker, size = C_GOAL, '*', 90
            elif name == 'D':
                color, marker, size = C_PATH, 'D', 55
            else:
                color, marker, size = '#374151', 'o', 35
            ax.scatter(nx, ny, s=size, c=color, marker=marker,
                       edgecolors='k', lw=0.5, zorder=9)
            offy = 0.05 if name != 'C' else -0.06
            ax.annotate(name, xy=(nx, ny + offy), fontsize=8, ha='center',
                        fontweight='bold', zorder=10,
                        path_effects=[patheffects.withStroke(
                            linewidth=2, foreground='white')])

        ax.set_title(title, fontsize=9, pad=5, fontweight='bold')
        ax.set_xlabel('$q_1$')
        if idx == 0:
            ax.set_ylabel('$q_2$')

    fig.tight_layout(w_pad=0.3)
    _save(fig, 'fig_rewiring.pdf')


# ═══════════════════════════════════════════════════════════════════════
#  Fig 6: Algorithm Pipeline Overview
# ═══════════════════════════════════════════════════════════════════════

def fig6_algorithm_pipeline():
    print('[Fig 6] Algorithm pipeline')
    fig, axes = plt.subplots(1, 4, figsize=(6.5, 2.2))

    xs, ys = 0.10, 0.50
    xg, yg = 0.90, 0.50
    cx, cy = 0.50, 0.50
    circles = [((0.40, 0.58), 0.08), ((0.62, 0.38), 0.08)]
    rng = np.random.default_rng(99)

    tree_nodes = np.array([
        [0.10, 0.50], [0.22, 0.62], [0.22, 0.38],
        [0.34, 0.72], [0.34, 0.28],
    ])
    tree_edges = [(0,1),(0,2),(1,3),(2,4)]

    titles = [
        '(a) Sample batch\nfrom $\\mathcal{I}_R$',
        '(b) Find $r_n^R$\nneighbors',
        '(c) Extend &\nrewire',
        '(d) Update $c_{best}$\n& prune'
    ]

    step_colors = ['#DBEAFE', '#D1FAE5', '#EDE9FE', '#FEF3C7']

    for idx, (ax, title, bg_c) in enumerate(zip(axes, titles, step_colors)):
        ax.set_facecolor(bg_c)
        _draw_obstacles(ax, circles, alpha=0.25)

        for a, b in tree_edges:
            ax.plot([tree_nodes[a,0], tree_nodes[b,0]],
                    [tree_nodes[a,1], tree_nodes[b,1]],
                    '-', color=C_TREE, lw=0.9, zorder=3)
        for ni, n in enumerate(tree_nodes):
            c = C_START if ni == 0 else C_TREE
            ax.scatter(n[0], n[1], s=18, c=c, edgecolors='k', lw=0.2, zorder=5)

        if idx == 0:
            _draw_ellipse(ax, cx, cy, 0.82, 0.40, angle=8,
                         color=C_RIT, lw=1.2, fill=True, alpha=0.10)
            _draw_ellipse(ax, cx, cy, 0.82, 0.40, angle=8, color=C_RIT, lw=1.2)
            cos_a = np.cos(np.radians(8))
            sin_a = np.sin(np.radians(8))
            for _ in range(30):
                t = rng.uniform(0, 2*np.pi)
                r = np.sqrt(rng.uniform(0, 1))
                lx = r * 0.41 * np.cos(t)
                ly = r * 0.20 * np.sin(t)
                px = cx + lx * cos_a - ly * sin_a
                py = cy + lx * sin_a + ly * cos_a
                ax.plot(px, py, '+', color=C_SAMPLE, ms=3.5, mew=0.7, zorder=4)

        elif idx == 1:
            new_pt = np.array([0.44, 0.34])
            ax.scatter(new_pt[0], new_pt[1], s=35, c=C_SAMPLE, marker='D',
                       edgecolors='k', lw=0.4, zorder=7)
            _draw_ellipse(ax, new_pt[0], new_pt[1], 0.26, 0.18, angle=0,
                         color=C_RIT, lw=1.0, ls='--', fill=True, alpha=0.10)
            for ni, n in enumerate(tree_nodes):
                d = np.linalg.norm(n - new_pt)
                if d < 0.20:
                    ax.plot([new_pt[0], n[0]], [new_pt[1], n[1]], ':',
                            color=C_RIT, lw=1.0, zorder=4)

        elif idx == 2:
            ext = np.array([[0.44,0.34], [0.54,0.68], [0.66,0.56]])
            ax.plot([tree_nodes[4,0], ext[0,0]], [tree_nodes[4,1], ext[0,1]],
                    '-', color=C_RIT, lw=1.5, zorder=4)
            ax.plot([tree_nodes[3,0], ext[1,0]], [tree_nodes[3,1], ext[1,1]],
                    '-', color=C_RIT, lw=1.5, zorder=4)
            ax.plot([ext[0,0], ext[2,0]], [ext[0,1], ext[2,1]],
                    '-', color=C_RIT, lw=1.5, zorder=4)
            for en in ext:
                ax.scatter(en[0], en[1], s=22, c=C_RIT, edgecolors='k',
                           lw=0.3, zorder=6)
            ax.annotate('', xy=(0.44, 0.34), xytext=(0.22, 0.38),
                        arrowprops=dict(arrowstyle='->', color=C_REWIRE,
                                        lw=1.5, ls='--'))
            ax.annotate('rewire', xy=(0.26, 0.30), fontsize=6.5,
                        color=C_REWIRE, fontstyle='italic', fontweight='bold')

        elif idx == 3:
            all_n = np.vstack([tree_nodes, [[0.44,0.34],[0.54,0.68],
                               [0.66,0.56],[0.78,0.50],[0.90,0.50]]])
            path_idx = [0, 2, 5, 7, 8, 9]
            for pi in range(len(path_idx)-1):
                a, b = path_idx[pi], path_idx[pi+1]
                ax.plot([all_n[a,0], all_n[b,0]], [all_n[a,1], all_n[b,1]],
                        '-', color=C_PATH, lw=2.5, zorder=6,
                        solid_capstyle='round')
            for a, b in [(0,1),(1,3)]:
                ax.plot([tree_nodes[a,0], tree_nodes[b,0]],
                        [tree_nodes[a,1], tree_nodes[b,1]],
                        '-', color='#D1D5DB', lw=0.5, zorder=2)
            ax.scatter(tree_nodes[1,0], tree_nodes[1,1], s=15, c='#D1D5DB',
                       edgecolors='#9CA3AF', lw=0.2, zorder=3, marker='x')
            ax.scatter(tree_nodes[3,0], tree_nodes[3,1], s=15, c='#D1D5DB',
                       edgecolors='#9CA3AF', lw=0.2, zorder=3, marker='x')
            for n in all_n[5:]:
                ax.scatter(n[0], n[1], s=18, c=C_PATH, edgecolors='k',
                           lw=0.2, zorder=7)
            ax.annotate('pruned', xy=(0.27, 0.74), fontsize=6.5,
                        color='#9CA3AF', fontstyle='italic')

        ax.scatter(xs, ys, s=45, c=C_START, marker='*', edgecolors='k',
                   lw=0.4, zorder=10)
        ax.scatter(xg, yg, s=45, c=C_GOAL, marker='*', edgecolors='k',
                   lw=0.4, zorder=10)
        ax.set_xlim(0, 1); ax.set_ylim(0.08, 0.92); ax.set_aspect('equal')
        ax.set_title(title, fontsize=8, pad=4, fontweight='bold')
        ax.set_xticks([]); ax.set_yticks([])

    fig.tight_layout(w_pad=0.15)
    _save(fig, 'fig_algorithm_pipeline.pdf')


# ═══════════════════════════════════════════════════════════════════════
#  Fig 7: Whitened Coordinate Transform
# ═══════════════════════════════════════════════════════════════════════

def fig7_whitening():
    print('[Fig 7] Whitened coordinate transform')
    fig, axes = plt.subplots(1, 3, figsize=(6.5, 2.4))

    xs, ys = -0.60, 0.0
    xg, yg = 0.60, 0.0

    titles = [
        '(a) Original space',
        '(b) Whitened: $\\tilde{x}=G^{1/2}x$',
        '(c) Sample & map back'
    ]
    bgs = ['#DBEAFE', '#D1FAE5', '#EDE9FE']

    for idx, (ax, title, bg) in enumerate(zip(axes, titles, bgs)):
        ax.set_facecolor(bg)

        if idx == 0:
            _draw_ellipse(ax, 0, 0, 1.44, 0.92, color=C_EUCL, lw=1.2,
                         ls='--', alpha=0.4, label='$\\mathcal{I}_E$')
            _draw_ellipse(ax, 0, 0, 1.44, 0.52, angle=0, color=C_RIT,
                         lw=2.0, fill=True, alpha=0.12,
                         label='$\\mathcal{I}_R$')
            _draw_ellipse(ax, 0, 0, 1.44, 0.52, angle=0, color=C_RIT, lw=2.0)
            ax.annotate('', xy=(0.74, 0), xytext=(0, 0),
                        arrowprops=dict(arrowstyle='->', color='gray', lw=0.8))
            ax.annotate('', xy=(0, 0.50), xytext=(0, 0),
                        arrowprops=dict(arrowstyle='->', color='gray', lw=0.8))
            ax.annotate('$q_1$ (cheap)', xy=(0.48, -0.10), fontsize=6, color='gray')
            ax.annotate('$q_2$ (costly)', xy=(0.04, 0.43), fontsize=6, color='gray')
            ax.legend(loc='upper left', fontsize=6.5, framealpha=0.9)

        elif idx == 1:
            _draw_ellipse(ax, 0, 0, 1.44, 1.0, angle=0, color=C_RIT,
                         lw=2.0, fill=True, alpha=0.12,
                         label='$\\tilde{\\mathcal{I}}_R$')
            _draw_ellipse(ax, 0, 0, 1.44, 1.0, angle=0, color=C_RIT, lw=2.0)
            ax.annotate('Standard prolate\nhyperspheroid', xy=(0, 0),
                        fontsize=7, color=C_RIT, ha='center',
                        fontweight='bold')
            # Transform label
            ax.annotate('$G^{1/2}$', xy=(-0.55, 0.48), fontsize=9,
                        fontweight='bold',
                        bbox=dict(fc='white', ec='k', lw=0.6, pad=2,
                                  boxstyle='round,pad=0.3'))
            ax.legend(loc='upper left', fontsize=6.5, framealpha=0.9)

        else:
            _draw_ellipse(ax, 0, 0, 1.44, 0.52, angle=0, color=C_RIT,
                         lw=2.0, fill=True, alpha=0.08)
            _draw_ellipse(ax, 0, 0, 1.44, 0.52, angle=0, color=C_RIT, lw=2.0)
            rng = np.random.default_rng(42)
            for _ in range(80):
                t = rng.uniform(0, 2*np.pi)
                r = np.sqrt(rng.uniform(0, 1))
                px = r * 0.72 * np.cos(t)
                py = r * 0.26 * np.sin(t)
                ax.plot(px, py, '.', color=C_RIT, ms=2.5, alpha=0.6, zorder=4)
            ax.annotate('$G^{-1/2}$', xy=(-0.55, 0.38), fontsize=9,
                        fontweight='bold',
                        bbox=dict(fc='white', ec='k', lw=0.6, pad=2,
                                  boxstyle='round,pad=0.3'))
            ax.annotate('Uniform sampling\nin $\\mathcal{I}_R$ via map-back',
                        xy=(0, -0.38), fontsize=7, ha='center', color=C_RIT,
                        fontstyle='italic',
                        bbox=dict(fc='white', ec=C_RIT, lw=0.4, pad=2,
                                  boxstyle='round,pad=0.2'))

        ax.scatter(xs, ys, s=55, c=C_START, marker='*', edgecolors='k',
                   lw=0.4, zorder=8)
        ax.scatter(xg, yg, s=55, c=C_GOAL, marker='*', edgecolors='k',
                   lw=0.4, zorder=8)
        ax.set_xlim(-0.88, 0.88); ax.set_ylim(-0.55, 0.60)
        ax.set_aspect('equal')
        ax.set_title(title, fontsize=8.5, pad=4, fontweight='bold')
        ax.set_xticks([]); ax.set_yticks([])

    fig.tight_layout(w_pad=0.3)
    _save(fig, 'fig_whitening.pdf')


# ═══════════════════════════════════════════════════════════════════════
#  Fig 8: Batch Shrinking
# ═══════════════════════════════════════════════════════════════════════

def fig8_batch_shrinking():
    print('[Fig 8] Batch shrinking')
    fig, axes = plt.subplots(1, 4, figsize=(6.5, 2.0))

    xs, ys = 0.12, 0.50
    xg, yg = 0.88, 0.50
    cx, cy = 0.50, 0.50
    circles = [((0.40, 0.58), 0.06), ((0.62, 0.38), 0.06)]

    configs = [
        ('$k{=}0$, $c_{best}{=}\\infty$', 0.45, 0.45, True),
        ('$k{=}50$, $c_{best}{=}1.8$', 0.42, 0.30, False),
        ('$k{=}100$, $c_{best}{=}1.4$', 0.36, 0.22, False),
        ('$k{=}150$, $c_{best}{=}1.1$', 0.28, 0.14, False),
    ]
    blues = ['#EFF6FF', '#DBEAFE', '#BFDBFE', '#93C5FD']

    for idx, (ax, (title, sm, sn, uniform), bg_c) in enumerate(
            zip(axes, configs, blues)):
        ax.set_facecolor(bg_c)
        _draw_obstacles(ax, circles, alpha=0.25)

        rng = np.random.default_rng(10 + idx)
        if uniform:
            ax.add_patch(mpatches.Rectangle((0,0), 1, 1, fc=C_RIT,
                                            alpha=0.04, zorder=1))
            for _ in range(45):
                ax.plot(rng.uniform(0.05, 0.95), rng.uniform(0.12, 0.88),
                        '+', color=C_SAMPLE, ms=2.5, mew=0.5, zorder=4)
        else:
            _draw_ellipse(ax, cx, cy, 2*sm, 2*sn, angle=12,
                         color=C_RIT, lw=1.2, fill=True, alpha=0.12)
            _draw_ellipse(ax, cx, cy, 2*sm, 2*sn, angle=12, color=C_RIT, lw=1.2)
            cos_a = np.cos(np.radians(12))
            sin_a = np.sin(np.radians(12))
            for _ in range(45):
                tt = rng.uniform(0, 2*np.pi)
                r = np.sqrt(rng.uniform(0, 1))
                lx = r * sm * np.cos(tt)
                ly = r * sn * np.sin(tt)
                px = cx + lx * cos_a - ly * sin_a
                py = cy + lx * sin_a + ly * cos_a
                ax.plot(px, py, '+', color=C_SAMPLE, ms=2.5, mew=0.5, zorder=4)

        ax.scatter(xs, ys, s=35, c=C_START, marker='*', edgecolors='k',
                   lw=0.4, zorder=10)
        ax.scatter(xg, yg, s=35, c=C_GOAL, marker='*', edgecolors='k',
                   lw=0.4, zorder=10)
        ax.set_xlim(0, 1); ax.set_ylim(0.08, 0.92); ax.set_aspect('equal')
        ax.set_title(title, fontsize=7.5, pad=3, fontweight='bold')
        ax.set_xticks([]); ax.set_yticks([])

    fig.suptitle('Riemannian informed set shrinks as $c_{best}$ improves',
                 fontsize=9.5, y=1.03, fontweight='bold')
    fig.tight_layout(w_pad=0.12)
    _save(fig, 'fig_batch_shrinking.pdf')


# ═══════════════════════════════════════════════════════════════════════
#  Fig 9: Cascading Lazy Edge Evaluation
# ═══════════════════════════════════════════════════════════════════════

def fig9_edge_evaluation():
    print('[Fig 9] Cascading edge evaluation')
    fig, ax = plt.subplots(figsize=(3.4, 2.8))
    ax.set_facecolor(C_BG)

    x1, y1 = 0.08, 0.55
    x2, y2 = 0.78, 0.65

    # Main edge
    ax.plot([x1, x2], [y1, y2], '-', color='#374151', lw=2.0, zorder=3,
            solid_capstyle='round')
    ax.scatter([x1, x2], [y1, y2], s=65, c=[C_TREE, C_SAMPLE],
               edgecolors='k', lw=0.5, zorder=7)
    ax.annotate('$u$', xy=(x1-0.05, y1+0.02), fontsize=10, fontweight='bold')
    ax.annotate('$v$', xy=(x2+0.02, y2+0.02), fontsize=10, fontweight='bold')

    # Level configurations with clear visual hierarchy
    levels = [
        ([0.50], 'Level 1: Midpoint check', '#16A34A', 'o', 9, '~ 1 eval'),
        ([0.33, 0.50, 0.67], "Level 2: Simpson's 3-pt", '#F59E0B', 's', 7, '~ 3 evals'),
        (np.linspace(0.08, 0.92, 10).tolist(), 'Level 3: Gauss-Legendre 10-pt',
         '#DC2626', '^', 5, '~ 10 evals'),
    ]

    y_offsets = [0.16, 0.04, -0.08]
    for li, (ts, label, color, marker, ms, cost) in enumerate(levels):
        for t in ts:
            px = x1 + t * (x2 - x1)
            py = y1 + t * (y2 - y1) + y_offsets[li]
            ax.scatter(px, py, s=ms**2/2, c=color, marker=marker,
                       edgecolors='k', lw=0.3, zorder=6)
        # Label with cost
        ax.annotate(f'{label}\n({cost})',
                    xy=(0.85, y1 + 0.5*(y2-y1) + y_offsets[li]),
                    fontsize=6.5, color=color, fontweight='bold',
                    va='center')

    # Decision flow arrows
    ax.annotate('pass $\\rightarrow$', xy=(0.57, y1+0.5*(y2-y1)+0.11),
                fontsize=6.5, color='#16A34A', fontweight='bold')
    ax.annotate('pass $\\rightarrow$', xy=(0.57, y1+0.5*(y2-y1)-0.01),
                fontsize=6.5, color='#F59E0B', fontweight='bold')

    ax.annotate('Cheap checks first; abort early if collision found',
                xy=(0.43, 0.15), fontsize=7, ha='center', fontstyle='italic',
                color=C_ANNOT,
                bbox=dict(fc='white', ec='#D1D5DB', lw=0.5, pad=3,
                          boxstyle='round,pad=0.3'))

    ax.set_xlim(-0.02, 1.32); ax.set_ylim(0.06, 0.84)
    ax.set_aspect('equal')
    ax.set_title('Cascading lazy edge evaluation', fontsize=9.5, pad=5,
                 fontweight='bold')
    ax.set_xticks([]); ax.set_yticks([])

    fig.tight_layout()
    _save(fig, 'fig_edge_evaluation.pdf')


# ═══════════════════════════════════════════════════════════════════════
#  Fig 10: Path Comparison
# ═══════════════════════════════════════════════════════════════════════

def fig10_path_comparison():
    print('[Fig 10] Path comparison')
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 3.0))

    xs, ys = 0.08, 0.50
    xg, yg = 0.92, 0.50

    circles = [
        ((0.30, 0.50), 0.10),
        ((0.55, 0.65), 0.08),
        ((0.55, 0.35), 0.08),
        ((0.75, 0.55), 0.07),
    ]
    centres = [np.array(c) for c, _ in circles]

    def local_metric(x, y):
        s = 1.0
        for c in centres:
            d2 = (x - c[0])**2 + (y - c[1])**2
            s += 6.0 * np.exp(-d2 / 0.12**2)
        return s

    # Paths
    path_eucl = np.array([
        [0.08, 0.50], [0.16, 0.56], [0.24, 0.66], [0.34, 0.72],
        [0.44, 0.76], [0.54, 0.80], [0.64, 0.76], [0.74, 0.70],
        [0.82, 0.62], [0.92, 0.50]
    ])
    path_riem = np.array([
        [0.08, 0.50], [0.16, 0.40], [0.24, 0.28], [0.34, 0.20],
        [0.44, 0.16], [0.55, 0.14], [0.66, 0.18], [0.76, 0.28],
        [0.84, 0.38], [0.92, 0.50]
    ])

    res = 180
    xx = np.linspace(0, 1, res)
    yy = np.linspace(0, 1, res)
    X, Y = np.meshgrid(xx, yy)
    Z = np.vectorize(local_metric)(X, Y)

    for i, (ax, title, path, path_color, other_path, other_color) in enumerate(zip(
            axes,
            ['(a) Euclidean shortest path', '(b) Riemannian safest-shortest path'],
            [path_eucl, path_riem],
            [C_EUCL, C_RIT],
            [path_riem, path_eucl],
            [C_RIT, C_EUCL])):

        ax.pcolormesh(X, Y, Z, cmap='YlOrRd', shading='gouraud',
                      alpha=0.30, zorder=1)
        _draw_obstacles(ax, circles, alpha=0.45, hatch='///')

        # Other path (ghost)
        ax.plot(other_path[:, 0], other_path[:, 1], '--',
                color=other_color, lw=1.5, alpha=0.4, zorder=5)
        # Main path
        ax.plot(path[:, 0], path[:, 1], '-', color=path_color, lw=3.0,
                zorder=7, solid_capstyle='round',
                path_effects=[patheffects.withStroke(linewidth=4.5,
                              foreground='white')])
        ax.plot(path[:, 0], path[:, 1], 'o', color=path_color, ms=4,
                zorder=8, markeredgecolor='k', markeredgewidth=0.3)

        # Cost annotation
        if i == 0:
            cost_label = 'Shorter path, but\npasses near obstacles'
            ax.annotate(cost_label, xy=(0.45, 0.76), xytext=(0.60, 0.90),
                        fontsize=7, color=C_EUCL,
                        arrowprops=dict(arrowstyle='->', color=C_EUCL, lw=0.8),
                        bbox=dict(fc='white', ec=C_EUCL, lw=0.5, pad=2,
                                  boxstyle='round,pad=0.2'))
        else:
            cost_label = 'Detours around high-cost\nregions $\\Rightarrow$ safer'
            ax.annotate(cost_label, xy=(0.44, 0.16), xytext=(0.15, 0.08),
                        fontsize=7, color=C_RIT,
                        arrowprops=dict(arrowstyle='->', color=C_RIT, lw=0.8),
                        bbox=dict(fc='white', ec=C_RIT, lw=0.5, pad=2,
                                  boxstyle='round,pad=0.2'))

        _endpoint_markers(ax, xs, ys, xg, yg)
        _styled_axis(ax)
        ax.set_title(title, fontsize=9.5, pad=6, fontweight='bold')
        ax.set_xlabel('$q_1$')
        if i == 0:
            ax.set_ylabel('$q_2$')

    fig.tight_layout(w_pad=0.5)
    _save(fig, 'fig_path_comparison.pdf')


# ═══════════════════════════════════════════════════════════════════════
#  Fig 11: Connection Radius vs Dimension
# ═══════════════════════════════════════════════════════════════════════

def fig11_connection_radius():
    print('[Fig 11] Connection radius')
    fig, ax = plt.subplots(figsize=(3.4, 2.8))
    ax.set_facecolor(C_BG)

    dims = np.arange(2, 13)
    kappa = 10
    n = 1000
    zeta = lambda d: np.pi**(d/2) / gamma_fn(d/2 + 1)

    r_eucl, r_riem = [], []
    for d in dims:
        base = (np.log(n) / n)**(1.0/d)
        vol_ratio = kappa**(-0.5)
        r_e = 2 * (1/d)**(1/d) * base
        r_r = r_e * vol_ratio**(1/d)
        r_eucl.append(r_e)
        r_riem.append(r_r)

    ax.plot(dims, r_eucl, 'o-', color=C_EUCL, lw=2.0, ms=5, mew=0.5,
            mec='k', label='$r_n^E$ (Euclidean)', zorder=5)
    ax.plot(dims, r_riem, 's-', color=C_RIT, lw=2.0, ms=5, mew=0.5,
            mec='k', label='$r_n^R$ (Riemannian)', zorder=5)
    ax.fill_between(dims, r_riem, r_eucl, alpha=0.15, color=C_RIT)
    ax.annotate('Fewer wasted\nedge evaluations',
                xy=(7, (r_eucl[5]+r_riem[5])/2), fontsize=7.5,
                ha='center', color=C_RIT, fontstyle='italic',
                fontweight='bold')

    ax.set_xlabel('Dimension $d$', fontsize=9)
    ax.set_ylabel('Connection radius $r_n$', fontsize=9)
    ax.set_title(f'Metric-adapted radius ($\\kappa={kappa}$, $n={n}$)',
                 fontsize=9.5, fontweight='bold')
    ax.legend(fontsize=7.5, framealpha=0.95, edgecolor='#D1D5DB')
    ax.grid(True, alpha=0.3, ls='--')
    ax.set_xticks(dims)

    fig.tight_layout()
    _save(fig, 'fig_connection_radius.pdf')


# ═══════════════════════════════════════════════════════════════════════
#  Fig 12: 3D Riemannian Metric Surface  (NEW)
# ═══════════════════════════════════════════════════════════════════════

def fig12_3d_riemannian_surface():
    print('[Fig 12] 3D Riemannian surface')
    fig = plt.figure(figsize=(6.5, 3.5))

    # (a) 3D surface
    ax1 = fig.add_subplot(121, projection='3d')
    res = 120
    xx = np.linspace(0, 1, res)
    yy = np.linspace(0, 1, res)
    X, Y = np.meshgrid(xx, yy)
    Z = np.vectorize(metric_scale)(X, Y)

    surf = ax1.plot_surface(X, Y, Z, cmap='inferno', alpha=0.85,
                            rstride=2, cstride=2, linewidth=0,
                            antialiased=True, zorder=2)

    # Draw obstacle cylinders on the surface
    for (cx, cy), r in OBSTACLES:
        theta = np.linspace(0, 2*np.pi, 30)
        xc = cx + r * np.cos(theta)
        yc = cy + r * np.sin(theta)
        zc = np.array([metric_scale(xc[j], yc[j]) for j in range(len(theta))])
        ax1.plot(xc, yc, zc, '-', color='white', lw=1.0, alpha=0.8, zorder=5)
        ax1.plot(xc, yc, np.zeros_like(zc), '--', color=C_OBS, lw=0.5,
                 alpha=0.4, zorder=1)

    # Mark start and goal
    z_s = metric_scale(0.12, 0.50)
    z_g = metric_scale(0.88, 0.50)
    ax1.scatter([0.12], [0.50], [z_s + 0.3], s=80, c=C_START, marker='*',
                edgecolors='k', lw=0.5, zorder=10)
    ax1.scatter([0.88], [0.50], [z_g + 0.3], s=80, c=C_GOAL, marker='*',
                edgecolors='k', lw=0.5, zorder=10)
    ax1.text(0.12, 0.50, z_s + 0.8, '$x_s$', fontsize=8, color=C_START,
             fontweight='bold')
    ax1.text(0.88, 0.50, z_g + 0.8, '$x_g$', fontsize=8, color=C_GOAL,
             fontweight='bold')

    # Draw a Riemannian geodesic-like path on the surface
    path_t = np.linspace(0, 1, 50)
    path_x = 0.12 + 0.76 * path_t
    # Path curves away from obstacles
    path_y = 0.50 - 0.20 * np.sin(np.pi * path_t) * (1 - 0.5*np.sin(2*np.pi*path_t))
    path_z = np.array([metric_scale(path_x[j], path_y[j]) for j in range(len(path_t))])
    ax1.plot(path_x, path_y, path_z + 0.1, '-', color=C_PATH, lw=2.5,
             zorder=8)

    ax1.set_xlabel('$q_1$', fontsize=8, labelpad=-2)
    ax1.set_ylabel('$q_2$', fontsize=8, labelpad=-2)
    ax1.set_zlabel('$g(x)$', fontsize=8, labelpad=-2)
    ax1.set_title('(a) Metric cost landscape', fontsize=9.5, pad=2,
                  fontweight='bold')
    ax1.view_init(elev=32, azim=-55)
    ax1.tick_params(labelsize=6, pad=-2)
    fig.colorbar(surf, ax=ax1, shrink=0.55, pad=0.08, aspect=15,
                 label='$g(x)$')

    # (b) Top-down contour view with optimal path
    ax2 = fig.add_subplot(122)
    res2 = 200
    xx2 = np.linspace(0, 1, res2)
    yy2 = np.linspace(0, 1, res2)
    X2, Y2 = np.meshgrid(xx2, yy2)
    Z2 = np.vectorize(metric_scale)(X2, Y2)

    ax2.pcolormesh(X2, Y2, Z2, cmap='inferno', shading='gouraud', zorder=1)
    cs = ax2.contour(X2, Y2, Z2, levels=[2, 4, 6, 8],
                     colors='white', linewidths=0.6, alpha=0.7)
    ax2.clabel(cs, inline=True, fontsize=6, fmt='%.0f')

    _draw_obstacles(ax2, ec='white', color='none', lw=1.0)

    # Draw two paths: Euclidean and Riemannian
    path_e_x = np.linspace(0.12, 0.88, 30)
    path_e_y = 0.50 + 0.22 * np.sin(np.pi * np.linspace(0, 1, 30))
    ax2.plot(path_e_x, path_e_y, '--', color=C_EUCL, lw=2.0, alpha=0.8,
             label='Euclidean path', zorder=5)

    ax2.plot(path_x, path_y, '-', color=C_PATH, lw=2.5,
             label='Riemannian path', zorder=6)

    _endpoint_markers(ax2, 0.12, 0.50, 0.88, 0.50)
    ax2.set_xlim(0, 1); ax2.set_ylim(0, 1); ax2.set_aspect('equal')
    ax2.set_xlabel('$q_1$'); ax2.set_ylabel('$q_2$')
    ax2.set_title('(b) Top view with paths', fontsize=9.5, pad=5,
                  fontweight='bold')
    ax2.legend(loc='upper right', fontsize=7, framealpha=0.95,
               edgecolor='#D1D5DB')

    fig.tight_layout(w_pad=0.5)
    _save(fig, 'fig_3d_riemannian_surface.pdf')


# ═══════════════════════════════════════════════════════════════════════
#  Fig 13: 3D Sampling in Riemannian Informed Set  (NEW)
# ═══════════════════════════════════════════════════════════════════════

def fig13_3d_sampling():
    print('[Fig 13] 3D sampling comparison')
    fig = plt.figure(figsize=(6.5, 3.2))

    xs, xg = np.array([0.12, 0.50, 0.50]), np.array([0.88, 0.50, 0.50])
    center = (xs + xg) / 2.0
    rng = np.random.default_rng(42)

    # (a) Euclidean informed set (3D ellipsoid)
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.set_facecolor('white')

    # Draw ellipsoid wireframe (Euclidean)
    u = np.linspace(0, 2*np.pi, 40)
    v = np.linspace(0, np.pi, 25)
    a_e, b_e, c_e = 0.42, 0.28, 0.28  # semi-axes
    ex = center[0] + a_e * np.outer(np.cos(u), np.sin(v))
    ey = center[1] + b_e * np.outer(np.sin(u), np.sin(v))
    ez = center[2] + c_e * np.outer(np.ones_like(u), np.cos(v))
    ax1.plot_surface(ex, ey, ez, alpha=0.08, color=C_EUCL, zorder=1)
    ax1.plot_wireframe(ex, ey, ez, color=C_EUCL, alpha=0.15,
                       rstride=4, cstride=4, linewidth=0.4)

    # Uniform samples inside ellipsoid
    n_samp = 200
    count = 0
    while count < n_samp:
        pt = center + rng.uniform(-1, 1, 3) * np.array([a_e, b_e, c_e])
        dx = (pt - center) / np.array([a_e, b_e, c_e])
        if np.sum(dx**2) <= 1.0:
            ax1.scatter(pt[0], pt[1], pt[2], s=3, c=C_EUCL, alpha=0.4,
                        zorder=3)
            count += 1

    ax1.scatter(*xs, s=80, c=C_START, marker='*', edgecolors='k', lw=0.5,
                zorder=10)
    ax1.scatter(*xg, s=80, c=C_GOAL, marker='*', edgecolors='k', lw=0.5,
                zorder=10)
    ax1.set_xlabel('$q_1$', fontsize=7, labelpad=-3)
    ax1.set_ylabel('$q_2$', fontsize=7, labelpad=-3)
    ax1.set_zlabel('$q_3$', fontsize=7, labelpad=-3)
    ax1.set_title('(a) Euclidean $\\mathcal{I}_E$ (3D)', fontsize=9,
                  pad=1, fontweight='bold')
    ax1.view_init(elev=20, azim=-60)
    ax1.tick_params(labelsize=5, pad=-3)

    # (b) Riemannian informed set (3D: narrower ellipsoid, rotated)
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.set_facecolor('white')

    # Draw Euclidean wireframe (reference)
    ax2.plot_wireframe(ex, ey, ez, color=C_EUCL, alpha=0.08,
                       rstride=4, cstride=4, linewidth=0.3)

    # Riemannian: narrower in costly directions
    a_r, b_r, c_r = 0.40, 0.16, 0.16
    # Slight rotation to show anisotropy
    angle = np.radians(12)
    cos_a, sin_a = np.cos(angle), np.sin(angle)

    rx = center[0] + a_r * np.outer(np.cos(u), np.sin(v))
    ry_raw = b_r * np.outer(np.sin(u), np.sin(v))
    rz_raw = c_r * np.outer(np.ones_like(u), np.cos(v))
    ry = center[1] + ry_raw * cos_a - rz_raw * sin_a
    rz = center[2] + ry_raw * sin_a + rz_raw * cos_a

    ax2.plot_surface(rx, ry, rz, alpha=0.12, color=C_RIT, zorder=1)
    ax2.plot_wireframe(rx, ry, rz, color=C_RIT, alpha=0.2,
                       rstride=4, cstride=4, linewidth=0.4)

    # Focused samples inside Riemannian ellipsoid
    count = 0
    while count < n_samp:
        pt_local = rng.uniform(-1, 1, 3) * np.array([a_r, b_r, c_r])
        if np.sum((pt_local / np.array([a_r, b_r, c_r]))**2) <= 1.0:
            # Apply rotation
            py_rot = pt_local[1] * cos_a - pt_local[2] * sin_a
            pz_rot = pt_local[1] * sin_a + pt_local[2] * cos_a
            pt = center + np.array([pt_local[0], py_rot, pz_rot])
            ax2.scatter(pt[0], pt[1], pt[2], s=3, c=C_RIT, alpha=0.5,
                        zorder=3)
            count += 1

    ax2.scatter(*xs, s=80, c=C_START, marker='*', edgecolors='k', lw=0.5,
                zorder=10)
    ax2.scatter(*xg, s=80, c=C_GOAL, marker='*', edgecolors='k', lw=0.5,
                zorder=10)
    ax2.set_xlabel('$q_1$', fontsize=7, labelpad=-3)
    ax2.set_ylabel('$q_2$', fontsize=7, labelpad=-3)
    ax2.set_zlabel('$q_3$', fontsize=7, labelpad=-3)
    ax2.set_title('(b) Riemannian $\\mathcal{I}_R$ (3D)', fontsize=9,
                  pad=1, fontweight='bold')
    ax2.view_init(elev=20, azim=-60)
    ax2.tick_params(labelsize=5, pad=-3)

    # Set consistent axis limits
    for ax in [ax1, ax2]:
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.15, 0.85)
        ax.set_zlim(0.15, 0.85)

    fig.tight_layout(w_pad=0.3)
    _save(fig, 'fig_3d_sampling.pdf')


# ═══════════════════════════════════════════════════════════════════════
#  Fig 14: 3D Nearest Neighbor (Sphere vs Ellipsoid)  (NEW)
# ═══════════════════════════════════════════════════════════════════════

def fig14_3d_nearest_neighbor():
    print('[Fig 14] 3D nearest neighbor')
    fig = plt.figure(figsize=(6.5, 3.2))

    rng = np.random.default_rng(17)
    # Tree vertices in 3D
    n_verts = 25
    verts = rng.uniform(0.15, 0.85, (n_verts, 3))
    query = np.array([0.50, 0.50, 0.50])
    r = 0.25

    # Anisotropic metric: λ = [1, 3, 3]
    lam = np.array([1.0, 3.0, 3.0])

    # Wireframe sphere/ellipsoid
    u = np.linspace(0, 2*np.pi, 30)
    v = np.linspace(0, np.pi, 15)

    for i, (ax_pos, title, use_riem) in enumerate([
            (121, '(a) Euclidean ball $\\|\\cdot\\|_2 \\leq r$', False),
            (122, '(b) Riemannian ball $d_R(\\cdot) \\leq r$', True)]):

        ax = fig.add_subplot(ax_pos, projection='3d')
        ax.set_facecolor('white')

        if use_riem:
            # Ellipsoid
            radii = r / np.sqrt(lam)
            sx = query[0] + radii[0] * np.outer(np.cos(u), np.sin(v))
            sy = query[1] + radii[1] * np.outer(np.sin(u), np.sin(v))
            sz = query[2] + radii[2] * np.outer(np.ones_like(u), np.cos(v))
            color = C_RIT
            # Also show Euclidean sphere as ghost
            sx_e = query[0] + r * np.outer(np.cos(u), np.sin(v))
            sy_e = query[1] + r * np.outer(np.sin(u), np.sin(v))
            sz_e = query[2] + r * np.outer(np.ones_like(u), np.cos(v))
            ax.plot_wireframe(sx_e, sy_e, sz_e, color=C_EUCL, alpha=0.06,
                              rstride=3, cstride=3, linewidth=0.3)
        else:
            # Sphere
            sx = query[0] + r * np.outer(np.cos(u), np.sin(v))
            sy = query[1] + r * np.outer(np.sin(u), np.sin(v))
            sz = query[2] + r * np.outer(np.ones_like(u), np.cos(v))
            color = C_EUCL

        ax.plot_surface(sx, sy, sz, alpha=0.06, color=color)
        ax.plot_wireframe(sx, sy, sz, color=color, alpha=0.15,
                          rstride=3, cstride=3, linewidth=0.4)

        # Classify and draw vertices
        n_inside = 0
        for vi in range(n_verts):
            diff = verts[vi] - query
            if use_riem:
                d = np.sqrt(np.sum(lam * diff**2))
            else:
                d = np.linalg.norm(diff)

            if d <= r:
                ax.scatter(verts[vi, 0], verts[vi, 1], verts[vi, 2],
                           s=25, c=color, marker='o', edgecolors='k',
                           lw=0.3, zorder=6)
                # Connection line
                ax.plot([query[0], verts[vi, 0]],
                        [query[1], verts[vi, 1]],
                        [query[2], verts[vi, 2]],
                        '--', color=color, lw=0.6, alpha=0.5)
                n_inside += 1
            else:
                ax.scatter(verts[vi, 0], verts[vi, 1], verts[vi, 2],
                           s=12, c=C_TREE, marker='o', edgecolors='k',
                           lw=0.2, alpha=0.5, zorder=4)

        # Query point
        ax.scatter(*query, s=80, c=C_PATH, marker='D', edgecolors='k',
                   lw=0.6, zorder=10)

        ax.set_xlabel('$q_1$', fontsize=7, labelpad=-3)
        ax.set_ylabel('$q_2$', fontsize=7, labelpad=-3)
        ax.set_zlabel('$q_3$', fontsize=7, labelpad=-3)
        ax.set_title(f'{title}\n({n_inside} neighbors)',
                     fontsize=8.5, pad=1, fontweight='bold')
        ax.view_init(elev=22, azim=-50)
        ax.tick_params(labelsize=5, pad=-3)
        ax.set_xlim(0.1, 0.9)
        ax.set_ylim(0.1, 0.9)
        ax.set_zlim(0.1, 0.9)

    fig.tight_layout(w_pad=0.2)
    _save(fig, 'fig_3d_nearest_neighbor.pdf')


# ═══════════════════════════════════════════════════════════════════════
#  Fig 15: 3D Tree Growth Through Metric Landscape  (NEW)
# ═══════════════════════════════════════════════════════════════════════

def fig15_3d_tree_growth():
    print('[Fig 15] 3D tree growth')
    fig = plt.figure(figsize=(6.5, 3.5))

    # Build a tree that grows through 3D space with obstacles
    rng = np.random.default_rng(55)
    xs = np.array([0.12, 0.50, 0.50])
    xg = np.array([0.88, 0.50, 0.50])

    # Spherical obstacles in 3D
    obstacles_3d = [
        (np.array([0.35, 0.50, 0.55]), 0.10),
        (np.array([0.55, 0.45, 0.40]), 0.09),
        (np.array([0.65, 0.60, 0.55]), 0.08),
    ]

    def collision_free_3d(x):
        for c, r in obstacles_3d:
            if np.linalg.norm(x - c) < r + 0.02:
                return False
        return True

    # Build a simple RRT-like tree for visualization
    tree_pts = [xs.copy()]
    tree_parents = [-1]
    step = 0.08

    for _ in range(250):
        if rng.random() < 0.1:
            target = xg
        else:
            target = rng.uniform(0.05, 0.95, 3)

        # Find nearest
        dists = [np.linalg.norm(np.array(tree_pts[j]) - target)
                 for j in range(len(tree_pts))]
        nearest_idx = int(np.argmin(dists))
        nearest = np.array(tree_pts[nearest_idx])

        diff = target - nearest
        d = np.linalg.norm(diff)
        if d < 1e-10:
            continue
        new_pt = nearest + (diff / d) * min(step, d)
        new_pt = np.clip(new_pt, 0.05, 0.95)

        if collision_free_3d(new_pt):
            tree_pts.append(new_pt.copy())
            tree_parents.append(nearest_idx)

            if np.linalg.norm(new_pt - xg) < 0.08:
                tree_pts.append(xg.copy())
                tree_parents.append(len(tree_pts) - 2)
                break

    tree_pts = np.array(tree_pts)

    # Find path to goal
    goal_idx = len(tree_pts) - 1
    if np.linalg.norm(tree_pts[goal_idx] - xg) < 0.1:
        path_indices = []
        idx = goal_idx
        while idx >= 0:
            path_indices.append(idx)
            idx = tree_parents[idx]
        path_indices.reverse()
        has_path = True
    else:
        has_path = False
        path_indices = []

    # (a) Full tree view
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.set_facecolor('white')

    # Draw obstacles as spheres
    u = np.linspace(0, 2*np.pi, 25)
    v = np.linspace(0, np.pi, 15)
    for center, radius in obstacles_3d:
        sx = center[0] + radius * np.outer(np.cos(u), np.sin(v))
        sy = center[1] + radius * np.outer(np.sin(u), np.sin(v))
        sz = center[2] + radius * np.outer(np.ones_like(u), np.cos(v))
        ax1.plot_surface(sx, sy, sz, alpha=0.2, color=C_OBS_FC, zorder=1)
        ax1.plot_wireframe(sx, sy, sz, color=C_OBS, alpha=0.15,
                           rstride=3, cstride=3, linewidth=0.3)

    # Draw tree edges
    for j in range(1, len(tree_pts)):
        pi = tree_parents[j]
        ax1.plot([tree_pts[pi, 0], tree_pts[j, 0]],
                 [tree_pts[pi, 1], tree_pts[j, 1]],
                 [tree_pts[pi, 2], tree_pts[j, 2]],
                 '-', color=C_TREE, lw=0.4, alpha=0.4, zorder=3)

    # Draw vertices
    ax1.scatter(tree_pts[1:, 0], tree_pts[1:, 1], tree_pts[1:, 2],
                s=3, c=C_TREE, alpha=0.5, zorder=4)

    # Draw path
    if has_path:
        path_pts = tree_pts[path_indices]
        ax1.plot(path_pts[:, 0], path_pts[:, 1], path_pts[:, 2],
                 '-', color=C_PATH, lw=3.0, zorder=8)
        ax1.scatter(path_pts[:, 0], path_pts[:, 1], path_pts[:, 2],
                    s=15, c=C_PATH, edgecolors='k', lw=0.3, zorder=9)

    ax1.scatter(*xs, s=100, c=C_START, marker='*', edgecolors='k',
                lw=0.5, zorder=10)
    ax1.scatter(*xg, s=100, c=C_GOAL, marker='*', edgecolors='k',
                lw=0.5, zorder=10)

    ax1.set_xlabel('$q_1$', fontsize=7, labelpad=-3)
    ax1.set_ylabel('$q_2$', fontsize=7, labelpad=-3)
    ax1.set_zlabel('$q_3$', fontsize=7, labelpad=-3)
    ax1.set_title(f'(a) Tree growth ({len(tree_pts)} vertices)',
                  fontsize=9, pad=1, fontweight='bold')
    ax1.view_init(elev=25, azim=-50)
    ax1.tick_params(labelsize=5, pad=-3)
    ax1.set_xlim(0.0, 1.0); ax1.set_ylim(0.0, 1.0); ax1.set_zlim(0.0, 1.0)

    # (b) Informed set shrinking with tree
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.set_facecolor('white')

    # Draw obstacles
    for center, radius in obstacles_3d:
        sx = center[0] + radius * np.outer(np.cos(u), np.sin(v))
        sy = center[1] + radius * np.outer(np.sin(u), np.sin(v))
        sz = center[2] + radius * np.outer(np.ones_like(u), np.cos(v))
        ax2.plot_surface(sx, sy, sz, alpha=0.2, color=C_OBS_FC, zorder=1)

    # Draw Riemannian informed set ellipsoid
    mid = (xs + xg) / 2.0
    a_r, b_r, c_r = 0.42, 0.18, 0.18
    ex = mid[0] + a_r * np.outer(np.cos(u), np.sin(v))
    ey = mid[1] + b_r * np.outer(np.sin(u), np.sin(v))
    ez = mid[2] + c_r * np.outer(np.ones_like(u), np.cos(v))
    ax2.plot_surface(ex, ey, ez, alpha=0.06, color=C_RIT)
    ax2.plot_wireframe(ex, ey, ez, color=C_RIT, alpha=0.12,
                       rstride=3, cstride=3, linewidth=0.4)

    # Draw only the path and nearby tree
    if has_path:
        path_pts = tree_pts[path_indices]
        ax2.plot(path_pts[:, 0], path_pts[:, 1], path_pts[:, 2],
                 '-', color=C_PATH, lw=3.0, zorder=8)
        ax2.scatter(path_pts[:, 0], path_pts[:, 1], path_pts[:, 2],
                    s=15, c=C_PATH, edgecolors='k', lw=0.3, zorder=9)

    # Samples inside the informed set
    rng2 = np.random.default_rng(77)
    for _ in range(80):
        pt_local = rng2.uniform(-1, 1, 3) * np.array([a_r, b_r, c_r])
        if np.sum((pt_local / np.array([a_r, b_r, c_r]))**2) <= 1.0:
            pt = mid + pt_local
            ax2.scatter(pt[0], pt[1], pt[2], s=4, c=C_SAMPLE, alpha=0.4,
                        zorder=3, marker='+')

    ax2.scatter(*xs, s=100, c=C_START, marker='*', edgecolors='k',
                lw=0.5, zorder=10)
    ax2.scatter(*xg, s=100, c=C_GOAL, marker='*', edgecolors='k',
                lw=0.5, zorder=10)

    ax2.set_xlabel('$q_1$', fontsize=7, labelpad=-3)
    ax2.set_ylabel('$q_2$', fontsize=7, labelpad=-3)
    ax2.set_zlabel('$q_3$', fontsize=7, labelpad=-3)
    ax2.set_title('(b) Riemannian informed set $\\mathcal{I}_R$',
                  fontsize=9, pad=1, fontweight='bold')
    ax2.view_init(elev=25, azim=-50)
    ax2.tick_params(labelsize=5, pad=-3)
    ax2.set_xlim(0.0, 1.0); ax2.set_ylim(0.0, 1.0); ax2.set_zlim(0.0, 1.0)

    fig.tight_layout(w_pad=0.2)
    _save(fig, 'fig_3d_tree_growth.pdf')


# ═══════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    print('='*60)
    print('  Generating improved paper figures (v2)')
    print('='*60)
    print()

    # Improved originals
    fig1_informed_sets()
    fig2_metric_field()
    fig3_sampling()
    fig4_nearest_neighbor()
    fig5_rewiring()
    fig6_algorithm_pipeline()
    fig7_whitening()
    fig8_batch_shrinking()
    fig9_edge_evaluation()
    fig10_path_comparison()
    fig11_connection_radius()

    # New 3D figures
    print()
    print('--- New 3D figures ---')
    fig12_3d_riemannian_surface()
    fig13_3d_sampling()
    fig14_3d_nearest_neighbor()
    fig15_3d_tree_growth()

    print()
    print('='*60)
    print(f'  All 15 figures saved to {OUT_DIR}/')
    print('='*60)
