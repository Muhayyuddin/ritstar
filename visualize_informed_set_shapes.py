#!/usr/bin/env python3
"""
visualize_informed_set_shapes.py — Compare BIT* ellipse vs RIT* geodesic surface

Key insight:
- BIT*: Uses a PERFECT ELLIPSE (ignores the Riemannian metric)
- RIT*: Uses a CURVED SURFACE shaped by Riemannian geodesic distances

This script ONLY visualizes the informed set shapes using a fixed cost,
without running the full planning algorithms (for speed).

Output: visualization/plots/informed_set_shapes.pdf (+ .png)
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

from rit_star.environments import env_2d_obstacle_inflated
from rit_star.geodesic import GeodesicComputer
from rit_star.informed_set import RiemannianInformedSet, EuclideanInformedSet

# ── Output directory ──
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       'visualization', 'plots')
os.makedirs(OUT_DIR, exist_ok=True)

# ── Setup environment ──
print("Setting up 2D obstacle-inflated environment...")
coll_free, edge_cost, metric, xs, xg, bounds = env_2d_obstacle_inflated()

print(f"Start: {xs}")
print(f"Goal: {xg}")

# ── Use a fixed cost for comparison ──
# Based on empirical knowledge from previous runs
c_fixed = 2.5  # Fixed cost threshold for visualization

print(f"\n=== Building informed sets at c = {c_fixed:.4f} ===")
gc = GeodesicComputer(metric, tier='diagonal', bounds=bounds)

# BIT* uses Euclidean informed set (PERFECT ELLIPSE)
eis_bit = EuclideanInformedSet(xs, xg, c_fixed, bounds=bounds)
print("  ✓ Built Euclidean informed set (BIT* - perfect ellipse)")

# RIT* uses Riemannian informed set (GEODESIC SURFACE)
ris_rit = RiemannianInformedSet(xs, xg, c_fixed, gc, bounds=bounds)
print("  ✓ Built Riemannian informed set (RIT* - geodesic surface)")

# Volume comparison
v_e = eis_bit.volume_estimate(10000)
v_r = ris_rit.volume_estimate(10000)
vol_reduction = (1.0 - v_r / v_e) * 100 if v_e > 1e-12 else 0
print(f"\n  Volume reduction: {vol_reduction:.1f}%")
print(f"  Vol(Euclidean) = {v_e:.6f}")
print(f"  Vol(Riemannian) = {v_r:.6f}")

# ── Obstacle circles ──
circles = [
    (np.array([0.30, 0.35]), 0.08),
    (np.array([0.30, 0.65]), 0.08),
    (np.array([0.50, 0.45]), 0.09),
    (np.array([0.50, 0.75]), 0.09),
    (np.array([0.70, 0.40]), 0.08),
    (np.array([0.70, 0.60]), 0.08),
]

# ── CARM metric heatmap ──
print("\n=== Computing metric field ===")
res = 200
gx = np.linspace(0, 1, res)
gy = np.linspace(0, 1, res)
GX, GY = np.meshgrid(gx, gy)
pts_grid = np.column_stack([GX.ravel(), GY.ravel()])
scale_field = np.array([metric.sqrt_det_G(p) for p in pts_grid]).reshape(res, res)
print(f"  ✓ Computed {res}x{res} metric field")

# ══════════════════════════════════════════════════════════════════
#  Figure: Side-by-side comparison
# ══════════════════════════════════════════════════════════════════
print("\n=== Generating comparison figure ===")
fig, (ax_bit, ax_rit) = plt.subplots(1, 2, figsize=(16, 8))

# ────────────────────────────────────────────────────────────────
#  LEFT PANEL: BIT* with PERFECT ELLIPSE
# ────────────────────────────────────────────────────────────────
print("  Drawing BIT* panel (Euclidean ellipse)...")

# 1) Metric heatmap (to show the ellipse IGNORES it)
im1 = ax_bit.pcolormesh(GX, GY, scale_field, cmap='YlOrRd', shading='gouraud',
                        alpha=0.35, zorder=0)

# 2) Euclidean informed set - PERFECT ELLIPSE
eis_bit.visualize_2d(ax_bit, resolution=250, color='#1976D2', alpha=0.30)

# 3) Obstacles
for c, r in circles:
    ax_bit.add_patch(Circle(c, r, fc='#546E7A', ec='#37474F',
                            lw=1.2, alpha=0.85, zorder=5))

# 4) Start / Goal
ax_bit.plot(*xs, 's', color='#2E7D32', ms=14, zorder=10,
            mec='white', mew=2.0, label='Start')
ax_bit.plot(*xg, '*', color='#C62828', ms=18, zorder=10,
            mec='white', mew=1.5, label='Goal')

# Formatting
ax_bit.set_xlim(-0.02, 1.02)
ax_bit.set_ylim(-0.02, 1.02)
ax_bit.set_aspect('equal')
ax_bit.set_xlabel('$q_1$', fontsize=15)
ax_bit.set_ylabel('$q_2$', fontsize=15)
ax_bit.set_title(
    'BIT* — Euclidean Informed Set\n'
    '(PERFECT ELLIPSE)\n'
    'Ignores the Riemannian metric field',
    fontsize=14, fontweight='bold', pad=15,
    color='#1976D2'
)

cbar1 = fig.colorbar(im1, ax=ax_bit, shrink=0.65, pad=0.02)
cbar1.set_label(r'$\sqrt{\det G(x)}$ (CARM metric)', fontsize=12)

# Grid for clarity
ax_bit.grid(True, alpha=0.2, linestyle='--', linewidth=0.5)

# ────────────────────────────────────────────────────────────────
#  RIGHT PANEL: RIT* with RIEMANNIAN GEODESIC SURFACE
# ────────────────────────────────────────────────────────────────
print("  Drawing RIT* panel (Riemannian geodesic surface)...")

# 1) Metric heatmap
im2 = ax_rit.pcolormesh(GX, GY, scale_field, cmap='YlOrRd', shading='gouraud',
                        alpha=0.35, zorder=0)

# 2) Show Euclidean baseline as dashed outline (for comparison)
eis_bit.visualize_2d(ax_rit, resolution=150, color='gray', alpha=0.15)

# 3) Riemannian informed set - CURVED SURFACE
ris_rit.visualize_2d(ax_rit, resolution=250, color='#00695C', alpha=0.30)

# 4) Obstacles
for c, r in circles:
    ax_rit.add_patch(Circle(c, r, fc='#546E7A', ec='#37474F',
                            lw=1.2, alpha=0.85, zorder=5))

# 5) Start / Goal
ax_rit.plot(*xs, 's', color='#2E7D32', ms=14, zorder=10,
            mec='white', mew=2.0, label='Start')
ax_rit.plot(*xg, '*', color='#C62828', ms=18, zorder=10,
            mec='white', mew=1.5, label='Goal')

# Formatting
ax_rit.set_xlim(-0.02, 1.02)
ax_rit.set_ylim(-0.02, 1.02)
ax_rit.set_aspect('equal')
ax_rit.set_xlabel('$q_1$', fontsize=15)
ax_rit.set_ylabel('$q_2$', fontsize=15)
ax_rit.set_title(
    'RIT* — Riemannian Informed Set\n'
    f'(CURVED GEODESIC SURFACE — {vol_reduction:.0f}% smaller volume)\n'
    'Warps with the Riemannian metric field',
    fontsize=14, fontweight='bold', pad=15,
    color='#00695C'
)

cbar2 = fig.colorbar(im2, ax=ax_rit, shrink=0.65, pad=0.02)
cbar2.set_label(r'$\sqrt{\det G(x)}$ (CARM metric)', fontsize=12)

# Grid for clarity
ax_rit.grid(True, alpha=0.2, linestyle='--', linewidth=0.5)

# ── Overall title ──
fig.suptitle(
    'BIT* vs RIT*: Perfect Ellipse vs Curved Geodesic Surface\n'
    f'Informed sets at cost threshold c = {c_fixed:.2f}',
    fontsize=16, fontweight='bold', y=0.98
)

# ── Common legend ──
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='#1976D2', alpha=0.5, edgecolor='#1976D2', linewidth=2,
          label='Euclidean ellipse (BIT*)'),
    Patch(facecolor='#00695C', alpha=0.5, edgecolor='#00695C', linewidth=2,
          label='Riemannian surface (RIT*)'),
    Patch(facecolor='gray', alpha=0.3, label='Euclidean baseline (dashed)'),
    Patch(facecolor='#546E7A', alpha=0.85, label='Obstacles'),
    Line2D([0], [0], marker='s', color='w', markerfacecolor='#2E7D32',
           ms=10, mew=1.5, markeredgecolor='white', label='Start'),
    Line2D([0], [0], marker='*', color='w', markerfacecolor='#C62828',
           ms=12, mew=1.5, markeredgecolor='white', label='Goal'),
]

# Place legend at the bottom
fig.legend(handles=legend_elements, loc='lower center', ncol=3,
           fontsize=11, framealpha=0.95, edgecolor='#424242',
           bbox_to_anchor=(0.5, -0.02), fancybox=True, shadow=True)

fig.tight_layout()
fig.subplots_adjust(top=0.92, bottom=0.10)

# ── Save ──
pdf_path = os.path.join(OUT_DIR, 'informed_set_shapes.pdf')
png_path = os.path.join(OUT_DIR, 'informed_set_shapes.png')
fig.savefig(pdf_path, dpi=300, bbox_inches='tight')
fig.savefig(png_path, dpi=200, bbox_inches='tight')
print(f"\n=== Output ===")
print(f"Saved: {pdf_path}")
print(f"Saved: {png_path}")
plt.close(fig)

print("\n" + "="*70)
print("KEY INSIGHT")
print("="*70)
print("BIT* uses a PERFECT ELLIPSE:")
print("  • Defined by Euclidean distances: ||x_s - x|| + ||x - x_g|| ≤ c")
print("  • Ignores the Riemannian metric completely")
print("  • Simple geometric shape, but includes expensive regions")
print()
print("RIT* uses a CURVED GEODESIC SURFACE:")
print("  • Defined by Riemannian distances: d_R(x_s,x) + d_R(x,x_g) ≤ c")
print("  • Warps according to the metric field")
print(f"  • {vol_reduction:.0f}% smaller volume - fewer wasted samples!")
print("  • Avoids high-cost regions near obstacles")
print("="*70)
