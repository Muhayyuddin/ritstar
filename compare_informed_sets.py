#!/usr/bin/env python3
"""
compare_informed_sets.py — Side-by-side comparison of RIT* vs BIT* informed sets

Shows the key difference:
- BIT*: Uses a PERFECT ELLIPSE (Euclidean informed set)
- RIT*: Uses a CURVED SURFACE (Riemannian geodesic disk)

The Riemannian surface warps around the obstacle metric, while the
ellipse ignores it completely.

Output: visualization/plots/informed_set_comparison.pdf (+ .png)
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

from rit_star.environments import env_2d_obstacle_inflated
from rit_star.rit_star import RITStar
from rit_star.baselines import BITStar
from rit_star.geodesic import GeodesicComputer
from rit_star.informed_set import RiemannianInformedSet, EuclideanInformedSet

# ── Output directory ──
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       'visualization', 'plots')
os.makedirs(OUT_DIR, exist_ok=True)

# ── Setup environment ──
print("Setting up 2D obstacle-inflated environment...")
coll_free, edge_cost, metric, xs, xg, bounds = env_2d_obstacle_inflated()

# ── Run BOTH planners to get solutions ──
print("\n=== Running RIT* planner ===")
rit_planner = RITStar(
    x_start=xs,
    x_goal=xg,
    c_space_bounds=bounds,
    collision_checker=coll_free,
    metric=metric,
    geodesic_tier='diagonal',
    batch_size=100,
    max_iterations=300,
    random_seed=42
)
rit_path, rit_cost = rit_planner.plan()
print(f"  RIT* Path cost: {rit_cost:.4f}")
print(f"  RIT* Vertices: {len(rit_planner.vertices)}")

print("\n=== Running BIT* planner ===")
bit_planner = BITStar(
    x_start=xs,
    x_goal=xg,
    c_space_bounds=bounds,
    collision_checker=coll_free,
    metric=metric,
    batch_size=100,
    max_iterations=300,
    random_seed=42
)
bit_path, bit_cost = bit_planner.plan()
print(f"  BIT* Path cost: {bit_cost:.4f}")
print(f"  BIT* Vertices: {len(bit_planner.vertices)}")

# ── Collect tree data ──
def get_tree_data(planner):
    """Extract tree edges and vertex positions."""
    tree_edges = []
    for v in planner.vertices:
        if v.parent is not None:
            tree_edges.append((v.parent.x, v.x))
    vertex_pts = np.array([v.x for v in planner.vertices])
    return tree_edges, vertex_pts

rit_edges, rit_verts = get_tree_data(rit_planner)
bit_edges, bit_verts = get_tree_data(bit_planner)

rit_path_arr = np.array(rit_path) if rit_path else None
bit_path_arr = np.array(bit_path) if bit_path else None

# ── Build informed sets ──
print("\n=== Computing informed sets for visualization ===")
gc = GeodesicComputer(metric, tier='diagonal', bounds=bounds)

# Use slightly inflated costs for clear visualization
c_vis_rit = rit_cost * 1.95
c_vis_bit = bit_cost * 1.95

print(f"  RIT* visualizing at c = {c_vis_rit:.4f}")
print(f"  BIT* visualizing at c = {c_vis_bit:.4f}")

# RIT* uses Riemannian informed set (geodesic surface)
ris_rit = RiemannianInformedSet(xs, xg, c_vis_rit, gc, bounds=bounds)

# BIT* uses Euclidean informed set (perfect ellipse)
eis_bit = EuclideanInformedSet(xs, xg, c_vis_bit, bounds=bounds)

# Also show the Euclidean baseline for RIT* for comparison
eis_rit = EuclideanInformedSet(xs, xg, c_vis_rit, bounds=bounds)

# Volume estimates
v_eis_rit = eis_rit.volume_estimate(10000)
v_ris = ris_rit.volume_estimate(10000)
vol_reduction = (1.0 - v_ris / v_eis_rit) * 100 if v_eis_rit > 1e-12 else 0
print(f"  RIT* Riemannian surface reduces volume by {vol_reduction:.1f}%")

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
res = 150
gx = np.linspace(0, 1, res)
gy = np.linspace(0, 1, res)
GX, GY = np.meshgrid(gx, gy)
pts_grid = np.column_stack([GX.ravel(), GY.ravel()])
scale_field = np.array([metric.sqrt_det_G(p) for p in pts_grid]).reshape(res, res)

# ══════════════════════════════════════════════════════════════════
#  Figure: Side-by-side comparison
# ══════════════════════════════════════════════════════════════════
print("\n=== Generating comparison figure ===")
fig, (ax_bit, ax_rit) = plt.subplots(1, 2, figsize=(16, 8))

# ────────────────────────────────────────────────────────────────
#  LEFT PANEL: BIT* with PERFECT ELLIPSE
# ────────────────────────────────────────────────────────────────
print("  Drawing BIT* panel (Euclidean ellipse)...")

# 1) Metric heatmap
im1 = ax_bit.pcolormesh(GX, GY, scale_field, cmap='YlOrRd', shading='gouraud',
                        alpha=0.35, zorder=0)

# 2) Euclidean informed set (PERFECT ELLIPSE for BIT*)
eis_bit.visualize_2d(ax_bit, resolution=200, color='#1976D2', alpha=0.25)

# 3) Obstacles
for c, r in circles:
    ax_bit.add_patch(Circle(c, r, fc='#546E7A', ec='#37474F',
                            lw=1.0, alpha=0.85, zorder=5))

# 4) Tree edges
for (p1, p2) in bit_edges:
    ax_bit.plot([p1[0], p2[0]], [p1[1], p2[1]],
                color='#90CAF9', lw=0.4, alpha=0.5, zorder=2)

# 5) Vertex samples
ax_bit.scatter(bit_verts[:, 0], bit_verts[:, 1],
               s=3, c='#1565C0', alpha=0.6, zorder=3, edgecolors='none')

# 6) Path
if bit_path_arr is not None:
    ax_bit.plot(bit_path_arr[:, 0], bit_path_arr[:, 1], '-',
                color='#C62828', lw=2.5, alpha=0.95, zorder=8,
                solid_capstyle='round')

# 7) Start / Goal
ax_bit.plot(*xs, 's', color='#2E7D32', ms=12, zorder=10,
            mec='white', mew=1.5)
ax_bit.plot(*xg, '*', color='#C62828', ms=16, zorder=10,
            mec='white', mew=1.0)

# Formatting
ax_bit.set_xlim(-0.02, 1.02)
ax_bit.set_ylim(-0.02, 1.02)
ax_bit.set_aspect('equal')
ax_bit.set_xlabel('$q_1$', fontsize=14)
ax_bit.set_ylabel('$q_2$', fontsize=14)
ax_bit.set_title(
    f'BIT* — Euclidean Informed Set\n'
    f'(PERFECT ELLIPSE)\n'
    f'Cost = {bit_cost:.4f}  |  {len(bit_planner.vertices)} vertices',
    fontsize=13, fontweight='bold', pad=12
)

cbar1 = fig.colorbar(im1, ax=ax_bit, shrink=0.6, pad=0.02)
cbar1.set_label(r'$\sqrt{\det G(x)}$', fontsize=11)

# ────────────────────────────────────────────────────────────────
#  RIGHT PANEL: RIT* with RIEMANNIAN GEODESIC SURFACE
# ────────────────────────────────────────────────────────────────
print("  Drawing RIT* panel (Riemannian geodesic surface)...")

# 1) Metric heatmap
im2 = ax_rit.pcolormesh(GX, GY, scale_field, cmap='YlOrRd', shading='gouraud',
                        alpha=0.35, zorder=0)

# 2) Show Euclidean baseline (for comparison)
eis_rit.visualize_2d(ax_rit, resolution=150, color='gray', alpha=0.15)

# 3) Riemannian informed set (CURVED SURFACE)
ris_rit.visualize_2d(ax_rit, resolution=200, color='#00695C', alpha=0.25)

# 4) Obstacles
for c, r in circles:
    ax_rit.add_patch(Circle(c, r, fc='#546E7A', ec='#37474F',
                            lw=1.0, alpha=0.85, zorder=5))

# 5) Tree edges
for (p1, p2) in rit_edges:
    ax_rit.plot([p1[0], p2[0]], [p1[1], p2[1]],
                color='#90CAF9', lw=0.4, alpha=0.5, zorder=2)

# 6) Vertex samples
ax_rit.scatter(rit_verts[:, 0], rit_verts[:, 1],
               s=3, c='#1565C0', alpha=0.6, zorder=3, edgecolors='none')

# 7) Path
if rit_path_arr is not None:
    ax_rit.plot(rit_path_arr[:, 0], rit_path_arr[:, 1], '-',
                color='#C62828', lw=2.5, alpha=0.95, zorder=8,
                solid_capstyle='round')

# 8) Start / Goal
ax_rit.plot(*xs, 's', color='#2E7D32', ms=12, zorder=10,
            mec='white', mew=1.5)
ax_rit.plot(*xg, '*', color='#C62828', ms=16, zorder=10,
            mec='white', mew=1.0)

# Formatting
ax_rit.set_xlim(-0.02, 1.02)
ax_rit.set_ylim(-0.02, 1.02)
ax_rit.set_aspect('equal')
ax_rit.set_xlabel('$q_1$', fontsize=14)
ax_rit.set_ylabel('$q_2$', fontsize=14)
ax_rit.set_title(
    f'RIT* — Riemannian Informed Set\n'
    f'(CURVED GEODESIC SURFACE — {vol_reduction:.0f}% smaller)\n'
    f'Cost = {rit_cost:.4f}  |  {len(rit_planner.vertices)} vertices',
    fontsize=13, fontweight='bold', pad=12
)

cbar2 = fig.colorbar(im2, ax=ax_rit, shrink=0.6, pad=0.02)
cbar2.set_label(r'$\sqrt{\det G(x)}$', fontsize=11)

# ── Common legend ──
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='#1976D2', alpha=0.4, label='Euclidean ellipse'),
    Patch(facecolor='#00695C', alpha=0.4, label='Riemannian geodesic surface'),
    Patch(facecolor='gray', alpha=0.3, label='Euclidean baseline (gray)'),
    Line2D([0], [0], color='#90CAF9', lw=1.5, alpha=0.7, label='Tree edges'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='#1565C0',
           ms=5, label='Samples'),
    Line2D([0], [0], color='#C62828', lw=2.5, label='Found path'),
    Line2D([0], [0], marker='s', color='w', markerfacecolor='#2E7D32',
           ms=8, label='Start'),
    Line2D([0], [0], marker='*', color='w', markerfacecolor='#C62828',
           ms=10, label='Goal'),
]

# Place legend below both panels
fig.legend(handles=legend_elements, loc='lower center', ncol=4,
           fontsize=10, framealpha=0.95, edgecolor='#E0E0E0',
           bbox_to_anchor=(0.5, -0.05))

fig.tight_layout()
fig.subplots_adjust(bottom=0.12)

# ── Save ──
pdf_path = os.path.join(OUT_DIR, 'informed_set_comparison.pdf')
png_path = os.path.join(OUT_DIR, 'informed_set_comparison.png')
fig.savefig(pdf_path, dpi=300, bbox_inches='tight')
fig.savefig(png_path, dpi=200, bbox_inches='tight')
print(f"\n=== Output ===")
print(f"Saved: {pdf_path}")
print(f"Saved: {png_path}")
plt.close(fig)

print("\n=== KEY INSIGHT ===")
print("BIT* uses a PERFECT ELLIPSE (Euclidean distances)")
print("RIT* uses a CURVED SURFACE (Riemannian geodesic distances)")
print("The surface warps around the metric, avoiding expensive regions!")
