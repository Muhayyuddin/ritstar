#!/usr/bin/env python3
"""
run_2d_obstacle_demo.py — Run RIT* on the 2D obstacle-inflated environment
and visualize: samples, tree, Riemannian informed set, and computed path.

Output: visualization/plots/demo_2d_obstacle.pdf (+ .png)
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

from rit_star.environments import env_2d_obstacle_inflated
from rit_star.rit_star import RITStar
from rit_star.geodesic import GeodesicComputer
from rit_star.informed_set import RiemannianInformedSet, EuclideanInformedSet

# ── Output directory ──
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       'visualization', 'plots')
os.makedirs(OUT_DIR, exist_ok=True)

# ── Setup environment ──
print("Setting up 2D obstacle-inflated environment...")
coll_free, edge_cost, metric, xs, xg, bounds = env_2d_obstacle_inflated()

# ── Run RIT* planner ──
print("Running RIT* planner...")
planner = RITStar(
    x_start=xs,
    x_goal=xg,
    c_space_bounds=bounds,
    collision_checker=coll_free,
    metric=metric,
    geodesic_tier='diagonal',  # Uses accurate integration for conformal metrics
    batch_size=100,
    max_iterations=300,
    random_seed=42
)
path, cost = planner.plan()
stats = planner.get_stats()
print(f"  Path cost: {cost:.4f}")
print(f"  Vertices: {len(planner.vertices)}")
print(f"  Iterations: {stats[-1]['iteration']}")

# ── Collect data ──
# Tree edges
tree_edges = []
for v in planner.vertices:
    if v.parent is not None:
        tree_edges.append((v.parent.x, v.x))

# All vertex positions (samples in the tree)
vertex_pts = np.array([v.x for v in planner.vertices])

# Path
path_arr = np.array(path) if path else None


def _polyline_length(pts: np.ndarray) -> float:
    """Return Euclidean length of a polyline represented by Nx2 points."""
    if pts is None or len(pts) < 2:
        return float('inf')
    diffs = np.diff(pts, axis=0)
    return float(np.sum(np.linalg.norm(diffs, axis=1)))

# ── Build informed sets for visualization ──
# Use between path cost and geodesic distance for visualization
print("Computing informed sets...")
gc = GeodesicComputer(metric, tier='jacobi', bounds=bounds)

# BIT* comparison ellipse (Euclidean PHS) should use Euclidean path cost
euclid_path_cost = _polyline_length(path_arr)
c_min_euclid = float(np.linalg.norm(xg - xs))
c_bit = max(euclid_path_cost, c_min_euclid + 1e-9)
print(f"  BIT* ellipse uses Euclidean c_best = {c_bit:.4f}")
print(f"  (Riemannian path cost from planner = {cost:.4f})")

# For visualization, use c_best larger than Riemannian path cost to ensure membership and show curved boundary
c_vis = cost * 1.5
print(f"  Using c_vis = {c_vis:.4f} for informed set visualization (cost × 1.5)")

ris = RiemannianInformedSet(xs, xg, c_vis, gc, bounds=bounds)
eis = EuclideanInformedSet(xs, xg, c_vis, bounds=bounds)

# Check start and goal membership
ris_has_start = ris.is_member(xs)
ris_has_goal = ris.is_member(xg)
eis_has_start = eis.is_member(xs)
eis_has_goal = eis.is_member(xg)

if ris_has_start and ris_has_goal and eis_has_start and eis_has_goal:
    print("  ✓ Start and goal are inside both informed sets")
else:
    print(f"  Membership: RIS start={ris_has_start}, goal={ris_has_goal} | "
          f"EIS start={eis_has_start}, goal={eis_has_goal}")

# Volume comparison
v_e = eis.volume_estimate(10000)
v_r = ris.volume_estimate(10000)
vol_reduction = (1.0 - v_r / v_e) * 100 if v_e > 1e-12 else 0
print(f"  Vol(I_R)/Vol(I_E) reduction: {vol_reduction:.1f}%")

# ── Obstacle circles (same as environment) ──
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
#  Figure
# ══════════════════════════════════════════════════════════════════
print("Generating figure...")
fig, ax = plt.subplots(figsize=(8, 8))

# 1) Metric heatmap (background)
im = ax.pcolormesh(GX, GY, scale_field, cmap='YlOrRd', shading='gouraud',
                   alpha=0.35, zorder=0)
cbar = fig.colorbar(im, ax=ax, shrink=0.6, pad=0.02)
cbar.set_label(r'$\sqrt{\det G(x)}$', fontsize=10)

# 2) Euclidean informed set (gray dashed)
eis.visualize_2d(ax, resolution=150, color='gray', alpha=0.15)

# 3) Riemannian informed set (teal filled)
ris.visualize_2d(ax, resolution=150, color='#00695C', alpha=0.20)

# 3.5) BIT* ellipse (Euclidean PHS at c_best)
diff = xg - xs
dist = np.linalg.norm(diff)
center = 0.5 * (xs + xg)
a = 0.5 * c_bit
c_focal = 0.5 * dist
b = np.sqrt(max(a * a - c_focal * c_focal, 1e-12))
phi = np.arctan2(diff[1], diff[0])
th = np.linspace(0.0, 2.0 * np.pi, 400)
ellipse_local = np.vstack((a * np.cos(th), b * np.sin(th)))
rot = np.array([
    [np.cos(phi), -np.sin(phi)],
    [np.sin(phi),  np.cos(phi)],
])
ellipse_world = rot @ ellipse_local + center.reshape(2, 1)
bit_line, = ax.plot(ellipse_world[0], ellipse_world[1], '--', color='#6A1B9A', lw=1.8,
            alpha=0.95, zorder=6)

# Clip BIT* ellipse to workspace bounds so it does not spill outside the plot box
xmin, xmax = bounds[0]
ymin, ymax = bounds[1]
# Add larger margin to keep ellipse inside bounds
margin = 0.1 * (xmax - xmin)
clip_rect = plt.Rectangle((xmin + margin, ymin + margin), 
                         xmax - xmin - 2*margin, ymax - ymin - 2*margin, 
                         transform=ax.transData)
bit_line.set_clip_path(clip_rect)

# 4) Obstacles
for c, r in circles:
    ax.add_patch(Circle(c, r, fc='#546E7A', ec='#37474F',
                        lw=1.0, alpha=0.85, zorder=5))

# 5) Tree edges
for (p1, p2) in tree_edges:
    ax.plot([p1[0], p2[0]], [p1[1], p2[1]],
            color='#90CAF9', lw=0.4, alpha=0.5, zorder=2)

# 6) Vertex samples
ax.scatter(vertex_pts[:, 0], vertex_pts[:, 1],
           s=3, c='#1565C0', alpha=0.6, zorder=3, edgecolors='none',
           label=f'Samples ({len(vertex_pts)})')

# 7) Path
if path_arr is not None:
    ax.plot(path_arr[:, 0], path_arr[:, 1], '-',
            color='#C62828', lw=2.5, alpha=0.95, zorder=8,
            solid_capstyle='round', label=f'Path (cost={cost:.3f})')

# 8) Start / Goal
ax.plot(*xs, 's', color='#2E7D32', ms=12, zorder=10,
        mec='white', mew=1.5, label='Start')
ax.plot(*xg, '*', color='#C62828', ms=16, zorder=10,
        mec='white', mew=1.0, label='Goal')

# ── Formatting ──
ax.set_xlim(-0.02, 1.02)
ax.set_ylim(-0.02, 1.02)
ax.set_aspect('equal')
ax.set_xlabel('$q_1$', fontsize=12)
ax.set_ylabel('$q_2$', fontsize=12)
ax.set_title(
    f'RIT* on 2D Obstacle Environment\n'
    f'Path cost = {cost:.4f}  |  {len(planner.vertices)} vertices  |  '
    f'Informed sets at $c = {c_vis:.3f}$ (visualized): '
    f'$\\mathrm{{Vol}}(\\mathcal{{I}}_R)$ is {vol_reduction:.0f}% smaller',
    fontsize=11, fontweight='bold', pad=10
)

# Legend
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='gray', alpha=0.3, label=r'$\mathcal{I}_E$ (Euclidean)'),
    Patch(facecolor='#00695C', alpha=0.4, label=r'$\mathcal{I}_R$ (Riemannian)'),
    Line2D([0], [0], color='#6A1B9A', lw=1.8, ls='--',
           label='BIT* ellipse ($c_{best}$)'),
    Line2D([0], [0], color='#90CAF9', lw=1.5, alpha=0.7, label='Tree edges'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='#1565C0',
           ms=5, label=f'Samples ({len(vertex_pts)})'),
    Line2D([0], [0], color='#C62828', lw=2.5, label=f'Path (cost={cost:.3f})'),
    Line2D([0], [0], marker='s', color='w', markerfacecolor='#2E7D32',
           ms=8, label='Start'),
    Line2D([0], [0], marker='*', color='w', markerfacecolor='#C62828',
           ms=10, label='Goal'),
]
ax.legend(handles=legend_elements, loc='upper left', fontsize=8,
          framealpha=0.9, edgecolor='#E0E0E0')

fig.tight_layout()

# ── Save ──
pdf_path = os.path.join(OUT_DIR, 'demo_2d_obstacle.pdf')
png_path = os.path.join(OUT_DIR, 'demo_2d_obstacle.png')
fig.savefig(pdf_path, dpi=300, bbox_inches='tight')
fig.savefig(png_path, dpi=200, bbox_inches='tight')
print(f"Saved: {pdf_path}")
print(f"Saved: {png_path}")
plt.close(fig)
