#!/usr/bin/env python3
"""
compare_before_after.py — Visual comparison of Riemannian informed sets
before and after the conformal geodesic fix.

This creates a side-by-side comparison showing how the informed set
changes with accurate vs. approximate geodesic computation.
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
from rit_star.metric_cache import MetricFieldCache

# Setup
print("Setting up comparison...")
coll_free, edge_cost, metric, xs, xg, bounds = env_2d_obstacle_inflated()

# Reasonable c_best for visualization
c_best = 2.8

print(f"Start: {xs}, Goal: {xg}")
print(f"Using c_best = {c_best}")

# Create metric cache
mc = MetricFieldCache(metric, bounds, resolution=32)

# OLD method: midpoint approximation
print("\nCreating OLD informed set (midpoint approximation)...")
gc_old = GeodesicComputer(metric, tier='diagonal', bounds=bounds)
gc_old._use_conformal = False  # Force old behavior
ris_old = RiemannianInformedSet(xs, xg, c_best, gc_old, bounds)
vol_old = ris_old.volume_estimate(10000)

# NEW method: accurate conformal integration
print("Creating NEW informed set (accurate integration)...")
gc_new = GeodesicComputer(metric, tier='diagonal', bounds=bounds, metric_cache=mc)
ris_new = RiemannianInformedSet(xs, xg, c_best, gc_new, bounds)
vol_new = ris_new.volume_estimate(10000)

# Euclidean baseline
eis = EuclideanInformedSet(xs, xg, c_best, bounds)
vol_euclid = eis.volume_estimate(10000)

print(f"\nVolume estimates:")
print(f"  Euclidean:           {vol_euclid:.6f}")
print(f"  Riemannian (OLD):    {vol_old:.6f} ({(1-vol_old/vol_euclid)*100:.1f}% reduction)")
print(f"  Riemannian (NEW):    {vol_new:.6f} ({(1-vol_new/vol_euclid)*100:.1f}% reduction)")
print(f"  Difference OLD→NEW:  {abs(vol_old-vol_new):.6f} ({abs(vol_old-vol_new)/vol_old*100:.1f}%)")

# Obstacles
circles = [
    (np.array([0.30, 0.35]), 0.08),
    (np.array([0.30, 0.65]), 0.08),
    (np.array([0.50, 0.45]), 0.09),
    (np.array([0.50, 0.75]), 0.09),
    (np.array([0.70, 0.40]), 0.08),
    (np.array([0.70, 0.60]), 0.08),
]

# Metric heatmap
res = 120
gx = np.linspace(0, 1, res)
gy = np.linspace(0, 1, res)
GX, GY = np.meshgrid(gx, gy)
pts_grid = np.column_stack([GX.ravel(), GY.ravel()])
scale_field = np.array([metric.sqrt_det_G(p) for p in pts_grid]).reshape(res, res)

# Create figure
print("\nGenerating comparison figure...")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

for ax, ris, title, color in [
    (ax1, ris_old, 'OLD: Midpoint Approximation', '#D32F2F'),
    (ax2, ris_new, 'NEW: Accurate Integration', '#388E3C')
]:
    # Metric heatmap
    im = ax.pcolormesh(GX, GY, scale_field, cmap='YlOrRd', shading='gouraud',
                       alpha=0.25, zorder=0)
    
    # Euclidean informed set (gray)
    eis.visualize_2d(ax, resolution=150, color='gray', alpha=0.12)
    
    # Riemannian informed set (colored)
    ris.visualize_2d(ax, resolution=150, color=color, alpha=0.25)
    
    # Obstacles
    for c, r in circles:
        ax.add_patch(Circle(c, r, fc='#546E7A', ec='#37474F',
                            lw=1.0, alpha=0.85, zorder=5))
    
    # Start/Goal
    ax.plot(xs[0], xs[1], 'gs', markersize=12, markeredgecolor='darkgreen',
            markeredgewidth=2, zorder=10, label='Start')
    ax.plot(xg[0], xg[1], 'r*', markersize=16, markeredgecolor='darkred',
            markeredgewidth=1.5, zorder=10, label='Goal')
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')
    ax.set_xlabel('$q_1$', fontsize=14)
    ax.set_ylabel('$q_2$', fontsize=14)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.2)

# Overall title
vol_change = abs(vol_old - vol_new) / vol_old * 100
fig.suptitle(f'Riemannian Informed Set Comparison (c_best = {c_best:.2f})\n' +
             f'Volume difference: {vol_change:.1f}% | ' +
             f'OLD: {vol_old:.4f}, NEW: {vol_new:.4f}',
             fontsize=16, fontweight='bold')

plt.tight_layout()

# Save
out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       'visualization', 'plots')
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, 'conformal_geodesic_comparison.png')
plt.savefig(out_path, dpi=150, bbox_inches='tight')
print(f"\nSaved: {out_path}")
print("Done!")
