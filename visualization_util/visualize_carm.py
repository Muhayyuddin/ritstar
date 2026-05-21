"""
visualize_carm.py — Visualize the Collision-Adaptive Riemannian Metric.

Generates figures showing:
  1. Learned CARM metric field (heat map of collision-driven scale factor)
  2. Collision points accumulated during planning
  3. Comparison of oracle vs CARM informed sets
  4. Tree growth under CARM with metric field evolution
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

from visualization_util.output_paths import IMAGES_DIR, PLOTS_DIR
from rit_star.rit_star import RITStar
from rit_star.metric import EuclideanMetric, CollisionAdaptiveMetric
from rit_star.environments import env_2d_obstacle_inflated, env_2d_maze


def visualize_carm_metric_field():
    """Show the learned metric field after CARM planning."""
    print('Generating CARM metric field visualization...')

    coll, _, oracle_metric, xs, xg, bounds = env_2d_obstacle_inflated()
    dim = len(xs)
    euclid = EuclideanMetric(dim)

    # Run CARM planner
    planner = RITStar(
        xs, xg, bounds, coll, euclid,
        geodesic_tier='diagonal', batch_size=80,
        max_iterations=100, random_seed=42,
        adaptive_metric=True,
        carm_sigma=0.08, carm_alpha=6.0,
        carm_rebuild_interval=10)
    path, cost = planner.plan()

    carm = planner._carm
    coll_pts = np.array(carm._collision_points) if carm._collision_points else np.zeros((0, 2))

    # Grid for metric field visualization
    res = 200
    xx = np.linspace(0, 1, res)
    yy = np.linspace(0, 1, res)
    XX, YY = np.meshgrid(xx, yy)
    pts_grid = np.column_stack([XX.ravel(), YY.ravel()])

    # Get CARM scale field
    carm_scale = carm._collision_scale_batch(pts_grid).reshape(XX.shape)

    # Get oracle scale field
    oracle_scale = np.array([oracle_metric.sqrt_det_G(p) for p in pts_grid]).reshape(XX.shape)

    # Obstacle regions (for overlay)
    circles = [
        (np.array([0.30, 0.35]), 0.08),
        (np.array([0.30, 0.65]), 0.08),
        (np.array([0.50, 0.45]), 0.09),
        (np.array([0.50, 0.75]), 0.09),
        (np.array([0.70, 0.40]), 0.08),
        (np.array([0.70, 0.60]), 0.08),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    # Panel 1: Oracle metric field
    ax = axes[0]
    im1 = ax.contourf(XX, YY, oracle_scale, levels=30, cmap='YlOrRd')
    for c, r in circles:
        ax.add_patch(Circle(c, r, fc='black', ec='black', alpha=0.7))
    ax.plot(*xs, 'gs', ms=10, zorder=5, label='Start')
    ax.plot(*xg, 'r*', ms=12, zorder=5, label='Goal')
    ax.set_title('Oracle metric √det(G)', fontsize=13)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_aspect('equal')
    ax.legend(fontsize=9)
    plt.colorbar(im1, ax=ax, shrink=0.8)

    # Panel 2: CARM learned metric field
    ax = axes[1]
    im2 = ax.contourf(XX, YY, carm_scale, levels=30, cmap='YlOrRd')
    for c, r in circles:
        ax.add_patch(Circle(c, r, fc='black', ec='black', alpha=0.7))
    if len(coll_pts) > 0:
        ax.scatter(coll_pts[:, 0], coll_pts[:, 1], c='cyan', s=3,
                   alpha=0.3, zorder=3, label=f'{len(coll_pts)} collision pts')
    ax.plot(*xs, 'gs', ms=10, zorder=5, label='Start')
    ax.plot(*xg, 'r*', ms=12, zorder=5, label='Goal')
    ax.set_title(f'CARM learned scale s(x) ({len(coll_pts)} collisions)', fontsize=13)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_aspect('equal')
    ax.legend(fontsize=8)
    plt.colorbar(im2, ax=ax, shrink=0.8)

    # Panel 3: CARM tree + path
    ax = axes[2]
    # Background: CARM scale (light)
    ax.contourf(XX, YY, carm_scale, levels=20, cmap='YlOrRd', alpha=0.3)
    for c, r in circles:
        ax.add_patch(Circle(c, r, fc='black', ec='black', alpha=0.7))
    # Tree edges
    for v in planner.vertices:
        if v.parent is not None:
            ax.plot([v.parent.x[0], v.x[0]], [v.parent.x[1], v.x[1]],
                    'b-', lw=0.3, alpha=0.3)
    # Path
    if path and len(path) > 1:
        px = [p[0] for p in path]
        py = [p[1] for p in path]
        ax.plot(px, py, 'r-', lw=3, zorder=4, label=f'CARM path (cost={cost:.3f})')
    ax.plot(*xs, 'gs', ms=10, zorder=5)
    ax.plot(*xg, 'r*', ms=12, zorder=5)
    ax.set_title('CARM tree & path', fontsize=13)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_aspect('equal')
    ax.legend(fontsize=9)

    fig.suptitle('Collision-Adaptive Riemannian Metric (CARM) — 2D Obstacle Environment',
                 fontsize=14, fontweight='bold', y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, 'carm_metric_field.png'),
                dpi=200, bbox_inches='tight')
    print(f'  → saved carm_metric_field.png')

    return fig


def visualize_carm_evolution():
    """Show metric evolution over iterations (4 snapshots)."""
    print('Generating CARM evolution visualization...')

    coll, _, oracle_metric, xs, xg, bounds = env_2d_obstacle_inflated()
    dim = len(xs)
    euclid = EuclideanMetric(dim)

    circles = [
        (np.array([0.30, 0.35]), 0.08),
        (np.array([0.30, 0.65]), 0.08),
        (np.array([0.50, 0.45]), 0.09),
        (np.array([0.50, 0.75]), 0.09),
        (np.array([0.70, 0.40]), 0.08),
        (np.array([0.70, 0.60]), 0.08),
    ]

    res = 150
    xx = np.linspace(0, 1, res)
    yy = np.linspace(0, 1, res)
    XX, YY = np.meshgrid(xx, yy)
    pts_grid = np.column_stack([XX.ravel(), YY.ravel()])

    # Run CARM step by step, capturing snapshots
    planner = RITStar(
        xs, xg, bounds, coll, euclid,
        geodesic_tier='diagonal', batch_size=80,
        max_iterations=120, random_seed=42,
        adaptive_metric=True,
        carm_sigma=0.08, carm_alpha=6.0,
        carm_rebuild_interval=10)

    snapshot_iters = [5, 20, 50, 100]
    snapshots = {}

    import time
    planner._t0 = time.time()
    for it in range(planner.max_iterations):
        samples = planner._sample_batch(it)
        planner._extend_tree(samples, it)
        if planner.c_best < np.inf:
            planner._prune()
            planner._update_informed_set()
            planner._update_stall_counter()
        if planner._adaptive_mode:
            planner._maybe_rebuild_carm_cache(it)
        elapsed = time.time() - planner._t0
        planner._record_stats(it, elapsed)

        if it + 1 in snapshot_iters:
            carm = planner._carm
            scale = carm._collision_scale_batch(pts_grid).reshape(XX.shape)
            coll_pts = np.array(carm._collision_points) if carm._collision_points else np.zeros((0, 2))
            snapshots[it + 1] = {
                'scale': scale.copy(),
                'coll_pts': coll_pts.copy(),
                'n_coll': len(coll_pts),
                'c_best': planner.c_best,
            }

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    for idx, it in enumerate(snapshot_iters):
        ax = axes[idx]
        snap = snapshots[it]
        im = ax.contourf(XX, YY, snap['scale'], levels=20, cmap='YlOrRd')
        for c, r in circles:
            ax.add_patch(Circle(c, r, fc='black', ec='black', alpha=0.7))
        if snap['n_coll'] > 0:
            ax.scatter(snap['coll_pts'][:, 0], snap['coll_pts'][:, 1],
                       c='cyan', s=2, alpha=0.3)
        ax.plot(*xs, 'gs', ms=8, zorder=5)
        ax.plot(*xg, 'r*', ms=10, zorder=5)
        c_str = f'{snap["c_best"]:.3f}' if np.isfinite(snap['c_best']) else '∞'
        ax.set_title(f'Iteration {it}\n{snap["n_coll"]} collisions, cost={c_str}',
                     fontsize=11)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.set_aspect('equal')
        plt.colorbar(im, ax=ax, shrink=0.75)

    fig.suptitle('CARM Metric Evolution — Learned Cost Field Over Time',
                 fontsize=14, fontweight='bold', y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, 'carm_evolution.png'),
                dpi=200, bbox_inches='tight')
    print(f'  → saved carm_evolution.png')
    return fig


if __name__ == '__main__':
    visualize_carm_metric_field()
    visualize_carm_evolution()
    print('\nAll CARM visualizations done!')
