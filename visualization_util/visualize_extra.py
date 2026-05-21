#!/usr/bin/env python
"""Generate metric field heatmap and RIT* tree growth GIF for 2D Obstacles."""
import sys
sys.stdout.reconfigure(line_buffering=True)

import gc
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import matplotlib.animation as animation

from visualization_util.output_paths import IMAGES_DIR, GIFS_DIR
from rit_star.environments import env_2d_obstacle_inflated
from rit_star.rit_star import RITStar


def plot_metric_heatmap():
    """Show the obstacle-inflated metric field as a heatmap."""
    coll, _, metric, xs, xg, bounds = env_2d_obstacle_inflated()

    res = 200
    x = np.linspace(0, 1, res)
    y = np.linspace(0, 1, res)
    X, Y = np.meshgrid(x, y)
    S = np.zeros_like(X)

    for i in range(res):
        for j in range(res):
            pt = np.array([X[i, j], Y[i, j]])
            G = metric.G(pt)
            S[i, j] = np.sqrt(G[0, 0])  # sqrt(s(x)) for conformal metric

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(S, origin='lower', extent=[0, 1, 0, 1],
                   cmap='hot_r', aspect='equal')
    plt.colorbar(im, ax=ax, label='Metric scale √s(x)')

    # Draw obstacle circles
    circles = [
        ([0.30, 0.35], 0.08), ([0.30, 0.65], 0.08),
        ([0.50, 0.45], 0.09), ([0.50, 0.75], 0.09),
        ([0.70, 0.40], 0.08), ([0.70, 0.60], 0.08),
    ]
    for c, r in circles:
        ax.add_patch(Circle(c, r, fc='none', ec='white', lw=2, ls='--'))

    ax.plot(*xs, 'go', ms=10, zorder=5, label='Start')
    ax.plot(*xg, 'r^', ms=10, zorder=5, label='Goal')
    ax.legend(loc='upper left', fontsize=10)
    ax.set_title('Obstacle-Inflated Metric Field\n(darker = higher cost → planner avoids)',
                 fontsize=12)
    ax.set_xlabel('x₁')
    ax.set_ylabel('x₂')

    fig.tight_layout()
    fname = os.path.join(IMAGES_DIR, 'metric_heatmap_2d_obstacles.png')
    fig.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close(fig)
    gc.collect()
    print('→ saved metric_heatmap_2d_obstacles.png')


def generate_tree_gif():
    """Generate GIF showing RIT* tree growth on 2D Obstacles."""
    coll, _, metric, xs, xg, bounds = env_2d_obstacle_inflated()

    planner = RITStar(xs, xg, bounds, coll, metric,
                      geodesic_tier='diagonal', batch_size=100,
                      max_iterations=200, random_seed=42)

    # Collect snapshots every 10 iterations
    snapshots = []
    path_snaps = []

    # Run iteration by iteration and capture state
    import time
    planner._t0 = time.time()
    for it in range(planner.max_iterations):
        samples = planner._sample_batch()
        planner._extend_tree(samples)
        if planner.c_best < np.inf:
            planner._prune()
            planner._update_informed_set()
        elapsed = time.time() - planner._t0
        planner._record_stats(it, elapsed)

        if it % 10 == 0 or it == planner.max_iterations - 1:
            verts = [v.x.copy() for v in planner.vertices]
            edges = [(v.parent.x.copy(), v.x.copy())
                     for v in planner.vertices if v.parent is not None]
            path = planner._extract_path()
            snapshots.append((it, verts, edges, planner.c_best))
            path_snaps.append(path)

    # Recompute exact final cost
    from rit_star.rit_star import riemannian_edge_cost
    final_path = planner._extract_path()
    if final_path and len(final_path) > 1:
        exact = sum(riemannian_edge_cost(final_path[i], final_path[i+1], metric)
                    for i in range(len(final_path) - 1))
        planner.c_best = exact

    print(f'  Final cost: {planner.c_best:.4f}, {len(snapshots)} frames')

    # Create GIF
    circles = [
        ([0.30, 0.35], 0.08), ([0.30, 0.65], 0.08),
        ([0.50, 0.45], 0.09), ([0.50, 0.75], 0.09),
        ([0.70, 0.40], 0.08), ([0.70, 0.60], 0.08),
    ]

    fig, ax = plt.subplots(figsize=(8, 7))

    def _frame(frame_idx):
        ax.cla()
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect('equal')

        # Draw obstacles
        for c, r in circles:
            ax.add_patch(Circle(c, r, fc='#333333', ec='black', alpha=0.7))

        it, verts, edges, c_best = snapshots[frame_idx]
        path = path_snaps[frame_idx]

        # Draw edges
        for p_x, c_x in edges:
            ax.plot([p_x[0], c_x[0]], [p_x[1], c_x[1]],
                    'gray', lw=0.15, alpha=0.3)

        # Draw vertices
        if verts:
            va = np.array(verts)
            ax.scatter(va[:, 0], va[:, 1], s=1, c='gray', alpha=0.3)

        # Draw path
        if path and len(path) > 1:
            pp = np.array(path)
            ax.plot(pp[:, 0], pp[:, 1], '#7B2FBE', lw=2.5, zorder=4)

        ax.plot(*xs, 'go', ms=8, zorder=5)
        ax.plot(*xg, 'r^', ms=8, zorder=5)

        cost_str = f'{c_best:.4f}' if np.isfinite(c_best) else '∞'
        ax.set_title(f'RIT* — iter {it}, vertices={len(verts)}, cost={cost_str}',
                     fontsize=12)

    anim = animation.FuncAnimation(fig, _frame, frames=len(snapshots),
                                   interval=300)
    anim.save(os.path.join(GIFS_DIR, 'rit_star_2d_obstacles.gif'), writer='pillow')
    plt.close(fig)
    del anim, snapshots, path_snaps, planner
    gc.collect()
    print('→ saved rit_star_2d_obstacles.gif')


if __name__ == '__main__':
    print('Generating metric heatmap...')
    plot_metric_heatmap()
    print('\nGenerating RIT* tree growth GIF...')
    generate_tree_gif()
    print('\nDone!')
