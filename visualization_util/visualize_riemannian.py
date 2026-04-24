#!/usr/bin/env python
"""Visualize the Riemannian metric field as 3-D surfaces and heatmaps.

Generates:
  1. 3-D surface plots of the metric scale field for each 2D environment
  2. Heatmaps with overlaid optimal paths
  3. Combined comparison figure showing all 4 environments side-by-side
"""
import sys
sys.stdout.reconfigure(line_buffering=True)

import gc
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
from matplotlib import cm
from matplotlib.collections import LineCollection
from matplotlib.animation import FuncAnimation, PillowWriter

try:
    from mpl_toolkits.mplot3d import Axes3D
except ImportError:
    # Fix for conflicting system vs pip mpl_toolkits
    import mpl_toolkits
    _user_mpl = os.path.join(os.path.expanduser('~'),
                             '.local/lib/python3.10/site-packages/mpl_toolkits')
    if os.path.isdir(_user_mpl):
        mpl_toolkits.__path__.insert(0, _user_mpl)
        from mpl_toolkits.mplot3d import Axes3D
    else:
        Axes3D = None

# Ensure 3D projection is registered even if matplotlib's init failed
if Axes3D is not None:
    from matplotlib.projections import projection_registry
    try:
        projection_registry.get_projection_class('3d')
    except (ValueError, KeyError):
        projection_registry.register(Axes3D)

from visualization_util.output_paths import IMAGES_DIR, GIFS_DIR

from rit_star.environments import (
    env_2d_obstacle_inflated,
    env_2d_diagonal_anisotropic,
    env_2d_narrow_passage,
    env_2d_maze,
    env_2d_bug_trap,
    env_2d_random_forest,
    env_2d_terrain,
    env_2d_hyper_dense,
    env_3d_diagonal_anisotropic,
    env_3d_sphere_field,
    env_3d_dense_labyrinth,
)
from rit_star.rit_star import RITStar


ENVS_2D = {
    '2D Obstacles':  env_2d_obstacle_inflated,
    '2D Diagonal':   env_2d_diagonal_anisotropic,
    '2D Narrow':     env_2d_narrow_passage,
    '2D Maze':       env_2d_maze,
    '2D Bug Trap':   env_2d_bug_trap,
    '2D Forest':     env_2d_random_forest,
    '2D Terrain':    env_2d_terrain,
    '2D Hyper-Dense': env_2d_hyper_dense,
}

ENVS_3D = {
    '3D Diagonal':  env_3d_diagonal_anisotropic,
    '3D Spheres':   env_3d_sphere_field,
    '3D Dense Lab': env_3d_dense_labyrinth,
}

# Backward compat
ENVS = ENVS_2D

OBSTACLE_DEFS = {
    '2D Obstacles': {
        'type': 'circles',
        'data': [
            ([0.30, 0.35], 0.08), ([0.30, 0.65], 0.08),
            ([0.50, 0.45], 0.09), ([0.50, 0.75], 0.09),
            ([0.70, 0.40], 0.08), ([0.70, 0.60], 0.08),
        ]
    },
    '2D Diagonal': {
        'type': 'rects',
        'data': [
            ([0.35, 0.30], [0.45, 0.70]),
            ([0.55, 0.30], [0.65, 0.70]),
        ]
    },
    '2D Narrow': {
        'type': 'rects',
        'data': [
            ([0.48, 0.00], [0.52, 0.47]),
            ([0.48, 0.53], [0.52, 1.00]),
            ([0.30, 0.15], [0.42, 0.35]),
            ([0.30, 0.65], [0.42, 0.85]),
            ([0.58, 0.15], [0.70, 0.35]),
            ([0.58, 0.65], [0.70, 0.85]),
        ]
    },
    '2D Maze': {
        'type': 'rects',
        'data': [
            ([0.00, 0.22], [0.70, 0.30]),
            ([0.30, 0.46], [1.00, 0.54]),
            ([0.00, 0.70], [0.70, 0.78]),
        ]
    },
    '2D Bug Trap': {
        'type': 'rects',
        'data': [
            ([0.15, 0.20], [0.60, 0.28]),
            ([0.15, 0.72], [0.60, 0.80]),
            ([0.15, 0.28], [0.23, 0.72]),
            ([0.52, 0.28], [0.60, 0.46]),
            ([0.52, 0.54], [0.60, 0.72]),
        ]
    },
    '2D Forest': {
        'type': 'circles',
        'data': None,  # generated at runtime
    },
    '2D Terrain': {
        'type': 'terrain_peaks',
        'data': [(cx, cy) for cx in [1/6, 0.5, 5/6] for cy in [1/6, 0.5, 5/6]],
    },
    '2D Hyper-Dense': {
        'type': 'circles',
        'data': None,  # generated at runtime
    },
}


def _get_forest_obstacles():
    """Reproduce the random forest obstacle positions."""
    rng = np.random.default_rng(12345)
    centres = []
    xs_forest = np.array([0.05, 0.05])
    xg_forest = np.array([0.95, 0.95])
    for _ in range(125):
        if len(centres) >= 25:
            break
        c = rng.uniform(0.1, 0.9, size=2)
        if np.linalg.norm(c - xs_forest) < 0.12:
            continue
        if np.linalg.norm(c - xg_forest) < 0.12:
            continue
        centres.append(c)
    return [(c.tolist(), 0.04) for c in centres[:25]]


# Lazily populate forest obstacles
OBSTACLE_DEFS['2D Forest']['data'] = _get_forest_obstacles()


def _get_hyper_dense_obstacles():
    """Reproduce the hyper-dense obstacle positions."""
    rng = np.random.default_rng(99999)
    centres = []
    xs_hd = np.array([0.05, 0.05])
    xg_hd = np.array([0.95, 0.95])
    radius = 0.03
    for _ in range(350):
        if len(centres) >= 35:
            break
        c = rng.uniform(0.08, 0.92, size=2)
        if np.linalg.norm(c - xs_hd) < 0.10:
            continue
        if np.linalg.norm(c - xg_hd) < 0.10:
            continue
        too_close = False
        for existing in centres:
            if np.linalg.norm(c - existing) < 2.2 * radius:
                too_close = True
                break
        if too_close:
            continue
        centres.append(c)
    return [(c.tolist(), radius) for c in centres[:35]]


OBSTACLE_DEFS['2D Hyper-Dense']['data'] = _get_hyper_dense_obstacles()

# Alias: env registry uses '2D Obstacle' but dict has '2D Obstacles'
OBSTACLE_DEFS['2D Obstacle'] = OBSTACLE_DEFS['2D Obstacles']

# Dividing wall obstacles (wall 0.47–0.53 + flanking blocks)
OBSTACLE_DEFS['2D Dividing Wall'] = {
    'type': 'rects',
    'data': [
        ([0.47, 0.00], [0.53, 0.10]),
        ([0.47, 0.13], [0.53, 0.85]),
        ([0.47, 0.88], [0.53, 1.00]),
        ([0.25, 0.70], [0.35, 0.85]),
        ([0.65, 0.15], [0.75, 0.30]),
    ]
}

# Joint arm: C-space obstacles can't be drawn as simple 2D shapes
OBSTACLE_DEFS['2D Joint Arm'] = {'type': 'none', 'data': None}


def _get_random_world_obstacles():
    """Reproduce the random-world obstacle positions (seed 2015_04)."""
    x_start = np.array([-0.1, -0.1])
    x_goal  = np.array([ 0.4,  0.4])
    rng = np.random.default_rng(2015_04)
    n_obs = 35
    rects = []
    for _ in range(n_obs * 50):
        if len(rects) >= n_obs:
            break
        ax_ = rng.uniform(-0.5, 0.5)
        ay_ = rng.uniform(-0.5, 0.5)
        w = rng.uniform(0.1, 0.2)
        h = rng.uniform(0.1, 0.2)
        lo = [ax_, ay_]
        hi = [ax_ + w, ay_ + h]
        clr = 0.06
        if (lo[0] <= x_start[0] + clr and hi[0] >= x_start[0] - clr and
            lo[1] <= x_start[1] + clr and hi[1] >= x_start[1] - clr):
            continue
        if (lo[0] <= x_goal[0] + clr and hi[0] >= x_goal[0] - clr and
            lo[1] <= x_goal[1] + clr and hi[1] >= x_goal[1] - clr):
            continue
        rects.append((lo, hi))
    return rects[:n_obs]


OBSTACLE_DEFS['2D Random World'] = {
    'type': 'rects',
    'data': _get_random_world_obstacles(),
}


def compute_metric_field(metric, bounds, res=150, collision_fn=None):
    """Compute sqrt of the max eigenvalue of G(x) on a grid.

    If *collision_fn* is supplied and the metric is (nearly) constant,
    obstacle cells are raised to create a visible "wall" effect so that
    the 3-D surface is not a flat plane.
    """
    x = np.linspace(bounds[0][0], bounds[0][1], res)
    y = np.linspace(bounds[1][0], bounds[1][1], res)
    X, Y = np.meshgrid(x, y)
    S = np.zeros_like(X)

    for i in range(res):
        for j in range(res):
            pt = np.array([X[i, j], Y[i, j]])
            G = metric.G(pt)
            eigvals = np.linalg.eigvalsh(G)
            S[i, j] = np.sqrt(np.max(eigvals))

    # Detect (near-)constant metric fields
    is_constant = (S.max() - S.min()) < 0.05 * S.mean()

    if is_constant and collision_fn is not None:
        # Build an obstacle-aware effective cost surface:
        # Free space keeps its metric value; obstacle cells get a tall
        # "wall" value so the surface clearly shows the environment.
        wall_height = S.mean() * 4.0
        for i in range(res):
            for j in range(res):
                pt = np.array([X[i, j], Y[i, j]])
                if not collision_fn(pt):
                    S[i, j] = wall_height
    return X, Y, S


def draw_obstacles_2d(ax, env_name, bounds=None):
    """Draw obstacles on a 2D axes, clipped to bounds if provided."""
    obs = OBSTACLE_DEFS.get(env_name)
    if obs is None or obs['type'] == 'none':
        return
    if obs['type'] == 'circles':
        for c, r in obs['data']:
            ax.add_patch(Circle(c, r, fc='#333333', ec='white', lw=1.5, ls='--', alpha=0.8))
    elif obs['type'] == 'rects':
        for lo, hi in obs['data']:
            x0, y0 = lo[0], lo[1]
            x1, y1 = hi[0], hi[1]
            if bounds is not None:
                x0 = max(x0, bounds[0][0])
                y0 = max(y0, bounds[1][0])
                x1 = min(x1, bounds[0][1])
                y1 = min(y1, bounds[1][1])
                if x1 <= x0 or y1 <= y0:
                    continue
            w, h = x1 - x0, y1 - y0
            ax.add_patch(Rectangle((x0, y0), w, h, fc='#333333', ec='white', lw=1.5, ls='--', alpha=0.8))
    elif obs['type'] == 'terrain_peaks':
        # These are cost peaks, not hard obstacles — draw just a dotted
        # outline so they don't look like walls the tree is violating.
        for cx, cy in obs['data']:
            ax.add_patch(Circle([cx, cy], 0.09, fc='none', ec='#E65100',
                                lw=1.0, ls=':', alpha=0.5, zorder=2))


def plot_3d_surface(env_name, env_fn, save_prefix):
    """Generate a 3-D surface plot of the Riemannian metric field.

    For constant-metric environments (e.g. DiagonalAnisotropic) the
    collision checker is used to raise obstacles into visible walls
    so the plot is not a flat grey plane.
    """
    coll, _, metric, xs, xg, bounds = env_fn()
    X, Y, S = compute_metric_field(metric, bounds, res=120,
                                   collision_fn=coll)

    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')

    # Normalize colors — use 'inferno' for better contrast than 'hot_r'
    norm = plt.Normalize(S.min(), S.max())
    colors = cm.inferno_r(norm(S))

    surf = ax.plot_surface(X, Y, S, facecolors=colors,
                           rstride=1, cstride=1,
                           antialiased=True, alpha=0.9,
                           shade=True)

    # Mark start and goal as vertical lines
    z_max = S.max() * 1.1
    ax.scatter([xs[0]], [xs[1]], [0], c='green', s=100, marker='o',
               zorder=10, depthshade=False, label='Start')
    ax.scatter([xg[0]], [xg[1]], [0], c='red', s=100, marker='^',
               zorder=10, depthshade=False, label='Goal')

    # Vertical lines from base to surface at start/goal
    for pt, color in [(xs, 'green'), (xg, 'red')]:
        ix = np.argmin(np.abs(X[0, :] - pt[0]))
        iy = np.argmin(np.abs(Y[:, 0] - pt[1]))
        z_val = S[iy, ix]
        ax.plot([pt[0], pt[0]], [pt[1], pt[1]], [0, z_val],
                color=color, lw=2, ls='--', alpha=0.7)

    ax.set_xlabel('x₁', fontsize=12)
    ax.set_ylabel('x₂', fontsize=12)
    ax.set_zlabel('Metric scale √λ_max(G)', fontsize=10)
    ax.set_title(f'Riemannian Metric Surface — {env_name}', fontsize=14, pad=20)
    ax.legend(loc='upper left', fontsize=10)

    # Good viewing angle
    ax.view_init(elev=35, azim=-45)

    # Add colorbar
    mappable = cm.ScalarMappable(norm=norm, cmap='inferno_r')
    fig.colorbar(mappable, ax=ax, shrink=0.5, label='√λ_max(G)', pad=0.1)

    fname = os.path.join(IMAGES_DIR, f'{save_prefix}_3d_surface.png')
    fig.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close(fig)
    del fig, surf, X, Y, S
    gc.collect()
    print(f'  → saved {fname}')
    return fname


def plot_heatmap_with_path(env_name, env_fn, save_prefix):
    """Generate a heatmap of the metric field with the RIT* path overlaid."""
    coll, _, metric, xs, xg, bounds = env_fn()
    X, Y, S = compute_metric_field(metric, bounds, res=150)

    # Run RIT* to get a path
    planner = RITStar(xs, xg, bounds, coll, metric,
                      geodesic_tier='diagonal', batch_size=100,
                      max_iterations=200, random_seed=42)
    path, cost = planner.plan()
    print(f'    RIT* path cost: {cost:.4f}')

    fig, ax = plt.subplots(figsize=(9, 8))
    extent = [bounds[0][0], bounds[0][1], bounds[1][0], bounds[1][1]]
    im = ax.imshow(S, origin='lower', extent=extent, cmap='hot_r', aspect='equal')
    plt.colorbar(im, ax=ax, label='Metric scale √λ_max(G)')

    draw_obstacles_2d(ax, env_name, bounds=bounds)

    # Draw path
    if path and len(path) > 1:
        pp = np.array(path)
        ax.plot(pp[:, 0], pp[:, 1], '#00FF00', lw=2.5, zorder=4,
                label=f'RIT* path (cost={cost:.3f})')

    ax.plot(*xs, 'go', ms=12, zorder=5, label='Start')
    ax.plot(*xg, 'r^', ms=12, zorder=5, label='Goal')
    ax.legend(loc='upper left', fontsize=10)
    ax.set_title(f'Metric Field & Optimal Path — {env_name}\n'
                 f'(darker = higher cost → planner avoids)', fontsize=12)
    ax.set_xlabel('x₁')
    ax.set_ylabel('x₂')

    fig.tight_layout()
    fname = os.path.join(IMAGES_DIR, f'{save_prefix}_heatmap.png')
    fig.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close(fig)
    del fig, planner, X, Y, S
    gc.collect()
    print(f'  → saved {fname}')
    return fname


def plot_combined_surfaces():
    """Generate a 2×2 grid of 3-D surfaces for all 4 environments."""
    fig = plt.figure(figsize=(20, 16))
    fig.suptitle('Riemannian Metric Surfaces — All 2D Environments', fontsize=18, y=0.95)

    for idx, (env_name, env_fn) in enumerate(ENVS.items()):
        print(f'  Computing {env_name}...')
        coll, _, metric, xs, xg, bounds = env_fn()
        X, Y, S = compute_metric_field(metric, bounds, res=100)

        ax = fig.add_subplot(2, 2, idx + 1, projection='3d')
        norm = plt.Normalize(S.min(), S.max())
        colors = cm.hot_r(norm(S))
        ax.plot_surface(X, Y, S, facecolors=colors,
                        rstride=2, cstride=2,
                        antialiased=True, alpha=0.9, shade=True)

        ax.scatter([xs[0]], [xs[1]], [0], c='green', s=80, marker='o',
                   depthshade=False)
        ax.scatter([xg[0]], [xg[1]], [0], c='red', s=80, marker='^',
                   depthshade=False)

        ax.set_title(env_name, fontsize=14, pad=10)
        ax.set_xlabel('x₁', fontsize=9)
        ax.set_ylabel('x₂', fontsize=9)
        ax.set_zlabel('√λ_max(G)', fontsize=9)
        ax.view_init(elev=30, azim=-50)

    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fname = os.path.join(IMAGES_DIR, 'riemannian_surfaces_combined.png')
    fig.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close(fig)
    del fig
    gc.collect()
    print(f'  → saved {fname}')


def plot_combined_heatmaps():
    """Generate a 2×2 grid of heatmaps for all 4 environments."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    fig.suptitle('Riemannian Metric Fields — All 2D Environments', fontsize=16, y=0.97)

    for idx, (env_name, env_fn) in enumerate(ENVS.items()):
        print(f'  Computing {env_name}...')
        coll, _, metric, xs, xg, bounds = env_fn()
        X, Y, S = compute_metric_field(metric, bounds, res=150)

        # Run RIT* for path
        planner = RITStar(xs, xg, bounds, coll, metric,
                          geodesic_tier='diagonal', batch_size=100,
                          max_iterations=200, random_seed=42)
        path, cost = planner.plan()
        del planner
        gc.collect()

        row, col = idx // 2, idx % 2
        ax = axes[row, col]
        extent = [bounds[0][0], bounds[0][1], bounds[1][0], bounds[1][1]]
        im = ax.imshow(S, origin='lower', extent=extent, cmap='hot_r', aspect='equal')
        fig.colorbar(im, ax=ax, shrink=0.8)

        draw_obstacles_2d(ax, env_name, bounds=bounds)

        if path and len(path) > 1:
            pp = np.array(path)
            ax.plot(pp[:, 0], pp[:, 1], '#00FF00', lw=2.5, zorder=4)

        ax.plot(*xs, 'go', ms=10, zorder=5)
        ax.plot(*xg, 'r^', ms=10, zorder=5)
        cost_str = f'{cost:.3f}' if np.isfinite(cost) else 'no sol'
        ax.set_title(f'{env_name} (RIT* cost={cost_str})', fontsize=12)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fname = os.path.join(IMAGES_DIR, 'riemannian_heatmaps_combined.png')
    fig.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close(fig)
    del fig
    gc.collect()
    print(f'  → saved {fname}')


def animate_tree_growth(env_name, env_fn, save_prefix,
                        max_iterations=80, batch_size=100,
                        frame_every=2, fps=8, res=100):
    """Create a GIF animation of the RIT* tree growing on the Riemannian surface.

    Shows the metric-field heatmap with the sampling tree edges growing
    and the current best path highlighted at each frame.

    Parameters
    ----------
    env_name : str
    env_fn : callable
    save_prefix : str
    max_iterations : int
    batch_size : int
    frame_every : int  — capture a frame every N iterations
    fps : int  — frames per second in the output GIF
    res : int  — heatmap grid resolution
    """
    coll, _, metric, xs, xg, bounds = env_fn()
    X, Y, S = compute_metric_field(metric, bounds, res=res)
    extent = [bounds[0][0], bounds[0][1], bounds[1][0], bounds[1][1]]

    planner = RITStar(xs, xg, bounds, coll, metric,
                      geodesic_tier='diagonal', batch_size=batch_size,
                      max_iterations=max_iterations, random_seed=42)

    # Collect snapshots
    snapshots = []
    for state in planner.plan_stepwise():
        if state['iteration'] % frame_every == 0 or state['iteration'] == max_iterations - 1:
            snapshots.append(state)

    if not snapshots:
        print(f'  No snapshots collected for {env_name}')
        return

    print(f'  Collected {len(snapshots)} frames for {env_name}')

    # Build the animation — match figure aspect ratio to domain
    x_range = bounds[0][1] - bounds[0][0]
    y_range = bounds[1][1] - bounds[1][0]
    base_size = 7
    if x_range >= y_range:
        fig_w = base_size
        fig_h = base_size * (y_range / x_range)
    else:
        fig_h = base_size
        fig_w = base_size * (x_range / y_range)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    # Static background: heatmap
    ax.imshow(S, origin='lower', extent=extent, cmap='hot_r', aspect='equal')
    draw_obstacles_2d(ax, env_name, bounds=bounds)
    ax.plot(*xs, 'go', ms=12, zorder=10)
    ax.plot(*xg, 'r^', ms=12, zorder=10)
    ax.set_xlim(bounds[0])
    ax.set_ylim(bounds[1])
    ax.autoscale(False)
    ax.set_xlabel('x₁', fontsize=14)
    ax.set_ylabel('x₂', fontsize=14)
    ax.tick_params(axis='both', labelsize=12)

    # Dynamic elements
    edge_collection = LineCollection([], colors='#4FC3F7', linewidths=0.5,
                                     alpha=0.6, zorder=2)
    ax.add_collection(edge_collection)
    path_line, = ax.plot([], [], '#00FF00', lw=3.5, zorder=5)
    vertex_scatter = ax.scatter([], [], c='#90CAF9', s=4, zorder=3, alpha=0.7)
    title = ax.set_title('', fontsize=14)
    fig.tight_layout(pad=0.5)

    def update(frame_idx):
        snap = snapshots[frame_idx]
        it = snap['iteration']
        c = snap['c_best']

        # Tree edges
        if snap['edges']:
            segments = [[(p[0], p[1]), (ch[0], ch[1])]
                        for p, ch in snap['edges']]
            edge_collection.set_segments(segments)
        else:
            edge_collection.set_segments([])

        # Vertices
        if snap['vertices']:
            verts = np.array(snap['vertices'])
            vertex_scatter.set_offsets(verts[:, :2])
        else:
            vertex_scatter.set_offsets(np.empty((0, 2)))

        # Best path
        path = snap['path']
        if path and len(path) > 1:
            pp = np.array(path)
            path_line.set_data(pp[:, 0], pp[:, 1])
        else:
            path_line.set_data([], [])

        cost_str = f'{c:.3f}' if np.isfinite(c) else '∞'
        n_verts = len(snap['vertices'])
        title.set_text(f'{env_name} — RIT* Tree Growth\n'
                       f'Iter {it+1}/{max_iterations}  |  '
                       f'Vertices: {n_verts}  |  Cost: {cost_str}')
        return edge_collection, vertex_scatter, path_line, title

    anim = FuncAnimation(fig, update, frames=len(snapshots),
                         interval=1000 // fps, blit=False)

    fname = os.path.join(GIFS_DIR, f'{save_prefix}_tree_growth.gif')
    anim.save(fname, writer=PillowWriter(fps=fps))
    plt.close(fig)
    del anim, fig, snapshots, planner, X, Y, S
    gc.collect()
    print(f'  → saved {fname}')
    return fname


def animate_tree_growth_carm(env_name, env_fn, save_prefix,
                             max_iterations=80, batch_size=100,
                             frame_every=2, fps=8, res=100,
                             carm_sigma=0.1, carm_alpha=5.0,
                             carm_rebuild_interval=5):
    """2-D tree-growth GIF with a LIVE CARM gradient that updates per frame.

    Identical to :func:`animate_tree_growth`, but enables RIT*'s
    Collision-Adaptive Riemannian Metric (CARM). At each captured snapshot,
    the metric field is recomputed from the *current* CARM state and the
    heat-map background is redrawn, so the viewer sees the gradient adapt
    around collision regions as the tree grows.

    Also saves a final PNG with scene + tree + path + final gradient.
    """
    coll, _, base_metric, xs, xg, bounds = env_fn()
    extent = [bounds[0][0], bounds[0][1], bounds[1][0], bounds[1][1]]

    planner = RITStar(xs, xg, bounds, coll, base_metric,
                      geodesic_tier='diagonal', batch_size=batch_size,
                      max_iterations=max_iterations, random_seed=42,
                      adaptive_metric=True,
                      carm_sigma=carm_sigma, carm_alpha=carm_alpha,
                      carm_rebuild_interval=carm_rebuild_interval)
    live_metric = planner.metric  # CARM instance — its G(x) changes over time

    # Precompute a grid of query points — we'll evaluate just the CARM
    # conformal scale s(x) on it each frame (not the full composite metric).
    gx = np.linspace(bounds[0][0], bounds[0][1], res)
    gy = np.linspace(bounds[1][0], bounds[1][1], res)
    GX, GY = np.meshgrid(gx, gy)
    grid_pts = np.column_stack([GX.ravel(), GY.ravel()])

    def _carm_scale_field():
        """Return the CARM-only scale factor s(x) as (res, res) array.

        Uses the vectorized batch interface when available. Regions with
        no nearby collision samples stay at s=1 (light in hot_r cmap).
        """
        if hasattr(live_metric, '_collision_scale_batch'):
            s = live_metric._collision_scale_batch(grid_pts)
        else:
            s = np.array([live_metric._collision_scale(p) for p in grid_pts])
        return s.reshape(res, res)

    snapshots = []
    for state in planner.plan_stepwise():
        if state['iteration'] % frame_every == 0 or state['iteration'] == max_iterations - 1:
            state = dict(state)
            state['metric_field'] = _carm_scale_field()
            snapshots.append(state)

    if not snapshots:
        print(f'  No snapshots collected for {env_name}')
        return

    print(f'  Collected {len(snapshots)} CARM frames for {env_name}')

    x_range = bounds[0][1] - bounds[0][0]
    y_range = bounds[1][1] - bounds[1][0]
    base_size = 7
    if x_range >= y_range:
        fig_w, fig_h = base_size, base_size * (y_range / x_range)
    else:
        fig_h, fig_w = base_size, base_size * (x_range / y_range)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    S0 = snapshots[0]['metric_field']
    # CARM scale is always ≥ 1 (s=1 means "no CARM inflation"). Anchor vmin
    # at 1.0 so untouched / sparse-sample regions render as light.
    vmin = 1.0
    vmax = max(max(s['metric_field'].max() for s in snapshots), 1.01)
    img = ax.imshow(S0, origin='lower', extent=extent, cmap='hot_r',
                    aspect='equal', vmin=vmin, vmax=vmax)
    draw_obstacles_2d(ax, env_name, bounds=bounds)
    ax.plot(*xs, 'go', ms=12, zorder=10)
    ax.plot(*xg, 'r^', ms=12, zorder=10)
    ax.set_xlim(bounds[0])
    ax.set_ylim(bounds[1])
    ax.autoscale(False)
    ax.set_xlabel('x₁', fontsize=14)
    ax.set_ylabel('x₂', fontsize=14)
    ax.tick_params(axis='both', labelsize=12)

    edge_collection = LineCollection([], colors='#4FC3F7', linewidths=0.5,
                                     alpha=0.6, zorder=2)
    ax.add_collection(edge_collection)
    path_line, = ax.plot([], [], '#00FF00', lw=3.5, zorder=5)
    vertex_scatter = ax.scatter([], [], c='#90CAF9', s=4, zorder=3, alpha=0.7)
    title = ax.set_title('', fontsize=14)
    fig.tight_layout(pad=0.5)

    def update(frame_idx):
        snap = snapshots[frame_idx]
        img.set_data(snap['metric_field'])

        if snap['edges']:
            segments = [[(p[0], p[1]), (ch[0], ch[1])]
                        for p, ch in snap['edges']]
            edge_collection.set_segments(segments)
        else:
            edge_collection.set_segments([])
        if snap['vertices']:
            verts = np.array(snap['vertices'])
            vertex_scatter.set_offsets(verts[:, :2])
        else:
            vertex_scatter.set_offsets(np.empty((0, 2)))
        path = snap['path']
        if path and len(path) > 1:
            pp = np.array(path)
            path_line.set_data(pp[:, 0], pp[:, 1])
        else:
            path_line.set_data([], [])

        c = snap['c_best']
        cost_str = f'{c:.3f}' if np.isfinite(c) else '∞'
        n_verts = len(snap['vertices'])
        carm_max = snap['metric_field'].max()
        title.set_text(f'{env_name} — RIT* + CARM (showing learned s(x))\n'
                       f'Iter {snap["iteration"]+1}/{max_iterations}  |  '
                       f'Vertices: {n_verts}  |  Cost: {cost_str}  |  '
                       f'max s(x)={carm_max:.2f}')
        return img, edge_collection, vertex_scatter, path_line, title

    anim = FuncAnimation(fig, update, frames=len(snapshots),
                         interval=1000 // fps, blit=False)
    gif_path = os.path.join(GIFS_DIR, f'{save_prefix}_tree_growth_carm.gif')
    anim.save(gif_path, writer=PillowWriter(fps=fps))
    plt.close(fig)
    print(f'  → saved {gif_path}')

    # Final summary PNG: scene + tree + path + final CARM gradient
    final = snapshots[-1]
    fig2, ax2 = plt.subplots(figsize=(fig_w, fig_h))
    im2 = ax2.imshow(final['metric_field'], origin='lower', extent=extent,
                     cmap='hot_r', aspect='equal', vmin=vmin, vmax=vmax)
    draw_obstacles_2d(ax2, env_name, bounds=bounds)
    if final['edges']:
        segs = [[(p[0], p[1]), (ch[0], ch[1])] for p, ch in final['edges']]
        ax2.add_collection(LineCollection(segs, colors='#4FC3F7',
                                          linewidths=0.6, alpha=0.7, zorder=2))
    if final['vertices']:
        verts = np.array(final['vertices'])
        ax2.scatter(verts[:, 0], verts[:, 1], c='#90CAF9', s=6, alpha=0.8, zorder=3)
    if final['path'] and len(final['path']) > 1:
        pp = np.array(final['path'])
        ax2.plot(pp[:, 0], pp[:, 1], '#00FF00', lw=3.5, zorder=5)
    ax2.plot(*xs, 'go', ms=12, zorder=10)
    ax2.plot(*xg, 'r^', ms=12, zorder=10)
    ax2.set_xlim(bounds[0]); ax2.set_ylim(bounds[1]); ax2.autoscale(False)
    ax2.set_xlabel('x₁', fontsize=14); ax2.set_ylabel('x₂', fontsize=14)
    c = final['c_best']
    cost_str = f'{c:.3f}' if np.isfinite(c) else '∞'
    ax2.set_title(f'{env_name} — Final CARM s(x) + RIT* tree + path\n'
                  f'Vertices: {len(final["vertices"])}  |  Cost: {cost_str}  '
                  f'|  max s(x)={final["metric_field"].max():.2f}',
                  fontsize=13)
    cbar = fig2.colorbar(im2, ax=ax2, shrink=0.85, pad=0.02)
    cbar.set_label('CARM scale s(x)  (1 = no inflation)', fontsize=11)
    fig2.tight_layout(pad=0.5)
    png_path = os.path.join(IMAGES_DIR, f'{save_prefix}_tree_carm_final.png')
    fig2.savefig(png_path, dpi=160, bbox_inches='tight')
    plt.close(fig2)
    print(f'  → saved {png_path}')

    del anim, fig, fig2, snapshots, planner
    gc.collect()
    return gif_path, png_path, final


def animate_3d_surface_tree(env_name, env_fn, save_prefix,
                            max_iterations=80, batch_size=100,
                            frame_every=2, fps=8, res=80):
    """Create a GIF showing the tree growing ON the 3-D Riemannian surface.

    Tree edges are projected onto the metric surface (z = sqrt(lambda_max))
    so the viewer sees the search exploring the manifold.

    Parameters
    ----------
    env_name : str
    env_fn : callable
    save_prefix : str
    max_iterations, batch_size, frame_every, fps, res : int
    """
    coll, _, metric, xs, xg, bounds = env_fn()
    X, Y, S = compute_metric_field(metric, bounds, res=res)

    planner = RITStar(xs, xg, bounds, coll, metric,
                      geodesic_tier='diagonal', batch_size=batch_size,
                      max_iterations=max_iterations, random_seed=42)

    # Collect snapshots
    snapshots = []
    for state in planner.plan_stepwise():
        if state['iteration'] % frame_every == 0 or state['iteration'] == max_iterations - 1:
            snapshots.append(state)

    if not snapshots:
        print(f'  No snapshots collected for {env_name}')
        return

    print(f'  Collected {len(snapshots)} 3D frames for {env_name}')

    # Precompute interpolator for lifting points onto the surface
    from scipy.interpolate import RegularGridInterpolator
    x_coords = np.linspace(bounds[0][0], bounds[0][1], res)
    y_coords = np.linspace(bounds[1][0], bounds[1][1], res)
    interp = RegularGridInterpolator((y_coords, x_coords), S,
                                     method='linear',
                                     bounds_error=False, fill_value=0.0)

    def lift_z(pts_2d):
        """Get surface height for 2D points."""
        pts = np.atleast_2d(pts_2d)
        return interp(pts[:, ::-1])  # interp expects (y, x)

    # Build animation
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    norm = plt.Normalize(S.min(), S.max())
    colors_surf = cm.hot_r(norm(S))

    def draw_frame(frame_idx):
        ax.cla()
        snap = snapshots[frame_idx]

        # Surface (semi-transparent)
        ax.plot_surface(X, Y, S, facecolors=colors_surf,
                        rstride=2, cstride=2,
                        antialiased=True, alpha=0.55, shade=True)

        # Start and goal
        zs = float(lift_z(xs.reshape(1, -1))[0])
        zg = float(lift_z(xg.reshape(1, -1))[0])
        ax.scatter([xs[0]], [xs[1]], [zs], c='green', s=100, marker='o',
                   depthshade=False, zorder=10)
        ax.scatter([xg[0]], [xg[1]], [zg], c='red', s=100, marker='^',
                   depthshade=False, zorder=10)

        # Tree edges on surface
        for p, ch in snap['edges']:
            zp = float(lift_z(p.reshape(1, -1))[0])
            zch = float(lift_z(ch.reshape(1, -1))[0])
            ax.plot([p[0], ch[0]], [p[1], ch[1]], [zp, zch],
                    color='#4FC3F7', lw=0.6, alpha=0.7)

        # Best path on surface
        path = snap['path']
        if path and len(path) > 1:
            pp = np.array(path)
            zz = lift_z(pp)
            ax.plot(pp[:, 0], pp[:, 1], zz, color='#00FF00',
                    lw=3.0, zorder=8)

        c = snap['c_best']
        cost_str = f'{c:.3f}' if np.isfinite(c) else '∞'
        ax.set_xlabel('x₁', fontsize=10)
        ax.set_ylabel('x₂', fontsize=10)
        ax.set_zlabel('√λ_max(G)', fontsize=9)
        ax.set_title(f'{env_name} — 3D Surface + Tree\n'
                     f'Iter {snap["iteration"]+1}/{max_iterations}  |  '
                     f'Cost: {cost_str}', fontsize=11)
        ax.view_init(elev=35, azim=-45)

    anim = FuncAnimation(fig, draw_frame, frames=len(snapshots),
                         interval=1000 // fps, blit=False)

    fname = os.path.join(GIFS_DIR, f'{save_prefix}_3d_tree_growth.gif')
    anim.save(fname, writer=PillowWriter(fps=fps))
    plt.close(fig)
    del anim, fig, snapshots, planner, X, Y, S
    gc.collect()
    print(f'  → saved {fname}')
    return fname


# ═══════════════════════════════════════════════════════════════════════
# 3-D environment animation (true 3D C-space)
# ═══════════════════════════════════════════════════════════════════════

_BLOCKED_PTS_CACHE = {}


def _probe_blocked_points(coll, bounds, res=22):
    """Voxel-probe the 3-D collision checker and return blocked cells.

    Works for any 3-D environment without hard-coding obstacle geometry.
    Results cached by id(coll) + bounds to avoid reprobing per frame.
    """
    key = (id(coll), tuple(tuple(b) for b in bounds), res)
    if key in _BLOCKED_PTS_CACHE:
        return _BLOCKED_PTS_CACHE[key]
    xs_ = np.linspace(bounds[0][0], bounds[0][1], res)
    ys_ = np.linspace(bounds[1][0], bounds[1][1], res)
    zs_ = np.linspace(bounds[2][0], bounds[2][1], res)
    blocked = []
    for x_ in xs_:
        for y_ in ys_:
            for z_ in zs_:
                pt = np.array([x_, y_, z_])
                if not coll(pt):
                    blocked.append((x_, y_, z_))
    arr = np.asarray(blocked) if blocked else np.empty((0, 3))
    _BLOCKED_PTS_CACHE[key] = arr
    return arr


def _draw_obstacles_3d_anim(ax, env_name, coll=None, bounds=None, res=22):
    """Draw 3D obstacles onto an Axes3D using the env's own collision checker.

    The collision checker is probed on a voxel grid; blocked cells are
    scattered as dark translucent squares. Generic — works for all 3-D
    envs (boxes, spheres, walls with holes, etc.).
    """
    if coll is None or bounds is None or len(bounds) != 3:
        return
    pts = _probe_blocked_points(coll, bounds, res=res)
    if pts.size == 0:
        return
    ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2],
               c='#2a2a2a', s=24, alpha=0.25, marker='s',
               depthshade=False, zorder=1)


def animate_3d_env(env_name, env_fn, save_prefix,
                   max_iterations=80, batch_size=100,
                   frame_every=2, fps=8):
    """Create a GIF showing RIT* tree growth in a true 3-D C-space.

    Parameters
    ----------
    env_name : str — '3D Diagonal' or '3D Spheres'
    env_fn : callable
    save_prefix : str
    max_iterations, batch_size, frame_every, fps : int
    """
    coll, _, metric, xs, xg, bounds = env_fn()

    planner = RITStar(xs, xg, bounds, coll, metric,
                      geodesic_tier='diagonal', batch_size=batch_size,
                      max_iterations=max_iterations, random_seed=42)

    # Collect snapshots
    snapshots = []
    for state in planner.plan_stepwise():
        if state['iteration'] % frame_every == 0 or state['iteration'] == max_iterations - 1:
            snapshots.append(state)

    if not snapshots:
        print(f'  No snapshots collected for {env_name}')
        return

    print(f'  Collected {len(snapshots)} frames for {env_name}')

    # Determine axis limits from bounds
    x_lim = (bounds[0][0], bounds[0][1])
    y_lim = (bounds[1][0], bounds[1][1])
    z_lim = (bounds[2][0], bounds[2][1])

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Prefer the rich solid-obstacle renderer used by the static PNGs
    # (matches config_<env>_rit_tree.png). Falls back to the voxel-probe
    # version for environments that aren't hand-coded there.
    try:
        from run_from_config import _draw_obstacles_3d_static
    except Exception:
        _draw_obstacles_3d_static = None

    _HANDCODED_3D = {
        '3D Diagonal', '3D Spheres', '3D Dense Lab', '3D Gauntlet',
        '3D Narrow', '3D Corridor', '3D Wall & Gaps', '3D Box Field',
    }

    def draw_frame(frame_idx):
        ax.cla()
        snap = snapshots[frame_idx]

        # Draw obstacles — use the rich static renderer when we have one
        # hand-coded for this env, otherwise voxel-probe the collision
        # checker so at least something shows up.
        if _draw_obstacles_3d_static is not None and env_name in _HANDCODED_3D:
            _draw_obstacles_3d_static(ax, env_name)
        else:
            _draw_obstacles_3d_anim(ax, env_name, coll=coll, bounds=bounds)

        # Start and goal
        ax.scatter([xs[0]], [xs[1]], [xs[2]], c='green', s=100, marker='o',
                   depthshade=False, zorder=10)
        ax.scatter([xg[0]], [xg[1]], [xg[2]], c='red', s=100, marker='^',
                   depthshade=False, zorder=10)

        # Tree edges
        for p, ch in snap['edges']:
            ax.plot([p[0], ch[0]], [p[1], ch[1]], [p[2], ch[2]],
                    color='#4FC3F7', lw=0.4, alpha=0.5)

        # Vertices
        if snap['vertices']:
            va = np.array(snap['vertices'])
            ax.scatter(va[:, 0], va[:, 1], va[:, 2],
                       s=2, c='#90CAF9', alpha=0.4)

        # Best path
        path = snap['path']
        if path and len(path) > 1:
            pp = np.array(path)
            ax.plot(pp[:, 0], pp[:, 1], pp[:, 2], '#7B2FBE',
                    lw=3.0, zorder=8)

        ax.set_xlim(*x_lim)
        ax.set_ylim(*y_lim)
        ax.set_zlim(*z_lim)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')

        c = snap['c_best']
        cost_str = f'{c:.3f}' if np.isfinite(c) else '∞'
        n_verts = len(snap['vertices'])
        ax.set_title(f'{env_name} — RIT* 3D Tree Growth\n'
                     f'Iter {snap["iteration"]+1}/{max_iterations}  |  '
                     f'Vertices: {n_verts}  |  Cost: {cost_str}', fontsize=11)

        # Slowly rotate camera
        azim = -45 + frame_idx * 1.5
        ax.view_init(elev=25, azim=azim)

    anim = FuncAnimation(fig, draw_frame, frames=len(snapshots),
                         interval=1000 // fps, blit=False)

    fname = os.path.join(GIFS_DIR, f'{save_prefix}_tree_growth.gif')
    anim.save(fname, writer=PillowWriter(fps=fps))
    plt.close(fig)
    del anim, fig, snapshots, planner
    gc.collect()
    print(f'  → saved {fname}')
    return fname


def animate_3d_env_carm(env_name, env_fn, save_prefix,
                        max_iterations=80, batch_size=100,
                        frame_every=2, fps=8,
                        carm_sigma=0.15, carm_alpha=5.0,
                        carm_rebuild_interval=5,
                        grid_res=14, s_threshold=1.10):
    """3-D tree-growth GIF with a LIVE CARM gradient cloud.

    Same as :func:`animate_3d_env` but enables CARM; at each frame a
    coarse grid of points is coloured by the current CARM scale s(x),
    and only points with s(x) > ``s_threshold`` are drawn (so untouched
    regions stay empty). The cloud evolves as the tree grows and more
    collision samples accumulate. Also saves a final summary PNG with
    scene + tree + path + final CARM cloud.
    """
    coll, _, base_metric, xs, xg, bounds = env_fn()

    planner = RITStar(xs, xg, bounds, coll, base_metric,
                      geodesic_tier='diagonal', batch_size=batch_size,
                      max_iterations=max_iterations, random_seed=42,
                      adaptive_metric=True,
                      carm_sigma=carm_sigma, carm_alpha=carm_alpha,
                      carm_rebuild_interval=carm_rebuild_interval)
    live_metric = planner.metric  # CollisionAdaptiveMetric

    # Precompute a 3-D query grid (cached in memory, evaluated each frame)
    gx = np.linspace(bounds[0][0], bounds[0][1], grid_res)
    gy = np.linspace(bounds[1][0], bounds[1][1], grid_res)
    gz = np.linspace(bounds[2][0], bounds[2][1], grid_res)
    GX, GY, GZ = np.meshgrid(gx, gy, gz, indexing='ij')
    grid_pts = np.column_stack([GX.ravel(), GY.ravel(), GZ.ravel()])

    # Keep only cells inside the free space — obstacle cells would
    # always stay at the base metric and clutter the cloud.
    free_mask = np.array([coll(p) for p in grid_pts])
    grid_pts = grid_pts[free_mask]

    def _carm_field():
        if hasattr(live_metric, '_collision_scale_batch'):
            return live_metric._collision_scale_batch(grid_pts)
        return np.array([live_metric._collision_scale(p) for p in grid_pts])

    snapshots = []
    for state in planner.plan_stepwise():
        if state['iteration'] % frame_every == 0 or state['iteration'] == max_iterations - 1:
            state = dict(state)
            state['s_field'] = _carm_field()
            snapshots.append(state)

    if not snapshots:
        print(f'  No snapshots collected for {env_name}')
        return

    print(f'  Collected {len(snapshots)} CARM frames for {env_name}')

    x_lim = (bounds[0][0], bounds[0][1])
    y_lim = (bounds[1][0], bounds[1][1])
    z_lim = (bounds[2][0], bounds[2][1])

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    try:
        from run_from_config import _draw_obstacles_3d_static
    except Exception:
        _draw_obstacles_3d_static = None

    _HANDCODED_3D = {
        '3D Diagonal', '3D Spheres', '3D Dense Lab', '3D Gauntlet',
        '3D Narrow', '3D Corridor', '3D Wall & Gaps', '3D Box Field',
    }

    # Global vmax for consistent colour mapping across frames
    vmax = max(float(s['s_field'].max()) for s in snapshots)
    vmax = max(vmax, s_threshold + 0.01)
    norm = plt.Normalize(vmin=s_threshold, vmax=vmax)
    cmap = cm.hot_r

    def draw_frame(frame_idx):
        ax.cla()
        snap = snapshots[frame_idx]

        if _draw_obstacles_3d_static is not None and env_name in _HANDCODED_3D:
            _draw_obstacles_3d_static(ax, env_name)
        else:
            _draw_obstacles_3d_anim(ax, env_name, coll=coll, bounds=bounds)

        # CARM cloud — only cells above threshold (so s≈1 regions stay empty)
        sfield = snap['s_field']
        mask = sfield > s_threshold
        if mask.any():
            pts = grid_pts[mask]
            vals = sfield[mask]
            ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2],
                       c=vals, cmap=cmap, norm=norm,
                       s=18, alpha=0.55, marker='o',
                       depthshade=False, zorder=1.5,
                       edgecolors='none')

        # Start / goal
        ax.scatter([xs[0]], [xs[1]], [xs[2]], c='green', s=100, marker='o',
                   depthshade=False, zorder=10)
        ax.scatter([xg[0]], [xg[1]], [xg[2]], c='red', s=100, marker='^',
                   depthshade=False, zorder=10)

        # Tree edges
        for p, ch in snap['edges']:
            ax.plot([p[0], ch[0]], [p[1], ch[1]], [p[2], ch[2]],
                    color='#4FC3F7', lw=0.4, alpha=0.5)

        if snap['vertices']:
            va = np.array(snap['vertices'])
            ax.scatter(va[:, 0], va[:, 1], va[:, 2],
                       s=2, c='#90CAF9', alpha=0.4)

        path = snap['path']
        if path and len(path) > 1:
            pp = np.array(path)
            ax.plot(pp[:, 0], pp[:, 1], pp[:, 2], '#7B2FBE',
                    lw=3.0, zorder=8)

        ax.set_xlim(*x_lim); ax.set_ylim(*y_lim); ax.set_zlim(*z_lim)
        ax.set_xlabel('x'); ax.set_ylabel('y'); ax.set_zlabel('z')

        c = snap['c_best']
        cost_str = f'{c:.3f}' if np.isfinite(c) else '∞'
        smax = sfield.max()
        n_cloud = int(mask.sum())
        ax.set_title(f'{env_name} — RIT* + CARM  (s(x) cloud)\n'
                     f'Iter {snap["iteration"]+1}/{max_iterations}  |  '
                     f'Verts {len(snap["vertices"])}  |  Cost {cost_str}  |  '
                     f'max s={smax:.2f}  |  inflated cells {n_cloud}',
                     fontsize=10)

        azim = -45 + frame_idx * 1.5
        ax.view_init(elev=25, azim=azim)

    anim = FuncAnimation(fig, draw_frame, frames=len(snapshots),
                         interval=1000 // fps, blit=False)
    fname = os.path.join(GIFS_DIR, f'{save_prefix}_tree_growth_carm.gif')
    anim.save(fname, writer=PillowWriter(fps=fps))
    plt.close(fig)
    print(f'  → saved {fname}')

    # Final summary PNG
    fig2 = plt.figure(figsize=(10, 8))
    ax2 = fig2.add_subplot(111, projection='3d')
    final = snapshots[-1]
    if _draw_obstacles_3d_static is not None and env_name in _HANDCODED_3D:
        _draw_obstacles_3d_static(ax2, env_name)
    else:
        _draw_obstacles_3d_anim(ax2, env_name, coll=coll, bounds=bounds)
    sfield = final['s_field']
    mask = sfield > s_threshold
    if mask.any():
        pts = grid_pts[mask]
        sc = ax2.scatter(pts[:, 0], pts[:, 1], pts[:, 2],
                         c=sfield[mask], cmap=cmap, norm=norm,
                         s=22, alpha=0.6, edgecolors='none',
                         depthshade=False, zorder=1.5)
        cbar = fig2.colorbar(sc, ax=ax2, shrink=0.7, pad=0.08)
        cbar.set_label('CARM scale s(x)  (1 = no inflation)', fontsize=10)
    ax2.scatter([xs[0]], [xs[1]], [xs[2]], c='green', s=100, marker='o',
                depthshade=False, zorder=10)
    ax2.scatter([xg[0]], [xg[1]], [xg[2]], c='red', s=100, marker='^',
                depthshade=False, zorder=10)
    for p, ch in final['edges']:
        ax2.plot([p[0], ch[0]], [p[1], ch[1]], [p[2], ch[2]],
                 color='#4FC3F7', lw=0.5, alpha=0.6)
    if final['vertices']:
        va = np.array(final['vertices'])
        ax2.scatter(va[:, 0], va[:, 1], va[:, 2],
                    s=4, c='#90CAF9', alpha=0.5)
    if final['path'] and len(final['path']) > 1:
        pp = np.array(final['path'])
        ax2.plot(pp[:, 0], pp[:, 1], pp[:, 2], '#7B2FBE',
                 lw=3.0, zorder=8)
    ax2.set_xlim(*x_lim); ax2.set_ylim(*y_lim); ax2.set_zlim(*z_lim)
    ax2.set_xlabel('x'); ax2.set_ylabel('y'); ax2.set_zlabel('z')
    c = final['c_best']
    cost_str = f'{c:.3f}' if np.isfinite(c) else '∞'
    ax2.set_title(f'{env_name} — Final CARM cloud + RIT* tree + path\n'
                  f'Vertices: {len(final["vertices"])}  |  Cost: {cost_str}  '
                  f'|  max s={sfield.max():.2f}', fontsize=11)
    ax2.view_init(elev=25, azim=-45 + len(snapshots) * 1.5)
    png_path = os.path.join(IMAGES_DIR, f'{save_prefix}_tree_carm_final.png')
    fig2.savefig(png_path, dpi=160, bbox_inches='tight')
    plt.close(fig2)
    print(f'  → saved {png_path}')

    del anim, fig, fig2, snapshots, planner
    gc.collect()
    return fname, png_path


if __name__ == '__main__':
    print('=' * 60)
    print('  Riemannian Metric Visualization')
    print('=' * 60)

    # Individual 3-D surfaces (2D envs rendered as 3D metric surface)
    for env_name, env_fn in ENVS_2D.items():
        print(f'\n[3D Surface] {env_name}')
        safe = env_name.lower().replace(' ', '_')
        plot_3d_surface(env_name, env_fn, f'riemannian_{safe}')
        gc.collect()

    # Individual heatmaps with paths
    for env_name, env_fn in ENVS_2D.items():
        print(f'\n[Heatmap] {env_name}')
        safe = env_name.lower().replace(' ', '_')
        plot_heatmap_with_path(env_name, env_fn, f'riemannian_{safe}')
        gc.collect()

    # Combined views
    print('\n[Combined 3D Surfaces]')
    plot_combined_surfaces()
    gc.collect()

    print('\n[Combined Heatmaps]')
    plot_combined_heatmaps()
    gc.collect()

    # Animated GIFs: 2D tree growth on heatmap
    for env_name, env_fn in ENVS_2D.items():
        print(f'\n[Animation — 2D Heatmap] {env_name}')
        safe = env_name.lower().replace(' ', '_')
        animate_tree_growth(env_name, env_fn, f'riemannian_{safe}')
        gc.collect()

    # Animated GIFs: 2D tree growth on 3-D Riemannian surface
    for env_name, env_fn in ENVS_2D.items():
        print(f'\n[Animation — 2D on 3D Surface] {env_name}')
        safe = env_name.lower().replace(' ', '_')
        animate_3d_surface_tree(env_name, env_fn, f'riemannian_{safe}')
        gc.collect()

    # Animated GIFs: true 3D environment tree growth
    for env_name, env_fn in ENVS_3D.items():
        print(f'\n[Animation — 3D Env] {env_name}')
        safe = env_name.lower().replace(' ', '_')
        animate_3d_env(env_name, env_fn, f'riemannian_{safe}')
        gc.collect()

    print('\nAll Riemannian visualizations complete!')
