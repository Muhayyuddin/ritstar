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


def draw_obstacles_2d(ax, env_name):
    """Draw obstacles on a 2D axes."""
    obs = OBSTACLE_DEFS.get(env_name)
    if obs is None or obs['type'] == 'none':
        return
    if obs['type'] == 'circles':
        for c, r in obs['data']:
            ax.add_patch(Circle(c, r, fc='#333333', ec='white', lw=1.5, ls='--', alpha=0.8))
    elif obs['type'] == 'rects':
        for lo, hi in obs['data']:
            w, h = hi[0] - lo[0], hi[1] - lo[1]
            ax.add_patch(Rectangle(lo, w, h, fc='#333333', ec='white', lw=1.5, ls='--', alpha=0.8))
    elif obs['type'] == 'terrain_peaks':
        for cx, cy in obs['data']:
            ax.add_patch(Circle([cx, cy], 0.09, fc='#FFB74D', ec='#E65100',
                                lw=1.2, ls='--', alpha=0.4, zorder=2))


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

    draw_obstacles_2d(ax, env_name)

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

        draw_obstacles_2d(ax, env_name)

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

    # Build the animation
    fig, ax = plt.subplots(figsize=(8, 7))

    # Static background: heatmap
    ax.imshow(S, origin='lower', extent=extent, cmap='hot_r', aspect='equal')
    draw_obstacles_2d(ax, env_name)
    ax.plot(*xs, 'go', ms=12, zorder=10)
    ax.plot(*xg, 'r^', ms=12, zorder=10)
    ax.set_xlabel('x₁')
    ax.set_ylabel('x₂')

    # Dynamic elements
    edge_collection = LineCollection([], colors='#4FC3F7', linewidths=0.5,
                                     alpha=0.6, zorder=2)
    ax.add_collection(edge_collection)
    path_line, = ax.plot([], [], '#00FF00', lw=2.5, zorder=5)
    vertex_scatter = ax.scatter([], [], c='#90CAF9', s=4, zorder=3, alpha=0.7)
    title = ax.set_title('', fontsize=12)
    fig.tight_layout()

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

def _draw_obstacles_3d_anim(ax, env_name):
    """Draw 3D obstacles onto an Axes3D for animation frames."""
    if env_name == '3D Diagonal':
        boxes = [
            ([0.25, 0.0, 0.0], [0.35, 0.6, 1.0]),
            ([0.45, 0.4, 0.0], [0.55, 1.0, 1.0]),
            ([0.65, 0.0, 0.0], [0.75, 0.6, 0.6]),
            ([0.65, 0.0, 0.7], [0.75, 0.6, 1.0]),
        ]
        for lo, hi in boxes:
            lo, hi = np.array(lo), np.array(hi)
            for s, e in [
                ([lo[0],lo[1],lo[2]], [hi[0],lo[1],lo[2]]),
                ([lo[0],hi[1],lo[2]], [hi[0],hi[1],lo[2]]),
                ([lo[0],lo[1],hi[2]], [hi[0],lo[1],hi[2]]),
                ([lo[0],hi[1],hi[2]], [hi[0],hi[1],hi[2]]),
                ([lo[0],lo[1],lo[2]], [lo[0],hi[1],lo[2]]),
                ([hi[0],lo[1],lo[2]], [hi[0],hi[1],lo[2]]),
                ([lo[0],lo[1],hi[2]], [lo[0],hi[1],hi[2]]),
                ([hi[0],lo[1],hi[2]], [hi[0],hi[1],hi[2]]),
                ([lo[0],lo[1],lo[2]], [lo[0],lo[1],hi[2]]),
                ([hi[0],lo[1],lo[2]], [hi[0],lo[1],hi[2]]),
                ([lo[0],hi[1],lo[2]], [lo[0],hi[1],hi[2]]),
                ([hi[0],hi[1],lo[2]], [hi[0],hi[1],hi[2]]),
            ]:
                ax.plot([s[0],e[0]], [s[1],e[1]], [s[2],e[2]],
                        color='#3a3a3a', alpha=0.7, linewidth=0.8)
    elif env_name == '3D Spheres':
        offsets = [-0.35, 0.35]
        centres = [[x, y, z]
                    for x in offsets for y in offsets for z in offsets]
        centres.append([0.0, 0.0, 0.0])
        r = 0.22
        u = np.linspace(0, 2 * np.pi, 12)
        v = np.linspace(0, np.pi, 8)
        for c in centres:
            xs_ = c[0] + r * np.outer(np.cos(u), np.sin(v))
            ys_ = c[1] + r * np.outer(np.sin(u), np.sin(v))
            zs_ = c[2] + r * np.outer(np.ones_like(u), np.cos(v))
            ax.plot_wireframe(xs_, ys_, zs_, color='#3a3a3a',
                              alpha=0.25, linewidth=0.3)
    elif env_name == '3D Dense Lab':
        centres = [
            [-0.5, -0.5, -0.5], [0.2, -0.6, -0.3], [0.6, -0.4, -0.6],
            [-0.3, 0.0, 0.0], [0.3, 0.1, 0.1], [0.0, 0.5, 0.0],
            [-0.6, 0.3, 0.2], [0.6, 0.3, -0.1], [-0.4, 0.6, 0.5],
            [0.2, 0.7, 0.4], [0.5, 0.5, 0.6], [-0.1, -0.2, 0.6],
            [0.0, 0.0, 0.5], [-0.6, -0.3, 0.3], [0.5, -0.1, 0.4],
        ]
        r = 0.18
        u = np.linspace(0, 2 * np.pi, 14)
        v = np.linspace(0, np.pi, 10)
        for c in centres:
            xs_ = c[0] + r * np.outer(np.cos(u), np.sin(v))
            ys_ = c[1] + r * np.outer(np.sin(u), np.sin(v))
            zs_ = c[2] + r * np.outer(np.ones_like(u), np.cos(v))
            ax.plot_surface(xs_, ys_, zs_, color='#555555',
                            alpha=0.65, shade=True, linewidth=0)


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

    def draw_frame(frame_idx):
        ax.cla()
        snap = snapshots[frame_idx]

        # Draw obstacles
        _draw_obstacles_3d_anim(ax, env_name)

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
