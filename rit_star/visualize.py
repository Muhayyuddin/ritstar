"""
visualize.py — Plotting and animation utilities for RIT*.

All functions accept optional matplotlib ``Axes`` objects so they can
be composed into multi-panel figures.  When *ax* is ``None`` a new
figure is created automatically.
"""

from __future__ import annotations

import gc
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.patches import Rectangle, Circle
import matplotlib.animation as animation
from typing import Optional, List

from .geodesic import GeodesicComputer
from .informed_set import RiemannianInformedSet, EuclideanInformedSet


# ═══════════════════════════════════════════════════════════════════════
# Obstacle drawing helpers
# ═══════════════════════════════════════════════════════════════════════

def _draw_obstacles_2d(ax, env_name: str):
    """Draw obstacles for a named 2-D environment onto *ax*.

    Parameters
    ----------
    ax : matplotlib Axes
    env_name : str
        One of '2d_diagonal', '2d_obstacle', '2d_arm'.
    """
    if env_name == '2d_diagonal':
        rects = [
            ([0.35, 0.30], [0.45, 0.70]),
            ([0.55, 0.30], [0.65, 0.70]),
        ]
        for lo, hi in rects:
            w, h = hi[0] - lo[0], hi[1] - lo[1]
            ax.add_patch(Rectangle(lo, w, h, fc='black', ec='black', alpha=0.7, zorder=6))

    elif env_name == '2d_obstacle':
        circles = [
            ([0.30, 0.35], 0.08),
            ([0.30, 0.65], 0.08),
            ([0.50, 0.45], 0.09),
            ([0.50, 0.75], 0.09),
            ([0.70, 0.40], 0.08),
            ([0.70, 0.60], 0.08),
        ]
        for c, r in circles:
            ax.add_patch(Circle(c, r, fc='black', ec='black', alpha=0.7, zorder=6))

    elif env_name == '2d_narrow_passage':
        gap_y_lo, gap_y_hi = 0.47, 0.53
        rects = [
            ([0.48, 0.00], [0.52, gap_y_lo]),
            ([0.48, gap_y_hi], [0.52, 1.00]),
            ([0.30, 0.15], [0.42, 0.35]),
            ([0.30, 0.65], [0.42, 0.85]),
            ([0.58, 0.15], [0.70, 0.35]),
            ([0.58, 0.65], [0.70, 0.85]),
        ]
        for lo, hi in rects:
            w, h = hi[0] - lo[0], hi[1] - lo[1]
            ax.add_patch(Rectangle(lo, w, h, fc='black', ec='black', alpha=0.7, zorder=6))

    elif env_name == '2d_maze':
        rects = [
            ([0.00, 0.22], [0.70, 0.30]),
            ([0.30, 0.46], [1.00, 0.54]),
            ([0.00, 0.70], [0.70, 0.78]),
        ]
        for lo, hi in rects:
            w, h = hi[0] - lo[0], hi[1] - lo[1]
            ax.add_patch(Rectangle(lo, w, h, fc='black', ec='black', alpha=0.7, zorder=6))

    elif env_name == '2d_bug_trap':
        rects = [
            ([0.15, 0.20], [0.60, 0.28]),
            ([0.15, 0.72], [0.60, 0.80]),
            ([0.15, 0.28], [0.23, 0.72]),
            ([0.52, 0.28], [0.60, 0.46]),
            ([0.52, 0.54], [0.60, 0.72]),
        ]
        for lo, hi in rects:
            w, h = hi[0] - lo[0], hi[1] - lo[1]
            ax.add_patch(Rectangle(lo, w, h, fc='black', ec='black', alpha=0.7, zorder=6))

    elif env_name == '2d_random_forest':
        rng = np.random.default_rng(12345)
        centres = []
        _xs = np.array([0.05, 0.05])
        _xg = np.array([0.95, 0.95])
        for _ in range(125):
            if len(centres) >= 25:
                break
            c = rng.uniform(0.1, 0.9, size=2)
            if np.linalg.norm(c - _xs) < 0.12:
                continue
            if np.linalg.norm(c - _xg) < 0.12:
                continue
            centres.append(c)
        for c in centres[:25]:
            ax.add_patch(Circle(c, 0.04, fc='black', ec='black', alpha=0.7, zorder=6))

    elif env_name == '2d_random_world':
        rng = np.random.default_rng(2015_04)
        n_obs = 35
        x_start = np.array([-0.1, -0.1])
        x_goal  = np.array([ 0.4,  0.4])
        clr = 0.06
        rects = []
        for _ in range(n_obs * 50):
            if len(rects) >= n_obs:
                break
            ax_ = rng.uniform(-0.5, 0.5)
            ay_ = rng.uniform(-0.5, 0.5)
            w = rng.uniform(0.1, 0.2)
            h = rng.uniform(0.1, 0.2)
            lo = np.array([ax_, ay_])
            hi = np.array([ax_ + w, ay_ + h])
            if (lo[0] <= x_start[0] + clr and hi[0] >= x_start[0] - clr and
                lo[1] <= x_start[1] + clr and hi[1] >= x_start[1] - clr):
                continue
            if (lo[0] <= x_goal[0] + clr and hi[0] >= x_goal[0] - clr and
                lo[1] <= x_goal[1] + clr and hi[1] >= x_goal[1] - clr):
                continue
            rects.append((lo, hi))

        # Mirror env_2d_random_world(): eliminate an ultra-narrow slit by
        # creating a slight overlap between the tightest vertical pair.
        min_gap = np.inf
        best_pair = None  # (upper_idx, gap)
        for i in range(len(rects)):
            lo_i, hi_i = rects[i]
            for j in range(i + 1, len(rects)):
                lo_j, hi_j = rects[j]
                x_overlap = min(hi_i[0], hi_j[0]) - max(lo_i[0], lo_j[0])
                if x_overlap < 0.04:
                    continue
                if hi_i[1] <= lo_j[1]:
                    gap = lo_j[1] - hi_i[1]
                    upper_idx = j
                elif hi_j[1] <= lo_i[1]:
                    gap = lo_i[1] - hi_j[1]
                    upper_idx = i
                else:
                    continue
                if gap < min_gap:
                    min_gap = gap
                    best_pair = (upper_idx, gap)

        if best_pair is not None and min_gap < 0.05:
            upper_idx, gap = best_pair
            target_gap = -0.02
            dy = target_gap - gap
            if abs(dy) > 1e-12:
                lo_u, hi_u = rects[upper_idx]
                new_lo = lo_u + np.array([0.0, dy])
                new_hi = hi_u + np.array([0.0, dy])
                s_ok = not (new_lo[0] <= x_start[0] + clr and new_hi[0] >= x_start[0] - clr and
                            new_lo[1] <= x_start[1] + clr and new_hi[1] >= x_start[1] - clr)
                g_ok = not (new_lo[0] <= x_goal[0] + clr and new_hi[0] >= x_goal[0] - clr and
                            new_lo[1] <= x_goal[1] + clr and new_hi[1] >= x_goal[1] - clr)
                if s_ok and g_ok:
                    rects[upper_idx] = (new_lo, new_hi)

        for lo, hi in rects[:n_obs]:
            w, h = hi[0] - lo[0], hi[1] - lo[1]
            ax.add_patch(Rectangle(lo, w, h, fc='#404040', ec='#303030', alpha=0.85, zorder=6))

    elif env_name == '2d_dividing_wall':
        wall_x_lo, wall_x_hi = 0.47, 0.53
        wall_segments = [
            (np.array([wall_x_lo, 0.00]), np.array([wall_x_hi, 0.10])),
            (np.array([wall_x_lo, 0.13]), np.array([wall_x_hi, 0.85])),
            (np.array([wall_x_lo, 0.88]), np.array([wall_x_hi, 1.00])),
        ]
        flanking = [
            (np.array([0.25, 0.70]), np.array([0.35, 0.85])),
            (np.array([0.65, 0.15]), np.array([0.75, 0.30])),
        ]
        for lo, hi in wall_segments:
            w, h = hi[0] - lo[0], hi[1] - lo[1]
            ax.add_patch(Rectangle(lo, w, h, fc='#333333', ec='#1a1a1a', alpha=0.9, zorder=6))
        for lo, hi in flanking:
            w, h = hi[0] - lo[0], hi[1] - lo[1]
            ax.add_patch(Rectangle(lo, w, h, fc='#404040', ec='#303030', alpha=0.85, zorder=6))

    elif env_name == '2d_terrain':
        # Terrain peaks at sin²(3πx)·sin²(3πy) maxima — high-cost zones (no hard obstacles)
        for cx in [1/6, 0.5, 5/6]:
            for cy in [1/6, 0.5, 5/6]:
                ax.add_patch(Circle([cx, cy], 0.09,
                                    fc='#909090', ec='#555555',
                                    lw=1.2, ls='--', alpha=0.55, zorder=6))


# ═══════════════════════════════════════════════════════════════════════
# Informed-set comparison
# ═══════════════════════════════════════════════════════════════════════

def plot_informed_set_comparison_2d(x_start, x_goal, c_best,
                                     geodesic_computer: GeodesicComputer,
                                     bounds, env_name: str = '',
                                     ax=None, resolution: int = 120):
    """Side-by-side overlay of I_euclid (gray) and I_R (purple).

    Parameters
    ----------
    x_start, x_goal : (2,) arrays
    c_best : float
    geodesic_computer : GeodesicComputer
    bounds : list of (lo, hi)
    env_name : str — for obstacle drawing
    ax : matplotlib Axes or None
    resolution : int — grid resolution for masks

    Returns
    -------
    ax : matplotlib Axes

    Notes
    -----
    Visualises Theorem 1: the Riemannian informed set is always a
    strict subset of the Euclidean one.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 6))

    eis = EuclideanInformedSet(x_start, x_goal, c_best, bounds=bounds)
    ris = RiemannianInformedSet(x_start, x_goal, c_best, geodesic_computer,
                                 bounds=bounds)

    eis.visualize_2d(ax, resolution=resolution, color='gray', alpha=0.2)
    ris.visualize_2d(ax, resolution=resolution, color='purple', alpha=0.25)

    _draw_obstacles_2d(ax, env_name)

    ax.plot(*x_start, 'go', ms=10, zorder=5, label='start')
    ax.plot(*x_goal, 'r^', ms=10, zorder=5, label='goal')

    # Volume annotation
    v_e = eis.volume_estimate(5000)
    v_r = ris.volume_estimate(5000)
    if v_e > 1e-12:
        pct = (1.0 - v_r / v_e) * 100
        ax.set_title(f'Informed set comparison  (I_R is {pct:.0f}% smaller)')
    ax.legend(loc='upper left', fontsize=8)
    ax.set_xlabel('x₁')
    ax.set_ylabel('x₂')
    ax.set_aspect('equal')
    return ax


# ═══════════════════════════════════════════════════════════════════════
# Cost convergence
# ═══════════════════════════════════════════════════════════════════════

def plot_cost_convergence(stats_rit: list, stats_irrt: list,
                          stats_bit: list, ax=None):
    """Cost-vs-samples convergence plot with uncertainty bands.

    Parameters
    ----------
    stats_rit, stats_irrt, stats_bit : list of list-of-dict
        Each outer list holds results from independent trials.
    ax : matplotlib Axes or None

    Returns
    -------
    ax

    Notes
    -----
    Plots median ± 25/75 percentile bands for each planner.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))

    def _band(trials, color, label):
        # trials is a list of stat-lists (one per trial)
        max_len = max(len(t) for t in trials)
        mat = np.full((len(trials), max_len), np.nan)
        longest_trial = max(trials, key=len)
        xs = np.array([r['n_samples_total'] for r in longest_trial])
        for i, trial in enumerate(trials):
            for j, row in enumerate(trial):
                mat[i, j] = row['c_best']
        med = np.nanmedian(mat, axis=0)
        q25 = np.nanpercentile(mat, 25, axis=0)
        q75 = np.nanpercentile(mat, 75, axis=0)
        ax.plot(xs, med, color=color, label=label, lw=2)
        ax.fill_between(xs, q25, q75, color=color, alpha=0.15)

    _band(stats_rit, 'purple', 'RIT*')
    _band(stats_irrt, 'steelblue', 'Informed RRT*')
    _band(stats_bit, 'darkorange', 'BIT*')

    # Estimate c_opt as best known final cost
    all_final = []
    for trials in [stats_rit, stats_irrt, stats_bit]:
        for t in trials:
            if t:
                all_final.append(t[-1]['c_best'])
    if all_final:
        c_opt = min(all_final)
        if np.isfinite(c_opt):
            ax.axhline(c_opt, ls='--', color='gray', lw=1, label=f'c* ≈ {c_opt:.3f}')

    ax.set_xlabel('Number of samples')
    ax.set_ylabel('Best cost (c_best)')
    ax.set_title('Cost convergence')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    return ax


# ═══════════════════════════════════════════════════════════════════════
# Tree animation (2-D)
# ═══════════════════════════════════════════════════════════════════════

def animate_planning_2d(vertices_per_iter: List[list],
                        path_per_iter: List[list],
                        x_start, x_goal, bounds, env_name: str,
                        filename: str = 'rit_star_2d.gif',
                        interval: int = 200):
    """Create a GIF of the tree growing iteration by iteration.

    Parameters
    ----------
    vertices_per_iter : list of lists of (2,) arrays — tree vertices
        at each saved iteration.
    path_per_iter : list of lists of (2,) arrays — best path (or empty).
    x_start, x_goal : (2,) arrays
    bounds : list of (lo, hi)
    env_name : str
    filename : str
    interval : int — ms between frames.

    Notes
    -----
    Saves as GIF using ``matplotlib.animation.PillowWriter``.
    """
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

    def _frame(i):
        ax.cla()
        ax.set_xlim(bounds[0])
        ax.set_ylim(bounds[1])
        ax.set_aspect('equal')
        _draw_obstacles_2d(ax, env_name)
        ax.plot(*x_start, 'go', ms=10, zorder=5)
        ax.plot(*x_goal, 'r^', ms=10, zorder=5)

        verts = vertices_per_iter[min(i, len(vertices_per_iter) - 1)]
        if verts:
            xs = [v[0] for v in verts]
            ys = [v[1] for v in verts]
            ax.scatter(xs, ys, s=3, c='gray', alpha=0.5)

        path = path_per_iter[min(i, len(path_per_iter) - 1)]
        if path and len(path) > 1:
            px = [p[0] for p in path]
            py = [p[1] for p in path]
            ax.plot(px, py, 'purple', lw=2.5, zorder=4)

        ax.set_title(f'RIT* — iteration {i}')

    n_frames = len(vertices_per_iter)
    anim = animation.FuncAnimation(fig, _frame, frames=n_frames,
                                    interval=interval)
    anim.save(filename, writer='pillow')
    plt.close(fig)
    del anim
    gc.collect()
    print(f'  Animation saved → {filename}')


# ═══════════════════════════════════════════════════════════════════════
# Volume-ratio heatmap
# ═══════════════════════════════════════════════════════════════════════

def plot_volume_heatmap(volume_ratios: np.ndarray,
                        kappa_values: np.ndarray,
                        c_ratio_values: np.ndarray,
                        ax=None):
    """2-D heatmap of Vol(I_R)/Vol(I_euclid).

    Parameters
    ----------
    volume_ratios : (n_kappa, n_c) array
    kappa_values : (n_kappa,) array — condition numbers of G.
    c_ratio_values : (n_c,) array — c_best / c_min ratios.
    ax : matplotlib Axes or None

    Returns
    -------
    ax

    Notes
    -----
    Empirically validates Theorem 1: the ratio is always < 1 when G ≠ I.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 5))

    im = ax.imshow(volume_ratios, origin='lower', aspect='auto',
                    cmap='viridis', vmin=0, vmax=1,
                    extent=[c_ratio_values[0], c_ratio_values[-1],
                            kappa_values[0], kappa_values[-1]])
    plt.colorbar(im, ax=ax, label='Vol(I_R) / Vol(I_euclid)')
    ax.set_xlabel('c_best / c_min')
    ax.set_ylabel('κ = λ_max / λ_min')
    ax.set_title('Volume ratio heatmap (Theorem 1)')
    return ax


# ═══════════════════════════════════════════════════════════════════════
# 3-D tree visualisation
# ═══════════════════════════════════════════════════════════════════════

def plot_3d_tree(tree_vertices: List[np.ndarray],
                 tree_edges: List[tuple],
                 path: List[np.ndarray],
                 obstacles: list,
                 ax=None):
    """3-D scatter + line plot of the search tree and solution path.

    Parameters
    ----------
    tree_vertices : list of (3,) arrays
    tree_edges : list of (parent_idx, child_idx) pairs
    path : list of (3,) arrays (best path, may be empty)
    obstacles : list of (centre, radius) for spheres, or
                (lo, hi) for boxes.
    ax : Axes3D or None

    Returns
    -------
    ax

    Notes
    -----
    Path is drawn in thick purple; tree edges in thin gray.
    """
    if ax is None:
        fig = plt.figure(figsize=(8, 7))
        ax = fig.add_subplot(111, projection='3d')

    verts = np.array(tree_vertices)
    if len(verts) > 0:
        ax.scatter(verts[:, 0], verts[:, 1], verts[:, 2],
                   s=2, c='gray', alpha=0.4)

    # Edges
    for i, j in tree_edges:
        p1, p2 = tree_vertices[i], tree_vertices[j]
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]],
                'gray', lw=0.3, alpha=0.3)

    # Path
    if path and len(path) > 1:
        pp = np.array(path)
        ax.plot(pp[:, 0], pp[:, 1], pp[:, 2], 'purple', lw=3, zorder=5)

    # Obstacles: draw wireframe spheres or boxes
    for obs in obstacles:
        if len(obs) == 2:
            a, b = obs
            a, b = np.asarray(a), np.asarray(b)
            if np.isscalar(b) or b.ndim == 0:
                # Sphere: (centre, radius)
                c, r = a, float(b)
                if len(c) == 3:
                    u = np.linspace(0, 2 * np.pi, 15)
                    v = np.linspace(0, np.pi, 10)
                    xs = c[0] + r * np.outer(np.cos(u), np.sin(v))
                    ys = c[1] + r * np.outer(np.sin(u), np.sin(v))
                    zs = c[2] + r * np.outer(np.ones_like(u), np.cos(v))
                    ax.plot_wireframe(xs, ys, zs, color='black', alpha=0.15,
                                      linewidth=0.4)
            elif len(a) == 3 and len(b) == 3:
                # Box: (lo, hi)
                lo, hi = a, b
                # Draw 12 edges of the box
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
                            color='black', alpha=0.4, linewidth=0.8)

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.set_title('3-D search tree')
    return ax


# ═══════════════════════════════════════════════════════════════════════
# Anisotropy speedup plot
# ═══════════════════════════════════════════════════════════════════════

def plot_anisotropy_speedup(kappa_values: np.ndarray,
                            speedup_factors: np.ndarray,
                            dim: int = 2,
                            ax=None):
    """Scatter of measured speedup vs κ with theoretical curve overlay.

    Parameters
    ----------
    kappa_values : (N,) array — anisotropy ratios.
    speedup_factors : (N,) array — measured speedup (iterations ratio).
    dim : int — dimensionality (for theoretical curve κ^(1/d)).
    ax : matplotlib Axes or None

    Returns
    -------
    ax

    Notes
    -----
    Validates Theorem 3: convergence speedup scales as κ^(1/d).
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 5))

    ax.scatter(kappa_values, speedup_factors, c='purple', s=50,
               zorder=4, label='measured')

    ks = np.linspace(1, kappa_values.max(), 100)
    theoretical = ks ** (1.0 / dim)
    ax.plot(ks, theoretical, 'k--', lw=1.5,
            label=f'κ^(1/{dim}) (Theorem 3)')

    ax.set_xlabel('κ = λ_max(G) / λ_min(G)')
    ax.set_ylabel('Speedup factor')
    ax.set_title('Anisotropy speedup (Theorem 3)')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    return ax


# ═══════════════════════════════════════════════════════════════════════
# Static 2-D tree + path plot
# ═══════════════════════════════════════════════════════════════════════

def plot_tree_2d(vertices: list, path: list,
                 x_start, x_goal, bounds, env_name: str = '',
                 ax=None, title: str = ''):
    """Plot the final 2-D tree and solution path.

    Parameters
    ----------
    vertices : list of Node objects (must have .x, .parent attributes)
    path : list of (2,) arrays
    x_start, x_goal : (2,) arrays
    bounds : list of (lo, hi)
    env_name : str
    ax : matplotlib Axes or None
    title : str

    Returns
    -------
    ax
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 6))

    ax.set_xlim(bounds[0])
    ax.set_ylim(bounds[1])
    ax.set_aspect('equal')
    _draw_obstacles_2d(ax, env_name)

    # Tree edges
    for v in vertices:
        if v.parent is not None:
            ax.plot([v.x[0], v.parent.x[0]], [v.x[1], v.parent.x[1]],
                    'gray', lw=0.3, alpha=0.4)

    # Path
    if path and len(path) > 1:
        px = [p[0] for p in path]
        py = [p[1] for p in path]
        ax.plot(px, py, 'purple', lw=2.5, zorder=4)

    ax.plot(*x_start, 'go', ms=10, zorder=5)
    ax.plot(*x_goal, 'r^', ms=10, zorder=5)
    if title:
        ax.set_title(title)
    ax.set_xlabel('x₁')
    ax.set_ylabel('x₂')
    return ax


# ═══════════════════════════════════════════════════════════════════════
# Animated arm GIF
# ═══════════════════════════════════════════════════════════════════════

def animate_arm_path(path: List[np.ndarray],
                     L1: float, L2: float,
                     obstacles: list,
                     filename: str = 'arm_motion.gif',
                     interval: int = 150):
    """Create a GIF of a 2-joint planar arm following *path*.

    Parameters
    ----------
    path : list of (2,) arrays — joint-angle waypoints.
    L1, L2 : float — link lengths.
    obstacles : list of (centre, radius) workspace obstacles.
    filename : str
    interval : int — ms between frames.
    """
    if not path or len(path) < 2:
        return

    fig, ax = plt.subplots(figsize=(6, 6))

    def _frame(i):
        ax.cla()
        ax.set_xlim(-1.1, 1.1)
        ax.set_ylim(-1.1, 1.1)
        ax.set_aspect('equal')
        for obs_c, obs_r in obstacles:
            ax.add_patch(Circle(obs_c, obs_r, fc='black', alpha=0.6))
        theta = path[i]
        elbow = np.array([L1 * np.cos(theta[0]), L1 * np.sin(theta[0])])
        ee = elbow + np.array([L2 * np.cos(theta[0] + theta[1]),
                               L2 * np.sin(theta[0] + theta[1])])
        ax.plot([0, elbow[0], ee[0]], [0, elbow[1], ee[1]],
                'o-', color='purple', lw=3, ms=6)
        ax.set_title(f'Arm waypoint {i + 1}/{len(path)}')

    anim = animation.FuncAnimation(fig, _frame, frames=len(path),
                                    interval=interval)
    anim.save(filename, writer='pillow')
    plt.close(fig)
    del anim
    gc.collect()
    print(f'  Animation saved → {filename}')


# ═══════════════════════════════════════════════════════════════════════
# Animated convergence GIF
# ═══════════════════════════════════════════════════════════════════════

def animate_convergence(stats_rit: list, stats_irrt: list,
                        stats_bit: list,
                        filename: str = 'convergence.gif',
                        interval: int = 120):
    """Create a GIF showing convergence curves building up over iterations.

    Parameters
    ----------
    stats_rit, stats_irrt, stats_bit : list of list-of-dicts
        Per-trial stats from each planner.
    filename : str
    interval : int — ms between frames.
    """
    # Compute median curves
    def _median_curve(all_stats):
        if not all_stats or not all_stats[0]:
            return [], []
        longest = max(all_stats, key=len)
        n_it = len(longest)
        iters = [longest[i]['n_samples_total'] for i in range(n_it)]
        mat = np.full((len(all_stats), n_it), np.nan)
        for i, trial in enumerate(all_stats):
            for j, s in enumerate(trial):
                mat[i, j] = s['c_best']
        med = np.nanmedian(mat, axis=0)
        return iters, med

    it_r, med_r = _median_curve(stats_rit)
    it_i, med_i = _median_curve(stats_irrt)
    it_b, med_b = _median_curve(stats_bit)

    if not it_r:
        return

    n_frames = min(50, len(it_r))
    step = max(1, len(it_r) // n_frames)
    frame_idxs = list(range(0, len(it_r), step))
    if frame_idxs[-1] != len(it_r) - 1:
        frame_idxs.append(len(it_r) - 1)

    all_costs = np.concatenate([med_r, med_i, med_b])
    finite_costs = all_costs[np.isfinite(all_costs)]
    y_max = finite_costs.max() * 1.1 if len(finite_costs) > 0 else 10
    y_min = finite_costs.min() * 0.95 if len(finite_costs) > 0 else 0

    fig, ax = plt.subplots(figsize=(8, 5))

    def _frame(fi):
        ax.cla()
        idx = frame_idxs[fi]
        ax.plot(it_r[:idx + 1], med_r[:idx + 1], 'purple', lw=2, label='RIT*')
        ax.plot(it_i[:idx + 1], med_i[:idx + 1], 'green', lw=2, label='Informed RRT*')
        ax.plot(it_b[:idx + 1], med_b[:idx + 1], 'orange', lw=2, label='BIT*')
        ax.set_xlim(0, it_r[-1])
        ax.set_ylim(y_min, y_max)
        ax.set_xlabel('Total samples')
        ax.set_ylabel('Cost (median)')
        ax.set_title('Convergence comparison')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    anim = animation.FuncAnimation(fig, _frame, frames=len(frame_idxs),
                                    interval=interval)
    anim.save(filename, writer='pillow')
    plt.close(fig)
    del anim
    gc.collect()
    print(f'  Animation saved → {filename}')


# ═══════════════════════════════════════════════════════════════════════
# Animated 3-D rotation GIF
# ═══════════════════════════════════════════════════════════════════════
# Animated 3-D tree-growing + rotation GIF
# ═══════════════════════════════════════════════════════════════════════

def _draw_obstacles_3d(ax, obstacles):
    """Render sphere and box obstacles onto an Axes3D."""
    for obs in obstacles:
        if len(obs) == 2:
            a, b = obs
            a, b = np.asarray(a), np.asarray(b)
            if np.isscalar(b) or b.ndim == 0:
                c, r = a, float(b)
                if len(c) == 3:
                    u = np.linspace(0, 2 * np.pi, 15)
                    v = np.linspace(0, np.pi, 10)
                    xs_ = c[0] + r * np.outer(np.cos(u), np.sin(v))
                    ys_ = c[1] + r * np.outer(np.sin(u), np.sin(v))
                    zs_ = c[2] + r * np.outer(np.ones_like(u), np.cos(v))
                    ax.plot_wireframe(xs_, ys_, zs_, color='black',
                                      alpha=0.15, linewidth=0.4)
            elif len(a) == 3 and len(b) == 3:
                lo, hi = a, b
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
                            color='black', alpha=0.4, linewidth=0.8)


def animate_planning_3d(verts_per_iter: List[List[np.ndarray]],
                        edges_per_iter: List[List[tuple]],
                        path_per_iter: List[List[np.ndarray]],
                        obstacles: list,
                        x_start: np.ndarray,
                        x_goal: np.ndarray,
                        filename: str = '3d_growing.gif',
                        n_grow_frames: int = 40,
                        n_rotate_frames: int = 18,
                        interval: int = 200):
    """Create a GIF of the 3-D tree growing, followed by a slow rotation.

    Parameters
    ----------
    verts_per_iter : list of lists of (3,) arrays — vertices per snapshot.
    edges_per_iter : list of lists of (i,j) tuples — edges per snapshot.
    path_per_iter : list of lists of (3,) arrays — best path per snapshot.
    obstacles : sphere/box obstacles for rendering.
    x_start, x_goal : (3,) arrays.
    filename : str
    n_grow_frames : int — number of growth frames to subsample from
        the iterations (default 40).
    n_rotate_frames : int — rotation frames appended after growth
        (default 18 → 20° steps).
    interval : int — ms between frames.
    """
    n_iters = len(verts_per_iter)
    if n_iters == 0:
        return

    # Subsample iterations for growth phase
    step = max(1, n_iters // n_grow_frames)
    grow_idxs = list(range(0, n_iters, step))
    if grow_idxs[-1] != n_iters - 1:
        grow_idxs.append(n_iters - 1)

    total_frames = len(grow_idxs) + n_rotate_frames

    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection='3d')

    def _frame(frame_i):
        ax.cla()
        _draw_obstacles_3d(ax, obstacles)
        ax.scatter(*x_start, c='green', s=60, zorder=5, marker='o')
        ax.scatter(*x_goal, c='red', s=60, zorder=5, marker='^')

        if frame_i < len(grow_idxs):
            # Growth phase
            it = grow_idxs[frame_i]
            verts = verts_per_iter[it]
            edges = edges_per_iter[it]
            path = path_per_iter[it]
            azim = 30
        else:
            # Rotation phase — show final tree and spin
            verts = verts_per_iter[-1]
            edges = edges_per_iter[-1]
            path = path_per_iter[-1]
            rot_i = frame_i - len(grow_idxs)
            azim = 30 + rot_i * (360 / n_rotate_frames)

        if verts:
            va = np.array(verts)
            ax.scatter(va[:, 0], va[:, 1], va[:, 2],
                       s=2, c='gray', alpha=0.4)
        for i, j in edges:
            p1, p2 = verts[i], verts[j]
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]],
                    'gray', lw=0.3, alpha=0.3)
        if path and len(path) > 1:
            pp = np.array(path)
            ax.plot(pp[:, 0], pp[:, 1], pp[:, 2], 'purple', lw=3, zorder=5)

        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        ax.view_init(elev=25, azim=azim)
        if frame_i < len(grow_idxs):
            it = grow_idxs[frame_i]
            ax.set_title(f'RIT* 3-D — iteration {it}')
        else:
            ax.set_title('RIT* 3-D — final solution')

    anim = animation.FuncAnimation(fig, _frame, frames=total_frames,
                                    interval=interval)
    anim.save(filename, writer='pillow')
    plt.close(fig)
    del anim
    gc.collect()
    print(f'  Animation saved → {filename}')


# ═══════════════════════════════════════════════════════════════════════
# Animated speedup accumulation GIF
# ═══════════════════════════════════════════════════════════════════════

def animate_speedup(kappa_values: np.ndarray,
                    speedup_factors: np.ndarray,
                    dim: int = 2,
                    filename: str = 'speedup.gif',
                    interval: int = 400):
    """Create a GIF showing speedup data points appearing one by one.

    Parameters
    ----------
    kappa_values, speedup_factors : arrays from experiment 3.
    dim : int
    filename : str
    interval : int — ms between frames.
    """
    fig, ax = plt.subplots(figsize=(7, 5))

    ks = np.linspace(1, kappa_values.max(), 100)
    theoretical = ks ** (1.0 / dim)

    def _frame(i):
        ax.cla()
        n = i + 1
        ax.scatter(kappa_values[:n], speedup_factors[:n], c='purple',
                   s=60, zorder=4, label='measured')
        ax.plot(ks, theoretical, 'k--', lw=1.5,
                label=f'κ^(1/{dim}) (Theorem 3)')
        ax.set_xlim(0, kappa_values.max() * 1.1)
        y_max = max(theoretical.max(), speedup_factors[:n].max()) * 1.2
        ax.set_ylim(0, max(y_max, 2))
        ax.set_xlabel('κ = λ_max(G) / λ_min(G)')
        ax.set_ylabel('Speedup factor')
        ax.set_title(f'Anisotropy speedup — κ={kappa_values[i]:.0f}')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    anim = animation.FuncAnimation(fig, _frame, frames=len(kappa_values),
                                    interval=interval)
    anim.save(filename, writer='pillow')
    plt.close(fig)
    del anim
    gc.collect()
    print(f'  Animation saved → {filename}')


# ═══════════════════════════════════════════════════════════════════════
# Theorem 1: Volume ratio validation plot
# ═══════════════════════════════════════════════════════════════════════

def plot_volume_ratio_validation(analytical: np.ndarray,
                                 mc_estimates: np.ndarray,
                                 kappa_vals: np.ndarray,
                                 labels: list = None,
                                 ax=None):
    """Scatter plot: x = analytical prediction, y = MC estimate.

    Perfect agreement → points on the diagonal. Validates Theorem 1.

    Parameters
    ----------
    analytical : (N,) array of analytical volume ratios.
    mc_estimates : (N,) array of MC-estimated volume ratios.
    kappa_vals : array of kappa values used (for coloring).
    labels : list of str labels per point.
    ax : optional matplotlib Axes.

    Returns
    -------
    fig : matplotlib Figure
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 6))
    else:
        fig = ax.figure

    ax.scatter(analytical, mc_estimates, c='purple', s=60, zorder=4,
               edgecolors='black', linewidths=0.5)
    lo = min(analytical.min(), mc_estimates.min(), 0.0)
    hi = max(analytical.max(), mc_estimates.max(), 1.0)
    ax.plot([lo, hi], [lo, hi], 'k--', lw=1.5, label='perfect agreement')

    if labels is not None:
        for i, lbl in enumerate(labels):
            ax.annotate(lbl, (analytical[i], mc_estimates[i]),
                        fontsize=7, textcoords='offset points',
                        xytext=(5, 5))

    ax.set_xlabel('Analytical Vol(I_R)/Vol(I_E)  (Theorem 1)')
    ax.set_ylabel('Monte Carlo Vol(I_R)/Vol(I_E)')
    ax.set_title('Volume Ratio Validation')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


# ═══════════════════════════════════════════════════════════════════════
# Theorem 3: Convergence rate separation plot
# ═══════════════════════════════════════════════════════════════════════

def plot_convergence_rate_separation(results: dict,
                                     kappa_values: list,
                                     dims: list,
                                     ax=None):
    """Three-panel figure for Theorem 3 validation.

    (a) Empirical speedup vs kappa with theoretical kappa^(1/d) overlay.
    (b) Heatmap of (kappa, dim) → measured speedup.

    Parameters
    ----------
    results : dict mapping (kappa, dim) -> dict with 'speedup' key.
    kappa_values : list of float.
    dims : list of int.

    Returns
    -------
    fig : matplotlib Figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Panel (a): normalized speedup vs kappa per dimension
    ax0 = axes[0]
    ks = np.linspace(1, max(kappa_values), 100)
    colors = ['purple', 'teal', 'orange', 'red', 'blue']
    for di, d in enumerate(dims):
        speedups = [results.get((k, d), {}).get('speedup_normalized',
                    results.get((k, d), {}).get('speedup', 1.0))
                    for k in kappa_values]
        c = colors[di % len(colors)]
        ax0.scatter(kappa_values, speedups, c=c, s=60, zorder=4,
                    label=f'd={d} (measured)')
        theory = ks ** (1.0 / d)
        ax0.plot(ks, theory, '--', color=c, lw=1.5,
                 label=f'κ^(1/{d}) (theory)')
    ax0.set_xlabel('κ = λ_max(G) / λ_min(G)')
    ax0.set_ylabel('Normalized Speedup (metric effect)')
    ax0.set_title('Convergence Rate Separation (Theorem 3)')
    ax0.legend(fontsize=8)
    ax0.grid(True, alpha=0.3)

    # Panel (b): heatmap of normalized speedup
    ax1 = axes[1]
    heatmap = np.ones((len(dims), len(kappa_values)))
    for di, d in enumerate(dims):
        for ki, k in enumerate(kappa_values):
            heatmap[di, ki] = results.get((k, d), {}).get('speedup_normalized',
                              results.get((k, d), {}).get('speedup', 1.0))
    im = ax1.imshow(heatmap, aspect='auto', cmap='Purples',
                    origin='lower')
    ax1.set_xticks(range(len(kappa_values)))
    ax1.set_xticklabels([f'{k:.0f}' for k in kappa_values])
    ax1.set_yticks(range(len(dims)))
    ax1.set_yticklabels([f'd={d}' for d in dims])
    ax1.set_xlabel('κ')
    ax1.set_title('Normalized Speedup (κ, d)')
    for di in range(len(dims)):
        for ki in range(len(kappa_values)):
            ax1.text(ki, di, f'{heatmap[di, ki]:.2f}',
                     ha='center', va='center', fontsize=8)
    fig.colorbar(im, ax=ax1, label='Speedup factor (normalized)')

    fig.tight_layout()
    return fig


# ═══════════════════════════════════════════════════════════════════════
# Theorem 3: Sample efficiency bar chart
# ═══════════════════════════════════════════════════════════════════════

def plot_sample_efficiency(n_E: np.ndarray, n_R: np.ndarray,
                           kappa_values: list, dims: list,
                           ax=None):
    """Bar chart: samples-to-threshold for Euclidean vs Riemannian.

    Parameters
    ----------
    n_E, n_R : arrays — samples to reach threshold for BIT* and RIT*.
    kappa_values : list of float.
    dims : list of int.

    Returns
    -------
    fig : matplotlib Figure
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 5))
    else:
        fig = ax.figure

    n_groups = len(n_E)
    x = np.arange(n_groups)
    width = 0.35

    ax.bar(x - width / 2, n_E, width, color='gray', alpha=0.7,
           label='BIT* (Euclidean)')
    ax.bar(x + width / 2, n_R, width, color='purple', alpha=0.7,
           label='RIT* (Riemannian)')

    labels = []
    for d in dims:
        for k in kappa_values:
            labels.append(f'd={d}\nκ={k:.0f}')
    if len(labels) == n_groups:
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel('Samples to reach 1.05 × c*')
    ax.set_title('Sample Efficiency: RIT* vs BIT*')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')
    fig.tight_layout()
    return fig
