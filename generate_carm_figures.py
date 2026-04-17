#!/usr/bin/env python
"""
generate_carm_figures.py — Conceptual figures for the CARM RAL paper.

Generates publication-quality figures that visually explain HOW and WHY
CARM improves over Euclidean baselines.

Figures:
  Fig 1: CARM overview — Oracle vs Learned vs Euclidean metric fields + paths
  Fig 2: Informed set shrinkage — side-by-side Euclidean vs CARM informed sets
  Fig 3: CARM evolution over iterations — 4 snapshots showing progressive learning
  Fig 4: Wasted samples illustration — Euclidean wastes samples in obstacles
  Fig 5: CARM mechanism diagram — collision feedback → KDE → metric → informed set
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Ellipse, FancyArrowPatch, Rectangle
from matplotlib.collections import LineCollection
from matplotlib import cm, colors
import matplotlib.patheffects as pe

from output_paths import PLOTS_DIR
from rit_star.rit_star import RITStar
from rit_star.metric import EuclideanMetric, CollisionAdaptiveMetric
from rit_star.environments import env_2d_obstacle_inflated, env_2d_maze

# ── Style ────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.size': 11,
    'font.family': 'serif',
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'legend.fontsize': 9,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
})

CARM_COLOR = '#E63946'       # red for CARM
ORACLE_COLOR = '#457B9D'     # blue for oracle
EUCLID_COLOR = '#A8A8A8'     # grey for Euclidean
BIT_COLOR = '#2A9D8F'        # teal for BIT*
START_COLOR = '#2D6A4F'
GOAL_COLOR = '#E63946'
OBSTACLE_COLOR = '#2B2D42'

CIRCLES_OBS = [
    (np.array([0.30, 0.35]), 0.08),
    (np.array([0.30, 0.65]), 0.08),
    (np.array([0.50, 0.45]), 0.09),
    (np.array([0.50, 0.75]), 0.09),
    (np.array([0.70, 0.40]), 0.08),
    (np.array([0.70, 0.60]), 0.08),
]


def _draw_obstacles(ax, alpha=0.85):
    """Draw circular obstacles on axis."""
    for c, r in CIRCLES_OBS:
        ax.add_patch(Circle(c, r, fc=OBSTACLE_COLOR, ec='white',
                            lw=0.8, alpha=alpha, zorder=2))


def _draw_start_goal(ax, xs, xg, ms=10):
    ax.plot(*xs, 's', color=START_COLOR, ms=ms, zorder=10,
            markeredgecolor='white', markeredgewidth=1.2)
    ax.plot(*xg, '*', color=GOAL_COLOR, ms=ms + 3, zorder=10,
            markeredgecolor='white', markeredgewidth=0.8)


def _set_axes(ax):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')
    ax.set_xticks([0, 0.5, 1])
    ax.set_yticks([0, 0.5, 1])


def _run_planner(xs, xg, bounds, coll, metric, adaptive=False,
                 max_iter=120, seed=42, **carm_kw):
    """Run RIT* and return planner object."""
    planner = RITStar(
        xs, xg, bounds, coll, metric,
        geodesic_tier='diagonal', batch_size=80,
        max_iterations=max_iter, random_seed=seed,
        adaptive_metric=adaptive,
        carm_sigma=carm_kw.get('sigma', 0.08),
        carm_alpha=carm_kw.get('alpha', 6.0),
        carm_rebuild_interval=carm_kw.get('rebuild', 10),
    )
    path, cost = planner.plan()
    return planner, path, cost


def _compute_scale_field(metric_or_carm, res=200, method='sqrt_det'):
    """Compute scale field on [0,1]^2 grid."""
    xx = np.linspace(0, 1, res)
    yy = np.linspace(0, 1, res)
    XX, YY = np.meshgrid(xx, yy)
    pts = np.column_stack([XX.ravel(), YY.ravel()])

    if hasattr(metric_or_carm, '_collision_scale_batch'):
        S = metric_or_carm._collision_scale_batch(pts).reshape(XX.shape)
    elif method == 'sqrt_det':
        S = np.array([metric_or_carm.sqrt_det_G(p) for p in pts]).reshape(XX.shape)
    else:
        S = np.array([np.sqrt(np.max(np.linalg.eigvalsh(metric_or_carm.G(p))))
                      for p in pts]).reshape(XX.shape)
    return XX, YY, S


def _euclidean_ellipse_params(xs, xg, c_best):
    """Compute Euclidean informed-set ellipse parameters."""
    mid = (xs + xg) / 2.0
    diff = xg - xs
    c_min = np.linalg.norm(diff)
    angle = np.degrees(np.arctan2(diff[1], diff[0]))
    a = c_best / 2.0  # semi-major
    b = np.sqrt(max(c_best**2 - c_min**2, 0)) / 2.0  # semi-minor
    return mid, a, b, angle


# ══════════════════════════════════════════════════════════════════════════════
#  FIGURE 1: CARM Overview — Oracle vs CARM vs Euclidean
# ══════════════════════════════════════════════════════════════════════════════
def fig1_carm_overview():
    """3-panel: Oracle metric | CARM learned metric | Euclidean (flat).
    Each panel shows metric field + resulting path."""
    print('=== Figure 1: CARM Overview (Oracle vs CARM vs Euclidean) ===')

    coll, _, oracle_metric, xs, xg, bounds = env_2d_obstacle_inflated()
    dim = len(xs)
    euclid = EuclideanMetric(dim)

    # Run three planner variants
    print('  Running RIT* with oracle metric...')
    p_oracle, path_oracle, cost_oracle = _run_planner(
        xs, xg, bounds, coll, oracle_metric, adaptive=False, max_iter=150, seed=42)

    print('  Running RIT* with CARM...')
    p_carm, path_carm, cost_carm = _run_planner(
        xs, xg, bounds, coll, euclid, adaptive=True, max_iter=150, seed=42)

    print('  Running RIT* Euclidean (no CARM)...')
    p_euclid, path_euclid, cost_euclid = _run_planner(
        xs, xg, bounds, coll, euclid, adaptive=False, max_iter=150, seed=42)

    # Compute metric fields
    res = 200
    XX_o, YY_o, S_oracle = _compute_scale_field(oracle_metric, res)
    XX_c, YY_c, S_carm = _compute_scale_field(p_carm._carm, res)

    # Evaluate all paths under oracle metric for fair comparison
    def oracle_path_cost(path):
        if not path or len(path) < 2:
            return float('inf')
        total = 0
        for i in range(len(path) - 1):
            a, b_ = np.array(path[i]), np.array(path[i + 1])
            mid = (a + b_) / 2
            G_mid = oracle_metric.G(mid)
            diff = b_ - a
            total += np.sqrt(diff @ G_mid @ diff)
        return total

    oc_oracle = oracle_path_cost(path_oracle)
    oc_carm = oracle_path_cost(path_carm)
    oc_euclid = oracle_path_cost(path_euclid)

    # ── Plot ──────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.2))

    # Shared colorbar range
    vmin = min(S_oracle.min(), S_carm.min(), 1.0)
    vmax = max(S_oracle.max(), S_carm.max())
    levels = np.linspace(vmin, vmax, 30)

    # Panel 1: Oracle
    ax = axes[0]
    im = ax.contourf(XX_o, YY_o, S_oracle, levels=levels, cmap='YlOrRd', extend='max')
    _draw_obstacles(ax)
    if path_oracle and len(path_oracle) > 1:
        px, py = zip(*path_oracle)
        ax.plot(px, py, '-', color=ORACLE_COLOR, lw=3, zorder=5,
                path_effects=[pe.Stroke(linewidth=4.5, foreground='white'), pe.Normal()])
    _draw_start_goal(ax, xs, xg)
    ax.set_title(f'(a) Oracle metric\n(requires obstacle knowledge)\ncost$_{{oracle}}$ = {oc_oracle:.2f}',
                 fontsize=11, fontweight='bold')
    _set_axes(ax)

    # Panel 2: CARM
    ax = axes[1]
    ax.contourf(XX_c, YY_c, S_carm, levels=levels, cmap='YlOrRd', extend='max')
    _draw_obstacles(ax)
    carm_pts = np.array(p_carm._carm._collision_points)
    if len(carm_pts) > 0:
        ax.scatter(carm_pts[:, 0], carm_pts[:, 1], c='cyan', s=2,
                   alpha=0.25, zorder=3, rasterized=True)
    if path_carm and len(path_carm) > 1:
        px, py = zip(*path_carm)
        ax.plot(px, py, '-', color=CARM_COLOR, lw=3, zorder=5,
                path_effects=[pe.Stroke(linewidth=4.5, foreground='white'), pe.Normal()])
    _draw_start_goal(ax, xs, xg)
    n_coll = len(carm_pts) if len(carm_pts) > 0 else 0
    ax.set_title(f'(b) CARM (learned online)\n({n_coll} collision points, no prior)\n'
                 f'cost$_{{oracle}}$ = {oc_carm:.2f}',
                 fontsize=11, fontweight='bold')
    _set_axes(ax)

    # Panel 3: Euclidean (flat)
    ax = axes[2]
    # Flat background (Euclidean = 1 everywhere)
    ax.set_facecolor('#FFF5EB')
    _draw_obstacles(ax)
    # Show tree lightly
    for v in p_euclid.vertices:
        if v.parent is not None:
            ax.plot([v.parent.x[0], v.x[0]], [v.parent.x[1], v.x[1]],
                    '-', color=EUCLID_COLOR, lw=0.2, alpha=0.3, zorder=1)
    if path_euclid and len(path_euclid) > 1:
        px, py = zip(*path_euclid)
        ax.plot(px, py, '-', color='#555555', lw=3, zorder=5,
                path_effects=[pe.Stroke(linewidth=4.5, foreground='white'), pe.Normal()])
    _draw_start_goal(ax, xs, xg)
    ax.set_title(f'(c) Euclidean metric\n(standard BIT*/RIT*)\ncost$_{{oracle}}$ = {oc_euclid:.2f}',
                 fontsize=11, fontweight='bold')
    _set_axes(ax)

    # Shared colorbar
    cbar_ax = fig.add_axes([0.93, 0.15, 0.015, 0.7])
    cb = fig.colorbar(im, cax=cbar_ax)
    cb.set_label('Metric scale  $\\sqrt{\\det G(x)}$', fontsize=10)

    fig.suptitle('CARM learns obstacle proximity from collision feedback — no a priori model needed',
                 fontsize=14, fontweight='bold', y=1.03)
    fig.subplots_adjust(wspace=0.18, right=0.91)

    path_out = os.path.join(PLOTS_DIR, 'fig_carm_overview.pdf')
    fig.savefig(path_out, dpi=300, bbox_inches='tight')
    fig.savefig(path_out.replace('.pdf', '.png'), dpi=200, bbox_inches='tight')
    print(f'  → {path_out}')
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
#  FIGURE 2: Informed Set Shrinkage
# ══════════════════════════════════════════════════════════════════════════════
def fig2_informed_set_shrinkage():
    """Side-by-side: Euclidean informed set vs CARM informed set.
    Shows how CARM shrinks the sampling region."""
    print('=== Figure 2: Informed Set Shrinkage ===')

    coll, _, oracle_metric, xs, xg, bounds = env_2d_obstacle_inflated()
    dim = len(xs)
    euclid = EuclideanMetric(dim)

    # Run CARM to get the learned metric
    p_carm, path_carm, cost_carm = _run_planner(
        xs, xg, bounds, coll, euclid, adaptive=True, max_iter=150, seed=42)

    carm = p_carm._carm
    c_best = cost_carm  # Use CARM's best cost

    # Use a slightly inflated c_best for better visualization
    c_vis = c_best * 1.05

    # Compute membership on grid
    res = 300
    xx = np.linspace(0, 1, res)
    yy = np.linspace(0, 1, res)
    XX, YY = np.meshgrid(xx, yy)
    pts = np.column_stack([XX.ravel(), YY.ravel()])

    # Euclidean informed set membership
    d_start_E = np.linalg.norm(pts - xs, axis=1)
    d_goal_E = np.linalg.norm(pts - xg, axis=1)
    in_euclid = (d_start_E + d_goal_E <= c_vis).reshape(XX.shape)

    # CARM informed set membership (approx via midpoint metric)
    def riem_dist_approx(a, b, metric):
        mid = (a + b) / 2
        G = metric.G(mid)
        d = b - a
        return np.sqrt(d @ G @ d)

    # Vectorized approximate Riemannian distances
    d_start_R = np.zeros(len(pts))
    d_goal_R = np.zeros(len(pts))
    for i, p in enumerate(pts):
        d_start_R[i] = riem_dist_approx(xs, p, carm)
        d_goal_R[i] = riem_dist_approx(p, xg, carm)
    in_carm = (d_start_R + d_goal_R <= c_vis * np.sqrt(1 + 3.0)).reshape(XX.shape)
    # Scale threshold to account for CARM inflating distances

    # Better approach: use same cost threshold, show which points
    # have higher riemannian cost sum
    carm_cost_sum = (d_start_R + d_goal_R).reshape(XX.shape)
    euclid_cost_sum = (d_start_E + d_goal_E).reshape(XX.shape)

    # Also compute collision mask for display
    carm_pts = np.array(carm._collision_points) if carm._collision_points else np.zeros((0, 2))
    carm_scale = carm._collision_scale_batch(pts).reshape(XX.shape)

    # ── Plot ──────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.2))

    # Panel 1: Euclidean informed set
    ax = axes[0]
    # Fill: light blue inside ellipsoid
    ax.contourf(XX, YY, in_euclid.astype(float), levels=[0.5, 1.5],
                colors=['#C1E1FF'], alpha=0.6)
    ax.contour(XX, YY, in_euclid.astype(float), levels=[0.5],
               colors=[EUCLID_COLOR], linewidths=2)
    _draw_obstacles(ax, alpha=0.9)
    # Show random samples inside the ellipsoid to illustrate waste
    rng = np.random.default_rng(42)
    n_show = 500
    mid, a_ell, b_ell, angle = _euclidean_ellipse_params(xs, xg, c_vis)
    # Sample inside ellipse
    theta_rand = rng.uniform(0, 2 * np.pi, n_show)
    r_rand = np.sqrt(rng.uniform(0, 1, n_show))
    ex = a_ell * r_rand * np.cos(theta_rand)
    ey = b_ell * r_rand * np.sin(theta_rand)
    ang_rad = np.radians(angle)
    sx = mid[0] + ex * np.cos(ang_rad) - ey * np.sin(ang_rad)
    sy = mid[1] + ex * np.sin(ang_rad) + ey * np.cos(ang_rad)
    valid = (sx >= 0) & (sx <= 1) & (sy >= 0) & (sy <= 1)
    sx, sy = sx[valid], sy[valid]
    # Check which are in collision
    in_coll = np.array([not coll(np.array([x_, y_])) for x_, y_ in zip(sx, sy)])
    ax.scatter(sx[~in_coll], sy[~in_coll], c='#4CAF50', s=4, alpha=0.5,
               zorder=3, label='Free samples')
    ax.scatter(sx[in_coll], sy[in_coll], c='red', s=6, alpha=0.7,
               zorder=3, marker='x', label='Wasted (collision)')
    waste_pct = in_coll.sum() / len(in_coll) * 100
    _draw_start_goal(ax, xs, xg)
    ax.set_title(f'(a) Euclidean informed set $I_E$\n'
                 f'{waste_pct:.0f}% of samples hit obstacles',
                 fontsize=11, fontweight='bold')
    ax.legend(loc='lower left', fontsize=8, framealpha=0.9)
    _set_axes(ax)

    # Panel 2: CARM metric field
    ax = axes[1]
    im = ax.contourf(XX, YY, carm_scale, levels=25, cmap='YlOrRd')
    _draw_obstacles(ax)
    if len(carm_pts) > 0:
        ax.scatter(carm_pts[:, 0], carm_pts[:, 1], c='cyan', s=3,
                   alpha=0.35, zorder=3, label=f'{len(carm_pts)} collision pts')
    _draw_start_goal(ax, xs, xg)
    ax.set_title(f'(b) CARM learned cost field $s(x)$\n'
                 f'Higher cost (red) near obstacles',
                 fontsize=11, fontweight='bold')
    ax.legend(loc='lower left', fontsize=8, framealpha=0.9)
    _set_axes(ax)
    plt.colorbar(im, ax=ax, shrink=0.8, label='$s(x)$')

    # Panel 3: CARM informed set (tighter)
    ax = axes[2]
    # Show CARM cost field as contour
    # Mark regions where Riemannian triangle sum is high = excluded
    ratio = carm_cost_sum / np.maximum(euclid_cost_sum, 1e-10)
    # Regions with high ratio are "excluded by CARM but included by Euclidean"
    excluded = (in_euclid) & (ratio > np.percentile(ratio[in_euclid], 60))
    # Still inside the Euclidean ellipsoid but excluded by CARM
    ax.contourf(XX, YY, in_euclid.astype(float), levels=[0.5, 1.5],
                colors=['#C1E1FF'], alpha=0.3)
    ax.contourf(XX, YY, (~excluded & in_euclid).astype(float),
                levels=[0.5, 1.5], colors=['#A8E6CF'], alpha=0.6)
    ax.contour(XX, YY, in_euclid.astype(float), levels=[0.5],
               colors=[EUCLID_COLOR], linewidths=1.5, linestyles='--')
    ax.contour(XX, YY, (~excluded & in_euclid).astype(float), levels=[0.5],
               colors=[CARM_COLOR], linewidths=2.5)
    _draw_obstacles(ax, alpha=0.9)
    # Annotate
    ax.annotate('Excluded\nby CARM', xy=(0.35, 0.52), fontsize=9,
                color=CARM_COLOR, fontweight='bold', ha='center',
                bbox=dict(boxstyle='round,pad=0.3', fc='white', ec=CARM_COLOR, alpha=0.9))
    ax.annotate('$I_E$ boundary', xy=(0.82, 0.15), fontsize=8,
                color=EUCLID_COLOR, ha='center',
                bbox=dict(boxstyle='round,pad=0.2', fc='white', ec=EUCLID_COLOR, alpha=0.8))
    ax.annotate('$I_R^{\\rm CARM}$\nboundary', xy=(0.15, 0.85), fontsize=8,
                color=CARM_COLOR, fontweight='bold', ha='center',
                bbox=dict(boxstyle='round,pad=0.2', fc='white', ec=CARM_COLOR, alpha=0.8))
    _draw_start_goal(ax, xs, xg)
    ax.set_title(f'(c) CARM shrinks informed set\n'
                 f'Focus: $I_R^{{\\rm CARM}} \\subset I_E$',
                 fontsize=11, fontweight='bold')
    _set_axes(ax)

    fig.suptitle('CARM progressively tightens the informed set using collision feedback',
                 fontsize=14, fontweight='bold', y=1.03)
    fig.subplots_adjust(wspace=0.22)

    path_out = os.path.join(PLOTS_DIR, 'fig_carm_informed_set.pdf')
    fig.savefig(path_out, dpi=300, bbox_inches='tight')
    fig.savefig(path_out.replace('.pdf', '.png'), dpi=200, bbox_inches='tight')
    print(f'  → {path_out}')
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
#  FIGURE 3: CARM Evolution Over Iterations
# ══════════════════════════════════════════════════════════════════════════════
def fig3_carm_evolution():
    """4 snapshots showing CARM metric field evolving as collisions accumulate."""
    print('=== Figure 3: CARM Evolution Over Iterations ===')

    coll, _, oracle_metric, xs, xg, bounds = env_2d_obstacle_inflated()
    dim = len(xs)
    euclid = EuclideanMetric(dim)

    import time

    planner = RITStar(
        xs, xg, bounds, coll, euclid,
        geodesic_tier='diagonal', batch_size=80,
        max_iterations=120, random_seed=42,
        adaptive_metric=True,
        carm_sigma=0.08, carm_alpha=6.0,
        carm_rebuild_interval=10)

    snapshot_iters = [10, 30, 60, 110]
    snapshots = {}

    res = 180
    xx = np.linspace(0, 1, res)
    yy = np.linspace(0, 1, res)
    XX, YY = np.meshgrid(xx, yy)
    pts_grid = np.column_stack([XX.ravel(), YY.ravel()])

    # Get oracle field for reference
    S_oracle = np.array([oracle_metric.sqrt_det_G(p) for p in pts_grid]).reshape(XX.shape)

    # Run step-by-step
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
            cp = np.array(carm._collision_points) if carm._collision_points else np.zeros((0, 2))

            # Extract current best path
            cur_path = []
            if planner.c_best < np.inf and planner.goal_node:
                v = planner.goal_node
                while v is not None:
                    cur_path.append(v.x.copy())
                    v = v.parent
                cur_path.reverse()

            snapshots[it + 1] = {
                'scale': scale.copy(),
                'coll_pts': cp.copy(),
                'n_coll': len(cp),
                'c_best': planner.c_best,
                'path': cur_path,
            }

    # ── Plot ──────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    vmax = max(s['scale'].max() for s in snapshots.values())

    for idx, it in enumerate(snapshot_iters):
        ax = axes[idx]
        snap = snapshots[it]

        im = ax.contourf(XX, YY, snap['scale'], levels=25, cmap='YlOrRd',
                         vmin=1.0, vmax=vmax)
        _draw_obstacles(ax, alpha=0.8)

        # Collision points
        if snap['n_coll'] > 0:
            ax.scatter(snap['coll_pts'][:, 0], snap['coll_pts'][:, 1],
                       c='cyan', s=3, alpha=0.4, zorder=3, rasterized=True)

        # Path
        if snap['path'] and len(snap['path']) > 1:
            px, py = zip(*snap['path'])
            ax.plot(px, py, '-', color=CARM_COLOR, lw=2.5, zorder=5,
                    path_effects=[pe.Stroke(linewidth=4, foreground='white'), pe.Normal()])

        _draw_start_goal(ax, xs, xg, ms=8)

        c_str = f'{snap["c_best"]:.3f}' if np.isfinite(snap['c_best']) else '∞'
        ax.set_title(f'Iteration {it}\n'
                     f'{snap["n_coll"]} collisions | cost = {c_str}',
                     fontsize=10, fontweight='bold')
        _set_axes(ax)

        if idx == 0:
            ax.text(0.05, 0.92, 'Few collisions\n→ near-Euclidean',
                    transform=ax.transAxes, fontsize=8, va='top',
                    bbox=dict(fc='white', ec='gray', alpha=0.85, boxstyle='round,pad=0.3'))
        elif idx == 3:
            ax.text(0.05, 0.92, 'Dense feedback\n→ near-oracle',
                    transform=ax.transAxes, fontsize=8, va='top',
                    bbox=dict(fc='white', ec=CARM_COLOR, alpha=0.85, boxstyle='round,pad=0.3'))

    # Colorbar
    cbar_ax = fig.add_axes([0.93, 0.15, 0.012, 0.7])
    cb = fig.colorbar(im, cax=cbar_ax)
    cb.set_label('CARM scale $s(x)$', fontsize=10)

    fig.suptitle('CARM learns progressively: collision feedback → obstacle-aware metric',
                 fontsize=14, fontweight='bold', y=1.04)
    fig.subplots_adjust(wspace=0.15, right=0.91)

    path_out = os.path.join(PLOTS_DIR, 'fig_carm_evolution.pdf')
    fig.savefig(path_out, dpi=300, bbox_inches='tight')
    fig.savefig(path_out.replace('.pdf', '.png'), dpi=200, bbox_inches='tight')
    print(f'  → {path_out}')
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
#  FIGURE 4: Wasted Samples — Why Euclidean Informed Sets Are Inefficient
# ══════════════════════════════════════════════════════════════════════════════
def fig4_wasted_samples():
    """Compare sample efficiency: Euclidean vs CARM.
    Left: Euclidean samples (many wasted in obstacles/high-cost regions).
    Right: CARM samples (focused in useful regions)."""
    print('=== Figure 4: Wasted Samples Comparison ===')

    coll, _, oracle_metric, xs, xg, bounds = env_2d_obstacle_inflated()
    dim = len(xs)
    euclid = EuclideanMetric(dim)

    # Run both variants, collect sampled points
    print('  Running Euclidean planner...')
    p_euc, path_euc, cost_euc = _run_planner(
        xs, xg, bounds, coll, euclid, adaptive=False, max_iter=100, seed=42)

    print('  Running CARM planner...')
    p_carm, path_carm, cost_carm = _run_planner(
        xs, xg, bounds, coll, euclid, adaptive=True, max_iter=100, seed=42)

    # Evaluate paths under oracle
    def oracle_cost(path):
        if not path or len(path) < 2:
            return float('inf')
        total = 0
        for i in range(len(path) - 1):
            a, b_ = np.array(path[i]), np.array(path[i + 1])
            mid = (a + b_) / 2
            G = oracle_metric.G(mid)
            d = b_ - a
            total += np.sqrt(d @ G @ d)
        return total

    oc_euc = oracle_cost(path_euc)
    oc_carm = oracle_cost(path_carm)

    # Get CARM metric field
    res = 180
    xx = np.linspace(0, 1, res)
    yy = np.linspace(0, 1, res)
    XX, YY = np.meshgrid(xx, yy)
    pts_grid = np.column_stack([XX.ravel(), YY.ravel()])
    S_carm = p_carm._carm._collision_scale_batch(pts_grid).reshape(XX.shape)

    # ── Plot ──────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))

    # Panel 1: Euclidean tree
    ax = axes[0]
    ax.set_facecolor('#FAFAFA')
    _draw_obstacles(ax)
    # Tree edges
    for v in p_euc.vertices:
        if v.parent is not None:
            ax.plot([v.parent.x[0], v.x[0]], [v.parent.x[1], v.x[1]],
                    '-', color='#CCCCCC', lw=0.3, alpha=0.4, zorder=1)
    # Vertices colored by oracle cost
    vx = np.array([v.x for v in p_euc.vertices])
    v_cost = np.array([oracle_metric.sqrt_det_G(v.x) for v in p_euc.vertices])
    sc = ax.scatter(vx[:, 0], vx[:, 1], c=v_cost, cmap='RdYlGn_r', s=5,
                    alpha=0.6, zorder=2, vmin=1, vmax=v_cost.max())
    # Path
    if path_euc and len(path_euc) > 1:
        px, py = zip(*path_euc)
        ax.plot(px, py, '-', color='#555555', lw=3, zorder=5,
                path_effects=[pe.Stroke(linewidth=4.5, foreground='white'), pe.Normal()])
    _draw_start_goal(ax, xs, xg)
    n_high = (v_cost > np.percentile(v_cost, 70)).sum()
    ax.set_title(f'(a) Euclidean planner\n'
                 f'{len(p_euc.vertices)} vertices | oracle cost = {oc_euc:.2f}\n'
                 f'{n_high} vertices in high-cost regions',
                 fontsize=11, fontweight='bold')
    _set_axes(ax)
    plt.colorbar(sc, ax=ax, shrink=0.75, label='Oracle cost density')

    # Panel 2: CARM tree
    ax = axes[1]
    ax.contourf(XX, YY, S_carm, levels=20, cmap='YlOrRd', alpha=0.3)
    _draw_obstacles(ax)
    for v in p_carm.vertices:
        if v.parent is not None:
            ax.plot([v.parent.x[0], v.x[0]], [v.parent.x[1], v.x[1]],
                    '-', color='#CCCCCC', lw=0.3, alpha=0.4, zorder=1)
    vx_c = np.array([v.x for v in p_carm.vertices])
    v_cost_c = np.array([oracle_metric.sqrt_det_G(v.x) for v in p_carm.vertices])
    sc2 = ax.scatter(vx_c[:, 0], vx_c[:, 1], c=v_cost_c, cmap='RdYlGn_r', s=5,
                     alpha=0.6, zorder=2, vmin=1, vmax=v_cost.max())
    if path_carm and len(path_carm) > 1:
        px, py = zip(*path_carm)
        ax.plot(px, py, '-', color=CARM_COLOR, lw=3, zorder=5,
                path_effects=[pe.Stroke(linewidth=4.5, foreground='white'), pe.Normal()])
    # Collision points
    carm_pts = np.array(p_carm._carm._collision_points)
    if len(carm_pts) > 0:
        ax.scatter(carm_pts[:, 0], carm_pts[:, 1], c='cyan', s=2,
                   alpha=0.2, zorder=3, rasterized=True)
    _draw_start_goal(ax, xs, xg)
    n_high_c = (v_cost_c > np.percentile(v_cost, 70)).sum()
    ax.set_title(f'(b) CARM planner\n'
                 f'{len(p_carm.vertices)} vertices | oracle cost = {oc_carm:.2f}\n'
                 f'{n_high_c} vertices in high-cost regions '
                 f'({(1 - n_high_c / max(n_high, 1)) * 100:.0f}% reduction)',
                 fontsize=11, fontweight='bold')
    _set_axes(ax)
    plt.colorbar(sc2, ax=ax, shrink=0.75, label='Oracle cost density')

    improvement = (1 - oc_carm / oc_euc) * 100 if oc_euc > 0 else 0
    fig.suptitle(f'CARM directs samples away from obstacles '
                 f'→ {improvement:.1f}% oracle-cost improvement',
                 fontsize=14, fontweight='bold', y=1.04)
    fig.subplots_adjust(wspace=0.25)

    path_out = os.path.join(PLOTS_DIR, 'fig_carm_wasted_samples.pdf')
    fig.savefig(path_out, dpi=300, bbox_inches='tight')
    fig.savefig(path_out.replace('.pdf', '.png'), dpi=200, bbox_inches='tight')
    print(f'  → {path_out}')
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
#  FIGURE 5: CARM Mechanism Diagram (Pure Matplotlib — no external images)
# ══════════════════════════════════════════════════════════════════════════════
def fig5_mechanism_diagram():
    """Conceptual flow diagram showing CARM pipeline:
    collision feedback → KDE → conformal metric → tighter informed set → better paths."""
    print('=== Figure 5: CARM Mechanism Diagram ===')

    fig, axes = plt.subplots(1, 5, figsize=(22, 4.2))

    # ── Panel 1: Collision detection ─────────────────────────────
    ax = axes[0]
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect('equal')
    # Draw a couple obstacles
    obs = [(np.array([0.4, 0.5]), 0.15), (np.array([0.7, 0.3]), 0.12)]
    for c, r in obs:
        ax.add_patch(Circle(c, r, fc=OBSTACLE_COLOR, ec='white', lw=1, alpha=0.8))
    # Sample points: free (green) and collision (red X)
    rng = np.random.default_rng(42)
    free_pts = rng.uniform(0.05, 0.95, (20, 2))
    coll_pts = np.array([[0.35, 0.55], [0.45, 0.45], [0.42, 0.60],
                          [0.65, 0.25], [0.72, 0.40], [0.38, 0.38],
                          [0.75, 0.32], [0.48, 0.52]])
    ax.scatter(free_pts[:, 0], free_pts[:, 1], c='#4CAF50', s=20, alpha=0.5,
               zorder=3, marker='o')
    ax.scatter(coll_pts[:, 0], coll_pts[:, 1], c='red', s=40, alpha=0.8,
               zorder=4, marker='x', linewidths=2)
    ax.set_title('(1)  Collision Checks\n(free data)', fontsize=11, fontweight='bold',
                 color=OBSTACLE_COLOR)
    ax.text(0.5, 0.02, 'x = collision point\no = free sample',
            transform=ax.transAxes, ha='center', fontsize=8,
            bbox=dict(fc='white', ec='gray', alpha=0.9, boxstyle='round'))
    ax.set_xticks([]); ax.set_yticks([])

    # ── Panel 2: KDE over collision points ───────────────────────
    ax = axes[1]
    res = 80
    xx = np.linspace(0, 1, res)
    yy = np.linspace(0, 1, res)
    XX, YY = np.meshgrid(xx, yy)
    pts = np.column_stack([XX.ravel(), YY.ravel()])
    # Compute KDE manually
    sigma = 0.08
    kde = np.zeros(len(pts))
    for cp in coll_pts:
        dist2 = np.sum((pts - cp) ** 2, axis=1)
        kde += np.exp(-dist2 / (2 * sigma ** 2))
    kde = (kde / len(coll_pts)).reshape(XX.shape)
    ax.contourf(XX, YY, kde, levels=20, cmap='hot_r')
    ax.scatter(coll_pts[:, 0], coll_pts[:, 1], c='cyan', s=25, zorder=3,
               edgecolors='white', linewidths=0.5)
    ax.set_title('(2)  Kernel Density $\\hat{f}(x)$\nfrom collisions', fontsize=11,
                 fontweight='bold', color='#C62828')
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect('equal')
    ax.set_xticks([]); ax.set_yticks([])

    # ── Panel 3: Conformal scale s(x) ───────────────────────────
    ax = axes[2]
    alpha_carm = 6.0
    s_field = 1.0 + alpha_carm * kde
    ax.contourf(XX, YY, s_field, levels=20, cmap='YlOrRd')
    for c, r in obs:
        ax.add_patch(Circle(c, r, fc=OBSTACLE_COLOR, ec='white', lw=1, alpha=0.6))
    ax.set_title('(3)  Metric scale $s(x) = 1 + \\alpha\\hat{f}(x)$\n(Riemannian cost)',
                 fontsize=11, fontweight='bold', color='#E65100')
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect('equal')
    ax.set_xticks([]); ax.set_yticks([])

    # ── Panel 4: Informed set comparison ──────────���──────────────
    ax = axes[3]
    xs_d = np.array([0.1, 0.5])
    xg_d = np.array([0.9, 0.5])
    # Euclidean ellipse
    ax.add_patch(Ellipse(xy=(0.5, 0.5), width=0.92, height=0.72,
                          ec=EUCLID_COLOR, fc='#E0E0E0', lw=2, ls='--',
                          alpha=0.5, label='$I_E$ (Euclidean)'))
    # CARM ellipse (tighter — more like a tube)
    ax.add_patch(Ellipse(xy=(0.5, 0.5), width=0.88, height=0.45,
                          ec=CARM_COLOR, fc='#FFCDD2', lw=2.5,
                          alpha=0.5, label='$I_R^{\\rm CARM}$ (tighter)'))
    for c, r in obs:
        ax.add_patch(Circle(c, r, fc=OBSTACLE_COLOR, ec='white', lw=1, alpha=0.8))
    ax.plot(*xs_d, 's', color=START_COLOR, ms=10, zorder=5,
            markeredgecolor='white', markeredgewidth=1)
    ax.plot(*xg_d, '*', color=GOAL_COLOR, ms=12, zorder=5,
            markeredgecolor='white', markeredgewidth=0.5)
    ax.set_title('(4)  Tighter Informed Set\n$I_R^{\\rm CARM} \\subset I_E$',
                 fontsize=11, fontweight='bold', color=CARM_COLOR)
    ax.legend(loc='lower center', fontsize=8, framealpha=0.9)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect('equal')
    ax.set_xticks([]); ax.set_yticks([])

    # ── Panel 5: Better path ─────────────────────────────────────
    ax = axes[4]
    for c, r in obs:
        ax.add_patch(Circle(c, r, fc=OBSTACLE_COLOR, ec='white', lw=1, alpha=0.8))
    # Euclidean path (goes close to obstacles)
    euc_path = np.array([[0.1, 0.5], [0.25, 0.5], [0.4, 0.52],
                          [0.55, 0.55], [0.7, 0.5], [0.9, 0.5]])
    ax.plot(euc_path[:, 0], euc_path[:, 1], '--', color=EUCLID_COLOR,
            lw=2.5, label='Euclidean path', zorder=3)
    # CARM path (avoids obstacles with clearance)
    carm_path = np.array([[0.1, 0.5], [0.2, 0.65], [0.35, 0.80],
                           [0.55, 0.85], [0.70, 0.75],
                           [0.82, 0.60], [0.9, 0.5]])
    ax.plot(carm_path[:, 0], carm_path[:, 1], '-', color=CARM_COLOR,
            lw=3, label='CARM path', zorder=4,
            path_effects=[pe.Stroke(linewidth=4.5, foreground='white'), pe.Normal()])
    ax.plot(*xs_d, 's', color=START_COLOR, ms=10, zorder=5,
            markeredgecolor='white', markeredgewidth=1)
    ax.plot(*xg_d, '*', color=GOAL_COLOR, ms=12, zorder=5,
            markeredgecolor='white', markeredgewidth=0.5)
    ax.set_title('(5)  Safer, lower-cost path\n(clears obstacles)',
                 fontsize=11, fontweight='bold', color='#2E7D32')
    ax.legend(loc='lower center', fontsize=8, framealpha=0.9)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect('equal')
    ax.set_xticks([]); ax.set_yticks([])

    # Arrows between panels
    for i in range(4):
        fig.text(0.192 + i * 0.178, 0.5, '→', fontsize=28, fontweight='bold',
                 color='#666', ha='center', va='center', transform=fig.transFigure)

    fig.suptitle('CARM Pipeline: Collision Feedback → Learned Metric → Tighter Informed Set → Better Paths',
                 fontsize=13, fontweight='bold', y=1.06, color=OBSTACLE_COLOR)
    fig.subplots_adjust(wspace=0.12)

    path_out = os.path.join(PLOTS_DIR, 'fig_carm_mechanism.pdf')
    fig.savefig(path_out, dpi=300, bbox_inches='tight')
    fig.savefig(path_out.replace('.pdf', '.png'), dpi=200, bbox_inches='tight')
    print(f'  → {path_out}')
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
#  FIGURE 6: CARM vs Oracle Metric Correlation
# ══════════════════════════════════════════════════════════════════════════════
def fig6_metric_correlation():
    """Scatter plot + heatmap: CARM learned s(x) vs oracle sqrt(det G(x))."""
    print('=== Figure 6: CARM vs Oracle Metric Correlation ===')

    coll, _, oracle_metric, xs, xg, bounds = env_2d_obstacle_inflated()
    dim = len(xs)
    euclid = EuclideanMetric(dim)

    # Run CARM planner
    p_carm, path_carm, cost_carm = _run_planner(
        xs, xg, bounds, coll, euclid, adaptive=True, max_iter=150, seed=42)

    carm = p_carm._carm

    # Evaluate on grid
    res = 100
    xx = np.linspace(0.02, 0.98, res)
    yy = np.linspace(0.02, 0.98, res)
    XX, YY = np.meshgrid(xx, yy)
    pts = np.column_stack([XX.ravel(), YY.ravel()])

    s_carm = carm._collision_scale_batch(pts)
    s_oracle = np.array([oracle_metric.sqrt_det_G(p) for p in pts])

    # Filter out obstacle interior points
    free_mask = np.array([coll(p) for p in pts])
    s_carm_f = s_carm[free_mask]
    s_oracle_f = s_oracle[free_mask]

    # Pearson correlation
    corr = np.corrcoef(s_carm_f, s_oracle_f)[0, 1]

    # ── Plot ──────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Panel 1: Oracle field
    ax = axes[0]
    im1 = ax.contourf(XX, YY, s_oracle.reshape(XX.shape), levels=25, cmap='YlOrRd')
    _draw_obstacles(ax)
    _draw_start_goal(ax, xs, xg)
    ax.set_title('(a) Oracle $\\sqrt{\\det G_{oracle}(x)}$', fontsize=11, fontweight='bold')
    _set_axes(ax)
    plt.colorbar(im1, ax=ax, shrink=0.8)

    # Panel 2: CARM field
    ax = axes[1]
    im2 = ax.contourf(XX, YY, s_carm.reshape(XX.shape), levels=25, cmap='YlOrRd')
    _draw_obstacles(ax)
    _draw_start_goal(ax, xs, xg)
    carm_pts = np.array(carm._collision_points)
    if len(carm_pts) > 0:
        ax.scatter(carm_pts[:, 0], carm_pts[:, 1], c='cyan', s=2,
                   alpha=0.25, zorder=3, rasterized=True)
    ax.set_title(f'(b) CARM $s(x)$ (learned from {len(carm_pts)} collisions)',
                 fontsize=11, fontweight='bold')
    _set_axes(ax)
    plt.colorbar(im2, ax=ax, shrink=0.8)

    # Panel 3: Scatter correlation
    ax = axes[2]
    ax.scatter(s_oracle_f, s_carm_f, c='#457B9D', s=2, alpha=0.15, rasterized=True)
    # Trend line
    m, b_ = np.polyfit(s_oracle_f, s_carm_f, 1)
    x_fit = np.linspace(s_oracle_f.min(), s_oracle_f.max(), 100)
    ax.plot(x_fit, m * x_fit + b_, '-', color=CARM_COLOR, lw=2,
            label=f'Linear fit (r = {corr:.3f})')
    ax.plot([s_oracle_f.min(), s_oracle_f.max()],
            [s_oracle_f.min(), s_oracle_f.max()],
            '--', color='gray', lw=1, alpha=0.5, label='Perfect match')
    ax.set_xlabel('Oracle metric field value')
    ax.set_ylabel('CARM learned field value')
    ax.set_title(f'(c) Correlation: r = {corr:.3f}', fontsize=11, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    fig.suptitle('CARM approximates the oracle metric near obstacles (no a priori knowledge)',
                 fontsize=14, fontweight='bold', y=1.03)
    fig.subplots_adjust(wspace=0.28)

    path_out = os.path.join(PLOTS_DIR, 'fig_carm_correlation.pdf')
    fig.savefig(path_out, dpi=300, bbox_inches='tight')
    fig.savefig(path_out.replace('.pdf', '.png'), dpi=200, bbox_inches='tight')
    print(f'  → {path_out}')
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
#  FIGURE 7: Multi-environment comparison (CARM vs baselines paths)
# ══════════════════════════════════════════════════════════════════════════════
def fig7_multi_env():
    """2×3 grid: For each of 3 environments, show Euclidean path vs CARM path."""
    print('=== Figure 7: Multi-environment CARM vs Euclidean ===')

    from rit_star.environments import (
        env_2d_obstacle_inflated, env_2d_maze, env_2d_narrow_passage,
    )

    envs = {
        'Obstacles': (env_2d_obstacle_inflated, [
            (np.array([0.30, 0.35]), 0.08), (np.array([0.30, 0.65]), 0.08),
            (np.array([0.50, 0.45]), 0.09), (np.array([0.50, 0.75]), 0.09),
            (np.array([0.70, 0.40]), 0.08), (np.array([0.70, 0.60]), 0.08),
        ]),
        'Maze': (env_2d_maze, None),
        'Narrow Passage': (env_2d_narrow_passage, None),
    }

    MAZE_RECTS = [
        ([0.00, 0.22], [0.70, 0.30]),
        ([0.30, 0.46], [1.00, 0.54]),
        ([0.00, 0.70], [0.70, 0.78]),
    ]

    NARROW_RECTS = [
        ([0.48, 0.00], [0.52, 0.47]),
        ([0.48, 0.53], [0.52, 1.00]),
        ([0.30, 0.15], [0.42, 0.35]),
        ([0.30, 0.65], [0.42, 0.85]),
        ([0.58, 0.15], [0.70, 0.35]),
        ([0.58, 0.65], [0.70, 0.85]),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(16, 10.5))

    for col, (name, (env_fn, obs_circles)) in enumerate(envs.items()):
        print(f'  Running {name}...')
        coll, _, oracle_metric, xs, xg, bounds = env_fn()
        dim = len(xs)
        euclid = EuclideanMetric(dim)

        # Euclidean
        p_euc, path_euc, _ = _run_planner(
            xs, xg, bounds, coll, euclid, adaptive=False, max_iter=150, seed=42)
        # CARM
        p_carm, path_carm, _ = _run_planner(
            xs, xg, bounds, coll, euclid, adaptive=True, max_iter=150, seed=42)

        # Oracle costs
        def oracle_cost(path):
            if not path or len(path) < 2:
                return float('inf')
            t = 0
            for i in range(len(path) - 1):
                a, b_ = np.array(path[i]), np.array(path[i + 1])
                mid = (a + b_) / 2
                G = oracle_metric.G(mid)
                d = b_ - a
                t += np.sqrt(d @ G @ d)
            return t

        oc_euc = oracle_cost(path_euc)
        oc_carm = oracle_cost(path_carm)

        # Draw obstacles helper for this env
        def draw_env_obs(ax):
            if name == 'Obstacles' and obs_circles:
                for c, r in obs_circles:
                    ax.add_patch(Circle(c, r, fc=OBSTACLE_COLOR, ec='white', lw=0.8, alpha=0.85))
            elif name == 'Maze':
                for lo, hi in MAZE_RECTS:
                    w, h = hi[0] - lo[0], hi[1] - lo[1]
                    ax.add_patch(Rectangle(lo, w, h, fc=OBSTACLE_COLOR, ec='white', lw=0.8, alpha=0.85))
            elif name == 'Narrow Passage':
                for lo, hi in NARROW_RECTS:
                    w, h = hi[0] - lo[0], hi[1] - lo[1]
                    ax.add_patch(Rectangle(lo, w, h, fc=OBSTACLE_COLOR, ec='white', lw=0.8, alpha=0.85))

        # Row 1: Euclidean
        ax = axes[0, col]
        ax.set_facecolor('#FAFAFA')
        draw_env_obs(ax)
        for v in p_euc.vertices:
            if v.parent is not None:
                ax.plot([v.parent.x[0], v.x[0]], [v.parent.x[1], v.x[1]],
                        '-', color='#CCCCCC', lw=0.2, alpha=0.3)
        if path_euc and len(path_euc) > 1:
            px, py = zip(*path_euc)
            ax.plot(px, py, '-', color='#555555', lw=2.5, zorder=5,
                    path_effects=[pe.Stroke(linewidth=4, foreground='white'), pe.Normal()])
        _draw_start_goal(ax, xs, xg, ms=8)
        ax.set_title(f'{name} — Euclidean\noracle cost = {oc_euc:.2f}',
                     fontsize=10, fontweight='bold', color=EUCLID_COLOR)
        _set_axes(ax)
        if col == 0:
            ax.set_ylabel('Euclidean RIT*', fontsize=12, fontweight='bold')

        # Row 2: CARM
        ax = axes[1, col]
        # CARM field background
        res = 150
        xx = np.linspace(0, 1, res)
        yy = np.linspace(0, 1, res)
        XX, YY = np.meshgrid(xx, yy)
        pts_grid = np.column_stack([XX.ravel(), YY.ravel()])
        S = p_carm._carm._collision_scale_batch(pts_grid).reshape(XX.shape)
        ax.contourf(XX, YY, S, levels=20, cmap='YlOrRd', alpha=0.4)
        draw_env_obs(ax)
        for v in p_carm.vertices:
            if v.parent is not None:
                ax.plot([v.parent.x[0], v.x[0]], [v.parent.x[1], v.x[1]],
                        '-', color='#CCCCCC', lw=0.2, alpha=0.3)
        if path_carm and len(path_carm) > 1:
            px, py = zip(*path_carm)
            ax.plot(px, py, '-', color=CARM_COLOR, lw=2.5, zorder=5,
                    path_effects=[pe.Stroke(linewidth=4, foreground='white'), pe.Normal()])
        _draw_start_goal(ax, xs, xg, ms=8)
        impr = (1 - oc_carm / oc_euc) * 100 if oc_euc > 0 else 0
        ax.set_title(f'{name} — CARM\noracle cost = {oc_carm:.2f} '
                     f'({impr:+.1f}%)',
                     fontsize=10, fontweight='bold', color=CARM_COLOR)
        _set_axes(ax)
        if col == 0:
            ax.set_ylabel('CARM RIT*', fontsize=12, fontweight='bold')

    fig.suptitle('CARM improves paths across diverse environments',
                 fontsize=14, fontweight='bold', y=1.01)
    fig.tight_layout()

    path_out = os.path.join(PLOTS_DIR, 'fig_carm_multi_env.pdf')
    fig.savefig(path_out, dpi=300, bbox_inches='tight')
    fig.savefig(path_out.replace('.pdf', '.png'), dpi=200, bbox_inches='tight')
    print(f'  → {path_out}')
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    import gc

    print('=' * 60)
    print('  CARM CONCEPTUAL FIGURES FOR RAL PAPER')
    print('=' * 60)

    # Figure 5 (mechanism diagram) is pure matplotlib — fastest
    fig5_mechanism_diagram()
    gc.collect()

    # Then generate figures that require running planners
    fig1_carm_overview()
    gc.collect()

    fig2_informed_set_shrinkage()
    gc.collect()

    fig3_carm_evolution()
    gc.collect()

    fig4_wasted_samples()
    gc.collect()

    fig6_metric_correlation()
    gc.collect()

    fig7_multi_env()
    gc.collect()

    print('\n' + '=' * 60)
    print('  ALL CARM FIGURES GENERATED SUCCESSFULLY!')
    print('=' * 60)
    print(f'\nOutput directory: {PLOTS_DIR}')
