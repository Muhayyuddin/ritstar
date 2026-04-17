"""
experiments.py — Benchmark runner for RIT* vs baselines.

Four experiments that empirically validate the theoretical properties
of Riemannian informed sampling:

  Experiment 1 — Convergence comparison (Theorem 3, empirical).
  Experiment 2 — Volume-ratio sweep (Theorem 1, empirical).
  Experiment 3 — Scaling with anisotropy κ (Theorem 3, κ^(1/d) curve).
  Experiment 4 — Full 3-D benchmark on the sphere-field environment.

All experiments accept a ``random_seed`` parameter so results are
fully reproducible.
"""

from __future__ import annotations

import gc
import os
import time
import numpy as np
import matplotlib.pyplot as plt

from output_paths import IMAGES_DIR, GIFS_DIR, PLOTS_DIR
from .rit_star import RITStar
from .baselines import InformedRRTStar, BITStar
from .environments import (
    env_2d_diagonal_anisotropic,
    env_2d_obstacle_inflated,
    env_3d_sphere_field,
    env_3d_diagonal_anisotropic,
    ALL_ENVS,
    _make_diagonal_env_nd,
)
from .metric import DiagonalAnisotropicMetric, EuclideanMetric
from .geodesic import GeodesicComputer
from .informed_set import (
    RiemannianInformedSet, EuclideanInformedSet, volume_ratio_bound,
)
from .visualize import (
    plot_cost_convergence,
    plot_volume_heatmap,
    plot_anisotropy_speedup,
    plot_3d_tree,
    animate_convergence,
    animate_speedup,
    animate_planning_3d,
    plot_volume_ratio_validation,
    plot_convergence_rate_separation,
    plot_sample_efficiency,
)


# ═══════════════════════════════════════════════════════════════════════
# Experiment 1 — Convergence comparison
# ═══════════════════════════════════════════════════════════════════════

def experiment_1_convergence_comparison(n_trials: int = 30,
                                        max_iterations: int = 300,
                                        random_seed: int = 0):
    """Run RIT*, Informed RRT*, BIT* on the 2-D diagonal-anisotropic
    environment and compare convergence speed.

    Parameters
    ----------
    n_trials : int
        Independent repetitions (default 30).
    max_iterations : int
        Per-planner iteration budget (default 300).
    random_seed : int
        Base seed for reproducibility.

    Returns
    -------
    fig : matplotlib Figure
        Two-panel figure: cost convergence + volume comparison.

    Notes
    -----
    Validates the empirical claim that RIT* converges faster per sample
    in anisotropic spaces (Theorem 3).
    """
    print('  Experiment 1: convergence comparison')
    coll, _, metric, xs, xg, bounds = env_2d_diagonal_anisotropic()

    all_rit, all_irrt, all_bit = [], [], []

    for trial in range(n_trials):
        seed = random_seed + trial
        print(f'    trial {trial + 1}/{n_trials}', end='\r')

        rit = RITStar(xs, xg, bounds, coll, metric,
                      geodesic_tier='diagonal', batch_size=100,
                      max_iterations=max_iterations, random_seed=seed)
        rit.plan()
        all_rit.append(rit.get_stats())
        del rit

        irrt = InformedRRTStar(xs, xg, bounds, coll, metric,
                               batch_size=100, max_iterations=max_iterations,
                               random_seed=seed)
        irrt.plan()
        all_irrt.append(irrt.get_stats())
        del irrt

        bit = BITStar(xs, xg, bounds, coll, metric,
                      batch_size=100, max_iterations=max_iterations,
                      random_seed=seed)
        bit.plan()
        all_bit.append(bit.get_stats())
        del bit
        gc.collect()

    print()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    plot_cost_convergence(all_rit, all_irrt, all_bit, ax=axes[0])

    # Volume comparison for a single representative trial
    stats_r = all_rit[0]
    iters = [s['iteration'] for s in stats_r]
    vol_r = [s['informed_set_volume'] for s in stats_r]
    vol_e = [s['euclidean_set_volume'] for s in stats_r]
    axes[1].plot(iters, vol_r, 'purple', lw=2, label='Vol(I_R)')
    axes[1].plot(iters, vol_e, 'gray', lw=2, label='Vol(I_euclid)')
    axes[1].set_xlabel('Iteration')
    axes[1].set_ylabel('Informed-set volume')
    axes[1].set_title('Volume: I_R vs I_euclid')
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, 'experiment_1_convergence.png'), dpi=150)
    print('  \u2192 saved experiment_1_convergence.png')

    # Animated convergence GIF
    animate_convergence(all_rit, all_irrt, all_bit,
                        filename=os.path.join(GIFS_DIR, 'experiment_1_convergence.gif'))

    return fig


# ═══════════════════════════════════════════════════════════════════════
# Experiment 2 — Volume ratio
# ═══════════════════════════════════════════════════════════════════════

def experiment_2_volume_ratio(n_mc: int = 10000,
                              random_seed: int = 42):
    """Compute Vol(I_R)/Vol(I_euclid) as a function of c_best/c_min
    and anisotropy ratio κ.

    Parameters
    ----------
    n_mc : int
        Monte Carlo samples per cell (default 10000).
    random_seed : int

    Returns
    -------
    fig : matplotlib Figure
        Heatmap of volume ratios.

    Notes
    -----
    Empirically validates Theorem 1: Vol(I_R) < Vol(I_euclid) when G≠I.
    """
    print('  Experiment 2: volume ratio heatmap')
    rng = np.random.default_rng(random_seed)
    xs = np.array([0.1, 0.5])
    xg = np.array([0.9, 0.5])
    bounds = [(0.0, 1.0), (0.0, 1.0)]

    kappa_values = np.array([1.0, 2.0, 4.0, 6.0, 8.0, 10.0, 15.0, 20.0])
    c_min_euclid = float(np.linalg.norm(xg - xs))
    c_ratio_values = np.array([1.05, 1.1, 1.2, 1.5, 2.0, 3.0, 5.0])

    ratios = np.zeros((len(kappa_values), len(c_ratio_values)))

    for ik, kappa in enumerate(kappa_values):
        metric = DiagonalAnisotropicMetric(weights=[kappa, 1.0])
        gc = GeodesicComputer(metric, tier='diagonal', bounds=bounds)
        c_min_riem = gc.distance(xs, xg)

        for ic, cr in enumerate(c_ratio_values):
            c_best = cr * c_min_riem
            ris = RiemannianInformedSet(xs, xg, c_best, gc, bounds=bounds)
            eis = EuclideanInformedSet(xs, xg, c_best, bounds=bounds)

            v_r = ris.volume_estimate(n_mc, rng=rng)
            v_e = eis.volume_estimate(n_mc, rng=rng)
            ratios[ik, ic] = v_r / max(v_e, 1e-12)
        print(f'    κ = {kappa:.0f} done', end='\r')

    print()

    fig, ax = plt.subplots(figsize=(8, 5))
    plot_volume_heatmap(ratios, kappa_values, c_ratio_values, ax=ax)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, 'experiment_2_volume_ratio.png'), dpi=150)
    print('  → saved experiment_2_volume_ratio.png')
    return fig


# ═══════════════════════════════════════════════════════════════════════
# Experiment 3 — Scaling with anisotropy
# ═══════════════════════════════════════════════════════════════════════

def experiment_3_scaling_with_anisotropy(n_trials: int = 20,
                                          random_seed: int = 0):
    """Vary anisotropy from κ = 1 to 20 and measure convergence rate
    advantage of Riemannian informed sampling over Euclidean.

    Parameters
    ----------
    n_trials : int
        Trials per κ value (default 20).
    random_seed : int

    Returns
    -------
    fig : matplotlib Figure
        Speedup factor vs κ with theoretical κ^(1/d) overlay.

    Notes
    -----
    Key validation of Theorem 3: RIT* converges faster by a factor
    that scales as κ^(1/d) compared to Informed RRT*.

    Measures the cost gap at a *fixed iteration budget* rather than
    iterations-to-target, which avoids the degeneracy where both
    planners reach c_opt in early iterations for easy problems.
    The metric: gap_IRRT(it) / gap_RIT(it) averaged over a range
    of iteration counts captures the sustained convergence advantage.
    """
    print('  Experiment 3: scaling with anisotropy')
    xs = np.array([0.1, 0.5])
    xg = np.array([0.9, 0.5])
    bounds = [(0.0, 1.0), (0.0, 1.0)]

    # Obstacles that force non-trivial path search, making informed
    # set quality matter.  Two rectangles create a narrow gap.
    obs = [
        (np.array([0.3, 0.0]), np.array([0.35, 0.40])),
        (np.array([0.3, 0.60]), np.array([0.35, 1.0])),
        (np.array([0.6, 0.0]), np.array([0.65, 0.35])),
        (np.array([0.6, 0.55]), np.array([0.65, 1.0])),
    ]

    def free(x):
        if not (np.all(x >= 0.0) and np.all(x <= 1.0)):
            return False
        for lo, hi in obs:
            if np.all(x >= lo) and np.all(x <= hi):
                return False
        return True

    kappa_list = [1.0, 2.0, 4.0, 8.0, 12.0, 16.0]
    speedups = []

    max_it = 200
    batch_sz = 30  # small batches → more iterations → finer measurement

    # Measure cost gap at these iteration indices (after initial solution)
    measure_iters = list(range(30, max_it, 10))

    for kappa in kappa_list:
        metric = DiagonalAnisotropicMetric(weights=[kappa, 1.0])

        trial_speedups = []

        for trial in range(n_trials):
            seed = random_seed + trial

            rit = RITStar(xs, xg, bounds, free, metric,
                          geodesic_tier='diagonal', batch_size=batch_sz,
                          max_iterations=max_it, random_seed=seed)
            rit.plan()
            stats_r = rit.get_stats()
            del rit

            irrt = InformedRRTStar(xs, xg, bounds, free, metric,
                                   batch_size=batch_sz, max_iterations=max_it,
                                   random_seed=seed)
            irrt.plan()
            stats_i = irrt.get_stats()
            del irrt
            gc.collect()

            # Best achievable cost as reference c*
            c_best_r = min((s['c_best'] for s in stats_r), default=np.inf)
            c_best_i = min((s['c_best'] for s in stats_i), default=np.inf)
            c_opt = min(c_best_r, c_best_i)
            if not np.isfinite(c_opt) or c_opt < 1e-8:
                continue

            # Compute gap ratio at each measurement iteration
            gap_ratios = []
            for mi in measure_iters:
                if mi >= len(stats_r) or mi >= len(stats_i):
                    break
                cr = stats_r[mi]['c_best']
                ci = stats_i[mi]['c_best']
                if not np.isfinite(cr) or not np.isfinite(ci):
                    continue
                gap_r = max((cr - c_opt) / c_opt, 1e-8)
                gap_i = max((ci - c_opt) / c_opt, 1e-8)
                gap_ratios.append(gap_i / gap_r)

            if gap_ratios:
                trial_speedups.append(float(np.median(gap_ratios)))

        if trial_speedups:
            speedup = float(np.median(trial_speedups))
        else:
            speedup = 1.0
        speedups.append(speedup)
        print(f'    κ = {kappa:.0f}  speedup = {speedup:.2f}')

    print()

    kappa_arr = np.array(kappa_list)
    speedup_arr = np.array(speedups)

    fig, ax = plt.subplots(figsize=(7, 5))
    plot_anisotropy_speedup(kappa_arr, speedup_arr, dim=2, ax=ax)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, 'experiment_3_anisotropy_speedup.png'), dpi=150)
    print('  \u2192 saved experiment_3_anisotropy_speedup.png')

    # Animated speedup GIF
    animate_speedup(kappa_arr, speedup_arr, dim=2,
                    filename=os.path.join(GIFS_DIR, 'experiment_3_speedup.gif'))

    return fig


# ═══════════════════════════════════════════════════════════════════════
# Experiment 4 — Full 3-D benchmark
# ═══════════════════════════════════════════════════════════════════════

def experiment_4_3d_full(n_trials: int = 20,
                         max_iterations: int = 500,
                         random_seed: int = 0):
    """Full 3-D benchmark on the sphere-field environment.

    Parameters
    ----------
    n_trials : int
    max_iterations : int
    random_seed : int

    Returns
    -------
    fig : matplotlib Figure
        Cost convergence + 3-D tree visualisation.

    Notes
    -----
    Demonstrates RIT* advantage in 3-D with obstacle-inflated metric.
    """
    print('  Experiment 4: 3-D sphere-field benchmark')
    coll, _, metric, xs, xg, bounds = env_3d_sphere_field()

    all_rit, all_irrt, all_bit = [], [], []
    best_rit_path = []
    best_rit_verts = []
    # Per-iteration snapshots for first trial's growing-tree GIF
    verts_per_it = []
    edges_per_it = []
    path_per_it = []

    for trial in range(n_trials):
        seed = random_seed + trial
        print(f'    trial {trial + 1}/{n_trials}', end='\r')

        rit = RITStar(xs, xg, bounds, coll, metric,
                      geodesic_tier='diagonal', batch_size=50,
                      max_iterations=max_iterations, random_seed=seed)

        if trial == 0:
            # Manual iteration loop to capture snapshots
            import time as _time
            rit._t0 = _time.time()
            for it in range(rit.max_iterations):
                samples = rit._sample_batch()
                rit._extend_tree(samples)
                if rit.c_best < np.inf:
                    rit._prune()
                    rit._update_informed_set()
                rit._record_stats(it, _time.time() - rit._t0)
                snap_v = [v.x.copy() for v in rit.vertices]
                snap_e = []
                vidx = {id(v): i for i, v in enumerate(rit.vertices)}
                for v in rit.vertices:
                    if v.parent is not None and id(v.parent) in vidx:
                        snap_e.append((vidx[id(v.parent)], vidx[id(v)]))
                verts_per_it.append(snap_v)
                edges_per_it.append(snap_e)
                path_per_it.append(rit._extract_path())
            path = rit._extract_path()
            cost = rit.c_best
        else:
            path, cost = rit.plan()

        all_rit.append(rit.get_stats())
        if trial == 0:
            best_rit_path = path
            best_rit_verts = rit.vertices[:]

        irrt = InformedRRTStar(xs, xg, bounds, coll, metric,
                               batch_size=50, max_iterations=max_iterations,
                               random_seed=seed)
        irrt.plan()
        all_irrt.append(irrt.get_stats())
        del irrt

        bit = BITStar(xs, xg, bounds, coll, metric,
                      batch_size=50, max_iterations=max_iterations,
                      random_seed=seed)
        bit.plan()
        all_bit.append(bit.get_stats())
        del bit
        gc.collect()

    print()

    fig = plt.figure(figsize=(14, 6))

    ax1 = fig.add_subplot(121)
    plot_cost_convergence(all_rit, all_irrt, all_bit, ax=ax1)
    ax1.set_title('3-D sphere field — cost convergence')

    ax2 = fig.add_subplot(122, projection='3d')
    # Build edges from Node objects
    vert_coords = [v.x for v in best_rit_verts]
    edges = []
    vert_idx = {id(v): i for i, v in enumerate(best_rit_verts)}
    for v in best_rit_verts:
        if v.parent is not None and id(v.parent) in vert_idx:
            edges.append((vert_idx[id(v.parent)], vert_idx[id(v)]))
    # Obstacles
    offsets = np.array([-0.35, 0.35])
    sphere_obs = []
    for sx in offsets:
        for sy in offsets:
            for sz in offsets:
                sphere_obs.append((np.array([sx, sy, sz]), 0.18))

    plot_3d_tree(vert_coords, edges, best_rit_path, sphere_obs, ax=ax2)
    ax2.set_title('RIT* — 3-D tree & path')

    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, 'experiment_4_3d_full.png'), dpi=150)
    print('  \u2192 saved experiment_4_3d_full.png')

    # Tree-growing + rotation GIF
    animate_planning_3d(verts_per_it, edges_per_it, path_per_it,
                        sphere_obs, xs, xg,
                        filename=os.path.join(GIFS_DIR, 'experiment_4_3d_growing.gif'),
                        n_grow_frames=40, n_rotate_frames=18,
                        interval=200)

    return fig


# ═══════════════════════════════════════════════════════════════════════
# Experiment 5 — Volume ratio validation (Theorem 1)
# ═══════════════════════════════════════════════════════════════════════

def experiment_volume_ratio_validation(
        kappa_values=None, dims=None,
        n_mc: int = 20000, random_seed: int = 42):
    """Validate Theorem 1: analytical Vol(I_R)/Vol(I_E) vs Monte Carlo.

    For each (kappa, dim):
      1. Create DiagonalAnisotropicMetric with kappa = w_max/w_min.
      2. Run RIT* briefly to get a finite c_best.
      3. Compute analytical volume ratio from Theorem 1.
      4. Compute Monte Carlo volume ratio.
      5. Compare — should match within sampling noise.

    Parameters
    ----------
    kappa_values : list of float
    dims : list of int
    n_mc : int
    random_seed : int

    Returns
    -------
    fig : matplotlib Figure
    """
    if kappa_values is None:
        kappa_values = [1.0, 2.0, 4.0, 8.0, 16.0]
    if dims is None:
        dims = [2, 3]

    print('  Experiment 5: volume ratio validation (Theorem 1)')
    rng = np.random.default_rng(random_seed)

    analytical_all = []
    mc_all = []
    labels = []

    for d in dims:
        for kappa in kappa_values:
            coll, _, metric, xs, xg, bounds = _make_diagonal_env_nd(
                d, kappa, seed=random_seed)
            gc_obj = GeodesicComputer(metric, tier='diagonal', bounds=bounds)

            c_min_riem = gc_obj.distance(xs, xg)
            c_min_euclid = float(np.linalg.norm(np.asarray(xg) - np.asarray(xs)))

            # Use same c_best for both (just above Riemannian c_min)
            c_best = c_min_riem * 1.05

            # Analytical (Theorem 1) — exact formula including eccentricity
            a_vr = volume_ratio_bound(metric, xs, xg, d, c_best=c_best)

            # Monte Carlo volume estimation.
            # Use a tight bounding box centered at the midpoint with
            # radius = c_best/2 (the Euclidean ellipsoid radius).
            # Both sets use the SAME bounding box so MC acceptance
            # rates are comparable.
            mid = 0.5 * (np.asarray(xs) + np.asarray(xg))
            r_box = c_best / 2.0 + 0.05  # slight margin
            mc_bounds = [(float(mid[k] - r_box), float(mid[k] + r_box))
                         for k in range(d)]
            ris = RiemannianInformedSet(xs, xg, c_best, gc_obj, bounds=mc_bounds)
            eis = EuclideanInformedSet(xs, xg, c_best, bounds=mc_bounds)
            v_r = ris.volume_estimate(n_mc, rng=rng)
            v_e = eis.volume_estimate(n_mc, rng=rng)
            mc_vr = v_r / max(v_e, 1e-12)

            analytical_all.append(a_vr)
            mc_all.append(mc_vr)
            labels.append(f'd={d},κ={kappa:.0f}')
            print(f'    d={d}, κ={kappa:.0f}: analytical={a_vr:.4f}, MC={mc_vr:.4f}')

            gc.collect()

    fig = plot_volume_ratio_validation(
        np.array(analytical_all), np.array(mc_all),
        np.array(kappa_values), labels)
    fig.savefig(os.path.join(PLOTS_DIR, 'experiment_5_volume_ratio_validation.png'),
                dpi=150)
    print('  → saved experiment_5_volume_ratio_validation.png')
    return fig


# ═══════════════════════════════════════════════════════════════════════
# Experiment 6 — AO validation (Theorem 2)
# ═══════════════════════════════════════════════════════════════════════

def experiment_ao_validation(
        n_samples_list=None, n_trials: int = 10,
        random_seed: int = 0):
    """Validate asymptotic optimality under the metric-adapted radius.

    Run RIT* with increasing sample budgets and verify:
      1. The planner finds a solution (probabilistic completeness).
      2. The cost monotonically decreases toward c*.
      3. The gap c_n - c* decreases at the predicted rate.

    Parameters
    ----------
    n_samples_list : list of int
    n_trials : int
    random_seed : int

    Returns
    -------
    fig : matplotlib Figure
    """
    if n_samples_list is None:
        n_samples_list = [100, 500, 1000, 3000, 5000]

    print('  Experiment 6: AO validation (Theorem 2)')
    coll, _, metric, xs, xg, bounds = env_2d_diagonal_anisotropic()

    # Reference: long run to approximate c*
    ref_planner = RITStar(xs, xg, bounds, coll, metric,
                          geodesic_tier='diagonal', batch_size=100,
                          max_iterations=800, random_seed=random_seed)
    ref_planner.plan()
    c_star = ref_planner.c_best
    del ref_planner
    gc.collect()

    mean_gaps = []
    std_gaps = []

    for budget in n_samples_list:
        max_it = max(budget // 100, 5)
        gaps = []
        for trial in range(n_trials):
            seed = random_seed + trial
            planner = RITStar(xs, xg, bounds, coll, metric,
                              geodesic_tier='diagonal', batch_size=100,
                              max_iterations=max_it, random_seed=seed)
            planner.plan()
            if np.isfinite(planner.c_best):
                gaps.append(planner.c_best - c_star)
            del planner
        gc.collect()
        mean_gaps.append(np.mean(gaps) if gaps else np.inf)
        std_gaps.append(np.std(gaps) if gaps else 0.0)
        print(f'    budget={budget}: mean gap={mean_gaps[-1]:.4f}')

    fig, ax = plt.subplots(figsize=(7, 5))
    budgets = np.array(n_samples_list, dtype=float)
    mg = np.array(mean_gaps)
    sg = np.array(std_gaps)
    finite = np.isfinite(mg)
    ax.errorbar(budgets[finite], mg[finite], yerr=sg[finite],
                fmt='o-', color='purple', capsize=4, label='RIT* gap')
    # Theoretical rate: C * n^(-2/d) for d=2
    if np.any(finite):
        C_fit = mg[finite][-1] * budgets[finite][-1]
        theory_line = C_fit / budgets
        ax.plot(budgets, theory_line, 'k--', lw=1.5,
                label='O(n⁻¹) (d=2 theory)')
    ax.set_xlabel('Total samples (n)')
    ax.set_ylabel('c_best − c*')
    ax.set_title('AO Validation — gap vs sample budget (Theorem 2)')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, 'experiment_6_ao_validation.png'), dpi=150)
    print('  → saved experiment_6_ao_validation.png')
    return fig


# ═══════════════════════════════════════════════════════════════════════
# Experiment 7 — Convergence rate separation (Theorem 3)
# ═══════════════════════════════════════════════════════════════════════

def experiment_convergence_rate_separation(
        kappa_values=None, dims=None,
        n_trials: int = 20, max_iterations: int = 300,
        random_seed: int = 0):
    """Validate Theorem 3: RIT* converges faster by factor kappa^(1/d).

    For each (kappa, d):
      1. Create a d-dimensional diagonal environment.
      2. Run RIT* and BIT* for n_trials.
      3. Record c_best per iteration, compute gap vs reference c*.
      4. Compare empirical convergence ratio against kappa^(1/d).

    Uses cost-gap-at-fixed-iteration methodology: the speedup is
    measured as the ratio of cost gaps (BIT* gap / RIT* gap) at a
    set of mid-run iteration checkpoints, then normalized by the
    kappa=1 baseline to isolate the metric effect.

    Parameters
    ----------
    kappa_values : list of float
    dims : list of int
    n_trials : int
    max_iterations : int
    random_seed : int

    Returns
    -------
    fig : matplotlib Figure
    """
    if kappa_values is None:
        kappa_values = [1.0, 2.0, 4.0, 8.0, 16.0]
    if dims is None:
        dims = [2, 3]

    print('  Experiment 7: convergence rate separation (Theorem 3)')

    results = {}  # (kappa, dim) -> dict

    # Checkpoint iterations for gap measurement (start early enough
    # that short runs from early-stopping still produce data)
    measure_range = list(range(5, min(max_iterations, 150), 5))

    for d in dims:
        for kappa in kappa_values:
            coll, _, metric, xs, xg, bounds = _make_diagonal_env_nd(
                d, kappa, seed=random_seed)

            # Run both planners for fixed budget, compare cost gap curves
            trial_speedups = []
            all_rit_curves = []
            all_bit_curves = []

            for trial in range(n_trials):
                seed = random_seed + trial

                # RIT* with Riemannian informed set (anisotropic)
                # Disable early stopping so both planners run the same
                # number of iterations for a fair gap comparison.
                rit = RITStar(xs, xg, bounds, coll, metric,
                              geodesic_tier='diagonal', batch_size=50,
                              max_iterations=max_iterations, random_seed=seed)
                rit._early_stop_window = max_iterations + 1  # effectively disable
                rit.plan()
                rit_stats = rit.get_stats()
                all_rit_curves.append([s['c_best'] for s in rit_stats])
                del rit

                # BIT* with Euclidean informed set (same cost metric,
                # different sampling region). This isolates the
                # informed-set volume effect for Theorem 3 validation.
                bit = BITStar(xs, xg, bounds, coll, metric,
                              batch_size=50, max_iterations=max_iterations,
                              random_seed=seed)
                bit.plan()
                bit_stats = bit.get_stats()
                all_bit_curves.append([s['c_best'] for s in bit_stats])
                del bit
                gc.collect()

                # Best of both as c* estimate
                c_opt_r = min((s['c_best'] for s in rit_stats), default=np.inf)
                c_opt_b = min((s['c_best'] for s in bit_stats), default=np.inf)
                c_opt = min(c_opt_r, c_opt_b)
                if not np.isfinite(c_opt) or c_opt < 1e-8:
                    continue

                # Measure gap ratio at each checkpoint
                gap_ratios = []
                for mi in measure_range:
                    if mi >= len(rit_stats) or mi >= len(bit_stats):
                        break
                    cr = rit_stats[mi]['c_best']
                    cb = bit_stats[mi]['c_best']
                    if not np.isfinite(cr) or not np.isfinite(cb):
                        continue
                    gap_r = max((cr - c_opt) / c_opt, 1e-6)
                    gap_b = max((cb - c_opt) / c_opt, 1e-6)
                    gap_ratios.append(gap_b / gap_r)

                if gap_ratios:
                    trial_speedups.append(float(np.median(gap_ratios)))

            mean_speedup = float(np.median(trial_speedups)) if trial_speedups else float('nan')
            # Also compute old-style final gaps for reporting
            rit_final_gaps = []
            bit_final_gaps = []
            for rc, bc in zip(all_rit_curves, all_bit_curves):
                c_opt = min(min(rc, default=np.inf), min(bc, default=np.inf))
                if np.isfinite(c_opt) and c_opt > 0:
                    rf = rc[-1] if rc else np.inf
                    bf = bc[-1] if bc else np.inf
                    if np.isfinite(rf):
                        rit_final_gaps.append((rf - c_opt) / c_opt)
                    if np.isfinite(bf):
                        bit_final_gaps.append((bf - c_opt) / c_opt)

            results[(kappa, d)] = {
                'speedup': mean_speedup,
                'rit_curves': all_rit_curves,
                'bit_curves': all_bit_curves,
                'mean_rit_gap': float(np.mean(rit_final_gaps)) if rit_final_gaps else float('nan'),
                'mean_bit_gap': float(np.mean(bit_final_gaps)) if bit_final_gaps else float('nan'),
            }
            print(f'    d={d}, κ={kappa:.0f}: speedup={mean_speedup:.2f} '
                  f'(theory κ^(1/d)={kappa**(1.0/d):.2f})')

    # Normalize by κ=1 baseline
    print('\n    Normalized speedups (metric effect only):')
    for d in dims:
        baseline = results.get((1.0, d), {}).get('speedup', 1.0)
        if not np.isfinite(baseline) or baseline == 0:
            baseline = 1.0
        for kappa in kappa_values:
            r = results.get((kappa, d), {})
            raw = r.get('speedup', float('nan'))
            normed = raw / baseline if np.isfinite(raw) else float('nan')
            r['speedup_normalized'] = normed
            print(f'      d={d}, κ={kappa:.0f}: raw={raw:.2f}, '
                  f'normalized={normed:.2f}, theory={kappa**(1.0/d):.2f}')

    fig = plot_convergence_rate_separation(results, kappa_values, dims)
    fig.savefig(os.path.join(PLOTS_DIR, 'experiment_7_convergence_separation.png'),
                dpi=150)
    print('  → saved experiment_7_convergence_separation.png')
    return fig


# ═══════════════════════════════════════════════════════════════════════
# Experiment 8 — CARM: Collision-Adaptive Riemannian Metric
# ═══════════════════════════════════════════════════════════════════════

def _path_cost_under_metric(path, metric):
    """Re-evaluate a path's total cost under a given metric.

    Parameters
    ----------
    path : list of (d,) arrays — waypoints from start to goal.
    metric : RiemannianMetric — the metric to use for evaluation.

    Returns
    -------
    float — total Riemannian arc length, or np.inf if path is empty.
    """
    if not path or len(path) < 2:
        return np.inf
    from .rit_star import riemannian_edge_cost
    return sum(riemannian_edge_cost(path[i], path[i + 1], metric)
               for i in range(len(path) - 1))


def experiment_carm_comparison(
        n_trials: int = 20,
        max_iterations: int = 300,
        random_seed: int = 0):
    """Compare RIT* variants to validate CARM's effectiveness.

    **All costs are evaluated under the oracle (obstacle-inflated) metric**
    so that results across variants are directly comparable.

    Variants compared:
      1. RIT* (oracle)    — a-priori known metric (ground truth).
      2. RIT* (CARM)      — starts from Euclidean, learns metric online.
      3. RIT* (Euclidean)  — Euclidean metric, no adaptation (ablation).
      4. BIT* (Euclidean)  — baseline without any Riemannian structure.

    Parameters
    ----------
    n_trials : int
        Independent repetitions.
    max_iterations : int
        Per-planner iteration budget.
    random_seed : int

    Returns
    -------
    fig : matplotlib Figure
    """
    from .environments import (
        env_2d_obstacle_inflated,
        env_2d_random_forest,
        env_2d_maze,
    )
    from .metric import EuclideanMetric

    print('  Experiment 8: CARM — Collision-Adaptive Riemannian Metric')
    print('  (all costs evaluated under oracle metric)')

    test_envs = {
        '2D Obstacles': env_2d_obstacle_inflated,
        '2D Forest':    env_2d_random_forest,
        '2D Maze':      env_2d_maze,
    }

    variant_colors = {
        'RIT* (oracle)':    '#7B2FBE',
        'RIT* (CARM)':      '#E91E63',
        'RIT* (Euclidean)': '#9E9E9E',
        'BIT* (Euclidean)': '#4CAF50',
    }
    variant_names = list(variant_colors.keys())

    all_results = {}

    for env_name, env_fn in test_envs.items():
        print(f'\n    Environment: {env_name}')
        coll, _, oracle_metric, xs, xg, bounds = env_fn()
        dim = len(xs)
        euclid = EuclideanMetric(dim)

        results = {name: [] for name in variant_names}

        for trial in range(n_trials):
            seed = random_seed + trial
            print(f'      trial {trial + 1}/{n_trials}', end='\r')

            # 1. RIT* with oracle metric (a priori known)
            rit_oracle = RITStar(
                xs, xg, bounds, coll, oracle_metric,
                geodesic_tier='diagonal', batch_size=100,
                max_iterations=max_iterations, random_seed=seed)
            oracle_curves = []
            for step in rit_oracle.plan_stepwise():
                c = _path_cost_under_metric(step['path'], oracle_metric)
                oracle_curves.append(c)
            final_path = rit_oracle._extract_path()
            final_oracle = _path_cost_under_metric(final_path, oracle_metric)
            results['RIT* (oracle)'].append({
                'oracle_curve': oracle_curves,
                'final_cost': final_oracle,
            })
            del rit_oracle

            # 2. RIT* with CARM (Euclidean base, learns online)
            rit_carm = RITStar(
                xs, xg, bounds, coll, euclid,
                geodesic_tier='diagonal', batch_size=100,
                max_iterations=max_iterations, random_seed=seed,
                adaptive_metric=True,
                carm_sigma=0.08, carm_alpha=6.0,
                carm_rebuild_interval=15)
            carm_curves = []
            for step in rit_carm.plan_stepwise():
                c = _path_cost_under_metric(step['path'], oracle_metric)
                carm_curves.append(c)
            final_path = rit_carm._extract_path()
            final_oracle = _path_cost_under_metric(final_path, oracle_metric)
            carm_pts = rit_carm._carm.n_collision_points if rit_carm._carm else 0
            results['RIT* (CARM)'].append({
                'oracle_curve': carm_curves,
                'final_cost': final_oracle,
                'carm_points': carm_pts,
            })
            del rit_carm

            # 3. RIT* with Euclidean metric (no adaptation — ablation)
            rit_euclid = RITStar(
                xs, xg, bounds, coll, euclid,
                geodesic_tier='diagonal', batch_size=100,
                max_iterations=max_iterations, random_seed=seed)
            euclid_curves = []
            for step in rit_euclid.plan_stepwise():
                c = _path_cost_under_metric(step['path'], oracle_metric)
                euclid_curves.append(c)
            final_path = rit_euclid._extract_path()
            final_oracle = _path_cost_under_metric(final_path, oracle_metric)
            results['RIT* (Euclidean)'].append({
                'oracle_curve': euclid_curves,
                'final_cost': final_oracle,
            })
            del rit_euclid

            # 4. BIT* baseline
            bit = BITStar(
                x_start=xs, x_goal=xg, c_space_bounds=bounds,
                collision_checker=coll, metric=euclid,
                batch_size=100, max_iterations=max_iterations,
                random_seed=seed)
            bit_curves = []
            for step in bit.plan_stepwise():
                c = _path_cost_under_metric(step['path'], oracle_metric)
                bit_curves.append(c)
            final_path = bit._extract_path()
            final_oracle = _path_cost_under_metric(final_path, oracle_metric)
            results['BIT* (Euclidean)'].append({
                'oracle_curve': bit_curves,
                'final_cost': final_oracle,
            })
            del bit
            gc.collect()

        print()
        all_results[env_name] = results

    # ── Plot results (oracle-metric cost convergence) ─────────────────
    n_envs = len(test_envs)
    fig, axes = plt.subplots(1, n_envs, figsize=(6 * n_envs, 5))
    if n_envs == 1:
        axes = [axes]

    for idx, (env_name, results) in enumerate(all_results.items()):
        ax = axes[idx]
        for vname in variant_names:
            trial_data = results[vname]
            if not trial_data:
                continue
            max_it = max(len(td['oracle_curve']) for td in trial_data)
            cost_matrix = np.full((len(trial_data), max_it), np.nan)
            for ti, td in enumerate(trial_data):
                for si, c in enumerate(td['oracle_curve']):
                    cost_matrix[ti, si] = c
            # Forward-fill NaN (keep last known cost)
            for ti in range(cost_matrix.shape[0]):
                for si in range(1, cost_matrix.shape[1]):
                    if np.isnan(cost_matrix[ti, si]):
                        cost_matrix[ti, si] = cost_matrix[ti, si - 1]

            iters = np.arange(max_it)
            mean_c = np.nanmean(cost_matrix, axis=0)
            std_c = np.nanstd(cost_matrix, axis=0)
            color = variant_colors[vname]
            ax.plot(iters, mean_c, color=color, lw=2, label=vname)
            ax.fill_between(iters, mean_c - std_c, mean_c + std_c,
                            color=color, alpha=0.12)

        ax.set_xlabel('Iteration')
        ax.set_ylabel('Oracle-metric cost')
        ax.set_title(f'{env_name} — CARM comparison')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        # Clip y range
        finite = []
        for vname in variant_names:
            for td in results[vname]:
                for c in td['oracle_curve']:
                    if np.isfinite(c):
                        finite.append(c)
        if finite:
            ax.set_ylim(min(finite) * 0.95,
                        np.percentile(finite, 90) * 1.05)

    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, 'experiment_8_carm_comparison.png'),
                dpi=150)
    print('  → saved experiment_8_carm_comparison.png')

    # ── Summary table (oracle-metric final cost) ──────────────────────
    print('\n  CARM Summary — oracle-metric final cost (mean ± std):')
    print(f'  {"Environment":<16} {"RIT*(oracle)":<18} {"RIT*(CARM)":<18} '
          f'{"RIT*(Euclid)":<18} {"BIT*(Euclid)":<18}')
    print('  ' + '-' * 88)
    for env_name, results in all_results.items():
        parts = [f'{env_name:<16}']
        for vname in variant_names:
            costs = [td['final_cost'] for td in results[vname]
                     if np.isfinite(td['final_cost'])]
            if costs:
                parts.append(f'{np.mean(costs):.4f}±{np.std(costs):.4f}'.ljust(18))
            else:
                parts.append('no sol.'.ljust(18))
        print('  ' + ' '.join(parts))

    return fig
