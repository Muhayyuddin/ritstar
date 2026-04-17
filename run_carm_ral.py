#!/usr/bin/env python3
"""
run_carm_ral.py — CARM-focused RAL experiment suite.

Headline: "A planner that learns its own Riemannian metric from collision
feedback, with no a priori cost model."

Experiments:
  1. CARM vs Oracle vs Euclidean vs BIT* (4 variants × 7 envs × 30 trials)
     - All paths evaluated under oracle metric for fair comparison
     - Per-iteration oracle-metric convergence curves
     - Statistical significance (Wilcoxon signed-rank test)
  2. CARM ablation: sigma sweep, alpha sweep, rebuild interval sweep
  3. CARM metric quality: learned vs oracle metric field correlation
  4. Scalability: CARM on 3D environments

Output:
  results/carm_ral_*.csv       — summary tables
  visualization/plots/carm_*   — figures (PDF for paper, PNG for quick view)
"""

from __future__ import annotations

import gc
import os
import pickle
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy.stats import wilcoxon, mannwhitneyu

from output_paths import RESULTS_DIR, PLOTS_DIR

from rit_star.rit_star import RITStar, riemannian_edge_cost
from rit_star.baselines import InformedRRTStar, BITStar, AITStar, EITStar, APTStar
from rit_star.metric import EuclideanMetric
from rit_star.environments import (
    env_2d_obstacle_inflated,
    env_2d_narrow_passage,
    env_2d_maze,
    env_2d_bug_trap,
    env_2d_random_forest,
    env_2d_terrain,
    env_2d_diagonal_anisotropic,
    env_3d_sphere_field,
    env_3d_diagonal_anisotropic,
)


# ═══════════════════════════════════════════════════════════════════════
#  Configuration
# ═══════════════════════════════════════════════════════════════════════

N_TRIALS = 30
MAX_ITERATIONS = 200
BATCH_SIZE = 100
BASE_SEED = 42

# Environments: name → (env_fn, has_obstacle_metric)
# has_obstacle_metric=True means the env has a non-trivial Riemannian metric
# that CARM should try to learn
ENVS = {
    '2D Obstacles':  (env_2d_obstacle_inflated, True),
    '2D Narrow':     (env_2d_narrow_passage,    True),
    '2D Maze':       (env_2d_maze,              True),
    '2D Bug Trap':   (env_2d_bug_trap,          True),
    '2D Forest':     (env_2d_random_forest,     True),
    '2D Terrain':    (env_2d_terrain,            True),
    '3D Spheres':    (env_3d_sphere_field,       True),
}

# CARM-focused variants
VARIANT_NAMES = [
    'RIT*-CARM',         # Our method: learns metric from collisions
    'RIT* (oracle)',      # Upper bound: a priori known metric
    'RIT* (Euclidean)',   # Ablation: no metric adaptation
    'BIT*',              # Strong baseline
    'Informed RRT*',     # Classic baseline
    'APT*',              # Recent baseline (RA-L 2025)
]

VARIANT_COLORS = {
    'RIT*-CARM':         '#E91E63',   # pink/red — our method (highlighted)
    'RIT* (oracle)':     '#7B2FBE',   # purple
    'RIT* (Euclidean)':  '#9E9E9E',   # grey
    'BIT*':              '#4CAF50',   # green
    'Informed RRT*':     '#2196F3',   # blue
    'APT*':              '#F44336',   # red
}

CARM_SIGMA = 0.08
CARM_ALPHA = 6.0
CARM_REBUILD = 15


# ═══════════════════════════════════════════════════════════════════════
#  Helper: path cost under arbitrary metric
# ═══════════════════════════════════════════════════════════════════════

def path_cost_under_metric(path, metric):
    """Evaluate path cost under a given metric (for fair oracle-metric comparison)."""
    if not path or len(path) < 2:
        return np.inf
    return sum(riemannian_edge_cost(path[i], path[i + 1], metric)
               for i in range(len(path) - 1))


# ═══════════════════════════════════════════════════════════════════════
#  Build a planner variant
# ═══════════════════════════════════════════════════════════════════════

def build_variant(name, xs, xg, bounds, coll, oracle_metric, seed):
    """Instantiate a planner variant.

    For CARM and Euclidean variants, we use EuclideanMetric as the
    planner's internal metric. For oracle, we use the true metric.
    For BIT*/IRRT*/APT*, we use the oracle metric (giving them the
    best possible advantage).
    """
    dim = len(xs)
    euclid = EuclideanMetric(dim)
    common = dict(
        x_start=xs, x_goal=xg, c_space_bounds=bounds,
        collision_checker=coll,
        batch_size=BATCH_SIZE, max_iterations=MAX_ITERATIONS,
        random_seed=seed,
    )

    if name == 'RIT*-CARM':
        return RITStar(
            xs, xg, bounds, coll, euclid,
            geodesic_tier='diagonal', batch_size=BATCH_SIZE,
            max_iterations=MAX_ITERATIONS, random_seed=seed,
            adaptive_metric=True,
            carm_sigma=CARM_SIGMA, carm_alpha=CARM_ALPHA,
            carm_rebuild_interval=CARM_REBUILD)

    elif name == 'RIT* (oracle)':
        return RITStar(
            xs, xg, bounds, coll, oracle_metric,
            geodesic_tier='diagonal', batch_size=BATCH_SIZE,
            max_iterations=MAX_ITERATIONS, random_seed=seed)

    elif name == 'RIT* (Euclidean)':
        return RITStar(
            xs, xg, bounds, coll, euclid,
            geodesic_tier='diagonal', batch_size=BATCH_SIZE,
            max_iterations=MAX_ITERATIONS, random_seed=seed)

    elif name == 'BIT*':
        return BITStar(metric=oracle_metric, **common)

    elif name == 'Informed RRT*':
        return InformedRRTStar(metric=oracle_metric, **common)

    elif name == 'APT*':
        return APTStar(metric=oracle_metric, **common)

    else:
        raise ValueError(f'Unknown variant: {name}')


# ═══════════════════════════════════════════════════════════════════════
#  Experiment 1: Main CARM comparison (headline results)
# ═══════════════════════════════════════════════════════════════════════

def experiment_1_carm_comparison(n_trials=N_TRIALS,
                                 envs=None,
                                 variants=None):
    """Run all variants on all environments with n_trials repetitions.

    All paths are evaluated under the ORACLE metric for fair comparison.
    CARM and Euclidean variants plan with Euclidean/learned metrics but
    their final paths are scored by the oracle.

    Returns
    -------
    all_results : dict[env_name -> dict[variant_name -> list[dict]]]
    """
    if envs is None:
        envs = ENVS
    if variants is None:
        variants = VARIANT_NAMES

    print('\n' + '=' * 65)
    print('  EXPERIMENT 1: CARM Main Comparison')
    print(f'  {len(variants)} variants × {len(envs)} environments × {n_trials} trials')
    print('=' * 65)

    cache_dir = os.path.join(RESULTS_DIR, '_carm_ral_cache')
    os.makedirs(cache_dir, exist_ok=True)

    all_results = {}
    t0 = time.time()

    for ei, (env_name, (env_fn, _)) in enumerate(envs.items()):
        safe_name = env_name.lower().replace(' ', '_')
        cache_path = os.path.join(cache_dir, f'{safe_name}.pkl')

        # Resume from cache if available
        if os.path.exists(cache_path):
            with open(cache_path, 'rb') as f:
                cached = pickle.load(f)
            cached_variants = set(cached.keys()) - {'_meta'}
            if cached_variants == set(variants):
                n_cached = min(len(cached[v]) for v in variants)
                if n_cached >= n_trials:
                    print(f'\n  [{ei+1}/{len(envs)}] {env_name}  (cached, {n_cached} trials)')
                    all_results[env_name] = cached
                    continue

        print(f'\n  [{ei+1}/{len(envs)}] {env_name}')
        coll, _, oracle_metric, xs, xg, bounds = env_fn()

        results = {name: [] for name in variants}

        for vname in variants:
            print(f'    {vname}:')
            for trial in range(n_trials):
                seed = BASE_SEED + ei * 1000 + trial
                print(f'      trial {trial+1}/{n_trials}', end='\r', flush=True)

                planner = build_variant(
                    vname, xs, xg, bounds, coll, oracle_metric, seed)

                # Use plan_stepwise for RIT* variants (oracle-metric convergence)
                if hasattr(planner, 'plan_stepwise') and vname.startswith('RIT*'):
                    oracle_curve = []
                    for step in planner.plan_stepwise():
                        c = path_cost_under_metric(step['path'], oracle_metric)
                        oracle_curve.append(c)
                    final_path = planner._extract_path()
                    final_oracle = path_cost_under_metric(final_path, oracle_metric)
                    carm_pts = (planner._carm.n_collision_points
                                if hasattr(planner, '_carm') and planner._carm else 0)
                    results[vname].append({
                        'final_cost': final_oracle,
                        'oracle_curve': oracle_curve,
                        'time_elapsed': planner._stats[-1]['time_elapsed'] if planner._stats else 0,
                        'carm_points': carm_pts,
                    })
                elif hasattr(planner, 'plan_stepwise'):
                    # BIT* also has plan_stepwise
                    oracle_curve = []
                    for step in planner.plan_stepwise():
                        c = path_cost_under_metric(step['path'], oracle_metric)
                        oracle_curve.append(c)
                    final_path = planner._extract_path()
                    final_oracle = path_cost_under_metric(final_path, oracle_metric)
                    results[vname].append({
                        'final_cost': final_oracle,
                        'oracle_curve': oracle_curve,
                        'time_elapsed': planner._stats[-1]['time_elapsed'] if planner._stats else 0,
                        'carm_points': 0,
                    })
                else:
                    # Non-stepwise planners (IRRT*, APT*): plan and evaluate final path
                    final_path, _ = planner.plan()
                    stats = planner.get_stats()
                    final_oracle = path_cost_under_metric(final_path, oracle_metric)

                    # Build oracle curve from stats (per-iteration c_best under internal metric,
                    # not oracle — but we evaluate final path under oracle)
                    oracle_curve = []
                    for s in stats:
                        oracle_curve.append(s['c_best'])  # internal metric cost

                    results[vname].append({
                        'final_cost': final_oracle,
                        'oracle_curve': oracle_curve,
                        'time_elapsed': stats[-1]['time_elapsed'] if stats else 0,
                        'carm_points': 0,
                    })

                del planner
            gc.collect()
            print(f'      done ({n_trials} trials)           ')

        all_results[env_name] = results

        # Save checkpoint
        with open(cache_path, 'wb') as f:
            pickle.dump(results, f, protocol=pickle.HIGHEST_PROTOCOL)

    total_time = time.time() - t0
    print(f'\n  Total experiment time: {total_time:.1f}s')
    return all_results


# ═══════════════════════════════════════════════════════════════════════
#  Experiment 2: CARM parameter sensitivity (ablation)
# ═══════════════════════════════════════════════════════════════════════

def experiment_2_carm_ablation(n_trials=15):
    """Sweep CARM hyperparameters: sigma, alpha, rebuild interval.

    Tested on 2D Obstacles (representative environment).
    """
    print('\n' + '=' * 65)
    print('  EXPERIMENT 2: CARM Parameter Sensitivity')
    print('=' * 65)

    coll, _, oracle_metric, xs, xg, bounds = env_2d_obstacle_inflated()
    dim = len(xs)
    euclid = EuclideanMetric(dim)

    # Parameter sweeps
    sigmas = [0.02, 0.04, 0.08, 0.12, 0.20]
    alphas = [1.0, 3.0, 6.0, 10.0, 20.0]
    rebuilds = [5, 10, 15, 25, 50]

    results = {'sigma': {}, 'alpha': {}, 'rebuild': {}}

    # Sigma sweep (alpha=6, rebuild=15)
    print('\n  Sigma sweep:')
    for sigma in sigmas:
        costs = []
        for trial in range(n_trials):
            seed = BASE_SEED + trial
            p = RITStar(xs, xg, bounds, coll, euclid,
                        geodesic_tier='diagonal', batch_size=BATCH_SIZE,
                        max_iterations=MAX_ITERATIONS, random_seed=seed,
                        adaptive_metric=True,
                        carm_sigma=sigma, carm_alpha=CARM_ALPHA,
                        carm_rebuild_interval=CARM_REBUILD)
            p.plan()
            path = p._extract_path()
            costs.append(path_cost_under_metric(path, oracle_metric))
            del p
        results['sigma'][sigma] = costs
        print(f'    σ={sigma:.2f}: {np.mean(costs):.4f} ± {np.std(costs):.4f}')
        gc.collect()

    # Alpha sweep (sigma=0.08, rebuild=15)
    print('\n  Alpha sweep:')
    for alpha in alphas:
        costs = []
        for trial in range(n_trials):
            seed = BASE_SEED + trial
            p = RITStar(xs, xg, bounds, coll, euclid,
                        geodesic_tier='diagonal', batch_size=BATCH_SIZE,
                        max_iterations=MAX_ITERATIONS, random_seed=seed,
                        adaptive_metric=True,
                        carm_sigma=CARM_SIGMA, carm_alpha=alpha,
                        carm_rebuild_interval=CARM_REBUILD)
            p.plan()
            path = p._extract_path()
            costs.append(path_cost_under_metric(path, oracle_metric))
            del p
        results['alpha'][alpha] = costs
        print(f'    α={alpha:.1f}: {np.mean(costs):.4f} ± {np.std(costs):.4f}')
        gc.collect()

    # Rebuild interval sweep (sigma=0.08, alpha=6)
    print('\n  Rebuild interval sweep:')
    for rb in rebuilds:
        costs = []
        for trial in range(n_trials):
            seed = BASE_SEED + trial
            p = RITStar(xs, xg, bounds, coll, euclid,
                        geodesic_tier='diagonal', batch_size=BATCH_SIZE,
                        max_iterations=MAX_ITERATIONS, random_seed=seed,
                        adaptive_metric=True,
                        carm_sigma=CARM_SIGMA, carm_alpha=CARM_ALPHA,
                        carm_rebuild_interval=rb)
            p.plan()
            path = p._extract_path()
            costs.append(path_cost_under_metric(path, oracle_metric))
            del p
        results['rebuild'][rb] = costs
        print(f'    rebuild={rb}: {np.mean(costs):.4f} ± {np.std(costs):.4f}')
        gc.collect()

    return results


# ═══════════════════════════════════════════════════════════════════════
#  Experiment 3: CARM metric recovery quality
# ═══════════════════════════════════════════════════════════════════════

def experiment_3_metric_recovery(n_trials=10):
    """Measure how well CARM approximates the oracle metric field.

    Evaluate on a grid: correlation between CARM scale s(x) and oracle sqrt(det G(x)).
    """
    print('\n' + '=' * 65)
    print('  EXPERIMENT 3: CARM Metric Recovery Quality')
    print('=' * 65)

    coll, _, oracle_metric, xs, xg, bounds = env_2d_obstacle_inflated()
    dim = len(xs)
    euclid = EuclideanMetric(dim)

    # Grid for evaluation
    res = 50
    grid_x = np.linspace(bounds[0][0], bounds[0][1], res)
    grid_y = np.linspace(bounds[1][0], bounds[1][1], res)

    # Oracle field
    oracle_field = np.zeros((res, res))
    for i, x in enumerate(grid_x):
        for j, y in enumerate(grid_y):
            pt = np.array([x, y])
            oracle_field[i, j] = oracle_metric.sqrt_det_G(pt)

    correlations = []
    recovery_fractions = []

    for trial in range(n_trials):
        seed = BASE_SEED + trial
        print(f'  trial {trial+1}/{n_trials}', end='\r', flush=True)

        p = RITStar(xs, xg, bounds, coll, euclid,
                    geodesic_tier='diagonal', batch_size=BATCH_SIZE,
                    max_iterations=MAX_ITERATIONS, random_seed=seed,
                    adaptive_metric=True,
                    carm_sigma=CARM_SIGMA, carm_alpha=CARM_ALPHA,
                    carm_rebuild_interval=CARM_REBUILD)
        p.plan()

        # Extract learned CARM field
        if p._carm is not None:
            carm_field = np.zeros((res, res))
            for i, x in enumerate(grid_x):
                for j, y in enumerate(grid_y):
                    pt = np.array([x, y])
                    carm_field[i, j] = p._carm.sqrt_det_G(pt)

            # Pearson correlation
            corr = np.corrcoef(oracle_field.ravel(), carm_field.ravel())[0, 1]
            correlations.append(corr)

            # Recovery fraction: % of oracle advantage recovered in path cost
            # (computed in experiment 1, here we just measure metric similarity)
            recovery_fractions.append(corr)

        del p
        gc.collect()

    print(f'\n  Metric correlation: {np.mean(correlations):.3f} ± {np.std(correlations):.3f}')
    return {
        'correlations': correlations,
        'oracle_field': oracle_field,
        'grid_x': grid_x,
        'grid_y': grid_y,
    }


# ═══════════════════════════════════════════════════════════════════════
#  Analysis & Table Generation
# ═══════════════════════════════════════════════════════════════════════

def generate_tables(all_results, variants=None):
    """Generate CSV tables from experiment results."""
    if variants is None:
        variants = VARIANT_NAMES

    # ── Cost table (mean ± std) ──────────────────────────────────────
    header = ['Environment'] + variants
    cost_rows = [header]
    for env_name, results in all_results.items():
        row = [env_name]
        for vname in variants:
            costs = [r['final_cost'] for r in results[vname]
                     if np.isfinite(r['final_cost'])]
            if costs:
                row.append(f'{np.mean(costs):.4f} ± {np.std(costs):.4f}')
            else:
                row.append('no sol.')
        cost_rows.append(row)

    _save_csv(cost_rows, os.path.join(RESULTS_DIR, 'carm_ral_cost.csv'))

    # ── Normalized improvement over BIT* ─────────────────────────────
    norm_rows = [header]
    for env_name, results in all_results.items():
        row = [env_name]
        bit_costs = [r['final_cost'] for r in results['BIT*']
                     if np.isfinite(r['final_cost'])]
        bit_mean = np.mean(bit_costs) if bit_costs else np.inf

        for vname in variants:
            costs = [r['final_cost'] for r in results[vname]
                     if np.isfinite(r['final_cost'])]
            if costs and np.isfinite(bit_mean) and bit_mean > 0:
                improvement = (bit_mean - np.mean(costs)) / bit_mean * 100
                row.append(f'{improvement:+.2f}%')
            else:
                row.append('n/a')
        norm_rows.append(row)

    _save_csv(norm_rows, os.path.join(RESULTS_DIR, 'carm_ral_normalized.csv'))

    # ── Statistical significance (Wilcoxon signed-rank: CARM vs each) ──
    baselines = [v for v in variants if v != 'RIT*-CARM']
    sig_header = ['Environment'] + baselines
    sig_rows = [sig_header]
    for env_name, results in all_results.items():
        row = [env_name]
        carm_costs = [r['final_cost'] for r in results['RIT*-CARM']
                      if np.isfinite(r['final_cost'])]
        for bname in baselines:
            b_costs = [r['final_cost'] for r in results[bname]
                       if np.isfinite(r['final_cost'])]
            n_pairs = min(len(carm_costs), len(b_costs))
            if n_pairs >= 5:
                try:
                    _, p = wilcoxon(carm_costs[:n_pairs], b_costs[:n_pairs])
                    sig = '***' if p < 0.001 else ('**' if p < 0.01 else ('*' if p < 0.05 else ''))
                    row.append(f'{p:.4f}{sig}')
                except ValueError:
                    # All differences are zero
                    row.append('1.0000')
            else:
                row.append('n/a')
        sig_rows.append(row)

    _save_csv(sig_rows, os.path.join(RESULTS_DIR, 'carm_ral_significance.csv'))

    # ── CARM recovery fraction ───────────────────────────────────────
    recovery_header = ['Environment', 'Oracle Cost', 'CARM Cost', 'Euclidean Cost',
                       'Recovery %', 'BIT* Cost']
    recovery_rows = [recovery_header]
    for env_name, results in all_results.items():
        oracle_costs = [r['final_cost'] for r in results['RIT* (oracle)']
                        if np.isfinite(r['final_cost'])]
        carm_costs = [r['final_cost'] for r in results['RIT*-CARM']
                      if np.isfinite(r['final_cost'])]
        euclid_costs = [r['final_cost'] for r in results['RIT* (Euclidean)']
                        if np.isfinite(r['final_cost'])]
        bit_costs = [r['final_cost'] for r in results['BIT*']
                     if np.isfinite(r['final_cost'])]

        if oracle_costs and carm_costs and euclid_costs:
            o_mean = np.mean(oracle_costs)
            c_mean = np.mean(carm_costs)
            e_mean = np.mean(euclid_costs)
            b_mean = np.mean(bit_costs) if bit_costs else np.nan

            # Recovery: how much of the gap (euclidean - oracle) does CARM close?
            gap = e_mean - o_mean
            if gap > 1e-8:
                recovery = (e_mean - c_mean) / gap * 100
            else:
                recovery = 100.0  # No gap to close

            recovery_rows.append([
                env_name,
                f'{o_mean:.4f}',
                f'{c_mean:.4f}',
                f'{e_mean:.4f}',
                f'{recovery:.1f}%',
                f'{b_mean:.4f}' if np.isfinite(b_mean) else 'n/a',
            ])

    _save_csv(recovery_rows, os.path.join(RESULTS_DIR, 'carm_ral_recovery.csv'))

    # Print summary
    print('\n  ═══ CARM RECOVERY SUMMARY ═══')
    for row in recovery_rows:
        print('  ' + '  '.join(str(c).ljust(16) for c in row))

    return cost_rows, norm_rows, sig_rows, recovery_rows


# ═══════════════════════════════════════════════════════════════════════
#  Plot Generation
# ═══════════════════════════════════════════════════════════════════════

def generate_plots(all_results, variants=None):
    """Generate publication-quality figures."""
    if variants is None:
        variants = VARIANT_NAMES

    # ── Figure 1: Oracle-metric convergence curves (multi-panel) ─────
    # Only use variants that have oracle_curve from stepwise planning
    stepwise_variants = ['RIT*-CARM', 'RIT* (oracle)', 'RIT* (Euclidean)', 'BIT*']
    n_envs = len(all_results)
    n_cols = min(4, n_envs)
    n_rows = (n_envs + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows),
                             squeeze=False)

    for idx, (env_name, results) in enumerate(all_results.items()):
        r, c = divmod(idx, n_cols)
        ax = axes[r][c]

        for vname in stepwise_variants:
            if vname not in results:
                continue
            trial_data = results[vname]
            if not trial_data or 'oracle_curve' not in trial_data[0]:
                continue

            max_it = max(len(td['oracle_curve']) for td in trial_data)
            cost_matrix = np.full((len(trial_data), max_it), np.nan)
            for ti, td in enumerate(trial_data):
                for si, c_val in enumerate(td['oracle_curve']):
                    cost_matrix[ti, si] = c_val
            # Forward-fill
            for ti in range(cost_matrix.shape[0]):
                for si in range(1, cost_matrix.shape[1]):
                    if np.isnan(cost_matrix[ti, si]):
                        cost_matrix[ti, si] = cost_matrix[ti, si - 1]

            iters = np.arange(max_it)
            mean_c = np.nanmean(cost_matrix, axis=0)
            std_c = np.nanstd(cost_matrix, axis=0)
            color = VARIANT_COLORS.get(vname, '#000000')
            lw = 2.5 if vname == 'RIT*-CARM' else 1.5
            ax.plot(iters, mean_c, color=color, lw=lw, label=vname)
            ax.fill_between(iters, mean_c - std_c, mean_c + std_c,
                            color=color, alpha=0.12)

        ax.set_xlabel('Iteration')
        ax.set_ylabel('Oracle-metric cost')
        ax.set_title(env_name, fontsize=10)
        ax.grid(True, alpha=0.3)

        # Clip y-axis
        finite = []
        for vname in stepwise_variants:
            if vname in results:
                for td in results[vname]:
                    if 'oracle_curve' in td:
                        finite.extend([c for c in td['oracle_curve'] if np.isfinite(c)])
        if finite:
            ax.set_ylim(min(finite) * 0.95, np.percentile(finite, 90) * 1.05)

    for idx in range(n_envs, n_rows * n_cols):
        r, c = divmod(idx, n_cols)
        axes[r][c].set_visible(False)

    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=len(stepwise_variants),
               fontsize=9, bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(os.path.join(PLOTS_DIR, 'carm_ral_convergence.pdf'),
                dpi=300, bbox_inches='tight')
    fig.savefig(os.path.join(PLOTS_DIR, 'carm_ral_convergence.png'),
                dpi=150, bbox_inches='tight')
    plt.close(fig)
    print('  → saved carm_ral_convergence.pdf')

    # ── Figure 2: Box plots of final oracle-metric cost ──────────────
    fig, axes = plt.subplots(1, n_envs, figsize=(3.5 * n_envs, 4), squeeze=False)

    for idx, (env_name, results) in enumerate(all_results.items()):
        ax = axes[0][idx]
        data, labels, colors = [], [], []
        for vname in variants:
            if vname not in results:
                continue
            costs = [r_['final_cost'] for r_ in results[vname]
                     if np.isfinite(r_['final_cost'])]
            if costs:
                data.append(costs)
                labels.append(vname.replace('RIT*-CARM', 'CARM\n(ours)'))
                colors.append(VARIANT_COLORS.get(vname, '#333'))

        if data:
            bp = ax.boxplot(data, labels=labels, patch_artist=True, widths=0.6)
            for patch, col in zip(bp['boxes'], colors):
                patch.set_facecolor(col)
                patch.set_alpha(0.6)
            ax.tick_params(axis='x', rotation=45, labelsize=7)

        ax.set_ylabel('Oracle-metric cost' if idx == 0 else '')
        ax.set_title(env_name, fontsize=9)
        ax.grid(True, alpha=0.3, axis='y')

    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, 'carm_ral_boxplots.pdf'),
                dpi=300, bbox_inches='tight')
    fig.savefig(os.path.join(PLOTS_DIR, 'carm_ral_boxplots.png'),
                dpi=150, bbox_inches='tight')
    plt.close(fig)
    print('  → saved carm_ral_boxplots.pdf')

    # ── Figure 3: Recovery bar chart ─────────────────────────────────
    env_names = list(all_results.keys())
    recoveries = []
    for env_name in env_names:
        results = all_results[env_name]
        oracle_costs = [r['final_cost'] for r in results['RIT* (oracle)']
                        if np.isfinite(r['final_cost'])]
        carm_costs = [r['final_cost'] for r in results['RIT*-CARM']
                      if np.isfinite(r['final_cost'])]
        euclid_costs = [r['final_cost'] for r in results['RIT* (Euclidean)']
                        if np.isfinite(r['final_cost'])]

        if oracle_costs and carm_costs and euclid_costs:
            o_mean = np.mean(oracle_costs)
            c_mean = np.mean(carm_costs)
            e_mean = np.mean(euclid_costs)
            gap = e_mean - o_mean
            if gap > 1e-8:
                rec = (e_mean - c_mean) / gap * 100
            else:
                rec = 100.0
            recoveries.append(min(rec, 100.0))
        else:
            recoveries.append(0)

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(env_names, recoveries, color='#E91E63', alpha=0.8, edgecolor='white')
    ax.axhline(y=100, color='#7B2FBE', linestyle='--', alpha=0.5, label='Oracle (100%)')
    ax.set_ylabel('Oracle advantage recovered (%)')
    ax.set_title('CARM: Fraction of Oracle Advantage Recovered')
    ax.set_ylim(0, 110)
    ax.tick_params(axis='x', rotation=20, labelsize=9)
    ax.grid(True, alpha=0.3, axis='y')
    ax.legend()

    # Add value labels
    for bar, val in zip(bars, recoveries):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2,
                f'{val:.0f}%', ha='center', va='bottom', fontsize=8)

    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, 'carm_ral_recovery.pdf'),
                dpi=300, bbox_inches='tight')
    fig.savefig(os.path.join(PLOTS_DIR, 'carm_ral_recovery.png'),
                dpi=150, bbox_inches='tight')
    plt.close(fig)
    print('  → saved carm_ral_recovery.pdf')


def generate_ablation_plots(ablation_results):
    """Plot CARM parameter sensitivity."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    # Sigma
    ax = axes[0]
    sigmas = sorted(ablation_results['sigma'].keys())
    means = [np.mean(ablation_results['sigma'][s]) for s in sigmas]
    stds = [np.std(ablation_results['sigma'][s]) for s in sigmas]
    ax.errorbar(sigmas, means, yerr=stds, marker='o', color='#E91E63',
                capsize=4, lw=2)
    ax.axvline(x=CARM_SIGMA, color='grey', linestyle='--', alpha=0.5, label=f'default={CARM_SIGMA}')
    ax.set_xlabel(r'Bandwidth $\sigma$')
    ax.set_ylabel('Oracle-metric cost')
    ax.set_title(r'Sensitivity to $\sigma$')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    # Alpha
    ax = axes[1]
    alphas = sorted(ablation_results['alpha'].keys())
    means = [np.mean(ablation_results['alpha'][a]) for a in alphas]
    stds = [np.std(ablation_results['alpha'][a]) for a in alphas]
    ax.errorbar(alphas, means, yerr=stds, marker='s', color='#E91E63',
                capsize=4, lw=2)
    ax.axvline(x=CARM_ALPHA, color='grey', linestyle='--', alpha=0.5, label=f'default={CARM_ALPHA}')
    ax.set_xlabel(r'Penalty strength $\alpha$')
    ax.set_ylabel('Oracle-metric cost')
    ax.set_title(r'Sensitivity to $\alpha$')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    # Rebuild interval
    ax = axes[2]
    rbs = sorted(ablation_results['rebuild'].keys())
    means = [np.mean(ablation_results['rebuild'][r]) for r in rbs]
    stds = [np.std(ablation_results['rebuild'][r]) for r in rbs]
    ax.errorbar(rbs, means, yerr=stds, marker='^', color='#E91E63',
                capsize=4, lw=2)
    ax.axvline(x=CARM_REBUILD, color='grey', linestyle='--', alpha=0.5,
               label=f'default={CARM_REBUILD}')
    ax.set_xlabel('Rebuild interval (iterations)')
    ax.set_ylabel('Oracle-metric cost')
    ax.set_title('Sensitivity to rebuild interval')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, 'carm_ral_ablation.pdf'),
                dpi=300, bbox_inches='tight')
    fig.savefig(os.path.join(PLOTS_DIR, 'carm_ral_ablation.png'),
                dpi=150, bbox_inches='tight')
    plt.close(fig)
    print('  → saved carm_ral_ablation.pdf')


# ═══════════════════════════════════════════════════════════════════════
#  Utilities
# ═══════════════════════════════════════════════════════════════════════

def _save_csv(rows, filename):
    with open(filename, 'w') as f:
        for row in rows:
            f.write(','.join(str(c) for c in row) + '\n')
    print(f'  → saved {filename}')


# ═══════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════

def run_all(n_trials=N_TRIALS, quick=False):
    """Run the complete CARM-focused RAL experiment suite.

    Parameters
    ----------
    n_trials : int
        Trials per variant per environment (default 30).
    quick : bool
        If True, use 5 trials and fewer environments for quick testing.
    """
    if quick:
        n_trials = 5
        envs = {
            '2D Obstacles': ENVS['2D Obstacles'],
            '2D Forest':    ENVS['2D Forest'],
            '2D Maze':      ENVS['2D Maze'],
        }
    else:
        envs = ENVS

    # Experiment 1: Main comparison
    all_results = experiment_1_carm_comparison(n_trials=n_trials, envs=envs)

    # Generate tables
    generate_tables(all_results)

    # Generate plots
    generate_plots(all_results)

    # Experiment 2: Ablation (only if not quick)
    if not quick:
        ablation = experiment_2_carm_ablation(n_trials=min(n_trials, 15))
        generate_ablation_plots(ablation)

    # Experiment 3: Metric recovery
    if not quick:
        recovery = experiment_3_metric_recovery(n_trials=min(n_trials, 10))
        print(f'\n  Metric correlation: {np.mean(recovery["correlations"]):.3f}')

    print('\n' + '=' * 65)
    print('  ALL CARM RAL EXPERIMENTS COMPLETE')
    print('=' * 65)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='CARM RAL Experiments')
    parser.add_argument('--quick', action='store_true', help='Quick test (5 trials, 3 envs)')
    parser.add_argument('--trials', type=int, default=N_TRIALS, help='Number of trials')
    args = parser.parse_args()
    run_all(n_trials=args.trials, quick=args.quick)
