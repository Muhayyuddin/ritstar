"""
comparison.py — Monte Carlo comparison of all 7 planners across environments.

Generates:
  1. LaTeX-formatted results tables (also saved as CSV)
  2. Convergence plots per environment (mean ± std)
  3. Box plots of final cost / time across MC trials
  4. Bar charts summarising relative performance

All plots are saved as high-res PNG files.

Usage
-----
    from rit_star.comparison import run_full_comparison
    run_full_comparison()
"""

from __future__ import annotations

import gc
import json
import os
import pickle
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu

from visualization_util.output_paths import RESULTS_DIR, PLOTS_DIR
from .rit_star import RITStar
from .baselines import (
    InformedRRTStar,
    GeometryAwareRRTStar,
    BITStar,
    AITStar,
    EITStar,
    APTStar,
)
from .environments import (
    env_2d_diagonal_anisotropic,
    env_2d_obstacle_inflated,
    env_2d_narrow_passage,
    env_2d_maze,
    env_3d_diagonal_anisotropic,
    env_3d_sphere_field,
    env_2d_bug_trap,
    env_2d_random_forest,
    env_2d_terrain,
    env_2d_hyper_dense,
    env_3d_dense_labyrinth,
    env_3d_anisotropic_corridor,
    env_3d_obstacle_gauntlet,
    env_6d_hyper_passage,
    ALL_6D_ENVS,
)
from .informed_set import volume_ratio_bound
from .metric import EuclideanMetric
from .environments import (
    _point_in_rect_2d, _point_in_circle_2d,
    _point_in_box_3d, _point_in_sphere_3d,
)


# ═══════════════════════════════════════════════════════════════════════
#  Planner registry
# ═══════════════════════════════════════════════════════════════════════

PLANNER_NAMES = [
    'RIT*',
    'GA-RRT*',
    'Informed RRT*',
    'BIT*',
    'AIT*',
    'EIT*',
    'APT*',
]

PLANNER_COLORS = {
    'RIT*':          '#7B2FBE',   # purple
    'GA-RRT*':       '#00695C',   # dark teal
    'Informed RRT*': '#2196F3',   # blue
    'BIT*':          '#4CAF50',   # green
    'AIT*':          '#FF9800',   # orange
    'EIT*':          '#00897B',   # teal
    'APT*':          '#F44336',   # red
}

# Maximum iteration shown on convergence x-axis (for readability)
_CONVERGENCE_ITER_LIMIT = 100


def _build_planner(name, xs, xg, bounds, coll, metric,
                   batch_size, max_iterations, seed):
    """Instantiate a planner by name with the standard interface."""
    common = dict(
        x_start=xs, x_goal=xg, c_space_bounds=bounds,
        collision_checker=coll, metric=metric,
        batch_size=batch_size, max_iterations=max_iterations,
        random_seed=seed,
    )
    if name == 'RIT*':
        return RITStar(xs, xg, bounds, coll, metric,
                       geodesic_tier='diagonal', batch_size=batch_size,
                       max_iterations=max_iterations, random_seed=seed)
    elif name == 'RIT*-CARM':
        dim = len(xs)
        euclid = EuclideanMetric(dim)
        return RITStar(xs, xg, bounds, coll, euclid,
                       geodesic_tier='diagonal', batch_size=batch_size,
                       max_iterations=max_iterations, random_seed=seed,
                       adaptive_metric=True,
                       carm_sigma=0.08, carm_alpha=6.0,
                       carm_rebuild_interval=15)
    elif name == 'Informed RRT*':
        return InformedRRTStar(**common)
    elif name == 'GA-RRT*':
        return GeometryAwareRRTStar(**common)
    elif name == 'BIT*':
        return BITStar(**common)
    elif name == 'AIT*':
        return AITStar(**common)
    elif name == 'EIT*':
        return EITStar(**common)
    elif name == 'APT*':
        return APTStar(**common)
    else:
        raise ValueError(f'Unknown planner: {name}')


# ═══════════════════════════════════════════════════════════════════════
#  Environment registry for comparison
# ═══════════════════════════════════════════════════════════════════════

COMPARISON_ENVS = {
    '2D Maze':      env_2d_maze,
    '2D Narrow':    env_2d_narrow_passage,
    '2D Forest':    env_2d_random_forest,
    '2D Terrain':   env_2d_terrain,
    '2D Bug Trap':  env_2d_bug_trap,
    '2D Hyper-Dense': env_2d_hyper_dense,
    '3D Spheres':   env_3d_sphere_field,
    '3D Dense Lab': env_3d_dense_labyrinth,
    '3D Corridor':  env_3d_anisotropic_corridor,
    '3D Gauntlet':  env_3d_obstacle_gauntlet,
    '6D Hyper-Passage': env_6d_hyper_passage,
}

# Add 6-D UR10e scenes when PyBullet is available
if ALL_6D_ENVS:
    from .environments import env_6d_tabletop, env_6d_shelf, env_6d_cluttered
    COMPARISON_ENVS['6D Tabletop'] = env_6d_tabletop
    COMPARISON_ENVS['6D Shelf'] = env_6d_shelf
    COMPARISON_ENVS['6D Cluttered'] = env_6d_cluttered


# ═══════════════════════════════════════════════════════════════════════
#  Core benchmark runner
# ═══════════════════════════════════════════════════════════════════════

def _run_single_env(env_name, env_fn, n_trials, max_iterations,
                    batch_size, base_seed):
    """Run all planners on a single environment for n_trials.

    Planners are run ONE AT A TIME to keep memory usage low and
    prevent the IDE from becoming unresponsive.  After each planner
    finishes all its trials, its results are stored and the planner
    objects are freed via ``gc.collect()``.

    Returns
    -------
    results : dict[planner_name -> list[dict]]
        Each dict has keys: final_cost, time_elapsed, stats (full list).
    """
    coll, _, metric, xs, xg, bounds = env_fn()
    results = {name: [] for name in PLANNER_NAMES}

    for name in PLANNER_NAMES:
        print(f'      planner: {name}')
        for trial in range(n_trials):
            seed = base_seed + trial
            print(f'        trial {trial + 1}/{n_trials}', end='\r')
            planner = _build_planner(
                name, xs, xg, bounds, coll, metric,
                batch_size, max_iterations, seed)
            planner.plan()
            stats = planner.get_stats()
            final_cost = stats[-1]['c_best'] if stats else np.inf
            elapsed = stats[-1]['time_elapsed'] if stats else 0.0
            results[name].append({
                'final_cost': final_cost,
                'time_elapsed': elapsed,
                'stats': stats,
            })
            del planner
        # Free memory between planners
        gc.collect()
        print()

    # ── Theory-vs-experiment tracking (Theorems 1-3) ──────────────
    theory_vs_experiment = {}
    dim = len(xs)
    # Analytical volume ratio (Theorem 1)
    a_vr = volume_ratio_bound(metric, xs, xg, dim)
    theory_vs_experiment['analytical_volume_ratio'] = a_vr
    theory_vs_experiment['predicted_speedup'] = 1.0 / max(a_vr, 1e-30)

    # Actual speedup: median time BIT* / median time RIT*
    rit_times = [r['time_elapsed'] for r in results['RIT*']
                 if np.isfinite(r['final_cost'])]
    bit_times = [r['time_elapsed'] for r in results['BIT*']
                 if np.isfinite(r['final_cost'])]
    if rit_times and bit_times:
        theory_vs_experiment['actual_time_speedup'] = (
            np.median(bit_times) / max(np.median(rit_times), 1e-6))
    else:
        theory_vs_experiment['actual_time_speedup'] = np.nan

    # Actual sample speedup: median iterations to reach 1.01*c* for BIT* vs RIT*
    all_best = []
    for pname in PLANNER_NAMES:
        for r in results[pname]:
            if np.isfinite(r['final_cost']):
                all_best.append(r['final_cost'])
    c_ref = min(all_best) if all_best else np.inf
    target = 1.05 * c_ref if np.isfinite(c_ref) else np.inf

    def _median_samples_to_target(pname):
        vals = []
        for r in results[pname]:
            for s in r['stats']:
                if s['c_best'] <= target:
                    vals.append(s['n_samples_total'])
                    break
        return np.median(vals) if vals else np.nan

    n_rit = _median_samples_to_target('RIT*')
    n_bit = _median_samples_to_target('BIT*')
    theory_vs_experiment['actual_sample_speedup'] = (
        n_bit / max(n_rit, 1) if np.isfinite(n_bit) and np.isfinite(n_rit) else np.nan)

    results['_theory'] = theory_vs_experiment

    return results


# ═══════════════════════════════════════════════════════════════════════
#  Table generation
# ═══════════════════════════════════════════════════════════════════════

def _generate_summary_table(all_results):
    """Build a summary table: rows = environments, columns = planners.

    Returns a formatted string (also usable as LaTeX) and a list of
    rows for CSV.
    """
    header = ['Environment'] + PLANNER_NAMES
    rows = [header]

    for env_name, results in all_results.items():
        row = [env_name]
        for pname in PLANNER_NAMES:
            costs = [r['final_cost'] for r in results[pname]]
            finite = [c for c in costs if np.isfinite(c)]
            if finite:
                mean = np.mean(finite)
                std = np.std(finite)
                row.append(f'{mean:.4f} ± {std:.4f}')
            else:
                row.append('no sol.')
        rows.append(row)
    return rows


def _generate_time_table(all_results):
    """Time-to-solution table (mean ± std seconds)."""
    header = ['Environment'] + PLANNER_NAMES
    rows = [header]

    for env_name, results in all_results.items():
        row = [env_name]
        for pname in PLANNER_NAMES:
            times = [r['time_elapsed'] for r in results[pname]]
            row.append(f'{np.mean(times):.2f} ± {np.std(times):.2f}')
        rows.append(row)
    return rows


def _generate_success_rate_table(all_results):
    """Success rate table (% of trials that found a path)."""
    header = ['Environment'] + PLANNER_NAMES
    rows = [header]

    for env_name, results in all_results.items():
        row = [env_name]
        for pname in PLANNER_NAMES:
            costs = [r['final_cost'] for r in results[pname]]
            rate = sum(1 for c in costs if np.isfinite(c)) / max(len(costs), 1)
            row.append(f'{rate * 100:.0f}%')
        rows.append(row)
    return rows


def _generate_aggregated_table(all_results):
    """Summary aggregated across environments — one row per planner.

    For each planner, average its per-env mean cost, per-env mean time, and
    per-env success rate over all environments in ``all_results``.
    """
    env_names = list(all_results.keys())
    header = [
        'Planner',
        'Mean cost (avg envs)',
        'Mean time (s, avg envs)',
        'Success rate (avg envs)',
        'N envs',
    ]
    rows = [header]

    for pname in PLANNER_NAMES:
        env_cost_means = []
        env_time_means = []
        env_success = []
        for env_name in env_names:
            trials = all_results[env_name].get(pname, [])
            finite = [t['final_cost'] for t in trials
                      if np.isfinite(t['final_cost'])]
            if finite:
                env_cost_means.append(float(np.mean(finite)))
            times = [t['time_elapsed'] for t in trials]
            if times:
                env_time_means.append(float(np.mean(times)))
            if trials:
                rate = sum(1 for t in trials
                           if np.isfinite(t['final_cost'])) / len(trials)
                env_success.append(rate)

        if env_cost_means:
            cost_str = f'{np.mean(env_cost_means):.4f} ± {np.std(env_cost_means):.4f}'
        else:
            cost_str = 'no sol.'
        time_str = (f'{np.mean(env_time_means):.2f} ± {np.std(env_time_means):.2f}'
                    if env_time_means else 'n/a')
        succ_str = (f'{np.mean(env_success) * 100:.0f}%'
                    if env_success else 'n/a')
        rows.append([pname, cost_str, time_str, succ_str, len(env_names)])

    return rows


def _plot_aggregated_summary(all_results,
                             filename='comparison_aggregated.png'):
    """2-panel bar chart: mean cost and mean time per planner, averaged
    across all environments (one bar per planner, not per env).
    """
    env_names = list(all_results.keys())
    if not env_names:
        return

    cost_means, cost_stds = [], []
    time_means, time_stds = [], []
    for pname in PLANNER_NAMES:
        env_cost_means = []
        env_time_means = []
        for env_name in env_names:
            trials = all_results[env_name].get(pname, [])
            finite = [t['final_cost'] for t in trials
                      if np.isfinite(t['final_cost'])]
            if finite:
                env_cost_means.append(float(np.mean(finite)))
            times = [t['time_elapsed'] for t in trials]
            if times:
                env_time_means.append(float(np.mean(times)))
        cost_means.append(np.mean(env_cost_means) if env_cost_means else 0.0)
        cost_stds.append(np.std(env_cost_means) if env_cost_means else 0.0)
        time_means.append(np.mean(env_time_means) if env_time_means else 0.0)
        time_stds.append(np.std(env_time_means) if env_time_means else 0.0)

    fig, (ax_c, ax_t) = plt.subplots(1, 2, figsize=(12, 5))
    xs = np.arange(len(PLANNER_NAMES))
    colors = [PLANNER_COLORS[p] for p in PLANNER_NAMES]

    ax_c.bar(xs, cost_means, yerr=cost_stds, color=colors, alpha=0.85,
             capsize=4, edgecolor='white', linewidth=0.5)
    ax_c.set_xticks(xs)
    ax_c.set_xticklabels(PLANNER_NAMES, rotation=15, fontsize=9)
    ax_c.set_ylabel('Mean final cost (avg over envs)')
    ax_c.set_title(f'Cost — averaged over {len(env_names)} environment(s)')
    ax_c.grid(True, alpha=0.3, axis='y')

    ax_t.bar(xs, time_means, yerr=time_stds, color=colors, alpha=0.85,
             capsize=4, edgecolor='white', linewidth=0.5)
    ax_t.set_xticks(xs)
    ax_t.set_xticklabels(PLANNER_NAMES, rotation=15, fontsize=9)
    ax_t.set_ylabel('Mean time (s, avg over envs)')
    ax_t.set_title(f'Time — averaged over {len(env_names)} environment(s)')
    ax_t.grid(True, alpha=0.3, axis='y')

    fig.suptitle('Aggregated planner comparison (env-mean of per-env means)',
                 fontsize=11, fontweight='bold')
    fig.tight_layout()
    fig.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  → saved {filename}')


def _save_csv(rows, filename):
    with open(filename, 'w') as f:
        for row in rows:
            f.write(','.join(str(c) for c in row) + '\n')
    print(f'  → saved {filename}')


def _print_table(rows, title=''):
    if title:
        print(f'\n  {title}')
        print('  ' + '-' * len(title))
    col_widths = [max(len(str(rows[r][c])) for r in range(len(rows)))
                  for c in range(len(rows[0]))]
    for r, row in enumerate(rows):
        line = '  | '.join(str(row[c]).ljust(col_widths[c])
                           for c in range(len(row)))
        print(f'  {line}')
        if r == 0:
            print('  ' + '-' * len(line))


# ═══════════════════════════════════════════════════════════════════════
#  Plot generation
# ═══════════════════════════════════════════════════════════════════════

def _plot_convergence_all_envs(all_results, filename='comparison_convergence.png'):
    """Multi-panel convergence curves (mean ± std shading)."""
    n_envs = len(all_results)
    n_cols = min(3, n_envs)
    n_rows = (n_envs + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows),
                             squeeze=False)

    for idx, (env_name, results) in enumerate(all_results.items()):
        r, c = divmod(idx, n_cols)
        ax = axes[r][c]

        for pname in PLANNER_NAMES:
            # Collect per-trial convergence curves, aligned by iteration
            trial_data = results[pname]
            if not trial_data or not trial_data[0]['stats']:
                continue

            max_it = max(len(td['stats']) for td in trial_data)
            cost_matrix = np.full((len(trial_data), max_it), np.nan)
            for ti, td in enumerate(trial_data):
                for si, s in enumerate(td['stats']):
                    cost_matrix[ti, si] = s['c_best']

            # Forward-fill NaN: if a trial ends early, carry last value
            for ti in range(cost_matrix.shape[0]):
                for si in range(1, cost_matrix.shape[1]):
                    if np.isnan(cost_matrix[ti, si]):
                        cost_matrix[ti, si] = cost_matrix[ti, si - 1]

            # Clip to iteration limit for readability
            clip = min(max_it, _CONVERGENCE_ITER_LIMIT)
            iters = np.arange(clip)
            mean_c = np.nanmean(cost_matrix[:, :clip], axis=0)
            std_c = np.nanstd(cost_matrix[:, :clip], axis=0)

            color = PLANNER_COLORS[pname]
            ax.plot(iters, mean_c, color=color, lw=2, label=pname)
            ax.fill_between(iters, mean_c - std_c, mean_c + std_c,
                            color=color, alpha=0.15)

        ax.set_xlabel('Iteration')
        ax.set_ylabel('Cost')
        ax.set_title(env_name)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, _CONVERGENCE_ITER_LIMIT)

        # Clip y axis to reasonable range (within displayed iterations)
        finite_costs = []
        for pname in PLANNER_NAMES:
            for td in results[pname]:
                for si, s in enumerate(td['stats']):
                    if si < _CONVERGENCE_ITER_LIMIT and np.isfinite(s['c_best']):
                        finite_costs.append(s['c_best'])
        if finite_costs:
            ymin = min(finite_costs) * 0.9
            ymax = np.percentile(finite_costs, 95) * 1.1
            ax.set_ylim(ymin, ymax)

    # Hide unused axes
    for idx in range(n_envs, n_rows * n_cols):
        r, c = divmod(idx, n_cols)
        axes[r][c].set_visible(False)

    # Single legend at top
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=len(PLANNER_NAMES),
               fontsize=9, bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  → saved {filename}')


def _plot_boxplots(all_results, filename='comparison_boxplots.png'):
    """Box plots of final cost per planner, one subplot per environment."""
    n_envs = len(all_results)
    n_cols = min(3, n_envs)
    n_rows = (n_envs + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows),
                             squeeze=False)

    for idx, (env_name, results) in enumerate(all_results.items()):
        r, c = divmod(idx, n_cols)
        ax = axes[r][c]

        data, labels, colors = [], [], []
        for pname in PLANNER_NAMES:
            costs = [r_['final_cost'] for r_ in results[pname]
                     if np.isfinite(r_['final_cost'])]
            if costs:
                data.append(costs)
                labels.append(pname)
                colors.append(PLANNER_COLORS[pname])

        if data:
            bp = ax.boxplot(data, labels=labels, patch_artist=True, widths=0.6)
            for patch, col in zip(bp['boxes'], colors):
                patch.set_facecolor(col)
                patch.set_alpha(0.6)
            ax.tick_params(axis='x', rotation=30)

        ax.set_ylabel('Final cost')
        ax.set_title(env_name)
        ax.grid(True, alpha=0.3, axis='y')

    for idx in range(n_envs, n_rows * n_cols):
        r, c = divmod(idx, n_cols)
        axes[r][c].set_visible(False)

    fig.tight_layout()
    fig.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  → saved {filename}')


def _plot_bar_summary(all_results, filename='comparison_bar_summary.png'):
    """Grouped bar chart: mean final cost per planner × environment."""
    env_names = list(all_results.keys())
    n_env = len(env_names)
    n_planners = len(PLANNER_NAMES)
    x = np.arange(n_env)
    width = 0.8 / n_planners

    fig, ax = plt.subplots(figsize=(max(10, 2 * n_env), 6))

    for pi, pname in enumerate(PLANNER_NAMES):
        means, stds = [], []
        for env_name in env_names:
            costs = [r_['final_cost'] for r_ in all_results[env_name][pname]
                     if np.isfinite(r_['final_cost'])]
            if costs:
                means.append(np.mean(costs))
                stds.append(np.std(costs))
            else:
                means.append(0)
                stds.append(0)
        ax.bar(x + pi * width - 0.4 + width / 2, means, width,
               yerr=stds, label=pname,
               color=PLANNER_COLORS[pname], alpha=0.8,
               capsize=3, edgecolor='white', linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(env_names, fontsize=9, rotation=15)
    ax.set_ylabel('Mean final cost')
    ax.set_title('Planner Comparison — Mean Final Cost (± σ)')
    ax.legend(fontsize=8, ncol=3)
    ax.grid(True, alpha=0.3, axis='y')

    fig.tight_layout()
    fig.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  → saved {filename}')


def _plot_time_bar(all_results, filename='comparison_time_bar.png'):
    """Grouped bar chart: mean planning time per planner × environment."""
    env_names = list(all_results.keys())
    n_env = len(env_names)
    n_planners = len(PLANNER_NAMES)
    x = np.arange(n_env)
    width = 0.8 / n_planners

    fig, ax = plt.subplots(figsize=(max(10, 2 * n_env), 6))

    for pi, pname in enumerate(PLANNER_NAMES):
        means, stds = [], []
        for env_name in env_names:
            times = [r_['time_elapsed'] for r_ in all_results[env_name][pname]]
            means.append(np.mean(times))
            stds.append(np.std(times))
        ax.bar(x + pi * width - 0.4 + width / 2, means, width,
               yerr=stds, label=pname,
               color=PLANNER_COLORS[pname], alpha=0.8,
               capsize=3, edgecolor='white', linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(env_names, fontsize=9, rotation=15)
    ax.set_ylabel('Mean time (s)')
    ax.set_title('Planner Comparison — Mean Planning Time (± σ)')
    ax.legend(fontsize=8, ncol=3)
    ax.grid(True, alpha=0.3, axis='y')

    fig.tight_layout()
    fig.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  → saved {filename}')


def _plot_mc_statistics(all_results, filename='comparison_mc_stats.png'):
    """Detailed Monte Carlo statistics table as a figure.

    Shows mean, std, median, min, max, success rate for each planner
    in each environment.
    """
    env_names = list(all_results.keys())

    # Build table data
    cell_text = []
    row_labels = []
    for env_name in env_names:
        for pname in PLANNER_NAMES:
            row_labels.append(f'{env_name}\n{pname}')
            costs = [r_['final_cost'] for r_ in all_results[env_name][pname]]
            finite = [c for c in costs if np.isfinite(c)]
            n_total = len(costs)
            n_ok = len(finite)
            if finite:
                cell_text.append([
                    f'{np.mean(finite):.4f}',
                    f'{np.std(finite):.4f}',
                    f'{np.median(finite):.4f}',
                    f'{np.min(finite):.4f}',
                    f'{np.max(finite):.4f}',
                    f'{n_ok}/{n_total}',
                ])
            else:
                cell_text.append(['—', '—', '—', '—', '—', f'0/{n_total}'])

    col_labels = ['Mean', 'Std', 'Median', 'Min', 'Max', 'Success']

    fig, ax = plt.subplots(figsize=(12, 0.5 + 0.35 * len(row_labels)))
    ax.axis('off')
    ax.set_title('Monte Carlo Statistics — Final Path Cost', fontsize=13, pad=12)

    table = ax.table(cellText=cell_text, colLabels=col_labels,
                     rowLabels=row_labels, loc='center',
                     cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(7)
    table.scale(1.0, 1.3)

    # Color-code rows by planner
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor('#E0E0E0')
        elif col == -1:
            pname_idx = (row - 1) % len(PLANNER_NAMES)
            pname = PLANNER_NAMES[pname_idx]
            cell.set_facecolor(PLANNER_COLORS[pname])
            cell.set_alpha(0.3)

    fig.tight_layout()
    fig.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  → saved {filename}')


def _plot_convergence_single_env(env_name, results, filename):
    """Convergence plot for a single environment."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: cost convergence
    ax = axes[0]
    for pname in PLANNER_NAMES:
        trial_data = results[pname]
        if not trial_data or not trial_data[0]['stats']:
            continue
        max_it = max(len(td['stats']) for td in trial_data)
        cost_matrix = np.full((len(trial_data), max_it), np.nan)
        for ti, td in enumerate(trial_data):
            for si, s in enumerate(td['stats']):
                cost_matrix[ti, si] = s['c_best']
        for ti in range(cost_matrix.shape[0]):
            for si in range(1, cost_matrix.shape[1]):
                if np.isnan(cost_matrix[ti, si]):
                    cost_matrix[ti, si] = cost_matrix[ti, si - 1]

        clip = min(max_it, _CONVERGENCE_ITER_LIMIT)
        iters = np.arange(clip)
        mean_c = np.nanmean(cost_matrix[:, :clip], axis=0)
        std_c = np.nanstd(cost_matrix[:, :clip], axis=0)
        color = PLANNER_COLORS[pname]
        ax.plot(iters, mean_c, color=color, lw=2, label=pname)
        ax.fill_between(iters, mean_c - std_c, mean_c + std_c,
                        color=color, alpha=0.15)

    ax.set_xlabel('Iteration')
    ax.set_ylabel('Cost')
    ax.set_title(f'{env_name} — Cost Convergence')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, _CONVERGENCE_ITER_LIMIT)

    # Clip y axis (within displayed iterations)
    finite_costs = []
    for pname in PLANNER_NAMES:
        for td in results[pname]:
            for si, s in enumerate(td['stats']):
                if si < _CONVERGENCE_ITER_LIMIT and np.isfinite(s['c_best']):
                    finite_costs.append(s['c_best'])
    if finite_costs:
        ymin = min(finite_costs) * 0.9
        ymax = np.percentile(finite_costs, 90) * 1.15
        ax.set_ylim(ymin, ymax)

    # Right: cost vs time
    ax2 = axes[1]
    for pname in PLANNER_NAMES:
        trial_data = results[pname]
        if not trial_data or not trial_data[0]['stats']:
            continue
        # Use first trial as representative
        stats = trial_data[0]['stats']
        times = [s['time_elapsed'] for s in stats]
        costs = [s['c_best'] for s in stats]
        ax2.plot(times, costs, color=PLANNER_COLORS[pname], lw=2, label=pname)

    ax2.set_xlabel('Computation time [s]')
    ax2.set_ylabel('Cost')
    ax2.set_title(f'{env_name} — Cost vs. Time')
    ax2.set_xscale('log')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)
    if finite_costs:
        ax2.set_ylim(ymin, ymax)

    fig.tight_layout()
    fig.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  → saved {filename}')


# ═══════════════════════════════════════════════════════════════════════
#  Convergence-vs-wall-clock-time plot  (multi-trial mean ± std)
# ═══════════════════════════════════════════════════════════════════════

def _plot_convergence_vs_time(all_results,
                              filename='comparison_convergence_vs_time.png'):
    """Multi-panel: cost vs. wall-clock time (mean ± std over trials)."""
    n_envs = len(all_results)
    n_cols = min(3, n_envs)
    n_rows = (n_envs + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(6 * n_cols, 5 * n_rows),
                             squeeze=False)

    # Common time grid for interpolation (log-spaced)
    t_max_global = 0.0
    t_min_pos = np.inf
    for results in all_results.values():
        for pname in PLANNER_NAMES:
            for td in results[pname]:
                if td['stats']:
                    t_max_global = max(t_max_global,
                                       td['stats'][-1]['time_elapsed'])
                    for s in td['stats']:
                        if s['time_elapsed'] > 0:
                            t_min_pos = min(t_min_pos, s['time_elapsed'])
    if t_min_pos == np.inf or t_min_pos <= 0:
        t_min_pos = t_max_global * 1e-3
    t_grid = np.geomspace(t_min_pos * 0.8, t_max_global * 1.05, 300)

    for idx, (env_name, results) in enumerate(all_results.items()):
        r, c = divmod(idx, n_cols)
        ax = axes[r][c]

        for pname in PLANNER_NAMES:
            trial_data = results[pname]
            if not trial_data or not trial_data[0]['stats']:
                continue

            # Interpolate each trial onto the common time grid
            interp_costs = []
            for td in trial_data:
                times = np.array([s['time_elapsed'] for s in td['stats']])
                costs = np.array([s['c_best'] for s in td['stats']])
                # Extend last cost value beyond trial end
                interp_c = np.interp(t_grid, times, costs,
                                     left=np.nan, right=costs[-1])
                interp_costs.append(interp_c)

            interp_costs = np.array(interp_costs)
            mean_c = np.nanmean(interp_costs, axis=0)
            std_c = np.nanstd(interp_costs, axis=0)

            color = PLANNER_COLORS[pname]
            ax.plot(t_grid, mean_c, color=color, lw=2, label=pname)
            ax.fill_between(t_grid, mean_c - std_c, mean_c + std_c,
                            color=color, alpha=0.15)

        ax.set_xlabel('Computation time [s]')
        ax.set_ylabel('Cost')
        ax.set_title(env_name)
        ax.set_xscale('log')
        ax.grid(True, alpha=0.3)

        # Clip y-axis
        finite_costs = []
        for pname in PLANNER_NAMES:
            for td in results[pname]:
                for s in td['stats']:
                    if np.isfinite(s['c_best']):
                        finite_costs.append(s['c_best'])
        if finite_costs:
            ymin = min(finite_costs) * 0.9
            ymax = np.percentile(finite_costs, 95) * 1.1
            ax.set_ylim(ymin, ymax)

    for idx in range(n_envs, n_rows * n_cols):
        r, c = divmod(idx, n_cols)
        axes[r][c].set_visible(False)

    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center',
               ncol=len(PLANNER_NAMES), fontsize=9,
               bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  → saved {filename}')


# ═══════════════════════════════════════════════════════════════════════
#  Statistical significance (Mann-Whitney U, RIT* vs each baseline)
# ═══════════════════════════════════════════════════════════════════════

def _statistical_significance_table(all_results,
                                    filename='comparison_significance.csv'):
    """Pairwise Mann-Whitney U test: RIT* costs vs. each baseline.

    Produces a CSV table where each cell is the p-value.  '*' marks
    p < 0.05, '**' marks p < 0.01.
    """
    baselines = [n for n in PLANNER_NAMES if n != 'RIT*']
    header = ['Environment'] + baselines
    rows = [header]

    for env_name, results in all_results.items():
        row = [env_name]
        rit_costs = [r_['final_cost'] for r_ in results['RIT*']
                     if np.isfinite(r_['final_cost'])]
        for bname in baselines:
            b_costs = [r_['final_cost'] for r_ in results[bname]
                       if np.isfinite(r_['final_cost'])]
            if len(rit_costs) >= 2 and len(b_costs) >= 2:
                _, p = mannwhitneyu(rit_costs, b_costs,
                                    alternative='two-sided')
                sig = '**' if p < 0.01 else ('*' if p < 0.05 else '')
                row.append(f'{p:.4f}{sig}')
            else:
                row.append('n/a')
        rows.append(row)

    _save_csv(rows, filename)
    _print_table(rows, 'STATISTICAL SIGNIFICANCE (Mann-Whitney U, RIT* vs baseline)')
    return rows


# ═══════════════════════════════════════════════════════════════════════
#  Normalized comparison table (% improvement over Informed RRT*)
# ═══════════════════════════════════════════════════════════════════════

def _normalized_comparison_table(all_results,
                                 filename='comparison_normalized.csv'):
    """Percentage improvement in cost over Informed RRT* baseline.

    For each planner p in environment e:
        improvement = (mean_cost_IRRT - mean_cost_p) / mean_cost_IRRT * 100

    Positive = better than Informed RRT*,  Negative = worse.
    """
    header = ['Environment'] + PLANNER_NAMES
    rows = [header]

    for env_name, results in all_results.items():
        row = [env_name]
        # Informed RRT* as baseline
        irrt_costs = [r_['final_cost'] for r_ in results['Informed RRT*']
                      if np.isfinite(r_['final_cost'])]
        irrt_mean = np.mean(irrt_costs) if irrt_costs else np.inf

        for pname in PLANNER_NAMES:
            costs = [r_['final_cost'] for r_ in results[pname]
                     if np.isfinite(r_['final_cost'])]
            if costs and np.isfinite(irrt_mean) and irrt_mean > 0:
                pmean = np.mean(costs)
                improvement = (irrt_mean - pmean) / irrt_mean * 100.0
                row.append(f'{improvement:+.2f}%')
            else:
                row.append('n/a')
        rows.append(row)

    _save_csv(rows, filename)
    _print_table(rows, 'NORMALIZED COST (% improvement over Informed RRT*)')
    return rows


# ═══════════════════════════════════════════════════════════════════════
#  Theory validation CSV (Theorems 1-3)
# ═══════════════════════════════════════════════════════════════════════

def _generate_theory_validation_csv(all_results, filename):
    """Generate a CSV comparing theoretical predictions vs measurements.

    Columns: Environment, Vol Ratio (theory), Speedup (theory),
    Time Speedup (actual), Sample Speedup (actual).
    """
    header = ['Environment', 'Vol Ratio (theory)', 'Speedup (theory)',
              'Time Speedup (actual)', 'Sample Speedup (actual)']
    rows = [header]

    for env_name, results in all_results.items():
        theory = results.get('_theory', {})
        row = [
            env_name,
            f"{theory.get('analytical_volume_ratio', 1.0):.4f}",
            f"{theory.get('predicted_speedup', 1.0):.2f}",
            f"{theory.get('actual_time_speedup', float('nan')):.2f}",
            f"{theory.get('actual_sample_speedup', float('nan')):.2f}",
        ]
        rows.append(row)

    _save_csv(rows, filename)
    _print_table(rows, 'THEORY VS EXPERIMENT (Theorems 1-3)')
    return rows


# ═══════════════════════════════════════════════════════════════════════
#  Environment visualization with samples + path
# ═══════════════════════════════════════════════════════════════════════

def _visualize_env_with_path(env_name, env_fn, plots_dir):
    """Run RIT* once via plan_stepwise and render environment + tree + path.

    Works for 2D (scatter), 3D (3D scatter), and high-D (pairwise projections).
    """
    coll, _, metric, xs, xg, bounds = env_fn()
    dim = len(xs)
    safe = env_name.lower().replace(' ', '_')

    # Run RIT* via stepwise to capture final tree state
    bs = 200 if dim >= 6 else 100
    mi = 200 if dim >= 6 else 150
    planner = RITStar(xs, xg, bounds, coll, metric,
                      geodesic_tier='diagonal',
                      batch_size=bs, max_iterations=mi, random_seed=42)
    last_state = None
    for state in planner.plan_stepwise():
        last_state = state

    if last_state is None:
        print(f'    [{env_name}] No iterations — skipping visualization')
        return

    vertices = np.array(last_state['vertices'])
    edges = last_state['edges']
    path = last_state['path']
    cost = last_state['c_best']

    # ── Sample obstacle boundaries for drawing ──
    if dim == 2:
        _viz_2d(env_name, safe, coll, bounds, xs, xg,
                vertices, edges, path, cost, plots_dir)
    elif dim == 3:
        _viz_3d(env_name, safe, coll, bounds, xs, xg,
                vertices, edges, path, cost, plots_dir)
    else:
        _viz_nd(env_name, safe, coll, bounds, xs, xg,
                vertices, edges, path, cost, dim, plots_dir)


def _sample_obstacle_grid(coll, bounds, dim, resolution):
    """Sample a grid and return points that are in collision."""
    if dim == 2:
        x = np.linspace(bounds[0][0], bounds[0][1], resolution)
        y = np.linspace(bounds[1][0], bounds[1][1], resolution)
        xx, yy = np.meshgrid(x, y)
        pts = np.column_stack([xx.ravel(), yy.ravel()])
    elif dim == 3:
        res3 = min(resolution, 50)  # coarser for 3D
        x = np.linspace(bounds[0][0], bounds[0][1], res3)
        y = np.linspace(bounds[1][0], bounds[1][1], res3)
        z = np.linspace(bounds[2][0], bounds[2][1], res3)
        xx, yy, zz = np.meshgrid(x, y, z)
        pts = np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])
    else:
        return None  # Too expensive for high-D
    in_coll = np.array([not coll(p) for p in pts])
    return pts[in_coll]


def _viz_2d(env_name, safe, coll, bounds, xs, xg,
            vertices, edges, path, cost, plots_dir):
    """2D environment: obstacles (gray), tree (light gray), path (purple)."""
    fig, ax = plt.subplots(figsize=(8, 7))

    # Obstacle regions
    obs_pts = _sample_obstacle_grid(coll, bounds, 2, 200)
    if obs_pts is not None and len(obs_pts) > 0:
        ax.scatter(obs_pts[:, 0], obs_pts[:, 1], s=1, c='lightgray',
                   marker='s', alpha=0.6, zorder=0)

    # Tree edges
    for p1, p2 in edges:
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]],
                'silver', lw=0.3, alpha=0.5, zorder=1)

    # Tree vertices
    ax.scatter(vertices[:, 0], vertices[:, 1], s=3, c='#cccccc',
               alpha=0.6, zorder=2)

    # Path
    if path and len(path) > 1:
        pp = np.array(path)
        ax.plot(pp[:, 0], pp[:, 1], color='#7B2FBE', lw=3, zorder=4,
                label=f'RIT* path (cost={cost:.3f})')

    # Start / goal
    ax.plot(xs[0], xs[1], 'go', ms=12, zorder=5, label='Start')
    ax.plot(xg[0], xg[1], 'r^', ms=12, zorder=5, label='Goal')

    ax.set_xlim(bounds[0])
    ax.set_ylim(bounds[1])
    ax.set_aspect('equal')
    ax.set_xlabel('$x_1$', fontsize=12)
    ax.set_ylabel('$x_2$', fontsize=12)
    ax.set_title(f'{env_name}', fontsize=13)
    ax.legend(fontsize=9, loc='upper left')
    ax.grid(True, alpha=0.2)

    fname = os.path.join(plots_dir, f'env_{safe}.png')
    fig.tight_layout()
    fig.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'    → saved {fname}')


def _viz_3d(env_name, safe, coll, bounds, xs, xg,
            vertices, edges, path, cost, plots_dir):
    """3D environment: obstacles, tree, path in 3D scatter + side views."""
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    fig = plt.figure(figsize=(16, 6))

    # ── Main 3D view ──
    ax3 = fig.add_subplot(131, projection='3d')
    obs_pts = _sample_obstacle_grid(coll, bounds, 3, 40)
    if obs_pts is not None and len(obs_pts) > 0:
        ax3.scatter(obs_pts[:, 0], obs_pts[:, 1], obs_pts[:, 2],
                    s=4, c='lightgray', alpha=0.15, zorder=0)

    # Tree edges (subsample for clarity)
    step = max(1, len(edges) // 500)
    for i in range(0, len(edges), step):
        p1, p2 = edges[i]
        ax3.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]],
                 'silver', lw=0.2, alpha=0.3)

    # Tree vertices (subsample)
    vs = max(1, len(vertices) // 1000)
    ax3.scatter(vertices[::vs, 0], vertices[::vs, 1], vertices[::vs, 2],
                s=2, c='#cccccc', alpha=0.4, zorder=2)

    if path and len(path) > 1:
        pp = np.array(path)
        ax3.plot(pp[:, 0], pp[:, 1], pp[:, 2],
                 color='#7B2FBE', lw=3, zorder=5,
                 label=f'RIT* (cost={cost:.3f})')

    ax3.scatter(*xs, c='green', s=80, zorder=6, marker='o', label='Start')
    ax3.scatter(*xg, c='red', s=80, zorder=6, marker='^', label='Goal')
    ax3.set_xlabel('$x_1$')
    ax3.set_ylabel('$x_2$')
    ax3.set_zlabel('$x_3$')
    ax3.set_title(f'{env_name} — 3D view')
    ax3.legend(fontsize=7, loc='upper left')
    ax3.view_init(elev=25, azim=135)

    # ── XY projection ──
    ax_xy = fig.add_subplot(132)
    if obs_pts is not None and len(obs_pts) > 0:
        ax_xy.scatter(obs_pts[:, 0], obs_pts[:, 1], s=1, c='lightgray',
                      marker='s', alpha=0.4)
    ax_xy.scatter(vertices[::vs, 0], vertices[::vs, 1], s=1, c='#cccccc', alpha=0.4)
    for i in range(0, len(edges), step):
        p1, p2 = edges[i]
        ax_xy.plot([p1[0], p2[0]], [p1[1], p2[1]], 'silver', lw=0.2, alpha=0.3)
    if path and len(path) > 1:
        pp = np.array(path)
        ax_xy.plot(pp[:, 0], pp[:, 1], color='#7B2FBE', lw=2.5)
    ax_xy.plot(xs[0], xs[1], 'go', ms=10)
    ax_xy.plot(xg[0], xg[1], 'r^', ms=10)
    ax_xy.set_xlabel('$x_1$')
    ax_xy.set_ylabel('$x_2$')
    ax_xy.set_title('XY projection')
    ax_xy.set_aspect('equal')
    ax_xy.grid(True, alpha=0.2)

    # ── XZ projection ──
    ax_xz = fig.add_subplot(133)
    if obs_pts is not None and len(obs_pts) > 0:
        ax_xz.scatter(obs_pts[:, 0], obs_pts[:, 2], s=1, c='lightgray',
                      marker='s', alpha=0.4)
    ax_xz.scatter(vertices[::vs, 0], vertices[::vs, 2], s=1, c='#cccccc', alpha=0.4)
    for i in range(0, len(edges), step):
        p1, p2 = edges[i]
        ax_xz.plot([p1[0], p2[0]], [p1[2], p2[2]], 'silver', lw=0.2, alpha=0.3)
    if path and len(path) > 1:
        pp = np.array(path)
        ax_xz.plot(pp[:, 0], pp[:, 2], color='#7B2FBE', lw=2.5)
    ax_xz.plot(xs[0], xs[2], 'go', ms=10)
    ax_xz.plot(xg[0], xg[2], 'r^', ms=10)
    ax_xz.set_xlabel('$x_1$')
    ax_xz.set_ylabel('$x_3$')
    ax_xz.set_title('XZ projection')
    ax_xz.set_aspect('equal')
    ax_xz.grid(True, alpha=0.2)

    fname = os.path.join(plots_dir, f'env_{safe}.png')
    fig.tight_layout()
    fig.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'    → saved {fname}')


def _viz_nd(env_name, safe, coll, bounds, xs, xg,
            vertices, edges, path, cost, dim, plots_dir):
    """High-D environment: pairwise 2D projections of dims 0-1, 0-2, 1-2."""
    pairs = [(0, 1), (0, 2), (1, 2)]
    if dim > 3:
        pairs.append((0, 3))
        if dim > 4:
            pairs.append((2, 4))
        if dim > 5:
            pairs.append((3, 5))

    n = len(pairs)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5))
    if n == 1:
        axes = [axes]

    vs = max(1, len(vertices) // 2000)
    step = max(1, len(edges) // 800)

    for ax, (di, dj) in zip(axes, pairs):
        # Tree vertices
        ax.scatter(vertices[::vs, di], vertices[::vs, dj],
                   s=2, c='#cccccc', alpha=0.4, zorder=1)

        # Tree edges
        for i in range(0, len(edges), step):
            p1, p2 = edges[i]
            ax.plot([p1[di], p2[di]], [p1[dj], p2[dj]],
                    'silver', lw=0.2, alpha=0.3)

        # Path
        if path and len(path) > 1:
            pp = np.array(path)
            ax.plot(pp[:, di], pp[:, dj], color='#7B2FBE', lw=2.5, zorder=4,
                    label=f'cost={cost:.3f}' if (di, dj) == pairs[0] else None)

        ax.plot(xs[di], xs[dj], 'go', ms=10, zorder=5)
        ax.plot(xg[di], xg[dj], 'r^', ms=10, zorder=5)
        ax.set_xlabel(f'$x_{{{di+1}}}$', fontsize=11)
        ax.set_ylabel(f'$x_{{{dj+1}}}$', fontsize=11)
        ax.set_title(f'Dims ({di+1},{dj+1})')
        ax.set_xlim(bounds[di])
        ax.set_ylim(bounds[dj])
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.2)

    fig.suptitle(f'{env_name} — RIT* (cost={cost:.3f})', fontsize=13, y=1.02)
    if path:
        axes[0].legend(fontsize=8)
    fname = os.path.join(plots_dir, f'env_{safe}.png')
    fig.tight_layout()
    fig.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'    → saved {fname}')


# ═══════════════════════════════════════════════════════════════════════
#  Main entry point
# ═══════════════════════════════════════════════════════════════════════

def run_full_comparison(n_trials: int = 10,
                        max_iterations: int = 150,
                        batch_size: int = 100,
                        base_seed: int = 42,
                        visualize: bool = True,
                        environments: dict = None):
    """Run the complete 7-planner × multi-environment comparison.

    Parameters
    ----------
    n_trials : int
        Monte Carlo repetitions per planner per environment (default 10).
    max_iterations : int
        Iteration budget for each planner run (default 150).
    batch_size : int
        Samples per iteration (default 100).
    base_seed : int
        Base random seed for reproducibility.
    visualize : bool
        If True (default), generate and save PNG plots, CSVs, and
        all heavy visualization artefacts.  If False, only print
        summary tables to stdout — much faster for benchmarking.
    environments : dict or None
        Optional dict of {env_name: env_fn} to run. If None, uses
        the default COMPARISON_ENVS registry.

    Returns
    -------
    all_results : dict[env_name -> dict[planner_name -> list[dict]]]
    """
    envs = environments if environments is not None else COMPARISON_ENVS

    print('\n' + '=' * 60)
    print('  FULL PLANNER COMPARISON')
    print(f'  {len(PLANNER_NAMES)} planners × {len(envs)} environments'
          f' × {n_trials} trials')
    print('=' * 60)

    t0 = time.time()
    all_results = {}

    # Intermediate results cache for resume support
    cache_dir = os.path.join(RESULTS_DIR, '_comparison_cache')
    os.makedirs(cache_dir, exist_ok=True)

    for ei, (env_name, env_fn) in enumerate(envs.items()):
        safe_name = env_name.lower().replace(' ', '_')
        cache_path = os.path.join(cache_dir, f'{safe_name}.pkl')

        # Resume: skip environments that were already completed
        if os.path.exists(cache_path):
            with open(cache_path, 'rb') as f:
                cached = pickle.load(f)
            # Validate that cached data matches current planner set
            if set(cached.keys()) - {'_theory'} == set(PLANNER_NAMES):
                print(f'\n  [{ei + 1}/{len(envs)}] {env_name}  (cached)')
                all_results[env_name] = cached
                continue
            else:
                os.remove(cache_path)  # stale cache, re-run

        print(f'\n  [{ei + 1}/{len(envs)}] {env_name}')
        results = _run_single_env(
            env_name, env_fn, n_trials, max_iterations,
            batch_size, base_seed + ei * 1000)
        all_results[env_name] = results

        # Save intermediate results so we can resume after interruption
        with open(cache_path, 'wb') as f:
            pickle.dump(results, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f'    → checkpoint saved')
        gc.collect()

    total_time = time.time() - t0
    print(f'\n  All trials done in {total_time:.1f}s')

    # ── Tables ────────────────────────────────────────────────────────
    print('\n  Generating tables...')
    cost_table = _generate_summary_table(all_results)
    _print_table(cost_table, 'FINAL COST (mean ± std)')

    time_table = _generate_time_table(all_results)
    _print_table(time_table, 'PLANNING TIME (mean ± std, seconds)')

    success_table = _generate_success_rate_table(all_results)
    _print_table(success_table, 'SUCCESS RATE')

    # Aggregated-across-envs summary (one row per planner)
    aggregated_table = _generate_aggregated_table(all_results)
    _print_table(aggregated_table,
                 f'AGGREGATED ACROSS {len(all_results)} ENVIRONMENT(S)')

    if visualize:
        _save_csv(cost_table, os.path.join(RESULTS_DIR, 'comparison_cost_table.csv'))
        _save_csv(time_table, os.path.join(RESULTS_DIR, 'comparison_time_table.csv'))
        _save_csv(success_table, os.path.join(RESULTS_DIR, 'comparison_success_table.csv'))
        _save_csv(aggregated_table,
                  os.path.join(RESULTS_DIR, 'comparison_aggregated.csv'))

        # ── Plots ─────────────────────────────────────────────────────
        print('\n  Generating plots...')
        _plot_convergence_all_envs(all_results, os.path.join(PLOTS_DIR, 'comparison_convergence.png'))
        _plot_convergence_vs_time(all_results,
                                  os.path.join(PLOTS_DIR, 'comparison_convergence_vs_time.png'))
        _plot_boxplots(all_results, os.path.join(PLOTS_DIR, 'comparison_boxplots.png'))
        _plot_bar_summary(all_results, os.path.join(PLOTS_DIR, 'comparison_bar_summary.png'))
        _plot_time_bar(all_results, os.path.join(PLOTS_DIR, 'comparison_time_bar.png'))
        _plot_mc_statistics(all_results, os.path.join(PLOTS_DIR, 'comparison_mc_stats.png'))
        _plot_aggregated_summary(
            all_results,
            os.path.join(PLOTS_DIR, 'comparison_aggregated.png'))

        # Per-environment convergence plots
        for env_name in all_results:
            safe = env_name.lower().replace(' ', '_')
            _plot_convergence_single_env(
                env_name, all_results[env_name],
                os.path.join(PLOTS_DIR, f'comparison_{safe}_convergence.png'))

        # ── Statistical analysis ──────────────────────────────────────
        print('\n  Statistical analysis...')
        _statistical_significance_table(all_results,
                                        os.path.join(RESULTS_DIR, 'comparison_significance.csv'))
        _normalized_comparison_table(all_results,
                                     os.path.join(RESULTS_DIR, 'comparison_normalized.csv'))

        # ── Theory validation table (Theorems 1-3) ───────────────────
        _generate_theory_validation_csv(all_results,
                                        os.path.join(RESULTS_DIR, 'theory_validation.csv'))

        # ── Per-environment visualizations (tree + path + obstacles) ──
        print('\n  Generating environment visualizations...')
        for env_name, env_fn in envs.items():
            _visualize_env_with_path(env_name, env_fn, PLOTS_DIR)

    print(f'\n  Comparison complete. Total time: {total_time:.1f}s')

    # Clean up cache after successful completion
    import shutil
    shutil.rmtree(cache_dir, ignore_errors=True)

    return all_results
