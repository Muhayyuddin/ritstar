#!/usr/bin/env python3
"""Run 6D UR10e environments only, then merge with existing 2D/3D results."""

import os, sys, gc, time, pickle
import numpy as np

# Verify pybullet
try:
    import pybullet
    print("pybullet OK")
except ImportError:
    print("ERROR: pybullet not installed"); sys.exit(1)

from rit_star.comparison import (
    _run_single_env, _generate_summary_table, _generate_time_table,
    _generate_success_rate_table, _save_csv, _print_table,
    _plot_convergence_all_envs, _plot_convergence_vs_time,
    _plot_boxplots, _plot_bar_summary, _plot_time_bar,
    _plot_mc_statistics, _plot_convergence_single_env,
    _statistical_significance_table, _normalized_comparison_table,
    _generate_theory_validation_csv, PLANNER_NAMES
)
from rit_star.environments import env_6d_tabletop, env_6d_shelf, env_6d_cluttered
from visualization_util.output_paths import RESULTS_DIR, PLOTS_DIR

# 6D environments
ENVS_6D = {
    '6D Shelf':     env_6d_shelf,
    '6D Cluttered': env_6d_cluttered,
    '6D Tabletop':  env_6d_tabletop,
}

N_TRIALS = 5
MAX_ITERATIONS = 150
BATCH_SIZE = 100
BASE_SEED = 42

if __name__ == '__main__':
    print(f"\nRunning 6D environments: {list(ENVS_6D.keys())}")
    print(f"  {len(PLANNER_NAMES)} planners × {len(ENVS_6D)} environments × {N_TRIALS} trials\n")

    t0 = time.time()
    results_6d = {}

    cache_dir = os.path.join(RESULTS_DIR, '_6d_cache')
    os.makedirs(cache_dir, exist_ok=True)

    for ei, (env_name, env_fn) in enumerate(ENVS_6D.items()):
        safe_name = env_name.lower().replace(' ', '_')
        cache_path = os.path.join(cache_dir, f'{safe_name}.pkl')

        if os.path.exists(cache_path):
            with open(cache_path, 'rb') as f:
                cached = pickle.load(f)
            if set(cached.keys()) - {'_theory'} == set(PLANNER_NAMES):
                print(f'  [{ei + 1}/{len(ENVS_6D)}] {env_name}  (cached)')
                results_6d[env_name] = cached
                continue

        print(f'  [{ei + 1}/{len(ENVS_6D)}] {env_name}')
        results = _run_single_env(
            env_name, env_fn, N_TRIALS, MAX_ITERATIONS,
            BATCH_SIZE, BASE_SEED + (ei + 8) * 1000)
        results_6d[env_name] = results

        with open(cache_path, 'wb') as f:
            pickle.dump(results, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f'    → checkpoint saved')
        gc.collect()

    total_time = time.time() - t0
    print(f'\n  6D trials done in {total_time:.1f}s')

    # ── Load existing 2D/3D results from CSVs and merge ───────────
    # We'll read the existing CSVs, append 6D rows, and regenerate everything
    import csv

    # Read existing cost table
    existing_cost_path = os.path.join(RESULTS_DIR, 'comparison_cost_table.csv')
    existing_rows = []
    if os.path.exists(existing_cost_path):
        with open(existing_cost_path) as f:
            existing_rows = list(csv.reader(f))

    # Build combined results dict for plotting/tables
    # For 6D, we have full results; for 2D/3D we only have summary CSVs
    # So let's just append to CSV files directly

    # Cost table
    header = existing_rows[0] if existing_rows else ['Environment'] + PLANNER_NAMES
    new_cost_rows = [header]
    # Keep existing 2D/3D rows
    for row in existing_rows[1:]:
        new_cost_rows.append(row)
    # Add 6D rows
    for env_name, results in results_6d.items():
        row = [env_name]
        for pname in PLANNER_NAMES:
            costs = [r['final_cost'] for r in results[pname]]
            finite = [c for c in costs if np.isfinite(c)]
            if finite:
                row.append(f'{np.mean(finite):.4f} ± {np.std(finite):.4f}')
            else:
                row.append('no sol.')
        new_cost_rows.append(row)

    _save_csv(new_cost_rows, os.path.join(RESULTS_DIR, 'comparison_cost_table.csv'))
    _print_table(new_cost_rows, 'FINAL COST (mean ± std)')

    # Time table
    existing_time_path = os.path.join(RESULTS_DIR, 'comparison_time_table.csv')
    existing_time_rows = []
    if os.path.exists(existing_time_path):
        with open(existing_time_path) as f:
            existing_time_rows = list(csv.reader(f))

    new_time_rows = [header]
    for row in existing_time_rows[1:]:
        new_time_rows.append(row)
    for env_name, results in results_6d.items():
        row = [env_name]
        for pname in PLANNER_NAMES:
            times = [r['time_elapsed'] for r in results[pname]]
            row.append(f'{np.mean(times):.2f} ± {np.std(times):.2f}')
        new_time_rows.append(row)

    _save_csv(new_time_rows, os.path.join(RESULTS_DIR, 'comparison_time_table.csv'))
    _print_table(new_time_rows, 'PLANNING TIME (mean ± std, seconds)')

    # Success table
    existing_success_path = os.path.join(RESULTS_DIR, 'comparison_success_table.csv')
    existing_success_rows = []
    if os.path.exists(existing_success_path):
        with open(existing_success_path) as f:
            existing_success_rows = list(csv.reader(f))

    new_success_rows = [header]
    for row in existing_success_rows[1:]:
        new_success_rows.append(row)
    for env_name, results in results_6d.items():
        row = [env_name]
        for pname in PLANNER_NAMES:
            costs = [r['final_cost'] for r in results[pname]]
            rate = sum(1 for c in costs if np.isfinite(c)) / max(len(costs), 1)
            row.append(f'{rate * 100:.0f}%')
        new_success_rows.append(row)

    _save_csv(new_success_rows, os.path.join(RESULTS_DIR, 'comparison_success_table.csv'))
    _print_table(new_success_rows, 'SUCCESS RATE')

    # Normalized table
    new_norm_rows = [header]
    for row in existing_rows[1:]:
        # Recompute from existing cost data (skip, just keep)
        pass

    # For 6D normalization
    for env_name, results in results_6d.items():
        row = [env_name]
        irrt_costs = [r['final_cost'] for r in results['Informed RRT*']
                      if np.isfinite(r['final_cost'])]
        irrt_mean = np.mean(irrt_costs) if irrt_costs else np.inf
        for pname in PLANNER_NAMES:
            costs = [r['final_cost'] for r in results[pname]
                     if np.isfinite(r['final_cost'])]
            if costs and np.isfinite(irrt_mean) and irrt_mean > 0:
                pmean = np.mean(costs)
                improvement = (irrt_mean - pmean) / irrt_mean * 100.0
                row.append(f'{improvement:+.2f}%')
            else:
                row.append('n/a')
        new_norm_rows.append(row)

    # Read and append to existing normalized CSV
    existing_norm_path = os.path.join(RESULTS_DIR, 'comparison_normalized.csv')
    if os.path.exists(existing_norm_path):
        with open(existing_norm_path) as f:
            existing_norm_rows = list(csv.reader(f))
        final_norm = existing_norm_rows
        for row in new_norm_rows[1:]:
            final_norm.append(row)
    else:
        final_norm = new_norm_rows

    _save_csv(final_norm, os.path.join(RESULTS_DIR, 'comparison_normalized.csv'))
    _print_table(final_norm, 'NORMALIZED COST (% improvement over Informed RRT*)')

    # Significance for 6D only
    from scipy.stats import mannwhitneyu
    baselines = [n for n in PLANNER_NAMES if n != 'RIT*']
    sig_header = ['Environment'] + baselines
    new_sig_rows = []
    for env_name, results in results_6d.items():
        row = [env_name]
        rit_costs = [r['final_cost'] for r in results['RIT*']
                     if np.isfinite(r['final_cost'])]
        for bname in baselines:
            b_costs = [r['final_cost'] for r in results[bname]
                       if np.isfinite(r['final_cost'])]
            if len(rit_costs) >= 2 and len(b_costs) >= 2:
                _, p = mannwhitneyu(rit_costs, b_costs, alternative='two-sided')
                sig = '**' if p < 0.01 else ('*' if p < 0.05 else '')
                row.append(f'{p:.4f}{sig}')
            else:
                row.append('n/a')
        new_sig_rows.append(row)

    existing_sig_path = os.path.join(RESULTS_DIR, 'comparison_significance.csv')
    if os.path.exists(existing_sig_path):
        with open(existing_sig_path) as f:
            existing_sig_rows = list(csv.reader(f))
        final_sig = existing_sig_rows
        for row in new_sig_rows:
            final_sig.append(row)
    else:
        final_sig = [sig_header] + new_sig_rows

    _save_csv(final_sig, os.path.join(RESULTS_DIR, 'comparison_significance.csv'))
    _print_table(final_sig, 'STATISTICAL SIGNIFICANCE (6D, Mann-Whitney U, RIT* vs baseline)')

    print('\n=== 6D analysis complete. All CSV files updated. ===')
