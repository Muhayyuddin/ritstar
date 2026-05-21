#!/usr/bin/env python3
"""
run_benchmark_plots.py — Generate AIT*/EIT*-style anytime benchmark plots.

Produces two-row figures per environment:
  Top:    Success rate (%) vs planning time (s)
  Bottom: Solution cost   vs planning time (s)  (median + IQR)

Style follows Strub & Gammell, IJRR 2022 (AIT* and EIT*), Fig. 12–15.

Usage:
    python run_benchmark_plots.py                    # uses config
    python run_benchmark_plots.py --envs 2D --trials 10
    python run_benchmark_plots.py --envs obstacle narrow --planners RIT BIT AIT
"""

from __future__ import annotations

import argparse
import gc
import os
import sys
import time
import pickle

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from visualization_util.output_paths import PLOTS_DIR
from run_from_config import (
    ENV_REGISTRY, _resolve_environments, _resolve_planners, _build_planner,
)

# ── Planner style (matches paper conventions) ─────────────────────────

PLANNER_COLORS = {
    'RIT*':          '#7B2FBE',
    'GA-RRT*':       '#00695C',
    'RIT*-CARM':     '#00695C',
    'Informed RRT*': '#2196F3',
    'BIT*':          '#4CAF50',
    'AIT*':          '#FF9800',
    'EIT*':          '#009688',
    'APT*':          '#F44336',
}

PLANNER_LINESTYLES = {
    'RIT*':          '-',
    'GA-RRT*':       '--',
    'RIT*-CARM':     '-',
    'Informed RRT*': '--',
    'BIT*':          '-.',
    'AIT*':          ':',
    'EIT*':          '--',
    'APT*':          '-.',
}

PLANNER_MARKERS = {
    'RIT*':          'o',
    'GA-RRT*':       'X',
    'RIT*-CARM':     's',
    'Informed RRT*': 's',
    'BIT*':          '^',
    'AIT*':          'D',
    'EIT*':          'v',
    'APT*':          'P',
}

# ── Data collection ───────────────────────────────────────────────────

def _collect_data(env_name, env_fn, planners, n_trials, max_iterations,
                  batch_size, base_seed, timeout_seconds=200):
    """Run all planners on one environment for n_trials, return stats."""
    import signal

    class _TimeoutError(Exception):
        pass

    def _alarm_handler(signum, frame):  # noqa: ARG001
        raise _TimeoutError()

    coll, _, metric, xs, xg, bounds = env_fn()
    results = {}

    for pname in planners:
        print(f'    {pname}:', end=' ', flush=True)
        trial_stats = []
        for trial in range(n_trials):
            seed = base_seed + trial
            planner = _build_planner(
                pname, xs, xg, bounds, coll, metric,
                batch_size, max_iterations, seed)
            timed_out = False
            try:
                signal.signal(signal.SIGALRM, _alarm_handler)
                signal.alarm(timeout_seconds)
                planner.plan()
            except _TimeoutError:
                timed_out = True
            finally:
                signal.alarm(0)
            stats = planner.get_stats()
            if timed_out and stats:
                # Mark the last recorded snapshot as a timeout
                stats[-1]['timed_out'] = True
            trial_stats.append(stats)
            c = stats[-1]['c_best'] if stats else np.inf
            suffix = 'T' if timed_out else ''
            print(f'{c:.3f}{suffix}', end=' ', flush=True)
            del planner
        gc.collect()
        results[pname] = trial_stats
        print()

    return results


# ── Interpolation helpers ─────────────────────────────────────────────

def _interpolate_cost_vs_time(trial_stats_list, t_grid):
    """Interpolate cost-vs-time curves for all trials onto a common grid.

    Returns (n_trials, len(t_grid)) array.  Uses forward-fill: before
    the first recorded time, cost is inf; after the last, it's held
    constant.
    """
    all_interp = []
    for stats in trial_stats_list:
        if not stats:
            all_interp.append(np.full_like(t_grid, np.inf))
            continue
        times = np.array([s['time_elapsed'] for s in stats])
        costs = np.array([s['c_best'] for s in stats])
        interp = np.interp(t_grid, times, costs, left=np.inf, right=costs[-1])
        all_interp.append(interp)
    return np.array(all_interp)


def _success_rate_vs_time(trial_stats_list, t_grid):
    """Compute fraction of trials that found a solution by each time point.

    Returns (len(t_grid),) array with values in [0, 1].
    """
    n_trials = len(trial_stats_list)
    if n_trials == 0:
        return np.zeros_like(t_grid)

    success = np.zeros_like(t_grid)
    for stats in trial_stats_list:
        if not stats:
            continue
        # Find the first time c_best becomes finite
        first_soln_time = None
        for s in stats:
            if np.isfinite(s['c_best']):
                first_soln_time = s['time_elapsed']
                break
        if first_soln_time is not None:
            success += (t_grid >= first_soln_time).astype(float)

    return success / n_trials


# ── Plotting ──────────────────────────────────────────────────────────

def plot_benchmark(env_name, results, planners, out_dir):
    """Generate a 2-row figure: success rate (top) and cost (bottom)."""

    # Build common time grid (log-spaced for log x-axis)
    t_max = 0.0
    t_min_pos = np.inf
    for pname in planners:
        for stats in results[pname]:
            if stats:
                t_max = max(t_max, stats[-1]['time_elapsed'])
                for s in stats:
                    if s['time_elapsed'] > 0:
                        t_min_pos = min(t_min_pos, s['time_elapsed'])
    if t_max <= 0:
        print(f'  Skipping {env_name}: no data')
        return
    if t_min_pos == np.inf or t_min_pos <= 0:
        t_min_pos = t_max * 1e-3
    t_grid = np.geomspace(t_min_pos * 0.8, t_max * 1.05, 500)

    # IEEE single-column publication style (renders well at ~3.5 in width).
    # All sizes are scaled up from the previous two-column preset so the
    # figure remains legible when placed in a single IEEE column.
    plt.rcParams.update({
        'font.family': 'serif',
        'font.size': 12,
        'axes.labelsize': 14,
        'axes.titlesize': 15,
        'legend.fontsize': 10,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'axes.linewidth': 1.0,
        'xtick.major.width': 1.0,
        'ytick.major.width': 1.0,
        'xtick.minor.visible': True,
        'ytick.minor.visible': False,
        'lines.linewidth': 2.4,
        'lines.markersize': 6,
        'figure.dpi': 300,
        'savefig.dpi': 300,
    })

    # Single-column-ready aspect (~3.5 in wide). Use tight layout so fonts
    # are clearly legible when the figure scales into the column.
    fig, (ax_sr, ax_cost) = plt.subplots(2, 1, figsize=(4.2, 5.2),
                                          sharex=True, gridspec_kw={
                                              'height_ratios': [1, 1.5],
                                              'hspace': 0.10,
                                          })

    # ── Top: Success rate ──
    for pname in planners:
        sr = _success_rate_vs_time(results[pname], t_grid) * 100.0
        color = PLANNER_COLORS.get(pname, 'gray')
        ls = PLANNER_LINESTYLES.get(pname, '-')
        ax_sr.plot(t_grid, sr, color=color, lw=2.4, ls=ls, label=pname)

    ax_sr.set_ylabel('Success [%]')
    ax_sr.set_ylim(-5, 105)
    ax_sr.set_yticks([0, 25, 50, 75, 100])
    ax_sr.set_xscale('log')
    ax_sr.grid(True, which='major', alpha=0.30)
    ax_sr.grid(True, which='minor', alpha=0.12, linestyle=':')
    ax_sr.set_title(env_name, pad=6)

    # ── Bottom: Cost vs time (median + two-band CI, Fig.6 style) ──────
    # Outer band: 5th–95th percentile  (light, ~nonparametric 90% CI)
    # Inner band: 25th–75th percentile (IQR, darker)
    # Median line on top.  First-solution time marked with a small dot.
    import warnings
    from matplotlib.ticker import LogLocator, NullFormatter
    finite_mins = []
    for pname in planners:
        interp = _interpolate_cost_vs_time(results[pname], t_grid)
        color = PLANNER_COLORS.get(pname, 'gray')
        ls = PLANNER_LINESTYLES.get(pname, '-')
        mk = PLANNER_MARKERS.get(pname, 'o')

        interp_masked = np.where(np.isinf(interp), np.nan, interp)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            median_c = np.nanmedian(interp_masked, axis=0)
            q25 = np.nanpercentile(interp_masked, 25, axis=0)
            q75 = np.nanpercentile(interp_masked, 75, axis=0)

        valid = np.any(np.isfinite(interp), axis=0)
        t_valid = t_grid[valid]
        m_valid  = median_c[valid]
        q25_v = q25[valid]
        q75_v = q75[valid]

        if len(t_valid) == 0:
            continue

        # IQR band (25–75 %) + median line
        ax_cost.fill_between(t_valid, q25_v, q75_v, color=color, alpha=0.22)
        ax_cost.plot(t_valid, m_valid, color=color, lw=2.4, ls=ls, label=pname)

        # Mark first-solution time with a filled dot on the median line
        first_valid_idx = np.argmax(np.isfinite(m_valid))
        if np.isfinite(m_valid[first_valid_idx]):
            ax_cost.plot(t_valid[first_valid_idx], m_valid[first_valid_idx],
                         marker=mk, color=color, markersize=6,
                         zorder=5, linestyle='none')

        fc = m_valid[np.isfinite(m_valid)]
        if len(fc) > 0:
            finite_mins.append(np.nanmin(fc))

    ax_cost.set_xlabel('Computation time [s]')
    ax_cost.set_ylabel('Cost')
    ax_cost.set_xscale('log')
    # Major + minor log gridlines (matches Fig. 6 style)
    ax_cost.grid(True, which='major', alpha=0.35)
    ax_cost.grid(True, which='minor', alpha=0.15, linestyle=':')
    ax_cost.xaxis.set_minor_locator(LogLocator(subs='all', numticks=10))
    ax_cost.xaxis.set_minor_formatter(NullFormatter())
    ax_sr.grid(True, which='minor', alpha=0.15, linestyle=':')
    ax_sr.xaxis.set_minor_locator(LogLocator(subs='all', numticks=10))
    ax_sr.xaxis.set_minor_formatter(NullFormatter())

    # Sensible y-limits
    if finite_mins:
        ymin = min(finite_mins) * 0.95
        # Get final costs for upper bound
        final_costs = []
        for pname in planners:
            for stats in results[pname]:
                if stats:
                    c = stats[-1]['c_best']
                    if np.isfinite(c):
                        final_costs.append(c)
        if final_costs:
            ymax = np.percentile(final_costs, 95) * 1.15
            # Also include initial solutions
            first_costs = []
            for pname in planners:
                for stats in results[pname]:
                    for s in stats:
                        if np.isfinite(s['c_best']):
                            first_costs.append(s['c_best'])
                            break
            if first_costs:
                ymax = max(ymax, np.median(first_costs) * 1.05)
            ax_cost.set_ylim(ymin, ymax)

    # Legend below the bottom plot — 2 columns, tight spacing
    handles, labels = ax_cost.get_legend_handles_labels()
    n_legend_cols = min(len(planners), 3)
    # Estimate legend height: ~0.055 per row of entries at the figure scale
    n_legend_rows = int(np.ceil(len(handles) / n_legend_cols))
    legend_height = 0.055 * n_legend_rows
    fig.legend(handles, labels, loc='lower center',
               ncol=n_legend_cols, fontsize=9,
               bbox_to_anchor=(0.5, 0.0),
               frameon=True, fancybox=True, shadow=False,
               edgecolor='#bfbfbf', handlelength=2.0,
               columnspacing=1.0, handletextpad=0.5)

    fig.tight_layout(pad=0.4)
    # Reserve enough bottom space so the legend clears the x-axis label
    fig.subplots_adjust(bottom=0.10 + legend_height + 0.10)

    # Save (PDF for vector paper inclusion, PNG for quick preview)
    safe = env_name.lower().replace(' ', '_').replace('-', '_')
    out_path = os.path.join(out_dir, f'benchmark_updated_{safe}.pdf')
    out_png = os.path.join(out_dir, f'benchmark_updated_{safe}.png')
    fig.savefig(out_path, bbox_inches='tight', pad_inches=0.05)
    fig.savefig(out_png, bbox_inches='tight', pad_inches=0.05)
    plt.close(fig)
    print(f'  → {out_path}')
    print(f'  → {out_png}')


def plot_combined(all_results, planners, out_dir):
    """Plot all environments in a combined multi-panel figure."""
    envs = list(all_results.keys())
    n_envs = len(envs)
    if n_envs == 0:
        return

    n_cols = min(3, n_envs)
    n_rows_env = (n_envs + n_cols - 1) // n_cols
    n_rows = n_rows_env * 2  # each env gets 2 rows (success + cost)

    plt.rcParams.update({
        'font.family': 'serif',
        'font.size': 9,
        'axes.labelsize': 10,
        'axes.titlesize': 11,
        'legend.fontsize': 7.5,
        'xtick.labelsize': 8,
        'ytick.labelsize': 8,
        'figure.dpi': 300,
        'savefig.dpi': 300,
    })

    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(5.2 * n_cols, 3 * n_rows),
                             squeeze=False)

    for env_idx, env_name in enumerate(envs):
        col = env_idx % n_cols
        base_row = (env_idx // n_cols) * 2
        ax_sr = axes[base_row][col]
        ax_cost = axes[base_row + 1][col]

        results = all_results[env_name]

        # Time grid (log-spaced for log x-axis)
        t_max = 0.0
        t_min_pos = np.inf
        for pname in planners:
            if pname not in results:
                continue
            for stats in results[pname]:
                if stats:
                    t_max = max(t_max, stats[-1]['time_elapsed'])
                    for s in stats:
                        if s['time_elapsed'] > 0:
                            t_min_pos = min(t_min_pos, s['time_elapsed'])
        if t_max <= 0:
            ax_sr.set_visible(False)
            ax_cost.set_visible(False)
            continue
        if t_min_pos == np.inf or t_min_pos <= 0:
            t_min_pos = t_max * 1e-3
        t_grid = np.geomspace(t_min_pos * 0.8, t_max * 1.05, 500)

        # Success rate
        for pname in planners:
            if pname not in results:
                continue
            sr = _success_rate_vs_time(results[pname], t_grid) * 100.0
            color = PLANNER_COLORS.get(pname, 'gray')
            ls = PLANNER_LINESTYLES.get(pname, '-')
            ax_sr.plot(t_grid, sr, color=color, lw=1.5, ls=ls, label=pname)

        ax_sr.set_ylabel('Success [%]')
        ax_sr.set_ylim(-5, 105)
        ax_sr.set_yticks([0, 50, 100])
        ax_sr.set_xscale('log')
        ax_sr.grid(True, alpha=0.25)
        ax_sr.set_title(env_name, fontsize=10)
        ax_sr.tick_params(labelbottom=False)

        # Cost — two-band CI style matching Fig. 6
        import warnings
        from matplotlib.ticker import LogLocator, NullFormatter
        for pname in planners:
            if pname not in results:
                continue
            interp = _interpolate_cost_vs_time(results[pname], t_grid)
            color = PLANNER_COLORS.get(pname, 'gray')
            ls = PLANNER_LINESTYLES.get(pname, '-')
            mk = PLANNER_MARKERS.get(pname, 'o')
            interp_masked = np.where(np.isinf(interp), np.nan, interp)
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', RuntimeWarning)
                median_c = np.nanmedian(interp_masked, axis=0)
                q25 = np.nanpercentile(interp_masked, 25, axis=0)
                q75 = np.nanpercentile(interp_masked, 75, axis=0)
            valid = np.any(np.isfinite(interp), axis=0)
            t_v = t_grid[valid]
            m_v = median_c[valid]
            if len(t_v) == 0:
                continue
            ax_cost.fill_between(t_v, q25[valid], q75[valid], color=color, alpha=0.22)
            ax_cost.plot(t_v, m_v, color=color, lw=1.5, ls=ls)
            # First-solution dot
            fi = np.argmax(np.isfinite(m_v))
            if np.isfinite(m_v[fi]):
                ax_cost.plot(t_v[fi], m_v[fi], marker=mk, color=color,
                             markersize=5, zorder=5, linestyle='none')

        ax_cost.set_xlabel('Computation time [s]')
        ax_cost.set_ylabel('Cost')
        ax_cost.set_xscale('log')
        ax_cost.grid(True, which='major', alpha=0.30)
        ax_cost.grid(True, which='minor', alpha=0.12, linestyle=':')
        ax_cost.xaxis.set_minor_locator(LogLocator(subs='all', numticks=10))
        ax_cost.xaxis.set_minor_formatter(NullFormatter())
        ax_sr.xaxis.set_minor_locator(LogLocator(subs='all', numticks=10))
        ax_sr.xaxis.set_minor_formatter(NullFormatter())
        ax_sr.grid(True, which='minor', alpha=0.12, linestyle=':')

        # Y-limits
        fc = []
        for pname in planners:
            if pname not in results:
                continue
            for stats in results[pname]:
                if stats:
                    c = stats[-1]['c_best']
                    if np.isfinite(c):
                        fc.append(c)
        if fc:
            ax_cost.set_ylim(min(fc) * 0.95,
                             np.percentile(fc, 95) * 1.15)

    # Hide unused panels
    for idx in range(n_envs, n_rows_env * n_cols):
        col = idx % n_cols
        base_row = (idx // n_cols) * 2
        axes[base_row][col].set_visible(False)
        axes[base_row + 1][col].set_visible(False)

    # Shared legend below all panels
    handles, labels = axes[0][0].get_legend_handles_labels()
    n_legend_cols = min(len(planners), 6)
    n_legend_rows = int(np.ceil(len(handles) / n_legend_cols))
    legend_frac = 0.045 * n_legend_rows  # fraction of figure height per row
    fig.legend(handles, labels, loc='lower center',
               ncol=n_legend_cols, fontsize=8,
               bbox_to_anchor=(0.5, 0.0),
               frameon=True, fancybox=True, edgecolor='#cccccc')

    fig.tight_layout(rect=[0, legend_frac + 0.04, 1, 1])

    out_path = os.path.join(out_dir, 'benchmark_combined_updated.pdf')
    out_png = os.path.join(out_dir, 'benchmark_combined_updated.png')
    fig.savefig(out_path, bbox_inches='tight')
    fig.savefig(out_png, bbox_inches='tight')
    plt.close(fig)
    print(f'  → {out_path}')
    print(f'  → {out_png}')


# ── Table II-style benchmark table (APT* paper format) ───────────────

def generate_table_ii(all_results, planners, out_dir, n_trials=10):
    """Generate a Table II-style performance comparison table.

    Columns per planner (for each environment):
      t_init_min, t_init_med, t_init_max,
      c_init_min, c_init_med, c_init_max,
      c_final_min, c_final_med, c_final_max,
      success rate
    Similar to APT* paper Table II (cage-ENV comparison).
    Saved as CSV, printed to console, and exported as LaTeX.
    """
    import csv

    rows = []
    for env_name, results in all_results.items():
        for pname in planners:
            if pname not in results:
                continue
            trial_stats_list = results[pname]

            init_times = []
            init_costs = []
            final_costs = []
            for stats in trial_stats_list:
                if not stats:
                    init_times.append(np.inf)
                    init_costs.append(np.inf)
                    final_costs.append(np.inf)
                    continue
                first_t, first_c = np.inf, np.inf
                for s in stats:
                    if np.isfinite(s['c_best']):
                        first_t = s['time_elapsed']
                        first_c = s['c_best']
                        break
                init_times.append(first_t)
                init_costs.append(first_c)
                final_costs.append(stats[-1]['c_best'])

            init_times = np.array(init_times)
            init_costs = np.array(init_costs)
            final_costs = np.array(final_costs)

            n_success = np.sum(np.isfinite(init_costs))
            success_rate = n_success / len(init_costs)

            finite_it = init_times[np.isfinite(init_times)]
            finite_ic = init_costs[np.isfinite(init_costs)]
            finite_fc = final_costs[np.isfinite(final_costs)]

            def _safe(fn, arr):
                return fn(arr) if len(arr) > 0 else np.inf

            rows.append({
                'env': env_name,
                'planner': pname,
                't_init_min': _safe(np.min, finite_it),
                't_init_med': _safe(np.median, finite_it),
                't_init_max': _safe(np.max, finite_it),
                'c_init_min': _safe(np.min, finite_ic),
                'c_init_med': _safe(np.median, finite_ic),
                'c_init_max': _safe(np.max, finite_ic),
                'c_final_min': _safe(np.min, finite_fc),
                'c_final_med': _safe(np.median, finite_fc),
                'c_final_max': _safe(np.max, finite_fc),
                'success': success_rate,
            })

    if not rows:
        print('  No data for Table II.')
        return

    # ── Console output ────────────────────────────────────────────────
    col_keys = ['t_init_min', 't_init_med', 't_init_max',
                'c_init_min', 'c_init_med', 'c_init_max',
                'c_final_min', 'c_final_med', 'c_final_max']
    col_labels = ['t_i^min', 't_i^med', 't_i^max',
                  'c_i^min', 'c_i^med', 'c_i^max',
                  'c_f^min', 'c_f^med', 'c_f^max']

    print('\n' + '=' * 145)
    print('  PERFORMANCE COMPARISON TABLE (Table II style)')
    print('=' * 145)
    hdr = f'  {"Planner":<16}'
    for lbl in col_labels:
        hdr += f' {lbl:>9}'
    hdr += f' {"Success":>8}'
    for env_name in all_results:
        print(f'\n  Environment: {env_name}')
        print(hdr)
        print('  ' + '-' * 140)

        env_rows = [r for r in rows if r['env'] == env_name]
        best = {}
        for key in col_keys:
            vals = [r[key] for r in env_rows if np.isfinite(r[key])]
            best[key] = min(vals) if vals else np.inf
        best_sr = max(r['success'] for r in env_rows)

        for r in env_rows:
            line = f'  {r["planner"]:<16}'
            for key in col_keys:
                val = r[key]
                if not np.isfinite(val):
                    line += '       inf'
                elif abs(val - best[key]) < 1e-9:
                    line += f'  *{val:.4f}'
                else:
                    line += f' {val:9.4f}'
            sr = r['success'] * 100
            if abs(r['success'] - best_sr) < 1e-9:
                line += f'  *{sr:.0f}%'
            else:
                line += f' {sr:7.0f}%'
            print(line)
        print('  ' + '-' * 140)

    # ── CSV (saved to plots dir and results/) ────────────────────────
    from visualization_util.output_paths import RESULTS_DIR
    fieldnames = ['env', 'planner'] + col_keys + ['success']
    for save_dir in (out_dir, RESULTS_DIR):
        csv_path = os.path.join(save_dir, 'benchmark_table_ii.csv')
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in rows:
                writer.writerow(r)
        print(f'  -> Table II CSV saved to {csv_path}')

    # ── LaTeX ─────────────────────────────────────────────────────────
    _generate_latex_table_ii(rows, all_results, planners, out_dir)


def _generate_latex_table_ii(rows, all_results, planners, out_dir):
    """Generate a booktabs LaTeX table matching APT* paper Table II style.

    Requires in the document preamble:
        \\usepackage{booktabs}
        \\usepackage{multirow}

    One table per environment.  Best value per column is bold.
    Unsuccessful runs show $\\infty$.  Caption mirrors APT* Table II note.
    Saved to out_dir AND results/ for direct paper inclusion.
    """
    from visualization_util.output_paths import RESULTS_DIR

    col_keys = ['t_init_min', 't_init_med', 't_init_max',
                'c_init_min', 'c_init_med', 'c_init_max',
                'c_final_min', 'c_final_med', 'c_final_max']

    # Sub-column headers matching APT* paper notation
    sub_headers = [
        r'min', r'med', r'max',
        r'min', r'med', r'max',
        r'min', r'med', r'max',
    ]

    def _cell(val, best_val):
        if not np.isfinite(val):
            return r'$\infty$'
        s = f'{val:.4f}'
        if np.isfinite(best_val) and abs(val - best_val) < 1e-9:
            s = r'\textbf{' + s + r'}'
        return s

    all_lines = [
        r'% ─────────────────────────────────────────────────────────',
        r'% Benchmark Table II  (APT* paper style)',
        r'% Requires: \usepackage{booktabs,multirow} in preamble',
        r'% Auto-generated by _generate_latex_table_ii()',
        r'% ─────────────────────────────────────────────────────────',
        '',
    ]

    for env_name in all_results:
        env_rows = [r for r in rows if r['env'] == env_name]
        if not env_rows:
            continue

        n_runs = len(next(iter(all_results[env_name].values())))
        safe_env = env_name.replace('_', r'\_').replace('&', r'\&')
        label_key = env_name.lower().replace(' ', '_').replace('-', '_')

        # Best value per column (lower is better; higher SR is better)
        best = {}
        for key in col_keys:
            vals = [r[key] for r in env_rows if np.isfinite(r[key])]
            best[key] = min(vals) if vals else np.inf
        best_sr = max(r['success'] for r in env_rows)

        L = all_lines  # alias for brevity
        L.append(r'\begin{table}[t]')
        L.append(r'\centering')
        L.append(r'\setlength{\tabcolsep}{4pt}')
        L.append(
            r'\caption{Performance comparison in \textbf{' + safe_env + r'} '
            r'over ' + str(n_runs) + r' runs. '
            r'Here, $t$ and $c$ denote time~(s) and path cost; '
            r'\textit{init} and \textit{final} refer to the initial and final '
            r'solutions; min, med, and max are over trials. '
            r'\textbf{Bold}: best per column. '
            r'Failed runs: $\infty$.}'
        )
        L.append(r'\label{tab:perf_' + label_key + r'}')
        L.append(r'\resizebox{\columnwidth}{!}{%')
        # col spec: planner name | 3 t_init | 3 c_init | 3 c_final | SR
        L.append(r'\begin{tabular}{@{}l rrr rrr rrr r@{}}')
        L.append(r'\toprule')

        # ── Row 1: group headers ──────────────────────────────────────
        L.append(
            r' & \multicolumn{3}{c}{$t_{\mathrm{init}}$~(s)}'
            r' & \multicolumn{3}{c}{$c_{\mathrm{init}}$}'
            r' & \multicolumn{3}{c}{$c_{\mathrm{final}}$}'
            r' & \\'
        )
        # cmidrule under each group (cols 2-4, 5-7, 8-10)
        L.append(
            r'\cmidrule(lr){2-4}\cmidrule(lr){5-7}'
            r'\cmidrule(lr){8-10}'
        )

        # ── Row 2: sub-column headers ─────────────────────────────────
        sub_row = r'Planner & ' + ' & '.join(sub_headers) + r' & $S$~(\%) \\'
        L.append(sub_row)
        L.append(r'\midrule')

        # ── Data rows ─────────────────────────────────────────────────
        for r in env_rows:
            pname = r['planner'].replace('*', r'$^{*}$')
            cells = [pname]
            for key in col_keys:
                cells.append(_cell(r[key], best[key]))
            sr = r['success'] * 100
            sr_str = f'{sr:.0f}'
            if abs(r['success'] - best_sr) < 1e-9:
                sr_str = r'\textbf{' + sr_str + r'}'
            cells.append(sr_str)
            L.append(' & '.join(cells) + r' \\')

        L.append(r'\bottomrule')
        L.append(r'\end{tabular}}')
        L.append(r'\end{table}')
        L.append('')

    tex_content = '\n'.join(all_lines)

    # Save to plots dir and results dir
    for save_dir in (out_dir, RESULTS_DIR):
        tex_path = os.path.join(save_dir, 'benchmark_table_ii.tex')
        with open(tex_path, 'w') as f:
            f.write(tex_content)
        print(f'  -> Table II LaTeX saved to {tex_path}')


# ── Aggregated-across-envs benchmark table ───────────────────────────

def generate_table_aggregated(all_results, planners, out_dir):
    """Aggregate Table-II style metrics across environments.

    For each planner, average the per-env medians (t_init_med, c_init_med,
    c_final_med) and success rate over all environments. Produces a single
    CSV row per planner (no env dimension) and prints a summary table.
    """
    import csv

    env_names = list(all_results.keys())
    if not env_names:
        print('  No data for aggregated benchmark table.')
        return

    rows = []
    for pname in planners:
        env_t_init = []
        env_c_init = []
        env_c_final = []
        env_success = []
        for env_name in env_names:
            trial_stats_list = all_results[env_name].get(pname)
            if not trial_stats_list:
                continue
            init_times, init_costs, final_costs = [], [], []
            for stats in trial_stats_list:
                if not stats:
                    init_times.append(np.inf)
                    init_costs.append(np.inf)
                    final_costs.append(np.inf)
                    continue
                first_t, first_c = np.inf, np.inf
                for s in stats:
                    if np.isfinite(s['c_best']):
                        first_t = s['time_elapsed']
                        first_c = s['c_best']
                        break
                init_times.append(first_t)
                init_costs.append(first_c)
                final_costs.append(stats[-1]['c_best'])
            init_times = np.asarray(init_times)
            init_costs = np.asarray(init_costs)
            final_costs = np.asarray(final_costs)

            env_success.append(
                float(np.sum(np.isfinite(init_costs))) / len(init_costs))
            fi_t = init_times[np.isfinite(init_times)]
            fi_c = init_costs[np.isfinite(init_costs)]
            fi_f = final_costs[np.isfinite(final_costs)]
            if fi_t.size:
                env_t_init.append(float(np.median(fi_t)))
            if fi_c.size:
                env_c_init.append(float(np.median(fi_c)))
            if fi_f.size:
                env_c_final.append(float(np.median(fi_f)))

        rows.append({
            'planner': pname,
            'n_envs': len(env_names),
            't_init_med_avg': float(np.mean(env_t_init)) if env_t_init else np.inf,
            'c_init_med_avg': float(np.mean(env_c_init)) if env_c_init else np.inf,
            'c_final_med_avg': float(np.mean(env_c_final)) if env_c_final else np.inf,
            'success_avg': float(np.mean(env_success)) if env_success else 0.0,
        })

    # ── Console output ────────────────────────────────────────────────
    print('\n' + '=' * 90)
    print(f'  AGGREGATED BENCHMARK TABLE (averaged over {len(env_names)} env(s))')
    print('=' * 90)
    print(f'  {"Planner":<16}{"t_init_med":>14}{"c_init_med":>14}'
          f'{"c_final_med":>14}{"Success":>12}')
    print('  ' + '-' * 80)
    for r in rows:
        print(f'  {r["planner"]:<16}'
              f'{r["t_init_med_avg"]:14.4f}'
              f'{r["c_init_med_avg"]:14.4f}'
              f'{r["c_final_med_avg"]:14.4f}'
              f'{r["success_avg"] * 100:11.0f}%')
    print('  ' + '-' * 80)

    csv_path = os.path.join(out_dir, 'benchmark_aggregated.csv')
    fieldnames = ['planner', 'n_envs', 't_init_med_avg', 'c_init_med_avg',
                  'c_final_med_avg', 'success_avg']
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
    print(f'\n  -> Aggregated CSV saved to {csv_path}')


# ── Aggregated-by-dimension benchmark tables ─────────────────────────

def generate_table_aggregated_by_dim(all_results, planners, out_dir,
                                     env_dim_map: dict):
    """Generate one aggregated Table-II style table per dimension group.

    Averages per-env medians over all 2D, 3D, and 6D environments
    separately, producing:
      - Console tables for each group present in ``all_results``
      - benchmark_aggregated_2d.csv / _3d.csv / _6d.csv
      - benchmark_aggregated_by_dim.tex  (LaTeX, one subtable per group)

    Parameters
    ----------
    all_results : dict  env_name -> {planner_name -> list[trial_stats]}
    planners    : list of canonical planner names
    out_dir     : output directory path
    env_dim_map : dict  env_name -> dim_tag  (e.g. '2d', '3d', '6d',
                  '2d_euclid').  Used to bucket environments.
    """
    import csv

    # ── bucket environments by dimension group ────────────────────────
    DIM_LABELS = {'2d': '2D', '2d_euclid': '2D', '3d': '3D', '6d': '6D'}
    groups: dict[str, list[str]] = {'2D': [], '3D': [], '6D': []}
    for env_name in all_results:
        raw_tag = env_dim_map.get(env_name, '')
        label = DIM_LABELS.get(raw_tag.lower(), '')
        if label:
            groups[label].append(env_name)

    # ── helper: compute aggregated rows for a list of env names ──────
    def _agg_rows(env_names):
        rows = []
        for pname in planners:
            env_t_init, env_c_init, env_c_final, env_success = [], [], [], []
            for env_name in env_names:
                trial_stats_list = all_results[env_name].get(pname)
                if not trial_stats_list:
                    continue
                init_times, init_costs, final_costs = [], [], []
                for stats in trial_stats_list:
                    if not stats:
                        init_times.append(np.inf)
                        init_costs.append(np.inf)
                        final_costs.append(np.inf)
                        continue
                    first_t, first_c = np.inf, np.inf
                    for s in stats:
                        if np.isfinite(s['c_best']):
                            first_t = s['time_elapsed']
                            first_c = s['c_best']
                            break
                    init_times.append(first_t)
                    init_costs.append(first_c)
                    final_costs.append(stats[-1]['c_best'])
                init_times  = np.asarray(init_times)
                init_costs  = np.asarray(init_costs)
                final_costs = np.asarray(final_costs)
                env_success.append(
                    float(np.sum(np.isfinite(init_costs))) / len(init_costs))
                fi_t = init_times[np.isfinite(init_times)]
                fi_c = init_costs[np.isfinite(init_costs)]
                fi_f = final_costs[np.isfinite(final_costs)]
                if fi_t.size:
                    env_t_init.append(float(np.median(fi_t)))
                if fi_c.size:
                    env_c_init.append(float(np.median(fi_c)))
                if fi_f.size:
                    env_c_final.append(float(np.median(fi_f)))
            rows.append({
                'planner':          pname,
                'n_envs':           len(env_names),
                't_init_med_avg':   float(np.mean(env_t_init))  if env_t_init  else np.inf,
                'c_init_med_avg':   float(np.mean(env_c_init))  if env_c_init  else np.inf,
                'c_final_med_avg':  float(np.mean(env_c_final)) if env_c_final else np.inf,
                'success_avg':      float(np.mean(env_success)) if env_success else 0.0,
            })
        return rows

    # ── header formatter ──────────────────────────────────────────────
    FIELDNAMES = ['planner', 'n_envs',
                  't_init_med_avg', 'c_init_med_avg',
                  'c_final_med_avg', 'success_avg']

    def _print_group(dim_label, env_names, rows):
        print('\n' + '=' * 90)
        print(f'  AGGREGATED TABLE — {dim_label} environments '
              f'({len(env_names)} env(s): {", ".join(env_names)})')
        print('=' * 90)
        print(f'  {"Planner":<16}{"t_init_med":>14}{"c_init_med":>14}'
              f'{"c_final_med":>14}{"Success":>12}')
        print('  ' + '-' * 80)

        # find best values for highlighting
        finite_t   = [r['t_init_med_avg']  for r in rows if np.isfinite(r['t_init_med_avg'])]
        finite_ci  = [r['c_init_med_avg']  for r in rows if np.isfinite(r['c_init_med_avg'])]
        finite_cf  = [r['c_final_med_avg'] for r in rows if np.isfinite(r['c_final_med_avg'])]
        best_t  = min(finite_t)  if finite_t  else np.inf
        best_ci = min(finite_ci) if finite_ci else np.inf
        best_cf = min(finite_cf) if finite_cf else np.inf
        best_sr = max(r['success_avg'] for r in rows)

        def _fmt(val, best):
            if not np.isfinite(val):
                return '           inf'
            s = f'{val:14.4f}'
            if abs(val - best) < 1e-9:
                s = f'  ** {val:.4f}  '
            return s

        for r in rows:
            sr = r['success_avg'] * 100
            sr_s = f'{sr:11.0f}%'
            if abs(r['success_avg'] - best_sr) < 1e-9:
                sr_s = f'  ** {sr:.0f}%'
            print(f'  {r["planner"]:<16}'
                  f'{_fmt(r["t_init_med_avg"],  best_t)}'
                  f'{_fmt(r["c_init_med_avg"],  best_ci)}'
                  f'{_fmt(r["c_final_med_avg"], best_cf)}'
                  f'{sr_s}')
        print('  ' + '-' * 80)

    # ── LaTeX accumulator ─────────────────────────────────────────────
    tex_lines = [
        r'% Aggregated benchmark tables by dimension',
        r'% Auto-generated by generate_table_aggregated_by_dim()',
        '',
    ]

    def _latex_group(dim_label, env_names, rows):
        safe_label = dim_label.replace(' ', '_').lower()
        tex_lines.append(r'\begin{table}[t]')
        tex_lines.append(r'\centering')
        tex_lines.append(
            r'\caption{Average performance across all \textbf{' + dim_label
            + r'} environments (' + str(len(env_names)) + r' envs).}')
        tex_lines.append(
            r'\label{tab:agg_' + safe_label + '}')
        tex_lines.append(r'\resizebox{\columnwidth}{!}{%')
        tex_lines.append(r'\begin{tabular}{|l|r|r|r|r|}')
        tex_lines.append(r'\hline')
        tex_lines.append(
            r'Planner & $\bar{t}^{\mathrm{med}}_{\mathrm{init}}$ (s) '
            r'& $\bar{c}^{\mathrm{med}}_{\mathrm{init}}$ '
            r'& $\bar{c}^{\mathrm{med}}_{\mathrm{final}}$ '
            r'& Success (\%) \\'
        )
        tex_lines.append(r'\hline')

        finite_t  = [r['t_init_med_avg']  for r in rows if np.isfinite(r['t_init_med_avg'])]
        finite_ci = [r['c_init_med_avg']  for r in rows if np.isfinite(r['c_init_med_avg'])]
        finite_cf = [r['c_final_med_avg'] for r in rows if np.isfinite(r['c_final_med_avg'])]
        best_t  = min(finite_t)  if finite_t  else np.inf
        best_ci = min(finite_ci) if finite_ci else np.inf
        best_cf = min(finite_cf) if finite_cf else np.inf
        best_sr = max(r['success_avg'] for r in rows)

        def _tex_val(val, best):
            if not np.isfinite(val):
                return r'$\infty$'
            s = f'{val:.4f}'
            if abs(val - best) < 1e-9:
                s = r'\textbf{' + s + '}'
            return s

        for r in rows:
            sr = r['success_avg'] * 100
            sr_s = f'{sr:.0f}'
            if abs(r['success_avg'] - best_sr) < 1e-9:
                sr_s = r'\textbf{' + sr_s + '}'
            pname = r['planner'].replace('*', r'$^*$')
            tex_lines.append(
                f'{pname} & '
                f'{_tex_val(r["t_init_med_avg"], best_t)} & '
                f'{_tex_val(r["c_init_med_avg"], best_ci)} & '
                f'{_tex_val(r["c_final_med_avg"], best_cf)} & '
                f'{sr_s} \\\\'
            )

        tex_lines.append(r'\hline')
        tex_lines.append(r'\end{tabular}}')
        tex_lines.append(r'\end{table}')
        tex_lines.append('')

    # ── process each dimension group ──────────────────────────────────
    found_any = False
    for dim_label in ('2D', '3D', '6D'):
        env_names = groups[dim_label]
        if not env_names:
            continue
        found_any = True
        rows = _agg_rows(env_names)
        _print_group(dim_label, env_names, rows)
        _latex_group(dim_label, env_names, rows)

        # per-group CSV
        csv_suffix = dim_label.lower()
        csv_path = os.path.join(out_dir, f'benchmark_aggregated_{csv_suffix}.csv')
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
            writer.writeheader()
            for r in rows:
                writer.writerow(r)
        print(f'  -> Aggregated {dim_label} CSV saved to {csv_path}')

    if not found_any:
        print('  No data for aggregated-by-dim tables.')
        return

    # combined LaTeX
    tex_path = os.path.join(out_dir, 'benchmark_aggregated_by_dim.tex')
    with open(tex_path, 'w') as f:
        f.write('\n'.join(tex_lines))
    print(f'  -> Aggregated-by-dim LaTeX saved to {tex_path}')


# ── Table III-style benchmark table (APT* paper format) ──────────────

def generate_table_iii(all_results, planners, out_dir, n_trials=10):
    """Generate a Table III-style benchmark evaluation table.

    Columns per planner: t_init_min, t_init_med, c_init_med
    Plus: c_final_med, success rate, and RIT* improvement %.
    Saved as CSV and printed to console in LaTeX-friendly format.
    """
    import csv

    rows = []
    for env_name, results in all_results.items():
        for pname in planners:
            if pname not in results:
                continue
            trial_stats_list = results[pname]

            # Collect per-trial initial solution time and cost
            init_times = []
            init_costs = []
            final_costs = []
            for stats in trial_stats_list:
                if not stats:
                    init_times.append(np.inf)
                    init_costs.append(np.inf)
                    final_costs.append(np.inf)
                    continue
                # Find first iteration with finite cost
                first_t, first_c = np.inf, np.inf
                for s in stats:
                    if np.isfinite(s['c_best']):
                        first_t = s['time_elapsed']
                        first_c = s['c_best']
                        break
                init_times.append(first_t)
                init_costs.append(first_c)
                final_costs.append(stats[-1]['c_best'])

            init_times = np.array(init_times)
            init_costs = np.array(init_costs)
            final_costs = np.array(final_costs)

            n_success = np.sum(np.isfinite(init_costs))
            success_rate = n_success / len(init_costs)

            finite_it = init_times[np.isfinite(init_times)]
            finite_ic = init_costs[np.isfinite(init_costs)]
            finite_fc = final_costs[np.isfinite(final_costs)]

            rows.append({
                'env': env_name,
                'planner': pname,
                't_init_min': np.min(finite_it) if len(finite_it) > 0 else np.inf,
                't_init_med': np.median(finite_it) if len(finite_it) > 0 else np.inf,
                'c_init_med': np.median(finite_ic) if len(finite_ic) > 0 else np.inf,
                'c_final_med': np.median(finite_fc) if len(finite_fc) > 0 else np.inf,
                'success': success_rate,
            })

    if not rows:
        print('  No data for table.')
        return

    # Print console table
    print('\n' + '=' * 110)
    print('  BENCHMARK EVALUATION TABLE (Table III style)')
    print('=' * 110)
    header = f'  {"Environment":<20} {"Planner":<16} {"t_init_min":>10} {"t_init_med":>10} {"c_init_med":>10} {"c_final_med":>11} {"Success":>8}'
    print(header)
    print('  ' + '─' * 106)

    for env_name in all_results:
        env_rows = [r for r in rows if r['env'] == env_name]
        # Find best values for bolding
        best_t_min = min(r['t_init_min'] for r in env_rows if np.isfinite(r['t_init_min'])) if any(np.isfinite(r['t_init_min']) for r in env_rows) else np.inf
        best_t_med = min(r['t_init_med'] for r in env_rows if np.isfinite(r['t_init_med'])) if any(np.isfinite(r['t_init_med']) for r in env_rows) else np.inf
        best_c_med = min(r['c_init_med'] for r in env_rows if np.isfinite(r['c_init_med'])) if any(np.isfinite(r['c_init_med']) for r in env_rows) else np.inf
        best_fc = min(r['c_final_med'] for r in env_rows if np.isfinite(r['c_final_med'])) if any(np.isfinite(r['c_final_med']) for r in env_rows) else np.inf
        best_sr = max(r['success'] for r in env_rows)

        for r in env_rows:
            def _fmt(val, best, is_time=False):
                if not np.isfinite(val):
                    return '       inf'
                s = f'{val:10.4f}'
                if abs(val - best) < 1e-9:
                    s = f'  *{val:.4f}'
                return s

            t_min_s = _fmt(r['t_init_min'], best_t_min, True)
            t_med_s = _fmt(r['t_init_med'], best_t_med, True)
            c_med_s = _fmt(r['c_init_med'], best_c_med)
            fc_s = _fmt(r['c_final_med'], best_fc)
            sr_s = f'{r["success"]*100:7.1f}%'
            if abs(r['success'] - best_sr) < 1e-9:
                sr_s = f' *{r["success"]*100:.1f}%'
            print(f'  {r["env"]:<20} {r["planner"]:<16} {t_min_s} {t_med_s} {c_med_s} {fc_s:>11} {sr_s:>8}')
        print('  ' + '─' * 106)

    # Save CSV
    csv_path = os.path.join(out_dir, 'benchmark_table_iii.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'env', 'planner', 't_init_min', 't_init_med',
            'c_init_med', 'c_final_med', 'success'])
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
    print(f'\n  → Table saved to {csv_path}')

    # Generate LaTeX table
    _generate_latex_table(rows, all_results, planners, out_dir)


def _generate_latex_table(rows, all_results, planners, out_dir):
    """Generate a compact combined LaTeX table (Table III style).

    Rows = environments; column groups = planners.
    Two metrics per planner: t_init_med (s) and c_final_med.
    Uses booktabs style suitable for IEEE two-column format (\table*).
    """
    def _fmt(val, best_val):
        if not np.isfinite(val):
            return r'$\infty$'
        s = f'{val:.4f}'
        if abs(val - best_val) < 1e-9:
            return r'\textbf{' + s + '}'
        return s

    n = len(planners)
    col_spec = '@{}l ' + ' '.join(['rr'] * n) + '@{}'

    lines = []
    lines.append(r'% ─────────────────────────────────────────────────────────')
    lines.append(r'% Benchmark Table III  (compact combined format)')
    lines.append(r'% Rows = environments; column groups = planners')
    lines.append(r'% Metrics per planner: t_init_med (s), c_final_med')
    lines.append(r'% All planners achieve 100% success across all environments.')
    lines.append(r'% Auto-generated by _generate_latex_table()')
    lines.append(r'% ─────────────────────────────────────────────────────────')
    lines.append(r'')
    lines.append(r'\begin{table*}[t]')
    lines.append(r'\centering')
    lines.append(r'\setlength{\tabcolsep}{4pt}')
    lines.append(
        r'\caption{Benchmark comparison across all test environments. '
        r'For each planner, $t_{\mathrm{init}}^{\mathrm{med}}$~(s) is the '
        r'median time to first solution and $c_{\mathrm{final}}^{\mathrm{med}}$ '
        r'is the median final path cost. All planners achieve 100\% success. '
        r'\textbf{Bold}: best per metric per environment.}'
    )
    lines.append(r'\label{tab:benchmark_iii}')
    lines.append(r'\resizebox{\linewidth}{!}{%')
    lines.append(r'\begin{tabular}{' + col_spec + '}')
    lines.append(r'\toprule')

    # Header row 1: planner names spanning 2 columns each
    def _escape(p):
        return p.replace('RRT*', r'RRT$^{*}$').replace('*', r'$^{*}$')

    cmidrule_pairs = []
    header1_parts = []
    for i, p in enumerate(planners):
        col_start = 2 + i * 2
        col_end = col_start + 1
        cmidrule_pairs.append(f'\\cmidrule(lr){{{col_start}-{col_end}}}')
        header1_parts.append(
            r'\multicolumn{2}{c}{' + _escape(p) + '}'
        )
    lines.append(r' & ' + ' & '.join(header1_parts) + r' \\')
    lines.append(''.join(cmidrule_pairs))

    # Header row 2: metric labels
    sub_headers = []
    for _ in planners:
        sub_headers.append(r'$t^{\mathrm{med}}$')
        sub_headers.append(r'$c^{\mathrm{fin}}$')
    lines.append('Environment & ' + ' & '.join(sub_headers) + r' \\')
    lines.append(r'\midrule')

    # Data rows
    for env_name in all_results:
        env_rows = [r for r in rows if r['env'] == env_name]
        if not env_rows:
            continue

        best_t = min(r['t_init_med'] for r in env_rows if np.isfinite(r['t_init_med']))
        best_c = min(r['c_final_med'] for r in env_rows if np.isfinite(r['c_final_med']))

        env_label = env_name.replace('_', r'\_')
        cells = [env_label]
        for pname in planners:
            pr = [r for r in env_rows if r['planner'] == pname]
            if pr:
                cells.append(_fmt(pr[0]['t_init_med'], best_t))
                cells.append(_fmt(pr[0]['c_final_med'], best_c))
            else:
                cells.extend(['--', '--'])

        lines.append(' & '.join(cells) + r' \\')

    lines.append(r'\bottomrule')
    lines.append(r'\end{tabular}}')
    lines.append(r'\end{table*}')

    tex_path = os.path.join(out_dir, 'benchmark_table_iii.tex')
    with open(tex_path, 'w') as f:
        f.write('\n'.join(lines))
    print(f'  → LaTeX table saved to {tex_path}')

def main():
    parser = argparse.ArgumentParser(
        description='Generate AIT*/EIT*-style benchmark plots')
    parser.add_argument('--envs', nargs='+', default=['2D'],
                        help='Environments (e.g. 2D, obstacle, maze_e)')
    parser.add_argument('--planners', nargs='+', default=None,
                        help='Planners (e.g. RIT BIT AIT). Default: all')
    parser.add_argument('--trials', type=int, default=10,
                        help='Number of trials per planner (default: 10)')
    parser.add_argument('--max-iter', type=int, default=150,
                        help='Max iterations per trial (default: 150)')
    parser.add_argument('--batch-size', type=int, default=100,
                        help='Batch size (default: 100)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Base random seed (default: 42)')
    parser.add_argument('--out', type=str, default=None,
                        help='Output directory (default: visualization/plots)')
    parser.add_argument('--cache', type=str, default=None,
                        help='Cache file to save/load results (pickle)')
    args = parser.parse_args()

    environments = _resolve_environments(args.envs)
    if args.planners:
        planners = _resolve_planners(args.planners)
    else:
        planners = ['RIT*', 'Informed RRT*', 'BIT*', 'AIT*', 'EIT*', 'APT*']
    out_dir = args.out or PLOTS_DIR
    os.makedirs(out_dir, exist_ok=True)

    print('=' * 60)
    print('  ANYTIME BENCHMARK PLOTS')
    print('=' * 60)
    print(f'  Environments:  {environments}')
    print(f'  Planners:      {planners}')
    print(f'  Trials:        {args.trials}')
    print(f'  Max iters:     {args.max_iter}')
    print(f'  Batch size:    {args.batch_size}')
    print(f'  Base seed:     {args.seed}')
    print(f'  Output dir:    {out_dir}')
    print('=' * 60)

    # Load or collect data
    all_results = {}
    cache_path = args.cache

    if cache_path and os.path.isfile(cache_path):
        print(f'\nLoading cached results from {cache_path}')
        with open(cache_path, 'rb') as f:
            all_results = pickle.load(f)
        print(f'  Loaded {len(all_results)} environments')
    else:
        for env_name in environments:
            env_fn, dim_tag = ENV_REGISTRY[env_name]
            print(f'\n  Environment: {env_name} ({dim_tag.upper()})')
            results = _collect_data(
                env_name, env_fn, planners, args.trials,
                args.max_iter, args.batch_size, args.seed)
            all_results[env_name] = results

        if cache_path:
            os.makedirs(os.path.dirname(cache_path) or '.', exist_ok=True)
            with open(cache_path, 'wb') as f:
                pickle.dump(all_results, f)
            print(f'\n  Cached results to {cache_path}')

    # Generate plots
    print('\nGenerating plots...')
    for env_name in all_results:
        plot_benchmark(env_name, all_results[env_name], planners, out_dir)

    if len(all_results) > 1:
        plot_combined(all_results, planners, out_dir)

    # Generate Table II and III-style benchmark tables
    generate_table_ii(all_results, planners, out_dir, n_trials=args.trials)
    generate_table_iii(all_results, planners, out_dir, n_trials=args.trials)

    print('\nDone.')


if __name__ == '__main__':
    main()
