#!/usr/bin/env python3
"""run_ablation.py — Ablation study for the RIT* paper Table~\\ref{tab:ablation}.

Runs five variants of RIT* on four environments (2D Maze, 3D Spheres,
6D Shelf, 6D Cluttered) and records mean final path cost.

Variants
--------
  full           : all features enabled
  no_riem_samp   : whitened / anisotropic sampling disabled
  no_cascading   : L1/L2 edge-cost filters bypassed (always full cost)
  no_carm        : CARM (adaptive metric) disabled
  no_smoothing   : post-processing shortcut smoothing disabled

Outputs
-------
  results/ablation.csv            — one row per (variant, env, trial)
  results/ablation_summary.csv    — means ± std per (variant, env)
  results/ablation_table.tex      — LaTeX fragment matching the paper table
"""
from __future__ import annotations

import csv
import os
import sys
import time
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from rit_star.rit_star import RITStar


# ── Variant definitions ────────────────────────────────────────────────
# (variant_id, display_label, kwargs-passed-to-build)
VARIANTS = [
    ('full',         'RIT* (full)',       dict(adaptive=True,  whitening=True,  smoothing=True,  cascading=True)),
    ('no_riem_samp', 'w/o Riem.\\ samp.', dict(adaptive=True,  whitening=False, smoothing=True,  cascading=True)),
    ('no_cascading', 'w/o cascading',     dict(adaptive=True,  whitening=True,  smoothing=True,  cascading=False)),
    ('no_carm',      'w/o CARM',          dict(adaptive=False, whitening=True,  smoothing=True,  cascading=True)),
    ('no_smoothing', 'w/o smoothing',     dict(adaptive=True,  whitening=True,  smoothing=False, cascading=True)),
]

# (env_id, display_label) — env_id must match ENV_REGISTRY in run_from_config
DEFAULT_ENVS = [
    ('2D Maze',            '2D Maze'),
    ('3D Spheres',         '3D Sph.'),
    ('6D Shelf',           '6D Shelf'),
    ('6D Cluttered',       '6D Clut.'),
    ('Tiago 14D simple',   '14D Tiago'),
]


# ── Variant application (monkey-patches a constructed RITStar) ────────

def _apply_variant(planner: RITStar, variant: dict) -> None:
    """Apply variant-specific modifications to a freshly-built planner."""
    if not variant['whitening']:
        # Disable anisotropic whitened-ellipsoid sampling. The planner
        # falls back to (Euclidean) informed-ellipsoid sampling, so this
        # isolates the contribution of Riemannian-aware sampling.
        planner._use_whitening = False
        planner._whitened_eis = None

    if not variant['cascading']:
        # Force every edge through the full metric evaluation by making
        # the cached L1/L2 upper bounds always 0 (so they never reject).
        planner._mc._no_cascading = True

    if not variant['smoothing']:
        # Skip the post-processing shortcut pass.
        planner._shortcut_path = lambda path, n_attempts=0: path


# ── Ablation runner ───────────────────────────────────────────────────

def _run_one_trial(env_name: str, env_fn, variant: dict,
                   trial: int, seed: int,
                   batch_size: int, max_iterations: int) -> dict:
    coll, _, metric, xs, xg, bounds = env_fn()
    planner = RITStar(
        x_start=xs, x_goal=xg, c_space_bounds=bounds,
        collision_checker=coll, metric=metric,
        geodesic_tier='diagonal',
        batch_size=batch_size, max_iterations=max_iterations,
        random_seed=seed,
        adaptive_metric=variant['adaptive'],
    )
    _apply_variant(planner, variant)

    t0 = time.time()
    path, cost = planner.plan()
    elapsed = time.time() - t0

    return {
        'env': env_name,
        'trial': trial,
        'seed': seed,
        'final_cost': float(cost) if np.isfinite(cost) else float('inf'),
        'time_s': float(elapsed),
        'path_len': int(len(path)) if path else 0,
    }


def run_ablation(cfg: dict) -> None:
    """Execute the ablation study and write CSV + LaTeX outputs."""
    from visualization_util.output_paths import RESULTS_DIR
    # Import ENV_REGISTRY lazily to avoid a circular import at module load
    from run_from_config import ENV_REGISTRY

    n_trials = int(cfg.get('ablation_n_trials', 10))
    max_iter = int(cfg.get('ablation_max_iterations', 150))
    batch_sz = int(cfg.get('ablation_batch_size', 100))
    base_seed = int(cfg.get('ablation_base_seed', 42))

    requested_envs = cfg.get('ablation_envs') or [e[0] for e in DEFAULT_ENVS]
    env_labels = dict(DEFAULT_ENVS)
    envs = []
    for name in requested_envs:
        if name not in ENV_REGISTRY:
            print(f'  [WARN] Unknown ablation env "{name}" — skipping.')
            continue
        envs.append((name, env_labels.get(name, name), ENV_REGISTRY[name][0]))

    if not envs:
        print('  [FATAL] No valid environments for ablation.')
        return

    print('\n' + '=' * 60)
    print('  ABLATION STUDY')
    print('=' * 60)
    print(f'  Variants:   {[v[0] for v in VARIANTS]}')
    print(f'  Envs:       {[e[0] for e in envs]}')
    print(f'  Trials:     {n_trials}')
    print(f'  Max iters:  {max_iter}')
    print(f'  Batch size: {batch_sz}')
    print(f'  Base seed:  {base_seed}')
    print('=' * 60)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    rows: list[dict] = []

    for v_id, v_label, v_kwargs in VARIANTS:
        for env_name, _env_lbl, env_fn in envs:
            print(f'\n  [{v_id}] on {env_name}')
            for t in range(n_trials):
                seed = base_seed + t
                print(f'    Trial {t+1}/{n_trials} ...', end=' ', flush=True)
                try:
                    r = _run_one_trial(env_name, env_fn, v_kwargs,
                                       trial=t, seed=seed,
                                       batch_size=batch_sz,
                                       max_iterations=max_iter)
                except Exception as exc:
                    print(f'FAILED: {exc}')
                    r = {'env': env_name, 'trial': t, 'seed': seed,
                         'final_cost': float('inf'), 'time_s': 0.0,
                         'path_len': 0}
                r['variant'] = v_id
                rows.append(r)
                print(f'cost={r["final_cost"]:.4f}  time={r["time_s"]:.2f}s')

    # ── Per-trial CSV ────────────────────────────────────────────────
    out_csv = os.path.join(RESULTS_DIR, 'ablation.csv')
    with open(out_csv, 'w', newline='') as f:
        w = csv.DictWriter(f,
            fieldnames=['variant', 'env', 'trial', 'seed',
                        'final_cost', 'time_s', 'path_len'])
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f'\n  -> {out_csv}')

    # ── Summary (mean ± std per variant × env) ───────────────────────
    summary = {}  # (variant, env) -> (mean_cost, std_cost, mean_time, n_ok)
    for v_id, _, _ in VARIANTS:
        for env_name, _env_lbl, _ in envs:
            ce = [r['final_cost'] for r in rows
                  if r['variant'] == v_id and r['env'] == env_name
                  and np.isfinite(r['final_cost'])]
            te = [r['time_s'] for r in rows
                  if r['variant'] == v_id and r['env'] == env_name]
            if ce:
                summary[(v_id, env_name)] = (
                    float(np.mean(ce)), float(np.std(ce)),
                    float(np.mean(te)), len(ce))
            else:
                summary[(v_id, env_name)] = (float('inf'), 0.0, 0.0, 0)

    sum_csv = os.path.join(RESULTS_DIR, 'ablation_summary.csv')
    with open(sum_csv, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['variant', 'env',
                    'cost_mean', 'cost_std', 'time_mean_s', 'n_success'])
        for v_id, _, _ in VARIANTS:
            for env_name, _env_lbl, _ in envs:
                m, s, tm, n = summary[(v_id, env_name)]
                w.writerow([v_id, env_name,
                            f'{m:.6f}', f'{s:.6f}', f'{tm:.4f}', n])
    print(f'  -> {sum_csv}')

    # ── LaTeX table (matches paper Table~\ref{tab:ablation}) ─────────
    tex_path = os.path.join(RESULTS_DIR, 'ablation_table.tex')
    with open(tex_path, 'w') as f:
        f.write('% Auto-generated by run_ablation.py — do not edit by hand.\n')
        f.write('\\begin{table}[t]\n')
        f.write('\\centering\n')
        f.write('\\caption{Ablation study: mean path cost ($\\pm$ std) per variant '
                + f'over {n_trials} trials, {max_iter} iterations each.'
                + '}\n')
        f.write('\\label{tab:ablation}\n')
        col_spec = 'l ' + 'c' * len(envs)
        f.write(f'\\begin{{tabular}}{{@{{}}{col_spec}@{{}}}}\n')
        f.write('\\toprule\n')
        header = ['\\textbf{Variant}'] + [f'\\textbf{{{lbl}}}' for _, lbl, _ in envs]
        f.write(' & '.join(header) + ' \\\\\n')
        f.write('\\midrule\n')
        for v_id, v_label, _ in VARIANTS:
            cells = [v_label]
            for env_name, _env_lbl, _ in envs:
                m, s, _, n = summary[(v_id, env_name)]
                if n == 0 or not np.isfinite(m):
                    cells.append('---')
                else:
                    cells.append(f'{m:.3f} $\\pm$ {s:.3f}')
            f.write(' & '.join(cells) + ' \\\\\n')
        f.write('\\bottomrule\n')
        f.write('\\end{tabular}\n')
        f.write('\\end{table}\n')
    print(f'  -> {tex_path}')

    # Console pretty-print
    print('\n  ABLATION RESULTS (mean cost ± std)')
    print('  ' + '-' * 68)
    head = f'  {"Variant":<18}' + ''.join(f' {lbl:>12}' for _, lbl, _ in envs)
    print(head)
    print('  ' + '-' * 68)
    for v_id, v_label, _ in VARIANTS:
        line = f'  {v_id:<18}'
        for env_name, _env_lbl, _ in envs:
            m, s, _, n = summary[(v_id, env_name)]
            cell = '---' if n == 0 else f'{m:.3f}±{s:.3f}'
            line += f' {cell:>12}'
        print(line)
    print('  ' + '-' * 68)


if __name__ == '__main__':
    # Stand-alone entry: read flags from YAML or use defaults.
    import yaml
    cfg_path = sys.argv[1] if len(sys.argv) > 1 else 'config/run_config.yaml'
    if os.path.isfile(cfg_path):
        with open(cfg_path) as fp:
            cfg = yaml.safe_load(fp) or {}
    else:
        cfg = {}
    run_ablation(cfg)
