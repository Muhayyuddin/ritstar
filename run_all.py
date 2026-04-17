#!/usr/bin/env python3
"""Master run script: run all analysis, save CSVs, PNGs, and GIFs."""
import sys
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

import gc
import os
import subprocess
import traceback
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from output_paths import RESULTS_DIR, IMAGES_DIR, GIFS_DIR, PLOTS_DIR


def run_experiments_1_4():
    """Experiments 1-4: convergence, volume ratio, anisotropy, 3D benchmark."""
    from rit_star.experiments import (
        experiment_1_convergence_comparison,
        experiment_2_volume_ratio,
        experiment_3_scaling_with_anisotropy,
        experiment_4_3d_full,
    )
    print('\n' + '=' * 60)
    print('  EXPERIMENTS 1–4')
    print('=' * 60)

    for name, fn, kwargs in [
        ('Exp 1', experiment_1_convergence_comparison, dict(n_trials=1, max_iterations=150, random_seed=0)),
        ('Exp 2', experiment_2_volume_ratio, dict(n_mc=3000, random_seed=42)),
        ('Exp 3', experiment_3_scaling_with_anisotropy, dict(n_trials=1, random_seed=0)),
        ('Exp 4', experiment_4_3d_full, dict(n_trials=1, max_iterations=200, random_seed=0)),
    ]:
        try:
            print(f'\n--- {name} ---')
            fig = fn(**kwargs)
            plt.close(fig)
            gc.collect()
            print(f'  {name} DONE')
        except Exception:
            traceback.print_exc()
            print(f'  {name} FAILED')


def run_experiments_5_7():
    """Theory experiments 5-7: volume ratio, AO, convergence separation."""
    from rit_star.experiments import (
        experiment_volume_ratio_validation,
        experiment_ao_validation,
        experiment_convergence_rate_separation,
    )
    print('\n' + '=' * 60)
    print('  THEORY EXPERIMENTS 5–7')
    print('=' * 60)

    for name, fn, kwargs in [
        ('Exp 5 (Thm 1)', experiment_volume_ratio_validation,
         dict(kappa_values=[1.0, 2.0, 4.0, 8.0, 16.0], dims=[2, 3], n_mc=5000, random_seed=42)),
        ('Exp 6 (Thm 2)', experiment_ao_validation,
         dict(n_samples_list=[100, 500, 1000, 3000, 5000], n_trials=1, random_seed=0)),
        ('Exp 7 (Thm 3)', experiment_convergence_rate_separation,
         dict(kappa_values=[1.0, 2.0, 4.0, 8.0, 16.0], dims=[2, 3], n_trials=1, max_iterations=150, random_seed=0)),
    ]:
        try:
            print(f'\n--- {name} ---')
            fig = fn(**kwargs)
            plt.close(fig)
            gc.collect()
            print(f'  {name} DONE')
        except Exception:
            traceback.print_exc()
            print(f'  {name} FAILED')


def run_mc_comparison():
    """Monte Carlo comparison — CSVs + plots."""
    from rit_star.comparison import run_full_comparison
    print('\n' + '=' * 60)
    print('  MONTE CARLO COMPARISON (CSVs + plots)')
    print('=' * 60)

    try:
        run_full_comparison(
            n_trials=2,
            max_iterations=150,
            batch_size=100,
            base_seed=42,
            visualize=True,
        )
        gc.collect()
        print('  MC comparison DONE')
    except Exception:
        traceback.print_exc()
        print('  MC comparison FAILED')


def run_visualizations():
    """Riemannian field + path comparison visualizations."""
    print('\n' + '=' * 60)
    print('  VISUALIZATION SCRIPTS')
    print('=' * 60)

    scripts = [
        ('Riemannian vis', 'visualize_riemannian.py'),
        ('Path vis', 'visualize_paths.py'),
        ('Extra vis', 'visualize_extra.py'),
        ('Theory vis', 'visualize_theory.py'),
    ]

    for label, script in scripts:
        script_path = os.path.join(os.path.dirname(__file__), script)
        if not os.path.exists(script_path):
            print(f'\n--- {label} --- SKIPPED (not found: {script})')
            continue
        try:
            print(f'\n--- {label} ({script}) ---')
            result = subprocess.run(
                [sys.executable, script_path],
                capture_output=True, text=True, timeout=600,
            )
            if result.stdout:
                for line in result.stdout.strip().split('\n')[-10:]:
                    print(f'    {line}')
            if result.returncode != 0:
                print(f'  STDERR: {result.stderr[-500:] if result.stderr else ""}')
                print(f'  {label} FAILED (exit code {result.returncode})')
            else:
                print(f'  {label} DONE')
        except subprocess.TimeoutExpired:
            print(f'  {label} TIMEOUT (600s)')
        except Exception:
            traceback.print_exc()
            print(f'  {label} FAILED')


def run_pybullet_gifs():
    """PyBullet final-path GIFs (not sampling)."""
    print('\n' + '=' * 60)
    print('  PYBULLET FINAL-PATH GIFs')
    print('=' * 60)

    try:
        import run_pybullet_gif
        run_pybullet_gif.main()
        gc.collect()
        print('  PyBullet GIFs DONE')
    except Exception:
        traceback.print_exc()
        print('  PyBullet GIFs FAILED')


def main():
    print('=' * 60)
    print('  RIT* FULL ANALYSIS PIPELINE')
    print('  Output: CSVs → results/')
    print('          PNGs → visualization/plots/ & images/')
    print('          GIFs → visualization/gifs/')
    print('=' * 60)

    run_experiments_1_4()
    run_experiments_5_7()
    run_mc_comparison()
    run_visualizations()
    run_pybullet_gifs()

    print('\n' + '=' * 60)
    print('  ALL DONE!')
    print('=' * 60)

    # List generated files
    for label, d in [('CSVs', RESULTS_DIR), ('Plots', PLOTS_DIR),
                     ('Images', IMAGES_DIR), ('GIFs', GIFS_DIR)]:
        files = sorted(os.listdir(d)) if os.path.isdir(d) else []
        if files:
            print(f'\n  {label} ({d}):')
            for f in files:
                size = os.path.getsize(os.path.join(d, f))
                print(f'    {f}  ({size / 1024:.0f} KB)')


if __name__ == "__main__":
    main()
