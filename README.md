# RIT* — Riemannian Informed Trees

Supplementary code for the IEEE RA-L submission:
**"RIT*: Asymptotically Optimal Motion Planning with Riemannian Informed Sampling"**

RIT* extends BIT* with a Riemannian metric field that encodes obstacle proximity, yielding shorter, smoother paths via anisotropic informed-set sampling, L1/L2 cascading edge filters, and optional collision-adaptive metric (CARM) refinement.

---

## Setup

### 1 — Create a virtual environment

```bash
python3 -m venv .venv
```

> **Note:** On Ubuntu/Debian you may need to install the `venv` package first:
> ```bash
> sudo apt install python3-venv python3-full
> ```

### 2 — Activate the virtual environment

```bash
# Linux / macOS
source .venv/bin/activate

# Windows (PowerShell)
.venv\Scripts\Activate.ps1
```

### 3 — Install dependencies

```bash
pip install -r requirements.txt
```

All required packages (`numpy`, `scipy`, `matplotlib`, `pyyaml`, `numba`, `pybullet`) are listed in [requirements.txt](requirements.txt).

---

## Quick Start

### Run the paper benchmark (Table II)

```bash
python run_from_config.py
```

This reads `config/run_config.yaml` and runs all seven planners
(RIT\*, GA-RRT\*, BIT\*, Informed RRT\*, AIT\*, EIT\*, APT\*) on the four paper environments.
Results and plots are written to `results/` and `visualization/`.

### Run the ablation study (Table III)

```bash
python run_ablation.py
```

Runs five RIT\* variants (full, w/o Riemannian sampling, w/o cascading,
w/o CARM, w/o smoothing) × 4 environments × 10 trials.
Outputs:

- `results/ablation.csv` — raw per-trial data
- `results/ablation_summary.csv` — mean ± std per variant/environment
- `results/ablation_table.tex` — paste-ready LaTeX table

### Run a single environment interactively

```bash
python run_2d_obstacle_demo.py          # animated 2-D demo
python run_manipulator.py               # UR10 pick-place (PyBullet GUI)
```

---

## Repository Structure

```
run_from_config.py      # config-driven benchmark runner (main entry point)
run_ablation.py         # ablation study (Table III)
run_all.py              # run benchmark + ablation + plots in sequence
run_benchmark_plots.py  # generate Fig. 6 success-rate / cost plots
run_carm_ral.py         # CARM overhead measurement (Table II footnote)
run_2d_obstacle_demo.py # animated 2-D planning demo
run_abstract_paths.py   # paths for paper graphical abstract
run_6d_analysis.py      # UR10 6-D analysis
run_full_analysis.py    # full multi-environment analysis
manipulation/
  run_manipulator.py          # UR10 PyBullet GUI demo
  run_pybullet_gif.py         # record PyBullet GIF
  UR10_grasp_can.py           # UR10 grasp demo (soup can)
  UR10_grasp_cube.py          # UR10 grasp demo (cube)
  UR10_pick_place_can.py      # UR10 pick-and-place (can)
  UR10_pick_place_cube.py     # UR10 pick-and-place (cube)
  UR10_pick_place_drill.py    # UR10 pick-and-place (drill)
  UR10_pick_place_shelf.py    # UR10 pick-and-place (shelf)
  UR10_pick_place_soup_box.py # UR10 pick-and-place (soup box)
  UR10_pick_shelf.py          # UR10 shelf grasp
  Tiago_pro_dual_grasp_box.py # Tiago dual-arm grasp
  tiago_pro.py                # Tiago single-arm demo
  visualize_tiago_simple_env.py # Tiago environment visualiser

config/
  run_config.yaml       # benchmark configuration (planners, envs, params)

rit_star/
  rit_star.py           # RIT* planner (core algorithm)
  metric.py             # Riemannian metric and CARM definitions
  informed_set.py       # whitened ellipsoidal informed set
  geodesic.py           # geodesic distance approximations
  metric_cache.py       # L1/L2 cascading metric cache
  environments.py       # 2-D and 3-D obstacle environments
  ur10_envs.py          # UR10 6-D C-space environments
  baselines.py          # BIT*, AIT*, EIT*, APT*, Informed RRT*
  comparison.py         # multi-planner comparison utilities
  experiments.py        # experiment runners
  visualize.py          # planner visualisation helpers

manipulator_env/
  pybullet_env.py       # PyBullet UR10 / Tiago environment wrapper
  planner_interface.py  # high-level planner interface for PyBullet

visualization_util/     # figure generation scripts for paper figures

results/                # CSV / LaTeX outputs (written at runtime)
visualization/          # plots and GIFs (written at runtime)

ycb_objects/            # YCB object meshes (for PyBullet scenes)
```

---

## Reproducing Paper Results

### Table II — Benchmark comparison

1. Set `config/run_config.yaml`:
   - `environments: ['2D Random World', '3D Diagonal', 'UR10_pick_place_drill', 'Tiago 14D simple']`
   - `n_trials: 10`, `max_iterations: 200`, `batch_size: 100`
2. Run: `python run_from_config.py`


### Table III — Ablation study

Run: `python run_ablation.py`

The script uses the same four environments and seeds as the paper.

---

## Key Planner Parameters

| Parameter | Description | Default |
|---|---|---|
| `max_iterations` | Number of BIT\*-style batches | 200 |
| `batch_size` | Samples added per batch | 100 |
| `adaptive_metric` | Enable CARM (collision-adaptive metric) | `True` |
| `geodesic_tier` | Geodesic approximation: `'diagonal'` or `'full'` | `'diagonal'` |
| `random_seed` | RNG seed for reproducibility | `42` |

Example usage:

```python
from rit_star.rit_star import RITStar
from rit_star.environments import env_2d_random_world

coll, _, metric, x_start, x_goal, bounds = env_2d_random_world()
planner = RITStar(
    x_start=x_start, x_goal=x_goal,
    c_space_bounds=bounds,
    collision_checker=coll,
    metric=metric,
    max_iterations=200,
    batch_size=100,
    random_seed=42,
)
path, stats = planner.plan()
```
