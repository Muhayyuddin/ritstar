#!/usr/bin/env python
"""Quick validation of RIT* improvements on key problem environments."""
import sys, time
sys.stdout.reconfigure(line_buffering=True)
import numpy as np

from rit_star.rit_star import RITStar
from rit_star.baselines import InformedRRTStar, BITStar
from rit_star.environments import (
    env_2d_obstacle_inflated,
    env_2d_maze,
    env_3d_sphere_field,
)

# Also try 6D if pybullet available
try:
    from rit_star.environments import env_6d_shelf, env_6d_cluttered
    HAS_6D = True
except ImportError:
    HAS_6D = False

ENVS = {
    '2D Obstacles': env_2d_obstacle_inflated,
    '2D Maze': env_2d_maze,
    '3D Spheres': env_3d_sphere_field,
}
if HAS_6D:
    ENVS['6D Shelf'] = env_6d_shelf
    ENVS['6D Cluttered'] = env_6d_cluttered

N_TRIALS = 3
MAX_ITER = 150
BATCH = 100

print(f"Testing RIT* vs Informed RRT* on {len(ENVS)} environments, {N_TRIALS} trials each\n")

for env_name, env_fn in ENVS.items():
    print(f"=== {env_name} ===")
    coll, _, metric, xs, xg, bounds = env_fn()

    rit_costs, rit_times = [], []
    irrt_costs, irrt_times = [], []

    for trial in range(N_TRIALS):
        seed = 42 + trial

        t0 = time.time()
        rit = RITStar(xs, xg, bounds, coll, metric,
                      geodesic_tier='diagonal', batch_size=BATCH,
                      max_iterations=MAX_ITER, random_seed=seed)
        path, cost = rit.plan()
        rit_time = time.time() - t0
        rit_costs.append(cost)
        rit_times.append(rit_time)
        del rit

        t0 = time.time()
        irrt = InformedRRTStar(x_start=xs, x_goal=xg, c_space_bounds=bounds,
                               collision_checker=coll, metric=metric,
                               batch_size=BATCH, max_iterations=MAX_ITER,
                               random_seed=seed)
        irrt.plan()
        stats = irrt.get_stats()
        irrt_cost = stats[-1]['c_best'] if stats else np.inf
        irrt_time = time.time() - t0
        irrt_costs.append(irrt_cost)
        irrt_times.append(irrt_time)
        del irrt

        print(f"  Trial {trial+1}: RIT*={cost:.4f} ({rit_time:.2f}s)  "
              f"IRRT*={irrt_cost:.4f} ({irrt_time:.2f}s)")

    print(f"  Summary: RIT* {np.mean(rit_costs):.4f}±{np.std(rit_costs):.4f} "
          f"({np.mean(rit_times):.2f}s)  "
          f"IRRT* {np.mean(irrt_costs):.4f}±{np.std(irrt_costs):.4f} "
          f"({np.mean(irrt_times):.2f}s)")
    diff_pct = (np.mean(rit_costs) - np.mean(irrt_costs)) / np.mean(irrt_costs) * 100
    print(f"  RIT* vs IRRT*: {diff_pct:+.2f}%\n")
