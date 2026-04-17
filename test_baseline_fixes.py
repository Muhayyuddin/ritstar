#!/usr/bin/env python3
"""Quick validation that all baseline fixes work and EIT* is functional."""

import sys
import time
import numpy as np

# Add project root
sys.path.insert(0, '.')

from rit_star.environments import (
    env_2d_diagonal_anisotropic,
    env_2d_obstacle_inflated,
    env_2d_narrow_passage,
    env_2d_maze,
)
from rit_star.baselines import BITStar, AITStar, EITStar

ENVS = {
    '2D Diagonal': env_2d_diagonal_anisotropic,
    '2D Obstacles': env_2d_obstacle_inflated,
    '2D Narrow': env_2d_narrow_passage,
    '2D Maze': env_2d_maze,
}

PLANNERS = {
    'BIT*': BITStar,
    'AIT*': AITStar,
    'EIT*': EITStar,
}

print("=" * 70)
print("  BASELINE FIX VALIDATION")
print("=" * 70)

results = {}
for env_name, env_fn in ENVS.items():
    coll, _, metric, xs, xg, bounds = env_fn()
    for pname, pcls in PLANNERS.items():
        t0 = time.time()
        planner = pcls(
            x_start=xs, x_goal=xg, c_space_bounds=bounds,
            collision_checker=coll, metric=metric,
            batch_size=100, max_iterations=30, random_seed=42,
        )
        path, cost = planner.plan()
        elapsed = time.time() - t0
        success = cost < np.inf
        status = f"cost={cost:.4f}" if success else "FAILED"
        print(f"  {env_name:15s} | {pname:8s} | {status:20s} | {elapsed:.1f}s")
        results[(env_name, pname)] = (success, cost, elapsed)

print("\n" + "=" * 70)
print("  SUMMARY")
print("=" * 70)

# Check BIT* fixes (was 0% on Obstacles, Maze, Bug Trap, Forest, Terrain)
bit_success = sum(1 for (e,p),(s,c,t) in results.items() if p == 'BIT*' and s)
print(f"BIT*:  {bit_success}/{len(ENVS)} environments solved (was 0-50% before)")

# Check ABIT* timing (was 91-767s on 2D)
abit_times = [t for (e,p),(s,c,t) in results.items() if p == 'ABIT*']
max_abit = max(abit_times)
print(f"ABIT*: max time = {max_abit:.1f}s (was 91-767s before at 150 iterations)")

# Check AIT* cost quality
ait_costs = {e: c for (e,p),(s,c,t) in results.items() if p == 'AIT*' and s}
print(f"AIT*:  solved {len(ait_costs)}/{len(ENVS)}, costs: {ait_costs}")

# Check EIT* works
eit_success = sum(1 for (e,p),(s,c,t) in results.items() if p == 'EIT*' and s)
print(f"EIT*:  {eit_success}/{len(ENVS)} environments solved (new baseline)")

all_ok = bit_success >= 3 and max_abit < 60 and eit_success >= 3
print(f"\nOverall: {'PASS' if all_ok else 'NEEDS INVESTIGATION'}")
