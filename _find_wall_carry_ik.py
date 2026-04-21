#!/usr/bin/env python3
"""
Offline IK search for run_wall_carry.py.

Finds a (q_start, q_goal) pair that is:
  - collision-free in the wall environment (with 180° base yaw)
  - both top-down grasps (one on −y side of wall, one on +y mirror)
  - minimum joint-space distance → easy motion plan
Prints the chosen configs so they can be hard-coded into run_wall_carry.py.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pybullet as p

from run_wall_carry import (
    build_wall_obstacles, find_ik, WALL_X, WALL_Y, WALL_W, ROBOT_BASE_Z,
    TABLE_SURFACE_Z, GRASP_OFFSET_Z,
)
from manipulator_env.pybullet_env import UR10eRobotiqEnv

# Start = run_wall_env goal (top-down grasp of the can on −y side).
# Goal  = mirror on +y side, same height.
side_clearance = WALL_W / 2 + 0.50
can_x = WALL_X
start_y = WALL_Y - side_clearance    # −0.591 (matches run_wall_env can_y)
goal_y  = WALL_Y + side_clearance    # +0.591
can_z = TABLE_SURFACE_Z + 0.055
grasp_z = can_z + GRASP_OFFSET_Z     # 0.970 — same as run_wall_env goal_target z

start_target = [can_x, start_y, grasp_z]
goal_target  = [can_x, goal_y,  grasp_z]

print(f"[CFG] start_target = {[round(v, 3) for v in start_target]}")
print(f"[CFG] goal_target  = {[round(v, 3) for v in goal_target]}")

env = UR10eRobotiqEnv(
    gui=False,
    obstacles=build_wall_obstacles(),
    base_position=[0.0, 0.0, ROBOT_BASE_Z],
    base_orientation=p.getQuaternionFromEuler([0, 0, np.pi]),
)

# Collect multiple collision-free IK candidates for start and goal by perturbing
# the seed search (the existing find_ik returns its best by orn/pos, we want
# options to minimise joint-space distance between start and goal).

def _collect_candidates(env, target, preferred_seeds):
    """Run find_ik with different preferred seeds to harvest distinct branches."""
    results = []
    for seed in preferred_seeds:
        q = find_ik(env, target, side_label="", n_random=50,
                    preferred_seed=np.array(seed))
        if q is not None and env.is_collision_free(q):
            results.append(np.asarray(q))
    # Deduplicate by joint proximity
    unique = []
    for q in results:
        if all(np.linalg.norm(q - u) > 0.1 for u in unique):
            unique.append(q)
    return unique

seeds = [
    [0.0, -1.2, 2.0, -2.4, -1.57, 0.0],
    [ np.pi/2, -1.2, 2.0, -2.4, -1.57, 0.0],
    [-np.pi/2, -1.2, 2.0, -2.4, -1.57, 0.0],
    [ np.pi,   -1.2, 2.0, -2.4, -1.57, 0.0],
    [0.0, -0.8, 1.5, -2.0, -1.57, 0.0],
    [ np.pi/4, -1.5, 2.37, -2.44, -1.57, -4.57],
    [-np.pi/4, -1.5, 2.37, -2.44, -1.57, -4.57],
]

print("\n[SEARCH] Harvesting start IK candidates ...")
start_cands = _collect_candidates(env, start_target, seeds)
print(f"  → {len(start_cands)} distinct start candidates")
print("\n[SEARCH] Harvesting goal IK candidates ...")
goal_cands = _collect_candidates(env, goal_target, seeds)
print(f"  → {len(goal_cands)} distinct goal candidates")

def _wrap(q):
    return (q + np.pi) % (2 * np.pi) - np.pi

def _jdist(a, b):
    return float(np.linalg.norm(_wrap(a - b)))

best = None
for qs in start_cands:
    for qg in goal_cands:
        d = _jdist(qs, qg)
        if best is None or d < best[0]:
            best = (d, qs, qg)

if best is None:
    print("\n[FAIL] No collision-free (start, goal) pair found.")
    env.disconnect()
    sys.exit(1)

d, qs, qg = best
print(f"\n[BEST] Joint-space distance Δq = {d:.4f} rad")
print(f"       q_start = {np.array2string(qs, precision=6, suppress_small=True)}")
print(f"       q_goal  = {np.array2string(qg, precision=6, suppress_small=True)}")
print()
print("--- copy–paste into run_wall_carry.py ---")
print(f"    q_start = np.array([{', '.join(f'{v:.6f}' for v in qs)}])")
print(f"    q_goal  = np.array([{', '.join(f'{v:.6f}' for v in qg)}])")

env.disconnect()
