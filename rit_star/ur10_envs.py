"""ur10_envs.py — wrap the UR10_* PyBullet demos as 6-D EnvTuples.

So the benchmark pipeline (2-D / 3-D style) can run any planner on them
just by listing the demo name in ``environments:``.

Each factory:
  * builds a headless PyBullet scene (obstacles + robot at 180° base yaw),
  * computes q_start / q_goal exactly the way the demo script does
    (import-level helpers; no ``main()`` execution),
  * returns the standard 6-tuple
    ``(collision_checker, edge_cost, metric, x_start, x_goal, bounds)``.

The PyBullet env is retained as an attribute on the returned collision
checker so it is not garbage-collected while the benchmark is running.
"""
from __future__ import annotations

import os
import sys
import numpy as np
import pybullet as p

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from manipulator_env.pybullet_env import UR10eRobotiqEnv
from manipulator_env.planner_interface import ManipulatorInertiaMetric
from rit_star.metric import DiagonalAnisotropicMetric

_BASE_ORN = p.getQuaternionFromEuler([0, 0, np.pi])

# Fast inertia-based diagonal metric — identical to the one the demo scripts
# use in their own main() (UR10_grasp_can.py / UR10_pick_place_*.py).
# Per-eval cost is a constant matrix-vector op, so the planner's 6-D
# metric-field cache (~260 k cells) builds in <1 s instead of ~minutes.
_UR10E_INERTIAS = np.array([7.778, 12.93, 3.87, 1.96, 1.96, 0.202])
_UR10E_WEIGHTS = (_UR10E_INERTIAS / _UR10E_INERTIAS.max()).tolist()


def _finalise(env: UR10eRobotiqEnv, q_start, q_goal, metric=None):
    if metric is None:
        metric = DiagonalAnisotropicMetric(weights=_UR10E_WEIGHTS)
    coll = env.is_collision_free   # bound method keeps env alive via __self__
    bounds = env.get_bounds()
    return coll, None, metric, np.asarray(q_start), np.asarray(q_goal), bounds


# ── UR10_grasp_can — top-down grasp of can next to kraft box wall ────

def env_ur10_grasp_can():
    """6-D UR10e: grasp a tomato-soup can next to a kraft-box wall."""
    import UR10_grasp_can as demo

    env = UR10eRobotiqEnv(
        gui=False, obstacles=demo.build_wall_obstacles(),
        base_position=[0.0, 0.0, demo.ROBOT_BASE_Z],
        base_orientation=_BASE_ORN,
    )

    # Start: canonical UR10e home config (matches demo's REAL_SETUP_Q_START).
    q_start = np.asarray(demo.REAL_SETUP_Q_START, dtype=float)

    # Goal: top-down grasp at the can — same targets the demo's main() uses.
    side_clearance = demo.WALL_W / 2 + 0.50
    can_x = demo.WALL_X
    can_y = demo.WALL_Y - side_clearance - 0.10
    can_z = demo.TABLE_SURFACE_Z + 0.055
    goal_target = [can_x, can_y, can_z + demo.GRASP_OFFSET_Z + 0.01]

    q_goal = demo.find_ik(env, goal_target, side_label="GOAL",
                          desired_orn=demo.TOP_DOWN_ORN, pos_tol=0.02)
    if q_goal is None:
        raise RuntimeError('UR10_grasp_can: goal IK failed')

    return _finalise(env, q_start, q_goal)


# ── UR10_pick_place_can — carry can over the wall ───────────────────
# Uses the same hard-coded (q_start, q_goal) pair that was generated offline
# with _find_wall_carry_ik.py and copied into the demo.

_UR10_PICK_PLACE_CAN_Q_START = np.array([
    0.664284, -0.985480, 1.677681,
    -2.263020, -1.570794, -2.477311,
])
_UR10_PICK_PLACE_CAN_Q_GOAL = np.array([
    -0.912412, -1.161520, 1.977143,
    -2.386444, -1.570793, 2.229174,
])


def env_ur10_pick_place_can():
    """6-D UR10e: pick-and-place can across the wall (fixed start/goal)."""
    import UR10_pick_place_can as demo

    env = UR10eRobotiqEnv(
        gui=False, obstacles=demo.build_wall_obstacles(),
        base_position=[0.0, 0.0, demo.ROBOT_BASE_Z],
        base_orientation=_BASE_ORN,
    )
    return _finalise(env,
                     _UR10_PICK_PLACE_CAN_Q_START,
                     _UR10_PICK_PLACE_CAN_Q_GOAL)


# ── UR10_pick_shelf — side-grasp mustard bottle on shelf ────────────

def env_ur10_pick_shelf():
    """6-D UR10e: side-grasp a mustard bottle from a 2-compartment shelf."""
    import UR10_pick_shelf as demo

    env = UR10eRobotiqEnv(
        gui=False, obstacles=demo.build_shelf_obstacles(),
        base_position=[0.0, 0.0, demo.ROBOT_BASE_Z],
        base_orientation=_BASE_ORN,
    )

    # Start: hard-coded home (matches demo line ~325).
    q_start = np.array([np.pi / 2, -np.pi / 2, np.pi / 2,
                        -np.pi / 2, -np.pi / 2, 0.0])

    # Bottle in upper compartment (same calc as demo).
    upper_floor_z = demo.SHELF_Z + demo.SHELF_H / 2 + demo.SHELF_T / 2
    bottle_pos = [demo.SHELF_X, demo.SHELF_Y, upper_floor_z + 0.08]
    q_goal, _grasp_pos = demo.compute_side_grasp_ik(env, bottle_pos)
    if q_goal is None:
        raise RuntimeError('UR10_pick_shelf: side-grasp IK failed')

    return _finalise(env, q_start, q_goal)


# ── UR10_pick_place_shelf — place bottle on top of shelf (Phase 2) ──

def env_ur10_pick_place_shelf():
    """6-D UR10e: starting from grasped bottle, place it on top of shelf.

    Builds a SINGLE headless PyBullet env and computes both Phase 1's
    side-grasp (used as ``q_start``) and Phase 2's place-on-shelf-top goal
    (``q_goal``) on it. Using only one physics client avoids the double-
    free that can occur when multiple headless PyBullet clients are alive
    simultaneously.
    """
    import UR10_pick_shelf as pick
    import UR10_pick_place_shelf as demo

    env = UR10eRobotiqEnv(
        gui=False, obstacles=demo.build_shelf_obstacles(),
        base_position=[0.0, 0.0, demo.ROBOT_BASE_Z],
        base_orientation=_BASE_ORN,
    )

    # Phase 1: compute the side-grasp config the same way UR10_pick_shelf does.
    upper_floor_z = demo.SHELF_Z + demo.SHELF_H / 2 + demo.SHELF_T / 2
    bottle_pos = [demo.SHELF_X, demo.SHELF_Y, upper_floor_z + 0.08]
    q_grasp_pose, _grasp_pt = pick.compute_side_grasp_ik(env, bottle_pos)
    if q_grasp_pose is None:
        raise RuntimeError('UR10_pick_place_shelf: Phase 1 grasp IK failed')

    # Phase 2: place on shelf top, seeded by the grasp pose so the planner
    # doesn't have to cross kinematic branches.
    q_goal = demo.compute_place_on_shelf_top_ik(env, q_seed=q_grasp_pose)
    if q_goal is None:
        raise RuntimeError('UR10_pick_place_shelf: place IK failed')

    return _finalise(env, q_grasp_pose, q_goal)


# ── UR10_pick_place_drill — carry power drill over kraft-box wall ────
# Hard-coded (q_start, q_goal) from UR10_pick_place_drill.py.
# q_start wrist_3 = IK base (-2.470412) + finger-rotation offset (-π/2)
# q_goal  wrist_3 = -4.086318 (same effective orientation as start)
_UR10_PICK_PLACE_DRILL_Q_START = np.array([
    0.671183, -0.993925,  1.692647,
   -2.269542, -1.570794, -4.041208,   # -2.470412 − π/2
])
_UR10_PICK_PLACE_DRILL_Q_GOAL = np.array([
   -0.944723, -1.190356,  2.023317,
   -2.403726, -1.570796, -4.086318,
])


def env_ur10_pick_place_drill():
    """6-D UR10e: pick power drill by handle and carry it over the wall."""
    import UR10_pick_place_drill as demo

    env = UR10eRobotiqEnv(
        gui=False, obstacles=demo.build_wall_obstacles(),
        base_position=[0.0, 0.0, demo.ROBOT_BASE_Z],
        base_orientation=_BASE_ORN,
    )
    return _finalise(env,
                     _UR10_PICK_PLACE_DRILL_Q_START,
                     _UR10_PICK_PLACE_DRILL_Q_GOAL)


# Public registry — consumed by run_from_config.py
UR10_ENV_REGISTRY = {
    'UR10_grasp_can':          env_ur10_grasp_can,
    'UR10_pick_place_can':     env_ur10_pick_place_can,
    'UR10_pick_shelf':         env_ur10_pick_shelf,
    'UR10_pick_place_shelf':   env_ur10_pick_place_shelf,
    'UR10_pick_place_drill':   env_ur10_pick_place_drill,
}
