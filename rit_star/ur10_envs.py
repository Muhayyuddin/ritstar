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
_MANIPULATION = os.path.join(_REPO, 'manipulation')
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)
if _MANIPULATION not in sys.path:
    sys.path.insert(0, _MANIPULATION)

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


# ── Tiago Pro 14-D dual-arm bimanual grasp ───────────────────────────

def env_tiago_14d():
    """14-D Tiago Pro: dual-arm bimanual pre-grasp of a tall box.

    Both 7-DOF arms are planned simultaneously in a joint 14-D
    configuration space.  The environment mirrors the headless version of
    Tiago_pro_dual_grasp_box.py:
      * q[:7]  = arm_left_1…7
      * q[7:]  = arm_right_1…7
      * torso is fixed at the midpoint of its travel range

    Returns the standard 6-tuple
    ``(collision_checker, edge_cost, metric, x_start, x_goal, bounds)``.
    """
    import sys
    import pybullet as p
    import pybullet_data

    # Import demo constants / helpers without running main()
    import importlib.util, os
    _spec = importlib.util.spec_from_file_location(
        'tiago_demo',
        os.path.join(_MANIPULATION, 'Tiago_pro_dual_grasp_box.py'))
    demo = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(demo)

    # Build headless PyBullet scene
    cid = p.connect(p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=cid)
    p.setGravity(0, 0, -9.81, physicsClientId=cid)

    tiago_orn = p.getQuaternionFromEuler([0, 0, demo.TIAGO_YAW])
    rid = p.loadURDF(demo.TIAGO_URDF,
                     basePosition=[demo.TIAGO_X, demo.TIAGO_Y, 0.0],
                     baseOrientation=tiago_orn,
                     useFixedBase=True,
                     physicsClientId=cid)

    jmap, lmap = demo.build_joint_maps(cid, rid)
    arm_left_idx  = [jmap[n] for n in demo.ARM_LEFT_JOINTS]
    arm_right_idx = [jmap[n] for n in demo.ARM_RIGHT_JOINTS]
    torso_idx     = jmap[demo.TORSO_JOINT]
    ee_left_idx   = lmap[demo.EE_LEFT_LINK]
    ee_right_idx  = lmap[demo.EE_RIGHT_LINK]

    # Fix torso at joint midpoint
    _ti = p.getJointInfo(rid, torso_idx, physicsClientId=cid)
    torso_mid = 0.5 * (float(_ti[8]) + float(_ti[9]))
    p.resetJointState(rid, torso_idx, torso_mid, physicsClientId=cid)
    p.setJointMotorControl2(rid, torso_idx, p.POSITION_CONTROL,
                            targetPosition=torso_mid, force=500,
                            physicsClientId=cid)

    # Apply home pose
    arm_home_right = demo.ARM_HOME * np.array([-1, 1, -1, 1, -1, 1, -1])
    demo.set_joint_values(cid, rid, arm_left_idx,  demo.ARM_HOME)
    demo.set_joint_values(cid, rid, arm_right_idx, arm_home_right)

    # Build scene, get obstacle IDs
    table_id, box_id, obs_left_id, obs_right_id = demo.build_scene(cid)
    obstacles = [table_id, box_id, obs_left_id, obs_right_id]

    # Compute IK goal configs for both arms
    all_movable = [i for i in range(p.getNumJoints(rid, physicsClientId=cid))
                   if p.getJointInfo(rid, i, physicsClientId=cid)[2]
                   != p.JOINT_FIXED]
    lpos, lorn, rpos, rorn = demo.compute_goal_targets()
    q_left, _, _ = demo.solve_arm_ik(
        cid, rid, arm_left_idx, ee_left_idx, lpos, lorn,
        all_movable, demo.ARM_HOME, obstacle_ids=obstacles,
        label='LEFT', n_seeds=200)
    demo.set_joint_values(cid, rid, arm_left_idx, q_left)
    q_right, _, _ = demo.solve_arm_ik(
        cid, rid, arm_right_idx, ee_right_idx, rpos, rorn,
        all_movable, arm_home_right, obstacle_ids=obstacles,
        label='RIGHT', n_seeds=200)

    # 14-D start / goal
    q_start_14 = np.concatenate([demo.ARM_HOME, arm_home_right])
    q_goal_14  = np.concatenate([q_left, q_right])

    # 14-D bounds (left-arm limits + right-arm limits)
    lower_l, upper_l = demo.get_joint_limits(cid, rid, arm_left_idx)
    lower_r, upper_r = demo.get_joint_limits(cid, rid, arm_right_idx)
    for lo, hi in [(lower_l, upper_l), (lower_r, upper_r)]:
        for i in range(len(lo)):
            if lo[i] >= hi[i]:
                lo[i] = -2 * np.pi
                hi[i] =  2 * np.pi
    bounds = (list(zip(lower_l.tolist(), upper_l.tolist())) +
              list(zip(lower_r.tolist(), upper_r.tolist())))

    # 14-D metric: per-arm inertia weights repeated for both arms
    weights_14 = demo.TIAGO_ARM_WEIGHTS + demo.TIAGO_ARM_WEIGHTS
    metric = DiagonalAnisotropicMetric(weights_14)

    # Collision checker: checks both arms simultaneously + self-collision
    fixed_joints = {torso_idx: torso_mid}
    collision_checker = demo.make_dual_arm_collision_checker(
        cid, rid, arm_left_idx, arm_right_idx, fixed_joints, obstacles)

    # Keep PyBullet client alive by attaching it to the checker closure
    collision_checker._cid = cid
    collision_checker._rid = rid

    return collision_checker, None, metric, q_start_14, q_goal_14, bounds


# ── Tiago Pro 14-D simple — no flanking obstacles ────────────────────

def env_tiago_14d_simple():
    """14-D Tiago Pro: dual-arm bimanual pre-grasp, no flanking obstacles.

    Mirrors tiago_pro.py — only the table and the central box are present
    (no red side obstacles).  Otherwise identical to env_tiago_14d.
    """
    import sys
    import pybullet as p
    import pybullet_data

    import importlib.util, os
    _spec = importlib.util.spec_from_file_location(
        'tiago_pro_demo',
        os.path.join(_MANIPULATION, 'tiago_pro.py'))
    demo = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(demo)

    # Build headless PyBullet scene
    cid = p.connect(p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=cid)
    p.setGravity(0, 0, -9.81, physicsClientId=cid)

    tiago_orn = p.getQuaternionFromEuler([0, 0, demo.TIAGO_YAW])
    rid = p.loadURDF(demo.TIAGO_URDF,
                     basePosition=[demo.TIAGO_X, demo.TIAGO_Y, 0.0],
                     baseOrientation=tiago_orn,
                     useFixedBase=True,
                     physicsClientId=cid)

    jmap, lmap = demo.build_joint_maps(cid, rid)
    arm_left_idx  = [jmap[n] for n in demo.ARM_LEFT_JOINTS]
    arm_right_idx = [jmap[n] for n in demo.ARM_RIGHT_JOINTS]
    torso_idx     = jmap[demo.TORSO_JOINT]
    ee_left_idx   = lmap[demo.EE_LEFT_LINK]
    ee_right_idx  = lmap[demo.EE_RIGHT_LINK]

    # Fix torso at joint midpoint
    _ti = p.getJointInfo(rid, torso_idx, physicsClientId=cid)
    torso_mid = 0.5 * (float(_ti[8]) + float(_ti[9]))
    p.resetJointState(rid, torso_idx, torso_mid, physicsClientId=cid)
    p.setJointMotorControl2(rid, torso_idx, p.POSITION_CONTROL,
                            targetPosition=torso_mid, force=500,
                            physicsClientId=cid)

    # Apply home pose
    arm_home_right = demo.ARM_HOME * np.array([-1, 1, -1, 1, -1, 1, -1])
    demo.set_joint_values(cid, rid, arm_left_idx,  demo.ARM_HOME)
    demo.set_joint_values(cid, rid, arm_right_idx, arm_home_right)

    # Build scene — returns only (table_id, box_id)
    table_id, box_id = demo.build_scene(cid)
    obstacles = [table_id, box_id]

    # Compute IK goal configs for both arms
    all_movable = [i for i in range(p.getNumJoints(rid, physicsClientId=cid))
                   if p.getJointInfo(rid, i, physicsClientId=cid)[2]
                   != p.JOINT_FIXED]
    lpos, lorn, rpos, rorn = demo.compute_goal_targets()
    q_left, _, _ = demo.solve_arm_ik(
        cid, rid, arm_left_idx, ee_left_idx, lpos, lorn,
        all_movable, demo.ARM_HOME, obstacle_ids=obstacles,
        label='LEFT', n_seeds=200)
    demo.set_joint_values(cid, rid, arm_left_idx, q_left)
    q_right, _, _ = demo.solve_arm_ik(
        cid, rid, arm_right_idx, ee_right_idx, rpos, rorn,
        all_movable, arm_home_right, obstacle_ids=obstacles,
        label='RIGHT', n_seeds=200)

    # 14-D start / goal
    q_start_14 = np.concatenate([demo.ARM_HOME, arm_home_right])
    q_goal_14  = np.concatenate([q_left, q_right])

    # 14-D bounds
    lower_l, upper_l = demo.get_joint_limits(cid, rid, arm_left_idx)
    lower_r, upper_r = demo.get_joint_limits(cid, rid, arm_right_idx)
    for lo, hi in [(lower_l, upper_l), (lower_r, upper_r)]:
        for i in range(len(lo)):
            if lo[i] >= hi[i]:
                lo[i] = -2 * np.pi
                hi[i] =  2 * np.pi
    bounds = (list(zip(lower_l.tolist(), upper_l.tolist())) +
              list(zip(lower_r.tolist(), upper_r.tolist())))

    # 14-D metric
    weights_14 = demo.TIAGO_ARM_WEIGHTS + demo.TIAGO_ARM_WEIGHTS
    metric = DiagonalAnisotropicMetric(weights_14)

    # Collision checker
    fixed_joints = {torso_idx: torso_mid}
    collision_checker = demo.make_dual_arm_collision_checker(
        cid, rid, arm_left_idx, arm_right_idx, fixed_joints, obstacles)

    collision_checker._cid = cid
    collision_checker._rid = rid

    return collision_checker, None, metric, q_start_14, q_goal_14, bounds


# Public registry — consumed by run_from_config.py
UR10_ENV_REGISTRY = {
    'UR10_grasp_can':          env_ur10_grasp_can,
    'UR10_pick_place_can':     env_ur10_pick_place_can,
    'UR10_pick_shelf':         env_ur10_pick_shelf,
    'UR10_pick_place_shelf':   env_ur10_pick_place_shelf,
    'UR10_pick_place_drill':   env_ur10_pick_place_drill,
    'Tiago 14D':               env_tiago_14d,
    'Tiago 14D simple':        env_tiago_14d_simple,
}
