#!/usr/bin/env python3
"""
UR10_pick_place_drill.py — Pick a power drill by its handle and carry it
across a box-shaped obstacle.

Setup (shares the kraft-box wall environment from UR10_grasp_can.py):
  - UR10e + Robotiq 85 mounted on the table, rotated 180° about world-Z.
  - Kraft-coloured box obstacle in front of the robot.
  - Start config: fingers straddle the drill's handle on the −y side of the
    box (one finger on each side of the handle). The drill sits flat on
    the table; only wrist_3 is yawed 90° so the finger opening direction
    is perpendicular to the handle axis.
  - Goal config:  mirror pose on the +y side of the box.
  - The YCB power drill is rigidly attached to the two finger-tip links
    via JOINT_FIXED constraints, so it travels with the gripper.
  - The planner must swing the arm — with the drill held by its handle —
    around or over the box.

Usage:
    python UR10_pick_place_drill.py                           # GUI
    python UR10_pick_place_drill.py --headless --save-gif ...
"""

import sys
import os
import time
import numpy as np
import pybullet as p

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from manipulator_env.pybullet_env import UR10eRobotiqEnv
from manipulator_env.planner_interface import plan_and_execute, interpolate_path
from rit_star.metric import DiagonalAnisotropicMetric

# Fast inertia-based diagonal metric
_UR10E_INERTIAS = np.array([7.778, 12.93, 3.87, 1.96, 1.96, 0.202])
_UR10E_WEIGHTS = (_UR10E_INERTIAS / _UR10E_INERTIAS.min()).tolist()

# ═══════════════════════════════════════════════════════════════════════
# Physical constants (metres) — identical to run_wall_env.py
# ═══════════════════════════════════════════════════════════════════════

TABLE_SURFACE_Z = 0.75
SLAB_THICKNESS  = 0.01990
ROBOT_BASE_Z    = TABLE_SURFACE_Z + SLAB_THICKNESS  # 0.76990

TABLE_LEN = 1.00
TABLE_WID = 1.50
TABLE_THK = 0.05
TABLE_CX  = -0.29
TABLE_CY  = -0.07

# Wall — identical to run_wall_env.py
# Wall: vertical barrier in front of robot, long axis along world x
WALL_L     = 0.57    # length along x-axis (57 cm)
WALL_X     = -0.60              # absolute wall centre x
WALL_Y     = 0.00               # absolute wall centre y
WALL_W     = 0.12     # thickness / depth (y-extent, 38 cm)
WALL_H     =  0.38  # height above table  (12 cm)
WALL_Z_BOT = ROBOT_BASE_Z
WALL_Z_MID = WALL_Z_BOT + WALL_H / 2

CLR_WALL  = [0.78, 0.64, 0.46, 0.85]  # kraft / Amazon-box tan (slightly transparent)
CLR_TABLE = [0.60, 0.60, 0.60, 1.0]
CLR_LEGS  = [0.25, 0.25, 0.28, 1.0]
CLR_SLAB  = [0.35, 0.35, 0.40, 1.0]

# YCB power drill (grasped by the handle in this demo)
DRILL_URDF   = os.path.join(_HERE, "ycb_objects", "ycb_assets",
                            "035_power_drill.urdf")
DRILL_SCALE  = 0.1
DRILL_HEIGHT = 0.15     # approximate overall drill height, reported in world state

# Offset (in the drill's local URDF frame) from the drill origin to the
# midpoint of the handle where the fingers should close.  Kept as a
# module-level constant so it can be tuned if the grasp needs to shift
# along the handle.
DRILL_HANDLE_OFFSET = np.array([0.0, 0.0, 0.0])

# Gripper depth from EE frame origin to fingertip centre
GRIPPER_DEPTH = 0.105

# Grasp offset along EE local-x (maps to world −z for top-down orientation)
GRASP_OFFSET_Z = GRIPPER_DEPTH + 0.06   # 0.165 m


# ═══════════════════════════════════════════════════════════════════════
# Obstacles
# ═══════════════════════════════════════════════════════════════════════

def build_wall_obstacles():
    return [
        {"type": "box", "color": CLR_TABLE,
         "pos": [TABLE_CX, TABLE_CY, TABLE_SURFACE_Z - TABLE_THK / 2],
         "half_extents": [TABLE_LEN / 2, TABLE_WID / 2, TABLE_THK / 2]},
        {"type": "box", "color": CLR_WALL,
         "pos": [WALL_X, WALL_Y, WALL_Z_MID],
         "half_extents": [WALL_L / 2, WALL_W / 2, WALL_H / 2]},
    ]


# ═══════════════════════════════════════════════════════════════════════
# Visual scenery
# ═══════════════════════════════════════════════════════════════════════

def _add_visual_box(cid, pos, half_extents, color):
    vis = p.createVisualShape(p.GEOM_BOX, halfExtents=half_extents,
                              rgbaColor=color, physicsClientId=cid)
    return p.createMultiBody(baseMass=0, baseVisualShapeIndex=vis,
                             basePosition=pos, physicsClientId=cid)


def add_scenery(cid):
    leg_h = TABLE_SURFACE_Z - TABLE_THK
    leg_he = [0.03, 0.03, leg_h / 2]
    inset = 0.06
    corners = [
        (TABLE_CX - TABLE_LEN / 2 + inset, TABLE_CY - TABLE_WID / 2 + inset),
        (TABLE_CX - TABLE_LEN / 2 + inset, TABLE_CY + TABLE_WID / 2 - inset),
        (TABLE_CX + TABLE_LEN / 2 - inset, TABLE_CY - TABLE_WID / 2 + inset),
        (TABLE_CX + TABLE_LEN / 2 - inset, TABLE_CY + TABLE_WID / 2 - inset),
    ]
    for lx, ly in corners:
        _add_visual_box(cid, pos=[lx, ly, leg_h / 2],
                        half_extents=leg_he, color=CLR_LEGS)
    _add_visual_box(cid, pos=[0.0, 0.0, TABLE_SURFACE_Z + SLAB_THICKNESS / 2],
                    half_extents=[0.10, 0.10, SLAB_THICKNESS / 2],
                    color=CLR_SLAB)


# ═══════════════════════════════════════════════════════════════════════
# IK solver (same as run_wall_env.py)
# ═══════════════════════════════════════════════════════════════════════

TOP_DOWN_ORN = list(p.getQuaternionFromEuler([0, np.pi / 2, 0]))


def find_ik(env, target_pos, side_label="", n_random=200, preferred_seed=None):
    cid = env.physics_client
    n_movable = len(env._all_joint_indices)
    ee_target = np.array(target_pos, dtype=float)
    desired_orn = np.array(TOP_DOWN_ORN)

    td_seed = [-1.50, 2.37, -2.44, -1.57, -4.57]
    j0_angles = np.linspace(-np.pi, np.pi, 17)
    seeds = []

    if preferred_seed is not None:
        seeds.append(list(preferred_seed[:6]))

    seeds.extend([[j0] + td_seed for j0 in j0_angles])

    for j0 in np.linspace(-np.pi, np.pi, 9):
        seeds.append([j0, -1.50, 2.37, -2.44, -1.57, -4.57])
        seeds.append([j0, -1.0, 1.5, -2.0, -1.57, 0.0])
        seeds.append([j0, -0.5, 1.0, -2.0, -1.57, 0.0])
        seeds.append([j0, -1.2, 2.0, -3.94, -1.57, 0.0])

    base_seed = [-np.pi / 2, np.pi / 2, -np.pi / 2, -np.pi / 2, 0.0]
    for j0 in np.linspace(-np.pi, np.pi, 13):
        seeds.append([j0] + base_seed)

    rng = np.random.RandomState(42)
    for _ in range(n_random):
        seeds.append(rng.uniform(-np.pi, np.pi, size=6).tolist())

    candidates = []
    for use_orn in [True, False]:
        if candidates and use_orn is False:
            break
        for si, seed in enumerate(seeds):
            env.set_joint_positions(np.array(seed[:6]))
            rest = list(seed[:6]) + [0.0] * (n_movable - 6)
            ik_kwargs = dict(
                bodyUniqueId=env.robot_id,
                endEffectorLinkIndex=env.ee_link_idx,
                targetPosition=ee_target.tolist(),
                lowerLimits=env.JOINT_LIMITS_LOWER.tolist(),
                upperLimits=env.JOINT_LIMITS_UPPER.tolist(),
                jointRanges=[4 * np.pi] * 6 + [0.01] * (n_movable - 6),
                restPoses=rest,
                maxNumIterations=500,
                residualThreshold=1e-4,
                physicsClientId=cid,
            )
            if use_orn:
                ik_kwargs['targetOrientation'] = list(desired_orn)
            q_ik = p.calculateInverseKinematics(**ik_kwargs)
            q_arm = np.array(q_ik[:6])
            ee_actual, ee_orn = env.get_ee_pose(q_arm)
            pos_err = np.linalg.norm(ee_target - ee_actual)

            if env.is_collision_free(q_arm) and pos_err < 0.05:
                orn_dot = abs(np.dot(desired_orn, np.array(ee_orn)))
                candidates.append((q_arm, pos_err, orn_dot, si))

    if not candidates:
        print(f"[IK]  ERROR: No collision-free {side_label} IK found!")
        return None

    candidates.sort(key=lambda c: (-c[2], c[1]))
    best_q, best_pos_err, best_orn_dot, best_si = candidates[0]

    orn_label = "GOOD" if best_orn_dot > 0.95 else "fair"
    print(f"[IK]  Collision-free {side_label} found  "
          f"(seed {best_si}, pos_err={best_pos_err:.4f}, "
          f"orn_dot={best_orn_dot:.4f} [{orn_label}])")
    print(f"      q   = [{', '.join(f'{v:.4f}' for v in best_q)}]")
    ee_pos, _ = env.get_ee_pose(best_q)
    print(f"      EE  = [{ee_pos[0]:.3f}, {ee_pos[1]:.3f}, {ee_pos[2]:.3f}]")
    print(f"      ({len(candidates)} candidates found, "
          f"best orn_dot={best_orn_dot:.4f})")
    return best_q


# ═══════════════════════════════════════════════════════════════════════
# Attach the YCB power drill BY ITS HANDLE between the gripper fingers
# ═══════════════════════════════════════════════════════════════════════

def attach_drill_to_gripper(env, q_grasp):
    """Spawn the drill in the gripper and weld it to both fingertips.

    The drill is placed so that the midpoint of its handle sits exactly
    between the two finger-tip links.  The handle axis is aligned with the
    gripper's approach axis (EE +x, which points world −z under
    TOP_DOWN_ORN), so the two fingers close on opposite sides of the
    handle.  Two JOINT_FIXED constraints — one per finger tip — pin the
    drill rigidly in place for the whole carry motion.
    """
    cid = env.physics_client
    rid = env.robot_id

    env.set_joint_positions(q_grasp)
    p.stepSimulation(physicsClientId=cid)

    # World-frame pose of the EE at the grasp configuration.
    ee_state = p.getLinkState(rid, env.ee_link_idx,
                              computeForwardKinematics=True,
                              physicsClientId=cid)
    ee_pos_w = np.array(ee_state[4])
    ee_orn_w = list(ee_state[5])

    # Drill POSITION.
    #   * drill_offset_local[0] = GRASP_OFFSET_Z − 0.01: drill sits 1 cm
    #     closer to the EE than the nominal fingertip plane, so from the
    #     gripper's view the fingers are effectively 2 cm LOWER on the
    #     drill than in the previous offset (= "gripper moves 2 cm down"
    #     for a proper grasp).
    #   * +2 cm in world +y afterwards: nudge the drill toward the kraft
    #     box obstacle as requested.
    drill_offset_local = [GRASP_OFFSET_Z - 0.01, 0.0, 0.0]
    drill_pos_w, _ = p.multiplyTransforms(
        ee_pos_w.tolist(), ee_orn_w,
        drill_offset_local, [0, 0, 0, 1],
    )
    drill_pos_w = [drill_pos_w[0],
                   drill_pos_w[1] + 0.03,   # 3 cm towards robot/box (+y)
                   drill_pos_w[2]]

    # Drill WORLD ORIENTATION — frozen at identity, independent of the
    # wrist_3 yaw.  In the original can-demo composition
    # (TOP_DOWN_ORN ∘ local [0,−π/2,0]) the can ended up with identity in
    # world frame, which places the drill flat on the table with its
    # handle pointing along world +z.  By setting it here directly the
    # wrist_3 offset in q_start / q_goal only rotates the fingers about
    # world-Z; the drill itself does not turn with them.
    drill_orn_w = [0.0, 0.0, 0.0, 1.0]

    # Shift the drill body so that DRILL_HANDLE_OFFSET (in drill-local
    # coordinates) ends up at the gripper grasp centre.
    if np.any(DRILL_HANDLE_OFFSET):
        _ho_world, _ = p.multiplyTransforms(
            [0.0, 0.0, 0.0], drill_orn_w,
            DRILL_HANDLE_OFFSET.tolist(), [0, 0, 0, 1])
        drill_pos_w = [drill_pos_w[i] - _ho_world[i] for i in range(3)]

    print(f"[DRILL] Spawning YCB power drill (grasp on handle) at "
          f"[{drill_pos_w[0]:.3f}, {drill_pos_w[1]:.3f}, {drill_pos_w[2]:.3f}]")

    drill_id = p.loadURDF(
        DRILL_URDF,
        basePosition=list(drill_pos_w),
        baseOrientation=list(drill_orn_w),
        globalScaling=DRILL_SCALE,
        useFixedBase=False,
        physicsClientId=cid,
    )

    # Disable robot-vs-drill self-collision (the fingers intentionally
    # overlap the drill handle); keep floor collision enabled.
    n_joints = p.getNumJoints(rid, physicsClientId=cid)
    for link_idx in range(-1, n_joints):
        p.setCollisionFilterPair(rid, drill_id, link_idx, -1, 0,
                                 physicsClientId=cid)

    # Fix the drill to each finger tip with a rigid JOINT_FIXED constraint.
    def _finger_constraint(joint_name, fallback_idx):
        fl_idx = env._joint_name_to_idx.get(joint_name, fallback_idx)
        fl_state = p.getLinkState(rid, fl_idx,
                                  computeForwardKinematics=True,
                                  physicsClientId=cid)
        drill_in_fl, drill_orn_in_fl = p.multiplyTransforms(
            *p.invertTransform(list(fl_state[4]), list(fl_state[5])),
            list(drill_pos_w), list(drill_orn_w),
        )
        cid_ = p.createConstraint(
            parentBodyUniqueId=rid,
            parentLinkIndex=fl_idx,
            childBodyUniqueId=drill_id,
            childLinkIndex=-1,
            jointType=p.JOINT_FIXED,
            jointAxis=[0, 0, 0],
            parentFramePosition=list(drill_in_fl),
            childFramePosition=[0, 0, 0],
            parentFrameOrientation=list(drill_orn_in_fl),
            childFrameOrientation=[0, 0, 0, 1],
            physicsClientId=cid,
        )
        p.changeConstraint(cid_, maxForce=10000, physicsClientId=cid)
        return fl_idx, cid_, list(drill_in_fl), list(drill_orn_in_fl)

    left_idx, _, drill_in_ft, drill_orn_in_ft = _finger_constraint(
        "robotiq_85_left_finger_tip_joint",  env.ee_link_idx)
    _, _, _, _ = _finger_constraint(
        "robotiq_85_right_finger_tip_joint", env.ee_link_idx)

    # Register the drill against the left finger tip for planner collision
    # checks (planner treats it as an attached body of the left finger).
    env.attach_body(drill_id, left_idx, drill_in_ft, drill_orn_in_ft)

    for _ in range(20):
        p.stepSimulation(physicsClientId=cid)

    print("[DRILL] Fixed to left & right finger-tip links via JOINT_FIXED")
    return drill_id, None, drill_in_ft, drill_orn_in_ft


# ═══════════════════════════════════════════════════════════════════════
# Path animation with the attached drill
# ═══════════════════════════════════════════════════════════════════════

def visualize_path_with_drill(env, path, delay=0.03, trail=True):
    cid = env.physics_client
    prev_pos = None
    for q in path:
        env.set_joint_positions(q)
        for _ in range(5):
            p.stepSimulation(physicsClientId=cid)
        if trail:
            pos, _ = env.get_ee_pose()
            if prev_pos is not None:
                p.addUserDebugLine(
                    prev_pos.tolist(), pos.tolist(),
                    lineColorRGB=[1, 0, 0], lineWidth=3, lifeTime=0,
                    physicsClientId=cid,
                )
            prev_pos = pos
        time.sleep(delay)


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

def main():
    from manipulator_env.demo_cli import parse_demo_args, append_demo_result_csv
    _args = parse_demo_args()
    wall_obstacles = build_wall_obstacles()

    mode = 'headless' if _args.headless else 'GUI'
    print(f"[ENV] Loading PyBullet ({mode}) ...")
    env = UR10eRobotiqEnv(
        gui=not _args.headless,
        obstacles=wall_obstacles,
        base_position=[0.0, 0.0, ROBOT_BASE_Z],
        base_orientation=p.getQuaternionFromEuler([0, 0, np.pi]),
    )
    cid = env.physics_client
    add_scenery(cid)

    # ── Infinite very light grey floor ────────────────────────────
    floor_he = [50.0, 50.0, 0.01]
    floor_vis = p.createVisualShape(p.GEOM_BOX, halfExtents=floor_he,
                                    rgbaColor=[0.78, 0.78, 0.78, 1.0],
                                    physicsClientId=cid)
    p.createMultiBody(baseMass=0, baseCollisionShapeIndex=-1,
                      baseVisualShapeIndex=floor_vis,
                      basePosition=[0.0, 0.0, 0.0],
                      physicsClientId=cid)
    p.changeVisualShape(env.plane_id, -1,
                        rgbaColor=[0.78, 0.78, 0.78, 1.0],
                        physicsClientId=cid)

    # ── Robot colours ─────────────────────────────────────────────
    UR_SILVER = [0.35, 0.35, 0.35, 1.0]
    UR_DARK   = [0.22, 0.22, 0.22, 1.0]
    UR_BLUE   = [0.00, 0.34, 0.68, 1.0]
    RQ_DARK   = [0.15, 0.15, 0.15, 1.0]
    RQ_BLACK  = [0.08, 0.08, 0.08, 1.0]

    rid = env.robot_id
    n = p.getNumJoints(rid, physicsClientId=cid)
    link_name_to_idx = {}
    for i in range(n):
        info = p.getJointInfo(rid, i, physicsClientId=cid)
        link_name_to_idx[info[12].decode("utf-8")] = i

    p.changeVisualShape(rid, -1, rgbaColor=UR_DARK, physicsClientId=cid)
    ur_colour_map = {
        "base_link_inertia": UR_DARK,
        "shoulder_link":     UR_BLUE,
        "upper_arm_link":    UR_SILVER,
        "forearm_link":      UR_SILVER,
        "wrist_1_link":      UR_DARK,
        "wrist_2_link":      UR_SILVER,
        "wrist_3_link":      UR_DARK,
        "flange":            UR_DARK,
        "tool0":             UR_DARK,
    }
    for lname, colour in ur_colour_map.items():
        if lname in link_name_to_idx:
            p.changeVisualShape(rid, link_name_to_idx[lname],
                                rgbaColor=colour, physicsClientId=cid)
    for lname, lidx in link_name_to_idx.items():
        if "robotiq" in lname:
            clr = RQ_BLACK if "finger_tip" in lname else RQ_DARK
            p.changeVisualShape(rid, lidx, rgbaColor=clr, physicsClientId=cid)

    # ── Hard-coded start / computed goal ──────────────────────────────────────
    # Start target: [-0.590, -0.691, 0.980] (−y side of wall)
    # Grasp orientation: wrist_3 offset −π/2 so fingers close ACROSS the handle.
    q_start = np.array([0.671183, -0.993925, 1.692647,
                        -2.269542, -1.570794, -2.470412 - np.pi / 2])

    # Goal: directly across the wall on the +y side, only 0.20 m past the wall's
    # +y face — x = WALL_X (inside the wall's x-extent), z = 0.980 (same height
    # as start).  Both start and goal are inside the wall's x-extent and below
    # the wall top (z=0.980 < z_top≈1.150), forcing the planner to pass over the
    # box rather than routing around it.
    goal_target = [WALL_X, WALL_Y + WALL_W / 2 + 0.20, 0.980]
    print(f"[IK] Solving goal IK for target {[round(v,3) for v in goal_target]} ...")
    q_goal = find_ik(env, goal_target, "goal (near box)")
    assert q_goal is not None, "Goal IK not found — adjust goal_target"

    assert env.is_collision_free(q_start), "Hard-coded start is in collision!"
    assert env.is_collision_free(q_goal),  "Goal is in collision!"

    # ── Attach drill between gripper fingers at start ─────────────
    drill_id, _, drill_in_ft, drill_orn_in_ft = attach_drill_to_gripper(env, q_start)

    pos_s, _ = env.get_ee_pose(q_start)
    pos_g, _ = env.get_ee_pose(q_goal)

    print(f"[CFG] Start q = [{', '.join(f'{v:.4f}' for v in q_start)}]")
    print(f"      EE    = [{pos_s[0]:.3f}, {pos_s[1]:.3f}, {pos_s[2]:.3f}]")
    print(f"[CFG] Goal  q = [{', '.join(f'{v:.4f}' for v in q_goal)}]")
    print(f"      EE    = [{pos_g[0]:.3f}, {pos_g[1]:.3f}, {pos_g[2]:.3f}]")

    # ── Camera & markers ──────────────────────────────────────────
    p.resetDebugVisualizerCamera(
        cameraDistance=1.60,
        cameraYaw=-114.60,
        cameraPitch=-40.20,
        cameraTargetPosition=[-0.449, -0.041, 0.892],
        physicsClientId=cid,
    )
    p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 0, physicsClientId=cid)

    def print_camera_view(tag=""):
        info = p.getDebugVisualizerCamera(physicsClientId=cid)
        yaw, pitch, dist, target = info[8], info[9], info[10], info[11]
        print(f"[CAM]{tag} yaw={yaw:+.2f}  pitch={pitch:+.2f}  "
              f"dist={dist:.3f}  target=[{target[0]:+.3f}, "
              f"{target[1]:+.3f}, {target[2]:+.3f}]")

    print_camera_view(" init")

    # Blue start marker only; red goal marker is drawn by plan_and_execute
    vis = p.createVisualShape(p.GEOM_SPHERE, radius=0.03,
                              rgbaColor=[0, 0, 1, 1], physicsClientId=cid)
    p.createMultiBody(baseMass=0, baseVisualShapeIndex=vis,
                      basePosition=pos_s.tolist(), physicsClientId=cid)

    print()
    print("=" * 62)
    print("  Wall Carry Demo — Grasp & Transport Over Wall")
    print("=" * 62)
    print(f"  Wall centre : [{WALL_X:.3f}, {WALL_Y:.3f}, {WALL_Z_MID:.3f}]")
    print(f"  Wall size   : {WALL_L:.2f}(x) x {WALL_W:.2f}(y) x {WALL_H:.2f}(z)")
    print(f"  Start EE    : [{pos_s[0]:.3f}, {pos_s[1]:.3f}, {pos_s[2]:.3f}]  (grasp, −y)")
    print(f"  Goal  EE    : [{pos_g[0]:.3f}, {pos_g[1]:.3f}, {pos_g[2]:.3f}]  (place, +y)")
    print(f"  Start cfg   : [{', '.join(f'{v:.3f}' for v in q_start)}]")
    print(f"  Goal  cfg   : [{', '.join(f'{v:.3f}' for v in q_goal)}]")
    print(f"  YCB drill   : grasped by handle (fixed joint to both fingers)")
    print("=" * 62)
    print()

    # ── Plan ──────────────────────────────────────────────────────
    fast_metric = DiagonalAnisotropicMetric(weights=_UR10E_WEIGHTS)
    import time as _time
    _t0 = _time.time()
    path, cost = plan_and_execute(
        env,
        q_start,
        q_goal,
        metric=fast_metric,
        batch_size=_args.batch_size,
        max_iterations=_args.max_iterations,
        smooth=True,
        animate=False,
        planner_name=_args.planner,
        seed=_args.seed,
    )
    _elapsed = _time.time() - _t0

    if _args.save_results:
        append_demo_result_csv({
            'demo': 'UR10_pick_place_drill',
            'planner': _args.planner,
            'seed': _args.seed,
            'max_iterations': _args.max_iterations,
            'batch_size': _args.batch_size,
            'final_cost': float(cost) if np.isfinite(cost) else float('inf'),
            'waypoints': len(path) if path else 0,
            'time_s': float(_elapsed),
            'success': bool(path),
        })

    if path:
        print(f"\n[RESULT] Path found — cost: {cost:.4f}, waypoints: {len(path)}")

        path_fine = interpolate_path(path, max_step=0.02)

        if _args.save_gif:
            from manipulator_env.demo_cli import save_path_gif
            _gif_tag = _args.planner.replace('*', '').replace(' ', '_')
            _gif_path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                'visualization', 'gifs',
                f'pybullet_UR10_pick_place_drill_{_gif_tag}.gif')
            save_path_gif(
                env, path_fine, _gif_path,
                cam_yaw=-114.60, cam_pitch=-40.20, cam_distance=1.60,
                cam_target=[-0.449, -0.041, 0.892],
                step=3, fps=20)
            print(f"[GIF] Saved {_gif_path}")

        if _args.headless:
            env.disconnect()
            return

        # Animate with drill attached
        print("[ANIM] Animating path with grasped drill ...")
        env.set_joint_positions(q_start)
        for _ in range(10):
            p.stepSimulation(physicsClientId=cid)
        time.sleep(0.5)
        visualize_path_with_drill(env, path_fine, delay=0.02, trail=True)

        os.makedirs("results", exist_ok=True)

        with open("results/UR10_pick_place_drill_world_state.txt", "w") as f:
            f.write("=" * 62 + "\n")
            f.write("  Wall Carry Demo — World State\n")
            f.write("=" * 62 + "\n\n")
            f.write("--- Robot ---\n")
            f.write(f"  Base position : [0.000, 0.000, {ROBOT_BASE_Z:.5f}]\n\n")
            f.write("--- Table ---\n")
            f.write(f"  Surface Z     : {TABLE_SURFACE_Z:.5f} m\n")
            f.write(f"  Centre (x,y)  : [{TABLE_CX:.3f}, {TABLE_CY:.3f}]\n")
            f.write(f"  Dimensions    : {TABLE_LEN:.2f} x {TABLE_WID:.2f} x "
                    f"{TABLE_THK:.2f} m\n\n")
            f.write("--- Wall ---\n")
            f.write(f"  Centre        : [{WALL_X:.3f}, {WALL_Y:.3f}, "
                    f"{WALL_Z_MID:.3f}]\n")
            f.write(f"  Dimensions    : {WALL_L:.3f}(x) x {WALL_W:.3f}(y) x "
                    f"{WALL_H:.3f}(z)\n\n")
            f.write("--- YCB Power Drill (035) ---\n")
            f.write(f"  Grasp         : fingers on either side of the handle\n")
            f.write(f"  Attachment    : JOINT_FIXED to left & right finger tips\n")
            f.write(f"  Height (approx): {DRILL_HEIGHT:.3f} m\n")
            f.write(f"  URDF scale    : {DRILL_SCALE}\n\n")
            f.write("--- Obstacle list (collision-checked) ---\n")
            for i, obs in enumerate(wall_obstacles):
                f.write(f"  [{i}] {obs['type']}  pos={[round(v,5) for v in obs['pos']]}  "
                        f"half_extents={[round(v,5) for v in obs['half_extents']]}\n")
            f.write("\n--- Start Configuration (grasp, −y side) ---\n")
            f.write(f"  q_start (rad) : [{', '.join(f'{v:.6f}' for v in q_start)}]\n")
            f.write(f"  EE position   : [{pos_s[0]:.5f}, {pos_s[1]:.5f}, {pos_s[2]:.5f}]\n\n")
            f.write("--- Goal Configuration (+y side, place) ---\n")
            f.write(f"  q_goal  (rad) : [{', '.join(f'{v:.6f}' for v in q_goal)}]\n")
            f.write(f"  EE position   : [{pos_g[0]:.5f}, {pos_g[1]:.5f}, {pos_g[2]:.5f}]\n\n")
            f.write("--- Planner ---\n")
            f.write(f"  Algorithm     : RIT*\n")
            f.write(f"  Metric        : DiagonalAnisotropicMetric\n")
            f.write(f"  Weights       : {[round(w,5) for w in _UR10E_WEIGHTS]}\n")
            f.write(f"  Batch size    : 200\n")
            f.write(f"  Max iters     : 300\n")
            f.write(f"  Path cost     : {cost:.6f}\n")
            f.write(f"  Waypoints     : {len(path)}\n")
        print("[FILE] Saved results/UR10_pick_place_drill_world_state.txt")

        with open("results/UR10_pick_place_drill_path.txt", "w") as f:
            f.write("=" * 62 + "\n")
            f.write("  Wall Carry Demo — Complete Path (Joint Configurations)\n")
            f.write("=" * 62 + "\n")
            f.write(f"  Waypoints : {len(path)}\n")
            f.write(f"  Path cost : {cost:.6f}\n")
            f.write(f"  DOF       : 6\n\n")
            f.write("  Each row: joint_1  joint_2  joint_3  joint_4  joint_5  joint_6  (radians)\n")
            f.write("-" * 62 + "\n")
            for i, q in enumerate(path):
                f.write(f"  {i:4d}  " + "  ".join(f"{v:+10.6f}" for v in q) + "\n")
            f.write("-" * 62 + "\n")
        print("[FILE] Saved results/UR10_pick_place_drill_path.txt")
    else:
        print("\n[RESULT] No path found.")

    # ── Loop path animation until window is closed ────────────
    if _args.headless:
        env.disconnect()
        return
    if path:
        print("\n[LOOP] Replaying path (close PyBullet window to exit) ...")
        finger_link_idx = env._joint_name_to_idx.get(
            "robotiq_85_left_finger_tip_joint", env.ee_link_idx
        )
        try:
            while p.isConnected(physicsClientId=cid):
                print_camera_view()
                env.set_joint_positions(q_start)
                ft_state = p.getLinkState(rid, finger_link_idx,
                                          computeForwardKinematics=True,
                                          physicsClientId=cid)
                drill_pos_w, drill_orn_w = p.multiplyTransforms(
                    list(ft_state[4]), list(ft_state[5]),
                    list(drill_in_ft), list(drill_orn_in_ft),
                )
                p.resetBasePositionAndOrientation(
                    drill_id, list(drill_pos_w), list(drill_orn_w),
                    physicsClientId=cid)
                for _ in range(10):
                    p.stepSimulation(physicsClientId=cid)
                time.sleep(0.5)
                visualize_path_with_drill(env, path_fine, delay=0.02, trail=False)
                time.sleep(1.0)
        except (KeyboardInterrupt, Exception):
            pass
    else:
        print("\n  Press Ctrl+C to exit.")
        try:
            while p.isConnected(physicsClientId=cid):
                p.stepSimulation(physicsClientId=cid)
                time.sleep(1 / 240)
        except (KeyboardInterrupt, Exception):
            pass
    print("Shutting down ...")
    env.disconnect()


if __name__ == "__main__":
    main()
