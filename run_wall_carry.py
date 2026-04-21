#!/usr/bin/env python3
"""
run_wall_carry.py — Carry-over-wall demo: grasp can, carry to other side.

Setup (same wall environment as run_wall_env.py):
  - UR10e mounted on table with a tall wall
  - Start: grasp config at the can (−y side) — same as run_wall_env goal
  - YCB tomato soup can attached between gripper fingers (fixed joint)
  - Goal:  mirror position on +y side of wall, same orientation, +1 cm height
  - The planner must swing the arm (with can) around/over the wall

Usage:
    python run_wall_carry.py
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
_UR10E_WEIGHTS = (_UR10E_INERTIAS / _UR10E_INERTIAS.max()).tolist()

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
WALL_L     = 0.54
WALL_X     = -0.60
WALL_Y     = 0.00
WALL_W     = 0.182
WALL_H     = 0.50
WALL_Z_BOT = ROBOT_BASE_Z
WALL_Z_MID = WALL_Z_BOT + WALL_H / 2

CLR_WALL  = [0.35, 0.45, 0.60, 0.85]
CLR_TABLE = [0.60, 0.60, 0.60, 1.0]
CLR_LEGS  = [0.25, 0.25, 0.28, 1.0]
CLR_SLAB  = [0.35, 0.35, 0.40, 1.0]

# YCB tomato soup can
CAN_URDF = os.path.join(_HERE, "ycb_objects", "ycb_assets",
                        "005_tomato_soup_can.urdf")
CAN_SCALE  = 0.1
CAN_HEIGHT = 0.11

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
# Attach YCB can between gripper fingers (same as run_wall_pick_place.py)
# ═══════════════════════════════════════════════════════════════════════

def attach_can_to_gripper(env, q_grasp):
    cid = env.physics_client
    rid = env.robot_id

    env.set_joint_positions(q_grasp)
    p.stepSimulation(physicsClientId=cid)

    ee_state = p.getLinkState(rid, env.ee_link_idx,
                              computeForwardKinematics=True,
                              physicsClientId=cid)
    ee_pos_w = np.array(ee_state[4])
    ee_orn_w = list(ee_state[5])

    # Place can in EE frame, then compute its pose in the finger tip frame
    can_offset_local = [GRASP_OFFSET_Z, 0.0, 0.0]
    can_local_orn = list(p.getQuaternionFromEuler([0, -np.pi / 2, 0]))
    can_pos_w, can_orn_w = p.multiplyTransforms(
        ee_pos_w.tolist(), ee_orn_w,
        can_offset_local, can_local_orn,
    )

    print(f"[CAN] Loading tomato soup can at "
          f"[{can_pos_w[0]:.3f}, {can_pos_w[1]:.3f}, {can_pos_w[2]:.3f}]")

    can_id = p.loadURDF(
        CAN_URDF,
        basePosition=list(can_pos_w),
        baseOrientation=list(can_orn_w),
        globalScaling=CAN_SCALE,
        useFixedBase=False,
        physicsClientId=cid,
    )

    # Disable robot–can self-collision; keep floor collision enabled
    n_joints = p.getNumJoints(rid, physicsClientId=cid)
    for link_idx in range(-1, n_joints):
        p.setCollisionFilterPair(rid, can_id, link_idx, -1, 0,
                                 physicsClientId=cid)

    # Compute can pose in each finger tip's local frame, then fix with both
    def _finger_constraint(joint_name, fallback_idx):
        fl_idx = env._joint_name_to_idx.get(joint_name, fallback_idx)
        fl_state = p.getLinkState(rid, fl_idx,
                                  computeForwardKinematics=True,
                                  physicsClientId=cid)
        can_in_fl, can_orn_in_fl = p.multiplyTransforms(
            *p.invertTransform(list(fl_state[4]), list(fl_state[5])),
            list(can_pos_w), list(can_orn_w),
        )
        cid_ = p.createConstraint(
            parentBodyUniqueId=rid,
            parentLinkIndex=fl_idx,
            childBodyUniqueId=can_id,
            childLinkIndex=-1,
            jointType=p.JOINT_FIXED,
            jointAxis=[0, 0, 0],
            parentFramePosition=list(can_in_fl),
            childFramePosition=[0, 0, 0],
            parentFrameOrientation=list(can_orn_in_fl),
            childFrameOrientation=[0, 0, 0, 1],
            physicsClientId=cid,
        )
        p.changeConstraint(cid_, maxForce=10000, physicsClientId=cid)
        return fl_idx, cid_, list(can_in_fl), list(can_orn_in_fl)

    left_idx, _, can_in_ft, can_orn_in_ft = _finger_constraint(
        "robotiq_85_left_finger_tip_joint",  env.ee_link_idx)
    _, _, _, _ = _finger_constraint(
        "robotiq_85_right_finger_tip_joint", env.ee_link_idx)

    # Register can against the left finger tip for planner collision checks
    env.attach_body(can_id, left_idx, can_in_ft, can_orn_in_ft)

    for _ in range(20):
        p.stepSimulation(physicsClientId=cid)

    print(f"[CAN] Fixed to left & right finger tip links")
    return can_id, None, can_in_ft, can_orn_in_ft


# ═══════════════════════════════════════════════════════════════════════
# Path animation with attached can
# ═══════════════════════════════════════════════════════════════════════

def visualize_path_with_can(env, path, delay=0.03, trail=True):
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
    wall_obstacles = build_wall_obstacles()

    print("[ENV] Loading PyBullet (GUI) ...")
    env = UR10eRobotiqEnv(
        gui=True,
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

    # ── Hard-coded start / goal (precomputed offline with _find_wall_carry_ik.py) ──
    # Start target: [-0.600, -0.691, 0.980] (10 cm toward −y table edge, +1 cm lift)
    # Goal target:  mirror on +y side at same height.
    # Both collision-free top-down grasps; Δq ≈ 2.26 rad (easy for planner).
    q_start = np.array([0.664284, -0.985480, 1.677681,
                        -2.263020, -1.570794, -2.477311])
    q_goal = np.array([-0.912412, -1.161520, 1.977143,
                       -2.386444, -1.570793, 2.229174])

    assert env.is_collision_free(q_start), "Hard-coded start is in collision!"
    assert env.is_collision_free(q_goal),  "Hard-coded goal  is in collision!"

    # ── Attach can between gripper fingers at start ───────────────
    can_id, _, can_in_ft, can_orn_in_ft = attach_can_to_gripper(env, q_start)

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
    print(f"  YCB can     : attached to gripper (fixed joint)")
    print("=" * 62)
    print()

    # ── Plan ──────────────────────────────────────────────────────
    fast_metric = DiagonalAnisotropicMetric(weights=_UR10E_WEIGHTS)
    path, cost = plan_and_execute(
        env,
        q_start,
        q_goal,
        metric=fast_metric,
        batch_size=200,
        max_iterations=300,
        smooth=True,
        animate=False,
    )

    if path:
        print(f"\n[RESULT] Path found — cost: {cost:.4f}, waypoints: {len(path)}")

        # Animate with can attached
        print("[ANIM] Animating path with grasped can ...")
        path_fine = interpolate_path(path, max_step=0.02)
        env.set_joint_positions(q_start)
        for _ in range(10):
            p.stepSimulation(physicsClientId=cid)
        time.sleep(0.5)
        visualize_path_with_can(env, path_fine, delay=0.02, trail=True)

        os.makedirs("results", exist_ok=True)

        with open("results/wall_carry_world_state.txt", "w") as f:
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
            f.write("--- Tomato Soup Can (YCB 005) ---\n")
            f.write(f"  Attached to EE with fixed joint\n")
            f.write(f"  Height        : {CAN_HEIGHT:.3f} m\n\n")
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
        print("[FILE] Saved results/wall_carry_world_state.txt")

        with open("results/wall_carry_path.txt", "w") as f:
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
        print("[FILE] Saved results/wall_carry_path.txt")
    else:
        print("\n[RESULT] No path found.")

    # ── Loop path animation until window is closed ────────────
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
                can_pos_w, can_orn_w = p.multiplyTransforms(
                    list(ft_state[4]), list(ft_state[5]),
                    list(can_in_ft), list(can_orn_in_ft),
                )
                p.resetBasePositionAndOrientation(
                    can_id, list(can_pos_w), list(can_orn_w),
                    physicsClientId=cid)
                for _ in range(10):
                    p.stepSimulation(physicsClientId=cid)
                time.sleep(0.5)
                visualize_path_with_can(env, path_fine, delay=0.02, trail=False)
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
