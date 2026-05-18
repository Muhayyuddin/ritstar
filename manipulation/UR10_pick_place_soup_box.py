#!/usr/bin/env python3
"""
UR10_pick_place_soup_box.py — Load a saved drill path and replay it with
a YCB potted-meat-can (010) welded to the gripper fingers instead of the
power drill.

The environment is identical to UR10_pick_place_drill.py (same wall, same
robot base, same start configuration).  The saved path is read from
  results/UR10_pick_place_drill_path.txt
and interpolated before animation.

Usage:
    python UR10_pick_place_soup_box.py               # GUI replay
    python UR10_pick_place_soup_box.py --headless    # headless (no window)
"""

import sys
import os
import time
import numpy as np
import pybullet as p

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))

from manipulator_env.pybullet_env import UR10eRobotiqEnv
from manipulator_env.planner_interface import interpolate_path

# ═══════════════════════════════════════════════════════════════════════
# Physical constants — identical to UR10_pick_place_drill.py
# ═══════════════════════════════════════════════════════════════════════

TABLE_SURFACE_Z = 0.75
SLAB_THICKNESS  = 0.01990
ROBOT_BASE_Z    = TABLE_SURFACE_Z + SLAB_THICKNESS  # 0.76990

TABLE_LEN = 1.00
TABLE_WID = 1.50
TABLE_THK = 0.05
TABLE_CX  = -0.29
TABLE_CY  = -0.07

WALL_L     = 0.57
WALL_X     = -0.60
WALL_Y     = 0.00
WALL_W     = 0.12
WALL_H     = 0.38
WALL_Z_BOT = ROBOT_BASE_Z
WALL_Z_MID = WALL_Z_BOT + WALL_H / 2

CLR_WALL  = [0.78, 0.64, 0.46, 0.85]
CLR_TABLE = [0.60, 0.60, 0.60, 1.0]
CLR_LEGS  = [0.25, 0.25, 0.28, 1.0]
CLR_SLAB  = [0.35, 0.35, 0.40, 1.0]

# YCB potted meat can (010)
CAN_URDF  = os.path.join(_HERE, "ycb_objects", "ycb_assets",
                         "010_potted_meat_can.urdf")
CAN_SCALE = 0.1   # mesh has scale="10 10 10" baked in; 0.1 restores real size

# Gripper depth from EE frame origin to fingertip centre
GRIPPER_DEPTH = 0.105
GRASP_OFFSET_Z = GRIPPER_DEPTH + 0.06   # 0.165 m  (same as drill demo)

PATH_FILE = os.path.join(_HERE, "results", "UR10_pick_place_drill_path.txt")


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
# Helpers
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
# Path loader
# ═══════════════════════════════════════════════════════════════════════

def load_path(filepath):
    """Parse the path text file saved by UR10_pick_place_drill.py."""
    waypoints = []
    with open(filepath, "r") as f:
        in_data = False
        for line in f:
            line = line.strip()
            if line.startswith("---"):
                in_data = not in_data
                continue
            if in_data and line:
                parts = line.split()
                if len(parts) == 7:          # index + 6 joints
                    try:
                        waypoints.append([float(v) for v in parts[1:]])
                    except ValueError:
                        pass
    return [np.array(w) for w in waypoints]


# ═══════════════════════════════════════════════════════════════════════
# Attach the potted-meat can between the gripper fingers
# ═══════════════════════════════════════════════════════════════════════

def attach_can_to_gripper(env, q_grasp):
    """Spawn the potted-meat can in the gripper and weld it to both fingertips."""
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

    # Place can so its centre sits at the gripper grasp point.
    can_offset_local = [GRASP_OFFSET_Z - 0.01, 0.0, 0.0]
    can_pos_w, _ = p.multiplyTransforms(
        ee_pos_w.tolist(), ee_orn_w,
        can_offset_local, [0, 0, 0, 1],
    )
    can_pos_w = [can_pos_w[0],
                 can_pos_w[1] + 0.01,   # 1 cm toward wall (3 cm − 2 cm away from box)
                 can_pos_w[2]]

    # Rotate -90° around world Z axis.
    can_orn_w = list(p.getQuaternionFromEuler([0.0, 0.0, -np.pi / 2]))

    print(f"[CAN] Spawning potted-meat can at "
          f"[{can_pos_w[0]:.3f}, {can_pos_w[1]:.3f}, {can_pos_w[2]:.3f}]")

    can_id = p.loadURDF(
        CAN_URDF,
        basePosition=list(can_pos_w),
        baseOrientation=list(can_orn_w),
        globalScaling=CAN_SCALE,
        useFixedBase=False,
        physicsClientId=cid,
    )

    # Disable robot-vs-can self-collision.
    n_joints = p.getNumJoints(rid, physicsClientId=cid)
    for link_idx in range(-1, n_joints):
        p.setCollisionFilterPair(rid, can_id, link_idx, -1, 0,
                                 physicsClientId=cid)

    # Weld to each fingertip.
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
    _finger_constraint(
        "robotiq_85_right_finger_tip_joint", env.ee_link_idx)

    # Register as attached body for planner collision checks.
    env.attach_body(can_id, left_idx, can_in_ft, can_orn_in_ft)

    for _ in range(20):
        p.stepSimulation(physicsClientId=cid)

    print("[CAN] Fixed to left & right finger-tip links via JOINT_FIXED")
    return can_id, can_in_ft, can_orn_in_ft


# ═══════════════════════════════════════════════════════════════════════
# Path animation
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
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--headless", action="store_true",
                        help="Run without GUI")
    args = parser.parse_args()

    wall_obstacles = build_wall_obstacles()

    mode = 'headless' if args.headless else 'GUI'
    print(f"[ENV] Loading PyBullet ({mode}) ...")
    env = UR10eRobotiqEnv(
        gui=not args.headless,
        obstacles=wall_obstacles,
        base_position=[0.0, 0.0, ROBOT_BASE_Z],
        base_orientation=p.getQuaternionFromEuler([0, 0, np.pi]),
    )
    cid = env.physics_client
    add_scenery(cid)

    # ── Floor & environment colours ───────────────────────────────
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

    # ── Load saved path ───────────────────────────────────────────
    print(f"[PATH] Loading path from {PATH_FILE} ...")
    if not os.path.exists(PATH_FILE):
        print(f"[ERROR] Path file not found: {PATH_FILE}")
        env.disconnect()
        return

    path = load_path(PATH_FILE)
    if not path:
        print("[ERROR] No waypoints parsed from path file.")
        env.disconnect()
        return

    print(f"[PATH] Loaded {len(path)} waypoints.")

    q_start = path[0]
    q_goal  = path[-1]

    pos_s, _ = env.get_ee_pose(q_start)
    pos_g, _ = env.get_ee_pose(q_goal)

    print(f"[CFG] Start q = [{', '.join(f'{v:.4f}' for v in q_start)}]")
    print(f"      EE    = [{pos_s[0]:.3f}, {pos_s[1]:.3f}, {pos_s[2]:.3f}]")
    print(f"[CFG] Goal  q = [{', '.join(f'{v:.4f}' for v in q_goal)}]")
    print(f"      EE    = [{pos_g[0]:.3f}, {pos_g[1]:.3f}, {pos_g[2]:.3f}]")

    # ── Attach potted-meat can to gripper at start pose ───────────
    can_id, can_in_ft, can_orn_in_ft = attach_can_to_gripper(env, q_start)

    # ── Camera & markers ──────────────────────────────────────────
    if not args.headless:
        p.resetDebugVisualizerCamera(
            cameraDistance=1.60,
            cameraYaw=-114.60,
            cameraPitch=-40.20,
            cameraTargetPosition=[-0.449, -0.041, 0.892],
            physicsClientId=cid,
        )
        p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 0, physicsClientId=cid)

        # Blue start sphere, green goal sphere
        for pos, clr in [(pos_s, [0, 0, 1, 1]), (pos_g, [0, 0.8, 0, 1])]:
            vis = p.createVisualShape(p.GEOM_SPHERE, radius=0.025,
                                      rgbaColor=clr, physicsClientId=cid)
            p.createMultiBody(baseMass=0, baseVisualShapeIndex=vis,
                              basePosition=pos.tolist(), physicsClientId=cid)

    print()
    print("=" * 62)
    print("  Soup Box Demo — Replay with Potted-Meat Can")
    print("=" * 62)
    print(f"  Path file   : {os.path.basename(PATH_FILE)}")
    print(f"  Waypoints   : {len(path)}")
    print(f"  Wall centre : [{WALL_X:.3f}, {WALL_Y:.3f}, {WALL_Z_MID:.3f}]")
    print(f"  Wall size   : {WALL_L:.2f}(x) x {WALL_W:.2f}(y) x {WALL_H:.2f}(z)")
    print(f"  Start EE    : [{pos_s[0]:.3f}, {pos_s[1]:.3f}, {pos_s[2]:.3f}]")
    print(f"  Goal  EE    : [{pos_g[0]:.3f}, {pos_g[1]:.3f}, {pos_g[2]:.3f}]")
    print(f"  Object      : YCB 010 potted-meat can (welded to fingers)")
    print("=" * 62)
    print()

    # Interpolate for smooth playback
    path_fine = interpolate_path(path, max_step=0.02)
    print(f"[PATH] Interpolated to {len(path_fine)} waypoints for animation.")

    if args.headless:
        env.disconnect()
        return

    finger_link_idx = env._joint_name_to_idx.get(
        "robotiq_85_left_finger_tip_joint", env.ee_link_idx
    )

    # ── Loop replay until window is closed ────────────────────────
    print("[ANIM] Replaying (close PyBullet window to exit) ...")
    try:
        while p.isConnected(physicsClientId=cid):
            # Reset to start and re-sync can position
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

            # Clear previous trail
            p.removeAllUserDebugItems(physicsClientId=cid)

            visualize_path_with_can(env, path_fine, delay=0.02, trail=True)
            time.sleep(1.0)
    except (KeyboardInterrupt, Exception):
        pass

    print("Shutting down ...")
    env.disconnect()


if __name__ == "__main__":
    main()
