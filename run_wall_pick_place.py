#!/usr/bin/env python3
"""
run_wall_pick_place.py — Same wall environment as run_wall_env.py, but with a
YCB tomato soup can grasped between the gripper fingers (fixed joint).

The can is attached to the EE at the start pose and carried through the
planned path around/over the wall to the goal pose.

Usage:
    python3 run_wall_pick_place.py
"""

import sys
import os
import time
import numpy as np
import pybullet as p

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from manipulator_env.pybullet_env import UR10eRobotiqEnv
from manipulator_env.planner_interface import plan_and_execute, interpolate_path
from rit_star.metric import DiagonalAnisotropicMetric

# Fast inertia-based diagonal metric
_UR10E_INERTIAS = np.array([7.778, 12.93, 3.87, 1.96, 1.96, 0.202])
_UR10E_WEIGHTS = (_UR10E_INERTIAS / _UR10E_INERTIAS.max()).tolist()

# =====================================================================
# Physical constants (metres) — identical to run_wall_env.py
# =====================================================================

TABLE_SURFACE_Z = 0.75
SLAB_THICKNESS  = 0.01990
ROBOT_BASE_Z    = TABLE_SURFACE_Z + SLAB_THICKNESS  # 0.76990

TABLE_LEN = 1.50
TABLE_WID = 1.40
TABLE_THK = 0.05
TABLE_CX  = 0.35
TABLE_CY  = 0.0

# Wall — exact same geometry as run_wall_env.py
_WALL_X_START = 0.35                            # forward of robot base
_WALL_X_END   = TABLE_CX + TABLE_LEN / 2        # far edge of table (1.10)
WALL_L     = _WALL_X_END - _WALL_X_START         # length along x-axis
WALL_X     = (_WALL_X_START + _WALL_X_END) / 2   # x-centre
WALL_Y     = 0.0
WALL_W     = 0.03
WALL_H     = 0.55
WALL_Z_BOT = ROBOT_BASE_Z
WALL_Z_MID = WALL_Z_BOT + WALL_H / 2

CLR_WALL  = [0.35, 0.45, 0.60, 0.85]
CLR_TABLE = [0.60, 0.60, 0.60, 1.0]
CLR_LEGS  = [0.25, 0.25, 0.28, 1.0]
CLR_SLAB  = [0.35, 0.35, 0.40, 1.0]

# YCB tomato soup can
_HERE = os.path.dirname(os.path.abspath(__file__))
CAN_URDF = os.path.join(_HERE, "ycb_objects", "ycb_assets",
                         "005_tomato_soup_can.urdf")
CAN_SCALE = 0.1   # URDF has internal scale=10; 0.1 gives real size
CAN_HEIGHT = 0.11  # approximate height of the can at real scale

# Vertical offset from EE frame to fingertip centre (gripper pointing down)
GRIPPER_DEPTH = 0.105


# =====================================================================
# Obstacles
# =====================================================================

def build_wall_obstacles():
    return [
        {"type": "box", "color": CLR_TABLE,
         "pos": [TABLE_CX, TABLE_CY, TABLE_SURFACE_Z - TABLE_THK / 2],
         "half_extents": [TABLE_LEN / 2, TABLE_WID / 2, TABLE_THK / 2]},
        {"type": "box", "color": CLR_WALL,
         "pos": [WALL_X, WALL_Y, WALL_Z_MID],
         "half_extents": [WALL_L / 2, WALL_W / 2, WALL_H / 2]},
    ]


# =====================================================================
# Visual scenery
# =====================================================================

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


# =====================================================================
# Attach YCB can between gripper fingers with a fixed joint
# =====================================================================

def attach_can_to_gripper(env, q_grasp):
    """Load the YCB tomato soup can and rigidly attach it between the
    gripper fingers at configuration *q_grasp* using a fixed constraint.

    Returns (can_id, constraint_id).
    """
    cid = env.physics_client
    rid = env.robot_id

    # Move robot to grasp config
    env.set_joint_positions(q_grasp)
    p.stepSimulation(physicsClientId=cid)

    # Get EE world pose
    ee_state = p.getLinkState(rid, env.ee_link_idx,
                              computeForwardKinematics=True,
                              physicsClientId=cid)
    ee_pos_w = np.array(ee_state[4])
    ee_orn_w = list(ee_state[5])

    # Can centre: offset along EE local x (which maps to world -z for the
    # top-down orientation euler=[0, pi/2, 0]), placing it between the fingers.
    can_offset_local = [GRIPPER_DEPTH + 0.05, 0.0, 0.0]
    # Counter-rotate so the can stays upright (vertical) in world frame.
    # The EE is rotated by euler=[0, pi/2, 0]; applying the inverse in the
    # local frame cancels it out:  EE_orn * local_orn = identity.
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

    # Disable collisions between can and all robot links
    n_joints = p.getNumJoints(rid, physicsClientId=cid)
    for link_idx in range(-1, n_joints):
        p.setCollisionFilterPair(rid, can_id, link_idx, -1, 0,
                                 physicsClientId=cid)

    # Disable can vs obstacles and ground plane
    for obs_id in env.obstacle_ids:
        p.setCollisionFilterPair(can_id, obs_id, -1, -1, 0,
                                 physicsClientId=cid)
    p.setCollisionFilterPair(can_id, env.plane_id, -1, -1, 0,
                             physicsClientId=cid)

    # Fixed constraint: EE -> can (can stays upright via local orientation)
    constraint_id = p.createConstraint(
        parentBodyUniqueId=rid,
        parentLinkIndex=env.ee_link_idx,
        childBodyUniqueId=can_id,
        childLinkIndex=-1,
        jointType=p.JOINT_FIXED,
        jointAxis=[0, 0, 0],
        parentFramePosition=can_offset_local,
        childFramePosition=[0, 0, 0],
        parentFrameOrientation=can_local_orn,
        childFrameOrientation=[0, 0, 0, 1],
        physicsClientId=cid,
    )
    p.changeConstraint(constraint_id, maxForce=10000, physicsClientId=cid)

    for _ in range(20):
        p.stepSimulation(physicsClientId=cid)

    print(f"[CAN] Fixed constraint created (id={constraint_id})")
    return can_id, constraint_id


# =====================================================================
# Path animation with attached can
# =====================================================================

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


# =====================================================================
# Main
# =====================================================================

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

    # -- Infinite very light grey floor --
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

    # -- Robot colours --
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

    # -- Hardcoded configs (same as run_wall_env.py) --
    # Collision-free with WALL_H=0.55 & wall x-start=0.35.
    # Top-down grasp orientation (euler=[0, pi/2, 0], fingers along -z).
    # Start: EE at [0.50,  0.40, 1.00] (+y side)
    # Goal:  EE at [0.50, -0.40, 1.00] (-y side, mirrored J0)
    q_start = np.array([ 0.3992, -1.0605, 1.7617, 0.8696, 1.5708, -1.1716])
    q_goal  = np.array([-0.9501, -1.0604, 1.7614, 0.8698, 1.5708, -2.5209])

    assert env.is_collision_free(q_start), "Start config is in collision!"
    assert env.is_collision_free(q_goal),  "Goal config is in collision!"

    pos_s, _ = env.get_ee_pose(q_start)
    pos_g, _ = env.get_ee_pose(q_goal)

    print(f"[CFG] Start q = [{', '.join(f'{v:.4f}' for v in q_start)}]")
    print(f"      EE    = [{pos_s[0]:.3f}, {pos_s[1]:.3f}, {pos_s[2]:.3f}]")
    print(f"[CFG] Goal  q = [{', '.join(f'{v:.4f}' for v in q_goal)}]")
    print(f"      EE    = [{pos_g[0]:.3f}, {pos_g[1]:.3f}, {pos_g[2]:.3f}]")

    # -- Attach YCB tomato soup can between gripper fingers --
    can_id, constraint_id = attach_can_to_gripper(env, q_start)

    # -- Camera & markers --
    p.resetDebugVisualizerCamera(
        cameraDistance=1.60,
        cameraYaw=-114.60,
        cameraPitch=-40.20,
        cameraTargetPosition=[-0.449, -0.041, 0.892],
        physicsClientId=cid,
    )
    p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 0, physicsClientId=cid)

    for tgt, clr in [(pos_s.tolist(), [0, 0, 1, 1]),
                      (pos_g.tolist(), [1, 0, 0, 1])]:
        vis = p.createVisualShape(p.GEOM_SPHERE, radius=0.03,
                                  rgbaColor=clr, physicsClientId=cid)
        p.createMultiBody(baseMass=0, baseVisualShapeIndex=vis,
                          basePosition=tgt, physicsClientId=cid)

    print()
    print("=" * 62)
    print("  Wall Pick-and-Place (Top Grasp) — RIT* Planning")
    print("=" * 62)
    print(f"  Wall centre : [{WALL_X:.3f}, {WALL_Y:.3f}, {WALL_Z_MID:.3f}]")
    print(f"  Wall size   : {WALL_L:.2f}(x) x {WALL_W:.2f}(y) x {WALL_H:.2f}(z)")
    print(f"  Start EE    : [{pos_s[0]:.3f}, {pos_s[1]:.3f}, {pos_s[2]:.3f}]  (+y side)")
    print(f"  Goal  EE    : [{pos_g[0]:.3f}, {pos_g[1]:.3f}, {pos_g[2]:.3f}]  (−y side)")
    print(f"  Start cfg   : [{', '.join(f'{v:.3f}' for v in q_start)}]")
    print(f"  Goal  cfg   : [{', '.join(f'{v:.3f}' for v in q_goal)}]")
    print(f"  YCB can     : attached to gripper (fixed joint)")
    print("=" * 62)
    print()

    # -- Plan (same parameters as run_wall_env.py) --
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

        with open("results/wall_pick_place_world_state.txt", "w") as f:
            f.write("=" * 62 + "\n")
            f.write("  Wall Pick-and-Place (Top Grasp) — World State\n")
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
            f.write("\n--- Start Configuration (+y side) ---\n")
            f.write(f"  q_start (rad) : [{', '.join(f'{v:.6f}' for v in q_start)}]\n")
            f.write(f"  EE position   : [{pos_s[0]:.5f}, {pos_s[1]:.5f}, {pos_s[2]:.5f}]\n\n")
            f.write("--- Goal Configuration (−y side) ---\n")
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
        print("[FILE] Saved results/wall_pick_place_world_state.txt")

        with open("results/wall_pick_place_path.txt", "w") as f:
            f.write("=" * 62 + "\n")
            f.write("  Wall Pick-and-Place — Complete Path (Joint Configurations)\n")
            f.write("=" * 62 + "\n")
            f.write(f"  Waypoints : {len(path)}\n")
            f.write(f"  Path cost : {cost:.6f}\n")
            f.write(f"  DOF       : 6\n\n")
            f.write(f"  q_start : [{', '.join(f'{v:+.6f}' for v in q_start)}]\n")
            f.write(f"  q_goal  : [{', '.join(f'{v:+.6f}' for v in q_goal)}]\n\n")
            f.write("  Each row: joint_1  joint_2  joint_3  joint_4  joint_5  joint_6  (radians)\n")
            f.write("-" * 62 + "\n")
            for i, q in enumerate(path):
                f.write(f"  {i:4d}  " +
                        "  ".join(f"{v:+10.6f}" for v in q) + "\n")
            f.write("-" * 62 + "\n")
        print("[FILE] Saved results/wall_pick_place_path.txt")
    else:
        print("\n[RESULT] No path found.")

    # ── Loop path animation until window is closed ────────────
    if path:
        print("\n[LOOP] Replaying path (close PyBullet window to exit) ...")
        try:
            while p.isConnected(physicsClientId=cid):
                env.set_joint_positions(q_start)
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
