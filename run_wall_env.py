#!/usr/bin/env python3
"""
run_wall_env.py — Wall environment: UR10e must plan around a tall wall.

Setup:
  - UR10e mounted on table (same physical setup as other demos)
  - A tall wall placed in front of the robot, extending along the world y-axis
  - Start: arm reaching to +y side of the wall
  - Goal:  arm reaching to −y side of the wall
  - The wall separates the two sides, forcing the planner to swing around/over

Usage:
    python run_wall_env.py
"""

import sys
import os
import time
import numpy as np
import pybullet as p

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from manipulator_env.pybullet_env import UR10eRobotiqEnv
from manipulator_env.planner_interface import plan_and_execute
from rit_star.metric import DiagonalAnisotropicMetric

# Fast inertia-based diagonal metric
_UR10E_INERTIAS = np.array([7.369, 13.051, 3.989, 2.1, 1.98, 0.615])
_UR10E_WEIGHTS = (_UR10E_INERTIAS / _UR10E_INERTIAS.max()).tolist()

# ═══════════════════════════════════════════════════════════════════════
# Physical constants (metres)
# ═══════════════════════════════════════════════════════════════════════

TABLE_SURFACE_Z = 0.75
SLAB_THICKNESS  = 0.01990
ROBOT_BASE_Z    = TABLE_SURFACE_Z + SLAB_THICKNESS  # 0.76990

TABLE_LEN = 1.50
TABLE_WID = 1.40
TABLE_THK = 0.05
TABLE_CX  = 0.35
TABLE_CY  = 0.0

# Wall: vertical barrier in front of robot, long axis along world x
# Starts just in front of robot base (x≈0.15) to far end of table (x=1.10)
_WALL_X_START = 0.15                            # just in front of robot
_WALL_X_END   = TABLE_CX + TABLE_LEN / 2        # far edge of table (1.10)
WALL_L     = _WALL_X_END - _WALL_X_START         # length along x-axis
WALL_X     = (_WALL_X_START + _WALL_X_END) / 2   # x-centre
WALL_Y     = 0.0     # y-centre: directly in front of robot
WALL_W     = 0.03    # thickness (y-extent)
WALL_H     = 0.55    # height above table surface
WALL_Z_BOT = ROBOT_BASE_Z          # base of wall = table surface
WALL_Z_MID = WALL_Z_BOT + WALL_H / 2

CLR_WALL  = [0.35, 0.45, 0.60, 0.85]
CLR_TABLE = [0.60, 0.60, 0.60, 1.0]
CLR_LEGS  = [0.25, 0.25, 0.28, 1.0]
CLR_SLAB  = [0.35, 0.35, 0.40, 1.0]


# ═══════════════════════════════════════════════════════════════════════
# Obstacles
# ═══════════════════════════════════════════════════════════════════════

def build_wall_obstacles():
    """Return obstacle list: table surface + wall."""
    return [
        # Table surface
        {"type": "box", "color": CLR_TABLE,
         "pos": [TABLE_CX, TABLE_CY, TABLE_SURFACE_Z - TABLE_THK / 2],
         "half_extents": [TABLE_LEN / 2, TABLE_WID / 2, TABLE_THK / 2]},

        # Wall — along world x-axis (rotated 90° about z)
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
    """Add table legs and mounting slab (visual only)."""
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
# IK solver with side-biased seeds + random fallback
# ═══════════════════════════════════════════════════════════════════════

def find_ik(env, target_pos, side_label="", n_random=200):
    """Position-only IK with deterministic seeds + random sampling.

    Returns the first collision-free solution with EE error < 5 cm.
    """
    cid = env.physics_client
    n_movable = len(env._all_joint_indices)
    ee_target = np.array(target_pos, dtype=float)

    # Deterministic seeds — cover many base-rotation angles
    base_seed = [-np.pi / 2, np.pi / 2, -np.pi / 2, -np.pi / 2, 0.0]
    j0_angles = np.linspace(-np.pi, np.pi, 13)  # 13 base angles
    seeds = [[j0] + base_seed for j0 in j0_angles]
    # Also add some with different elbow configs
    for j0 in [-1.0, -0.5, 0.0, 0.5, 1.0]:
        seeds.append([j0, -1.0, 1.5, -2.0, -1.57, 0.0])
        seeds.append([j0, -0.8, 0.8, -1.5, -1.57, 0.0])
        seeds.append([j0, -1.5, 1.0, -1.0, -1.57, 0.0])

    # Add random seeds
    rng = np.random.RandomState(42)
    for _ in range(n_random):
        q_rand = rng.uniform(-np.pi, np.pi, size=6).tolist()
        seeds.append(q_rand)

    best_q = None
    best_err = float('inf')

    for si, seed in enumerate(seeds):
        env.set_joint_positions(np.array(seed[:6]))

        rest = list(seed[:6]) + [0.0] * (n_movable - 6)
        q_ik = p.calculateInverseKinematics(
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
        q_arm = np.array(q_ik[:6])

        ee_actual, _ = env.get_ee_pose(q_arm)
        err = np.linalg.norm(ee_target - ee_actual)

        if env.is_collision_free(q_arm) and err < 0.05:
            print(f"[IK]  Collision-free {side_label} found  (seed {si}, err={err:.4f})")
            print(f"      q   = [{', '.join(f'{v:.4f}' for v in q_arm)}]")
            print(f"      EE  = [{ee_actual[0]:.3f}, {ee_actual[1]:.3f}, {ee_actual[2]:.3f}]")
            return q_arm

        if env.is_collision_free(q_arm) and err < best_err:
            best_err = err
            best_q = q_arm

    if best_q is not None:
        print(f"[IK]  WARNING: best collision-free {side_label} has err={best_err:.4f}")
        return best_q

    print(f"[IK]  ERROR: No collision-free {side_label} IK found!")
    return None


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
    )
    cid = env.physics_client
    add_scenery(cid)

    # ── Realistic lab floor ────────────────────────────────────────
    # Create a large flat box as the lab floor with a light grey colour
    floor_he = [3.0, 3.0, 0.01]
    floor_col = p.createCollisionShape(p.GEOM_BOX, halfExtents=floor_he,
                                       physicsClientId=cid)
    floor_vis = p.createVisualShape(p.GEOM_BOX, halfExtents=floor_he,
                                    rgbaColor=[0.85, 0.85, 0.82, 1.0],
                                    physicsClientId=cid)
    p.createMultiBody(baseMass=0, baseCollisionShapeIndex=-1,
                      baseVisualShapeIndex=floor_vis,
                      basePosition=[0.0, 0.0, -0.01],
                      physicsClientId=cid)
    # Change the default checkerboard plane to a neutral colour
    p.changeVisualShape(env.plane_id, -1,
                        rgbaColor=[0.75, 0.75, 0.72, 1.0],
                        physicsClientId=cid)

    # ── Realistic UR10e + Robotiq colours ─────────────────────────
    # UR10e: light silver body links, dark charcoal joint housings
    # Robotiq 85: dark grey body, black fingers
    UR_SILVER  = [0.75, 0.75, 0.75, 1.0]   # silver aluminium body
    UR_DARK    = [0.22, 0.22, 0.22, 1.0]    # dark joint housings
    UR_BLUE    = [0.00, 0.34, 0.68, 1.0]    # UR blue accent (caps)
    RQ_DARK    = [0.15, 0.15, 0.15, 1.0]    # Robotiq dark grey
    RQ_BLACK   = [0.08, 0.08, 0.08, 1.0]    # Robotiq finger tips

    rid = env.robot_id
    n = p.getNumJoints(rid, physicsClientId=cid)
    link_name_to_idx = {}
    for i in range(n):
        info = p.getJointInfo(rid, i, physicsClientId=cid)
        link_name_to_idx[info[12].decode("utf-8")] = i

    # Base link (link index -1)
    p.changeVisualShape(rid, -1, rgbaColor=UR_DARK, physicsClientId=cid)

    # UR10e arm links
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

    # Robotiq gripper links
    for lname, lidx in link_name_to_idx.items():
        if "robotiq" in lname:
            if "finger_tip" in lname:
                p.changeVisualShape(rid, lidx, rgbaColor=RQ_BLACK,
                                    physicsClientId=cid)
            else:
                p.changeVisualShape(rid, lidx, rgbaColor=RQ_DARK,
                                    physicsClientId=cid)

    # ── Target points ──────────────────────────────────────────────
    # # [OLD] Left/right of wall (along x-axis):
    # x_offset = WALL_L / 2 + 0.10     # 10 cm past each end of the wall
    # target_y = WALL_Y                 # same y as wall centre
    # target_z = ROBOT_BASE_Z + 0.30   # comfortable height above table
    # start_target = [WALL_X - x_offset, target_y, target_z]   # left (−x) side
    # goal_target  = [WALL_X + x_offset, target_y, target_z]   # right (+x) side

    # [NEW] Front/back of wall (along y-axis):
    # Wall spans full table length in x, offset in y from robot.
    # Targets on opposite y-sides of the wall.
    target_x = 0.50                   # in front of robot, within wall x-span
    target_z = ROBOT_BASE_Z + 0.35   # comfortable height above table
    y_near = WALL_Y - 0.25            # −y side (robot side)
    y_far  = WALL_Y + 0.25            # +y side (far side)

    start_target = [target_x, y_far,  target_z]   # +y side (far from robot)
    goal_target  = [target_x, y_near, target_z]   # −y side (robot side)

    print(f"[IK]  Computing start IK  target={[round(v,3) for v in start_target]}")
    q_start = find_ik(env, start_target, side_label="START (+y)")
    print(f"[IK]  Computing goal  IK  target={[round(v,3) for v in goal_target]}")
    q_goal  = find_ik(env, goal_target,  side_label="GOAL  (−y)")

    if q_start is None or q_goal is None:
        print("[FATAL] Could not find collision-free IK for both sides.")
        env.disconnect()
        return

    assert env.is_collision_free(q_start), "Start config is in collision!"
    assert env.is_collision_free(q_goal),  "Goal config is in collision!"

    pos_s, _ = env.get_ee_pose(q_start)
    pos_g, _ = env.get_ee_pose(q_goal)

    # ── Camera & markers ──────────────────────────────────────────
    p.resetDebugVisualizerCamera(
        cameraDistance=2.2,
        cameraYaw=45,
        cameraPitch=-25,
        cameraTargetPosition=[WALL_X, 0.0, 1.0],
        physicsClientId=cid,
    )
    p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 0, physicsClientId=cid)

    for tgt, clr in [(start_target, [0, 0, 1, 1]), (goal_target, [1, 0, 0, 1])]:
        vis = p.createVisualShape(p.GEOM_SPHERE, radius=0.03,
                                  rgbaColor=clr, physicsClientId=cid)
        p.createMultiBody(baseMass=0, baseVisualShapeIndex=vis,
                          basePosition=tgt, physicsClientId=cid)

    print("=" * 62)
    print("  Wall Environment — RIT* Planning")
    print("=" * 62)
    print(f"  Wall centre : [{WALL_X:.3f}, {WALL_Y:.3f}, {WALL_Z_MID:.3f}]")
    print(f"  Wall size   : {WALL_L:.2f}(long-x) x {WALL_W:.2f}(thick-y) x {WALL_H:.2f}(tall)")
    print(f"  Start EE    : [{pos_s[0]:.3f}, {pos_s[1]:.3f}, {pos_s[2]:.3f}]  (+y side)")
    print(f"  Goal  EE    : [{pos_g[0]:.3f}, {pos_g[1]:.3f}, {pos_g[2]:.3f}]  (−y side)")
    print(f"  Start cfg   : [{', '.join(f'{v:.3f}' for v in q_start)}]")
    print(f"  Goal  cfg   : [{', '.join(f'{v:.3f}' for v in q_goal)}]")
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
        animate=True,
        animate_delay=0.02,
    )

    if path:
        print(f"\n[RESULT] Path found — cost: {cost:.4f}, waypoints: {len(path)}")

        os.makedirs("results", exist_ok=True)

        with open("results/wall_env_world_state.txt", "w") as f:
            f.write("=" * 62 + "\n")
            f.write("  Wall Environment — World State\n")
            f.write("=" * 62 + "\n\n")
            f.write("--- Robot ---\n")
            f.write(f"  Base position : [0.000, 0.000, {ROBOT_BASE_Z:.5f}]\n\n")
            f.write("--- Table ---\n")
            f.write(f"  Surface Z     : {TABLE_SURFACE_Z:.5f} m\n")
            f.write(f"  Centre (x,y)  : [{TABLE_CX:.3f}, {TABLE_CY:.3f}]\n")
            f.write(f"  Dimensions    : {TABLE_LEN:.2f} x {TABLE_WID:.2f} x {TABLE_THK:.2f} m\n\n")
            f.write("--- Wall (along world x-axis) ---\n")
            f.write(f"  Centre        : [{WALL_X:.3f}, {WALL_Y:.3f}, {WALL_Z_MID:.3f}]\n")
            f.write(f"  Thickness (x) : {WALL_W:.3f} m\n")
            f.write(f"  Length (y)    : {WALL_L:.3f} m\n")
            f.write(f"  Height (z)    : {WALL_H:.3f} m\n\n")
            f.write("--- Obstacles ---\n")
            for i, obs in enumerate(wall_obstacles):
                f.write(f"  [{i}] {obs['type']}  pos={[round(v,5) for v in obs['pos']]}  "
                        f"he={[round(v,5) for v in obs['half_extents']]}\n")
            f.write(f"\n--- Start (+y side) ---\n")
            f.write(f"  q  : [{', '.join(f'{v:.6f}' for v in q_start)}]\n")
            f.write(f"  EE : [{pos_s[0]:.5f}, {pos_s[1]:.5f}, {pos_s[2]:.5f}]\n\n")
            f.write(f"--- Goal (−y side) ---\n")
            f.write(f"  q  : [{', '.join(f'{v:.6f}' for v in q_goal)}]\n")
            f.write(f"  EE : [{pos_g[0]:.5f}, {pos_g[1]:.5f}, {pos_g[2]:.5f}]\n\n")
            f.write(f"--- Path ---\n")
            f.write(f"  Cost      : {cost:.6f}\n")
            f.write(f"  Waypoints : {len(path)}\n")
        print("[FILE] Saved results/wall_env_world_state.txt")

        with open("results/wall_env_path.txt", "w") as f:
            f.write("=" * 62 + "\n")
            f.write("  Wall Environment — Path\n")
            f.write("=" * 62 + "\n")
            f.write(f"  Waypoints : {len(path)}   Cost : {cost:.6f}\n\n")
            for i, q in enumerate(path):
                f.write(f"  {i:4d}  " + "  ".join(f"{v:+10.6f}" for v in q) + "\n")
        print("[FILE] Saved results/wall_env_path.txt")
    else:
        print("\n[RESULT] No path found.")

    print("\n  Press Ctrl+C to exit.")
    try:
        while True:
            p.stepSimulation(physicsClientId=cid)
            time.sleep(1 / 240)
    except KeyboardInterrupt:
        print("\nShutting down ...")
    finally:
        env.disconnect()


if __name__ == "__main__":
    main()
