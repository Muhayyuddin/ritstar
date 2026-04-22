#!/usr/bin/env python3
"""
run_wall_env.py — Wall environment: UR10e must plan around a tall wall.

Setup:
  - UR10e mounted on table (same physical setup as other demos)
  - A tall wall placed in front of the robot, extending along the world y-axis
  - Start: arm at home configuration
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
from manipulator_env.planner_interface import plan_and_execute, interpolate_path
from rit_star.metric import DiagonalAnisotropicMetric

# Fast inertia-based diagonal metric
_UR10E_INERTIAS = np.array([7.369, 13.051, 3.989, 2.1, 1.98, 0.615])
_UR10E_WEIGHTS = (_UR10E_INERTIAS / _UR10E_INERTIAS.max()).tolist()
REAL_SETUP_Q_START = np.array([-np.pi / 2, -np.pi / 2, np.pi / 2,
                               -np.pi / 2, -np.pi / 2, 0.0])
TOP_DOWN_ORN = list(p.getQuaternionFromEuler([0, np.pi / 2, 0]))
GRIPPER_DEPTH = 0.105
GRASP_OFFSET_Z = GRIPPER_DEPTH + 0.06

# ═══════════════════════════════════════════════════════════════════════
# Physical constants (metres)
# ═══════════════════════════════════════════════════════════════════════

TABLE_SURFACE_Z = 0.75
SLAB_THICKNESS  = 0.01990
ROBOT_BASE_Z    = TABLE_SURFACE_Z + SLAB_THICKNESS  # 0.76990

TABLE_LEN = 1.00
TABLE_WID = 1.50
TABLE_THK = 0.05
TABLE_CX  = -0.29
TABLE_CY  = -0.07

# Wall: vertical barrier in front of robot, long axis along world x
WALL_L     = 0.54    # length along x-axis (54 cm)
WALL_X     = -0.60              # absolute wall centre x
WALL_Y     = 0.00               # absolute wall centre y
WALL_W     = 0.182   # thickness / depth (y-extent, 18.2 cm)
WALL_H     = 0.50    # height above table surface (50 cm)
WALL_Z_BOT = ROBOT_BASE_Z          # base of wall = table surface
WALL_Z_MID = WALL_Z_BOT + WALL_H / 2

CLR_WALL  = [0.78, 0.64, 0.46, 0.85]  # kraft / Amazon-box tan (slightly transparent)
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

def find_ik(env, target_pos, side_label="", n_random=200,
            desired_orn=None, pos_tol=0.05):
    """IK with deterministic seeds + random sampling.

    If desired_orn is provided, prefer solutions matching that orientation.
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
    candidates = []

    use_orientation = desired_orn is not None
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
        if use_orientation:
            ik_kwargs["targetOrientation"] = list(desired_orn)

        q_ik = p.calculateInverseKinematics(**ik_kwargs)
        q_arm = np.array(q_ik[:6])

        ee_actual, ee_orn = env.get_ee_pose(q_arm)
        err = np.linalg.norm(ee_target - ee_actual)

        if env.is_collision_free(q_arm) and err < pos_tol:
            if use_orientation:
                orn_dot = abs(np.dot(np.array(desired_orn), np.array(ee_orn)))
                candidates.append((q_arm, err, orn_dot, si, ee_actual))
            else:
                print(f"[IK]  Collision-free {side_label} found  (seed {si}, err={err:.4f})")
                print(f"      q   = [{', '.join(f'{v:.4f}' for v in q_arm)}]")
                print(f"      EE  = [{ee_actual[0]:.3f}, {ee_actual[1]:.3f}, {ee_actual[2]:.3f}]")
                return q_arm

        if env.is_collision_free(q_arm) and err < best_err:
            best_err = err
            best_q = q_arm

    if use_orientation and candidates:
        candidates.sort(key=lambda c: (-c[2], c[1]))
        best_q, best_err, best_orn_dot, best_si, best_ee = candidates[0]
        print(f"[IK]  Collision-free {side_label} found  "
              f"(seed {best_si}, err={best_err:.4f}, orn_dot={best_orn_dot:.4f})")
        print(f"      q   = [{', '.join(f'{v:.4f}' for v in best_q)}]")
        print(f"      EE  = [{best_ee[0]:.3f}, {best_ee[1]:.3f}, {best_ee[2]:.3f}]")
        return best_q

    if best_q is not None:
        print(f"[IK]  WARNING: best collision-free {side_label} has err={best_err:.4f}")
        return best_q

    print(f"[IK]  ERROR: No collision-free {side_label} IK found!")
    return None


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

    # ── Infinite dark grey floor ───────────────────────────────────
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

    # ── Realistic UR10e + Robotiq colours ─────────────────────────
    # UR10e: light silver body links, dark charcoal joint housings
    # Robotiq 85: dark grey body, black fingers
    UR_SILVER  = [0.35, 0.35, 0.35, 1.0]   # dark grey aluminium body
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

    # Place can and grasp target on the -y side of the wall in this frame.
    # Can shifted 10 cm toward the −y table edge so the goal pose here
    # matches the start pose of run_wall_carry.py exactly.
    side_clearance = WALL_W / 2 + 0.50
    can_x = WALL_X
    can_y = WALL_Y - side_clearance - 0.10   # −0.691
    can_z = TABLE_SURFACE_Z + 0.055          # half can height
    can_pos = [can_x, can_y, can_z]

    # Goal is a top grasp of the can with fingers pointing straight downward.
    # +1 cm lift so EE z = 0.980 — identical to run_wall_carry.py start.
    goal_target = [can_x, can_y, can_z + GRASP_OFFSET_Z + 0.01]
    print(f"[IK]  Computing goal IK   target={[round(v,3) for v in goal_target]}")
    q_goal = find_ik(env, goal_target, side_label="GOAL (-y)",
                     desired_orn=TOP_DOWN_ORN, pos_tol=0.02)

    if q_goal is None:
        print("[FATAL] Could not find collision-free goal IK.")
        env.disconnect()
        return

    # Start = goal with shoulder-pan (joint 1) flipped by π, so the arm
    # is the mirror image on the opposite (+y) side of the wall. Only
    # joint 1 needs to rotate to get from start to goal.
    q_start = q_goal.copy()
    q_start[0] = q_goal[0] + np.pi
    if q_start[0] > np.pi:
        q_start[0] -= 2 * np.pi
    if not env.is_collision_free(q_start):
        # fall back: try rotating the other way
        q_start = q_goal.copy()
        q_start[0] = q_goal[0] - np.pi
        if q_start[0] < -np.pi:
            q_start[0] += 2 * np.pi
        if not env.is_collision_free(q_start):
            print("[FATAL] Mirror start config is in collision.")
            env.disconnect()
            return
    print(f"[START] Using start configuration: [{', '.join(f'{v:.4f}' for v in q_start)}]")

    assert env.is_collision_free(q_start), "Start config is in collision!"
    assert env.is_collision_free(q_goal),  "Goal config is in collision!"

    pos_s, _ = env.get_ee_pose(q_start)
    pos_g, _ = env.get_ee_pose(q_goal)

    # ── Place YCB tomato soup can on the table ────────────────────
    CAN_URDF = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "ycb_objects", "ycb_assets",
                            "005_tomato_soup_can.urdf")
    can_id = p.loadURDF(
        CAN_URDF, basePosition=can_pos, baseOrientation=[0, 0, 0, 1],
        globalScaling=0.1, useFixedBase=True, physicsClientId=cid,
    )
    n_joints_r = p.getNumJoints(env.robot_id, physicsClientId=cid)
    for link_idx in range(-1, n_joints_r):
        p.setCollisionFilterPair(env.robot_id, can_id, link_idx, -1, 0,
                                 physicsClientId=cid)
    print(f"[OBJ] Placed tomato soup can at "
          f"[{can_x:.3f}, {can_y:.3f}, {can_z:.3f}]")

    # Fully open gripper
    if "finger_joint" in env._joint_name_to_idx:
        fj = env._joint_name_to_idx["finger_joint"]
        p.resetJointState(env.robot_id, fj, 0.0, physicsClientId=cid)

    # ── Camera & markers ──────────────────────────────────────────
    p.resetDebugVisualizerCamera(
        cameraDistance=1.60,
        cameraYaw=-114.60,
        cameraPitch=-40.20,
        cameraTargetPosition=[-0.449, -0.041, 0.892],
        physicsClientId=cid,
    )
    p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 0, physicsClientId=cid)

    # Blue start marker, red goal marker
    for tgt, clr in [(pos_s.tolist(), [0, 0, 1, 1]), (pos_g.tolist(), [1, 0, 0, 1])]:
        vis = p.createVisualShape(p.GEOM_SPHERE, radius=0.03,
                                  rgbaColor=clr, physicsClientId=cid)
        p.createMultiBody(baseMass=0, baseVisualShapeIndex=vis,
                          basePosition=tgt, physicsClientId=cid)

    print("=" * 62)
    print("  Wall Environment — RIT* Planning")
    print("=" * 62)
    print(f"  Wall centre : [{WALL_X:.3f}, {WALL_Y:.3f}, {WALL_Z_MID:.3f}]")
    print(f"  Wall size   : {WALL_L:.2f}(long-x) x {WALL_W:.2f}(thick-y) x {WALL_H:.2f}(tall)")
    print(f"  Can pos     : [{can_x:.3f}, {can_y:.3f}, {can_z:.3f}]  (−y side)")
    print(f"  Start EE    : [{pos_s[0]:.3f}, {pos_s[1]:.3f}, {pos_s[2]:.3f}]  (+y side)")
    print(f"  Goal  EE    : [{pos_g[0]:.3f}, {pos_g[1]:.3f}, {pos_g[2]:.3f}]  (−y side, grasp)")
    print(f"  Start cfg   : [{', '.join(f'{v:.3f}' for v in q_start)}]")
    print(f"  Goal  cfg   : [{', '.join(f'{v:.3f}' for v in q_goal)}]")
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
        animate_delay=0.02,
        planner_name=_args.planner,
        seed=_args.seed,
    )
    _elapsed = _time.time() - _t0

    if _args.save_results:
        append_demo_result_csv({
            'demo': 'UR10_grasp_can',
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

        os.makedirs("results", exist_ok=True)

        with open("results/UR10_grasp_can_world_state.txt", "w") as f:
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
            f.write(f"\n--- Start Configuration (+y side) ---\n")
            f.write(f"  q_start (rad) : [{', '.join(f'{v:.6f}' for v in q_start)}]\n")
            f.write(f"  EE position   : [{pos_s[0]:.5f}, {pos_s[1]:.5f}, {pos_s[2]:.5f}]\n\n")
            f.write(f"--- Goal Configuration (−y side) ---\n")
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
        print("[FILE] Saved results/UR10_grasp_can_world_state.txt")

        with open("results/UR10_grasp_can_path.txt", "w") as f:
            f.write("=" * 62 + "\n")
            f.write("  Wall Environment — Complete Path (Joint Configurations)\n")
            f.write("=" * 62 + "\n")
            f.write(f"  Waypoints : {len(path)}\n")
            f.write(f"  Path cost : {cost:.6f}\n")
            f.write(f"  DOF       : 6\n\n")
            f.write(f"  q_start : [{', '.join(f'{v:+.6f}' for v in q_start)}]\n")
            f.write(f"  q_goal  : [{', '.join(f'{v:+.6f}' for v in q_goal)}]\n\n")
            f.write("  Each row: joint_1  joint_2  joint_3  joint_4  joint_5  joint_6  (radians)\n")
            f.write("-" * 62 + "\n")
            for i, q in enumerate(path):
                f.write(f"  {i:4d}  " + "  ".join(f"{v:+10.6f}" for v in q) + "\n")
            f.write("-" * 62 + "\n")
        print("[FILE] Saved results/UR10_grasp_can_path.txt")

        # Animate path in GUI after saving so that an interrupted animation
        # does not prevent the world-state / path files from being written.
        path_fine = interpolate_path(path, max_step=0.02)

        if _args.save_gif:
            from manipulator_env.demo_cli import save_path_gif
            _gif_tag = _args.planner.replace('*', '').replace(' ', '_')
            _gif_path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                'visualization', 'gifs',
                f'pybullet_UR10_grasp_can_{_gif_tag}.gif')
            save_path_gif(
                env, path_fine, _gif_path,
                cam_yaw=-114.60, cam_pitch=-40.20, cam_distance=1.60,
                cam_target=[-0.449, -0.041, 0.892],
                step=3, fps=20)
            print(f"[GIF] Saved {_gif_path}")

        if _args.headless:
            env.disconnect()
            return

        env.set_joint_positions(q_start)
        time.sleep(0.5)
        print("[ANIM] Animating path (first pass with trail) ...")
        env.visualize_path(path_fine, delay=0.02, trail=True)
        time.sleep(1.0)
        print("\n[LOOP] Replaying path (close PyBullet window or Ctrl+C to exit) ...")
        try:
            while p.isConnected(physicsClientId=cid):
                env.set_joint_positions(q_start)
                p.stepSimulation(physicsClientId=cid)
                time.sleep(0.5)
                env.visualize_path(path_fine, delay=0.02, trail=False)
                time.sleep(1.0)
        except (KeyboardInterrupt, Exception):
            print("\nShutting down ...")
        finally:
            env.disconnect()
    else:
        print("\n[RESULT] No path found.")
        if _args.headless:
            env.disconnect()
            return
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
