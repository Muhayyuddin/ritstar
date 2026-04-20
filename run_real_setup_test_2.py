#!/usr/bin/env python3
"""
run_real_setup_test_2.py — Phase 2: Place mustard bottle on top of the shelf.

Continues from the end of run_real_setup_test.py:
  - Start config = grasp pose from Phase 1 (robot holding bottle inside shelf)
  - Bottle is fixed-jointed to the gripper EE link
  - Goal  config = place pose above the shelf top (bottle on top panel)
  - Collision checking includes the attached bottle vs obstacles

Usage:
    python run_real_setup_test_2.py
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

# Fast inertia-based diagonal metric (no PyBullet calls per evaluation)
_UR10E_INERTIAS = np.array([7.778, 12.93, 3.87, 1.96, 1.96, 0.202])
_UR10E_WEIGHTS = (_UR10E_INERTIAS / _UR10E_INERTIAS.max()).tolist()

# ═══════════════════════════════════════════════════════════════════════
# Physical constants (all in metres) — same as run_real_setup_test.py
# ═══════════════════════════════════════════════════════════════════════

TABLE_SURFACE_Z = 0.75
SLAB_THICKNESS  = 0.01990
ROBOT_BASE_Z    = TABLE_SURFACE_Z + SLAB_THICKNESS  # 0.76990

SHELF_REL_X = -0.67138
SHELF_REL_Y = -0.68403
SHELF_REL_Z = -0.01590

SHELF_X = SHELF_REL_X
SHELF_Y = SHELF_REL_Y
SHELF_Z = ROBOT_BASE_Z + SHELF_REL_Z  # ≈ 0.75400

SHELF_W = 0.32
SHELF_D = 0.24
SHELF_H = 0.54
SHELF_T = 0.02

TABLE_LEN = 1.00
TABLE_WID = 1.50
TABLE_THK = 0.05

CLR_TABLE = [0.60, 0.60, 0.60, 1.0]
CLR_SLAB  = [0.35, 0.35, 0.40, 1.0]
CLR_SHELF = [0.92, 0.92, 0.92, 1.0]
CLR_LEGS  = [0.25, 0.25, 0.28, 1.0]

# Table centre: only 16 cm of the table extends along +x from the robot base
TABLE_CX = -0.34
TABLE_CY = -0.07

# Gripper depth (EE origin to finger tips along +x_ee)
GRIPPER_DEPTH = 0.1045

# ═══════════════════════════════════════════════════════════════════════
# Shelf obstacles (same as Phase 1)
# ═══════════════════════════════════════════════════════════════════════

def build_shelf_obstacles():
    """Shelf rotated 90°: width→x, depth→y, open face at +y (toward robot)."""
    sx, sy, sz = SHELF_X, SHELF_Y, SHELF_Z
    W, D, H, t = SHELF_W, SHELF_D, SHELF_H, SHELF_T
    c = CLR_SHELF
    obstacles = [
        {"type": "box", "color": CLR_TABLE,
         "pos": [TABLE_CX, TABLE_CY, TABLE_SURFACE_Z - TABLE_THK / 2],
         "half_extents": [TABLE_LEN / 2, TABLE_WID / 2, TABLE_THK / 2]},
        # Back wall (at −y end, full height)
        {"type": "box", "color": c,
         "pos": [sx, sy - D / 2 + t / 2, sz + H / 2],
         "half_extents": [W / 2, t / 2, H / 2]},
        # Left side wall (−x side)
        {"type": "box", "color": c,
         "pos": [sx - W / 2 + t / 2, sy, sz + H / 2],
         "half_extents": [t / 2, D / 2, H / 2]},
        # Right side wall (+x side)
        {"type": "box", "color": c,
         "pos": [sx + W / 2 - t / 2, sy, sz + H / 2],
         "half_extents": [t / 2, D / 2, H / 2]},
        # Bottom panel
        {"type": "box", "color": c,
         "pos": [sx, sy, sz + t / 2],
         "half_extents": [W / 2, D / 2, t / 2]},
        # Middle shelf
        {"type": "box", "color": c,
         "pos": [sx, sy, sz + H / 2],
         "half_extents": [W / 2, D / 2, t / 2]},
        # Top panel
        {"type": "box", "color": c,
         "pos": [sx, sy, sz + H - t / 2],
         "half_extents": [W / 2, D / 2, t / 2]},
    ]
    return obstacles


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
# Attach bottle to gripper & augmented collision checking
# ═══════════════════════════════════════════════════════════════════════

def attach_bottle_to_gripper(env, mustard_id):
    """Rigidly attach the mustard bottle to the gripper.

    1. Creates a fixed constraint (for physics stepping).
    2. Monkey-patches set_joint_positions so the bottle teleports with
       the EE on every resetJointState call — this keeps the bottle
       rigidly attached during collision checking AND animation
       (neither of which calls stepSimulation).
    """
    cid = env.physics_client

    # Get current EE pose and bottle pose to compute relative transform
    ee_state = p.getLinkState(env.robot_id, env.ee_link_idx,
                              computeForwardKinematics=True,
                              physicsClientId=cid)
    ee_pos = np.array(ee_state[4])
    ee_orn = np.array(ee_state[5])

    bottle_pos, bottle_orn = p.getBasePositionAndOrientation(mustard_id,
                                                              physicsClientId=cid)

    # Compute relative transform: bottle in EE frame
    ee_inv_pos, ee_inv_orn = p.invertTransform(ee_pos, ee_orn)
    rel_pos, rel_orn = p.multiplyTransforms(ee_inv_pos, ee_inv_orn,
                                             bottle_pos, bottle_orn)

    # Create fixed constraint (used when stepSimulation runs)
    constraint_id = p.createConstraint(
        parentBodyUniqueId=env.robot_id,
        parentLinkIndex=env.ee_link_idx,
        childBodyUniqueId=mustard_id,
        childLinkIndex=-1,
        jointType=p.JOINT_FIXED,
        jointAxis=[0, 0, 0],
        parentFramePosition=list(rel_pos),
        parentFrameOrientation=list(rel_orn),
        childFramePosition=[0, 0, 0],
        childFrameOrientation=[0, 0, 0, 1],
        physicsClientId=cid,
    )
    p.changeConstraint(constraint_id, maxForce=1e6, physicsClientId=cid)

    # ── Monkey-patch set_joint_positions to also move the bottle ──
    import types
    original_set = env.set_joint_positions.__func__

    def set_joint_positions_with_bottle(self, q):
        # Move robot joints
        original_set(self, q)
        # Recompute EE world pose (after joints teleported)
        state = p.getLinkState(self.robot_id, self.ee_link_idx,
                               computeForwardKinematics=True,
                               physicsClientId=cid)
        new_ee_pos = state[4]
        new_ee_orn = state[5]
        # Compute bottle world pose from stored relative transform
        new_bottle_pos, new_bottle_orn = p.multiplyTransforms(
            new_ee_pos, new_ee_orn, rel_pos, rel_orn)
        p.resetBasePositionAndOrientation(
            mustard_id, new_bottle_pos, new_bottle_orn,
            physicsClientId=cid)

    env.set_joint_positions = types.MethodType(set_joint_positions_with_bottle, env)

    return constraint_id


def patch_collision_check(env, mustard_id):
    """Monkey-patch env.is_collision_free to also check attached bottle
    against obstacles and ground plane."""
    original_check = env.is_collision_free.__func__
    cid = env.physics_client

    _dbg_count = [0]  # mutable counter for debug

    def augmented_collision_free(self, q):
        # Run original robot collision check
        if not original_check(self, q):
            return False

        # Check attached bottle vs obstacles
        p.performCollisionDetection(physicsClientId=cid)

        for obs_id in self.obstacle_ids:
            contacts = p.getContactPoints(bodyA=mustard_id, bodyB=obs_id,
                                          physicsClientId=cid)
            if contacts:
                if _dbg_count[0] < 5:
                    bpos, _ = p.getBasePositionAndOrientation(
                        mustard_id, physicsClientId=cid)
                    print(f"  [DBG] Bottle at [{bpos[0]:.3f},{bpos[1]:.3f},{bpos[2]:.3f}] "
                          f"collides with obstacle {obs_id}")
                    _dbg_count[0] += 1
                return False

        # Bottle vs ground
        contacts = p.getContactPoints(bodyA=mustard_id, bodyB=self.plane_id,
                                      physicsClientId=cid)
        if contacts:
            return False

        return True

    import types
    env.is_collision_free = types.MethodType(augmented_collision_free, env)


# ═══════════════════════════════════════════════════════════════════════
# Compute a "place on top of shelf" IK
# ═══════════════════════════════════════════════════════════════════════

def compute_place_on_shelf_top_ik(env):
    """Compute IK for placing the bottle on top of the shelf.

    The gripper keeps the SAME orientation as the start (side-grasp):
        x_ee = [0, -1, 0]  (approach = −y, into shelf)
        y_ee = [±1, 0, 0]  (fingers open horizontally along ±x)

    The EE position is above the shelf top panel.  Since x_ee = −y,
    the finger tips are GRIPPER_DEPTH ahead in −y from the EE origin.
    We position the grasp centre (finger tips) above the shelf top,
    then offset the EE back by GRIPPER_DEPTH along +y.
    """
    from scipy.spatial.transform import Rotation as Rot
    cid = env.physics_client
    n_movable = len(env._all_joint_indices)

    # Target: bottle centre above shelf top panel
    shelf_top_z = SHELF_Z + SHELF_H  # 1.294 m
    place_z = shelf_top_z + 0.10     # 10 cm above shelf top

    # Grasp centre above shelf centre
    grasp_centre = np.array([SHELF_X, SHELF_Y, place_z])

    # EE is GRIPPER_DEPTH behind the grasp centre along +y (opposite of approach −y)
    ee_target = grasp_centre.copy()
    ee_target[1] += GRIPPER_DEPTH

    # Same side-grasp orientation as Phase 1:
    #   x_ee = [0,-1,0], y_ee = [±1,0,0], z_ee = cross(x_ee, y_ee)
    orns = []
    for sign in [+1, -1]:
        x_ee = np.array([0.0, -1.0, 0.0])
        y_ee = np.array([sign * 1.0, 0.0, 0.0])
        z_ee = np.cross(x_ee, y_ee)
        R = np.column_stack([x_ee, y_ee, z_ee])
        q_orn = Rot.from_matrix(R).as_quat().tolist()
        orns.append(q_orn)

    seeds = [
        [0.0, -np.pi / 2, np.pi / 2, -np.pi / 2, -np.pi / 2, 0.0],
        [-np.pi / 4, -np.pi / 2, np.pi / 2, -np.pi / 2, -np.pi / 2, 0.0],
        [-np.pi / 2, -np.pi / 3, np.pi / 3, -np.pi / 2, -np.pi / 2, 0.0],
        [-np.pi / 3, -np.pi / 4, np.pi / 3, -np.pi / 3, -np.pi / 2, 0.0],
        [-0.8, -0.5, 1.0, -1.0, -1.57, 0.0],
        [-1.2, -1.5, 1.5, -1.5, -1.57, 0.0],
        [-1.5, -1.2, 1.0, -1.0, -1.57, 0.0],
        [-1.0, -2.0, 1.8, -1.4, -1.57, 0.0],
        [-1.2, -0.8, 1.2, -0.4, -1.2, 0.0],
        [-0.5, -1.0, 0.8, -1.5, -1.57, 1.57],
        [-1.0, -1.0, 1.0, -1.57, -1.57, 1.57],
    ]

    best_q = None

    for oi, orn_q in enumerate(orns):
        for si, seed in enumerate(seeds):
            env.set_joint_positions(np.array(seed))

            rest = list(seed) + [0.0] * (n_movable - 6)
            q_ik = p.calculateInverseKinematics(
                bodyUniqueId=env.robot_id,
                endEffectorLinkIndex=env.ee_link_idx,
                targetPosition=ee_target.tolist(),
                targetOrientation=orn_q,
                lowerLimits=env.JOINT_LIMITS_LOWER.tolist(),
                upperLimits=env.JOINT_LIMITS_UPPER.tolist(),
                jointRanges=[4 * np.pi] * 6 + [0.01] * (n_movable - 6),
                restPoses=rest,
                maxNumIterations=500,
                residualThreshold=1e-4,
                physicsClientId=cid,
            )
            q_arm = np.array(q_ik[:6])

            if env.is_collision_free(q_arm):
                ee_actual, ee_orn_q = env.get_ee_pose(q_arm)
                err = np.linalg.norm(ee_target - ee_actual)
                if err < 0.05:
                    R = np.array(p.getMatrixFromQuaternion(ee_orn_q)).reshape(3, 3)
                    print(f"[IK]  Collision-free PLACE pose found!")
                    print(f"      orn_idx={oi}, seed={si}")
                    print(f"      EE pos : [{ee_actual[0]:.3f}, {ee_actual[1]:.3f}, {ee_actual[2]:.3f}]")
                    print(f"      x_ee (approach): [{R[0,0]:.3f}, {R[1,0]:.3f}, {R[2,0]:.3f}]")
                    print(f"      y_ee (open):     [{R[0,1]:.3f}, {R[1,1]:.3f}, {R[2,1]:.3f}]")
                    return q_arm

            if best_q is None:
                best_q = q_arm

    print("[IK]  WARNING: No collision-free place IK found, using best fallback.")
    return best_q


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

def main():
    # Build shelf obstacles (includes table)
    shelf_obstacles = build_shelf_obstacles()

    # ── Start config: goal config from Phase 1 ──
    q_start = np.array([-2.813, -0.892, 1.726, -3.975, -0.329, 1.570])

    # ── Phase 1: Headless IK computation for place pose ──────────
    print("[IK] Loading PyBullet (headless) for place IK computation ...")
    ik_env = UR10eRobotiqEnv(
        gui=False,
        obstacles=shelf_obstacles,
        base_position=[0.0, 0.0, ROBOT_BASE_Z],
    )
    ik_cid = ik_env.physics_client

    # Get EE pose at start config to place bottle at fingertips
    ik_env.set_joint_positions(q_start)
    ee_pos_start, ee_orn_start = ik_env.get_ee_pose(q_start)
    R_start = np.array(p.getMatrixFromQuaternion(ee_orn_start)).reshape(3, 3)
    x_ee = R_start[:, 0]  # approach/finger direction

    # Bottle position: at the grasp centre, shifted 3.5 cm further along approach to avoid penetration
    bottle_start_pos = ee_pos_start + (GRIPPER_DEPTH + 0.035) * x_ee
    bottle_start_orn = p.getQuaternionFromEuler([0, 0, np.deg2rad(-60)])

    # Load mustard bottle in headless env
    mustard_urdf = "/home/muhayy/Documents/forsight-tamp/assets/ycb_objects/ycb_assets/006_mustard_bottle.urdf"
    ik_bottle_id = p.loadURDF(
        mustard_urdf,
        basePosition=bottle_start_pos.tolist(),
        baseOrientation=bottle_start_orn,
        useFixedBase=False,
        globalScaling=0.1,
        physicsClientId=ik_cid,
    )

    # Attach bottle and patch collision checking in headless env
    attach_bottle_to_gripper(ik_env, ik_bottle_id)
    # Disable robot-bottle collision (they're attached)
    n_joints_ik = p.getNumJoints(ik_env.robot_id, physicsClientId=ik_cid)
    for link_idx in range(-1, n_joints_ik):
        p.setCollisionFilterPair(
            ik_env.robot_id, ik_bottle_id, link_idx, -1, 0,
            physicsClientId=ik_cid,
        )
    patch_collision_check(ik_env, ik_bottle_id)

    # Verify start is collision-free
    assert ik_env.is_collision_free(q_start), "Start config is in collision!"

    # Compute place goal
    q_goal = compute_place_on_shelf_top_ik(ik_env)
    assert ik_env.is_collision_free(q_goal), "Goal config is in collision!"

    pos_g, orn_g = ik_env.get_ee_pose(q_goal)
    print(f"[IK] Goal found — q_goal: [{', '.join(f'{v:.3f}' for v in q_goal)}]")
    print(f"[IK] Goal EE pos: [{pos_g[0]:.3f}, {pos_g[1]:.3f}, {pos_g[2]:.3f}]")

    ik_env.disconnect()
    print("[IK] Headless env disconnected.\n")

    # ── Phase 2: GUI environment + planner ────────────────────────
    print("[ENV] Loading PyBullet (GUI) ...")
    env = UR10eRobotiqEnv(
        gui=True,
        obstacles=shelf_obstacles,
        base_position=[0.0, 0.0, ROBOT_BASE_Z],
    )
    cid = env.physics_client

    # Add visual scenery
    add_scenery(cid)

    # Set robot to start config (grasp pose)
    env.set_joint_positions(q_start)

    # Load mustard bottle at the grasp position
    mustard_id = p.loadURDF(
        mustard_urdf,
        basePosition=bottle_start_pos.tolist(),
        baseOrientation=bottle_start_orn,
        useFixedBase=False,
        globalScaling=0.1,
        physicsClientId=cid,
    )

    # Attach bottle to gripper with fixed constraint
    constraint_id = attach_bottle_to_gripper(env, mustard_id)
    print(f"[ATTACH] Bottle fixed to gripper (constraint={constraint_id})")

    # Step simulation to settle the constraint
    for _ in range(10):
        p.stepSimulation(physicsClientId=cid)

    # Disable collision between robot and bottle (they're attached)
    n_joints = p.getNumJoints(env.robot_id, physicsClientId=cid)
    for link_idx in range(-1, n_joints):
        p.setCollisionFilterPair(
            env.robot_id, mustard_id, link_idx, -1, 0,
            physicsClientId=cid,
        )

    # Patch collision checking to include bottle vs obstacles
    patch_collision_check(env, mustard_id)
    print("[PATCH] Collision check now includes attached bottle vs obstacles")

    # Verify start and goal are collision-free in GUI env
    assert env.is_collision_free(q_start), "Start config is in collision!"
    assert env.is_collision_free(q_goal), "Goal config is in collision!"

    # Get poses for display
    pos_s, _ = env.get_ee_pose(q_start)
    pos_g, orn_g = env.get_ee_pose(q_goal)
    R_g = np.array(p.getMatrixFromQuaternion(orn_g)).reshape(3, 3)

    # Put robot back at start
    env.set_joint_positions(q_start)
    for _ in range(5):
        p.stepSimulation(physicsClientId=cid)

    # Configure camera
    p.resetDebugVisualizerCamera(
        cameraDistance=1.4,
        cameraYaw=-36.0,
        cameraPitch=-20.6,
        cameraTargetPosition=[0.363, -0.358, 1.21],
        physicsClientId=cid,
    )
    p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 0, physicsClientId=cid)

    # ── Infinite white tiled floor ─────────────────────────────────
    floor_he = [50.0, 50.0, 0.01]
    floor_vis = p.createVisualShape(p.GEOM_BOX, halfExtents=floor_he,
                                    rgbaColor=[1.0, 1.0, 1.0, 1.0],
                                    physicsClientId=cid)
    p.createMultiBody(baseMass=0, baseCollisionShapeIndex=-1,
                      baseVisualShapeIndex=floor_vis,
                      basePosition=[0.0, 0.0, 0.0],
                      physicsClientId=cid)
    p.changeVisualShape(env.plane_id, -1,
                        rgbaColor=[1.0, 1.0, 1.0, 1.0],
                        physicsClientId=cid)

    # ── Robot colours ─────────────────────────────────────────────
    UR_SILVER = [0.75, 0.75, 0.75, 1.0]
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

    shelf_top_z = SHELF_Z + SHELF_H
    print("=" * 62)
    print("  Real Setup 2 — Place Bottle on Shelf Top")
    print("=" * 62)
    print(f"  Bottle at start : [{bottle_start_pos[0]:.3f}, {bottle_start_pos[1]:.3f}, {bottle_start_pos[2]:.3f}]")
    print(f"  Shelf top Z     : {shelf_top_z:.3f}")
    print(f"  Start EE  : [{pos_s[0]:.3f}, {pos_s[1]:.3f}, {pos_s[2]:.3f}]")
    print(f"  Goal  EE  : [{pos_g[0]:.3f}, {pos_g[1]:.3f}, {pos_g[2]:.3f}]")
    print(f"  x_ee (approach) : [{R_g[0,0]:.3f}, {R_g[1,0]:.3f}, {R_g[2,0]:.3f}]")
    print(f"  y_ee (open)     : [{R_g[0,1]:.3f}, {R_g[1,1]:.3f}, {R_g[2,1]:.3f}]")
    print(f"  Start cfg : [{', '.join(f'{v:.3f}' for v in q_start)}]")
    print(f"  Goal  cfg : [{', '.join(f'{v:.3f}' for v in q_goal)}]")
    print("=" * 62)
    print()

    # Run RIT* planner
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

        # Save world state
        with open("results/real_setup_2_world_state.txt", "w") as f:
            f.write("=" * 62 + "\n")
            f.write("  Real Setup 2 — Place Bottle on Shelf Top (World State)\n")
            f.write("=" * 62 + "\n\n")

            f.write("--- Robot ---\n")
            f.write(f"  Base position (world) : [0.000, 0.000, {ROBOT_BASE_Z:.5f}]\n\n")

            f.write("--- Shelf ---\n")
            f.write(f"  Position (bottom-centre) : [{SHELF_X:.5f}, {SHELF_Y:.5f}, {SHELF_Z:.5f}]\n")
            f.write(f"  Top panel Z              : {shelf_top_z:.5f}\n")
            f.write(f"  Dimensions (W x D x H)   : {SHELF_W:.2f} x {SHELF_D:.2f} x {SHELF_H:.2f} m\n\n")

            f.write("--- Mustard Bottle (attached to gripper) ---\n")
            f.write(f"  Start position (world) : [{bottle_start_pos[0]:.5f}, {bottle_start_pos[1]:.5f}, {bottle_start_pos[2]:.5f}]\n")
            f.write(f"  Scale                  : 0.1\n")
            f.write(f"  Attached to EE link    : {env.ee_link_idx}\n\n")

            f.write("--- Obstacle list (collision-checked) ---\n")
            for i, obs in enumerate(shelf_obstacles):
                f.write(f"  [{i}] {obs['type']}  pos={[round(v,5) for v in obs['pos']]}  "
                        f"half_extents={[round(v,5) for v in obs['half_extents']]}\n")

            f.write("\n--- Start Configuration (grasp from Phase 1) ---\n")
            f.write(f"  q_start (rad) : [{', '.join(f'{v:.6f}' for v in q_start)}]\n")
            f.write(f"  EE position   : [{pos_s[0]:.5f}, {pos_s[1]:.5f}, {pos_s[2]:.5f}]\n\n")

            f.write("--- Goal Configuration (place on shelf top) ---\n")
            f.write(f"  q_goal  (rad) : [{', '.join(f'{v:.6f}' for v in q_goal)}]\n")
            f.write(f"  EE position   : [{pos_g[0]:.5f}, {pos_g[1]:.5f}, {pos_g[2]:.5f}]\n")
            f.write(f"  x_ee (approach) : [{R_g[0,0]:.5f}, {R_g[1,0]:.5f}, {R_g[2,0]:.5f}]\n")
            f.write(f"  y_ee (open)     : [{R_g[0,1]:.5f}, {R_g[1,1]:.5f}, {R_g[2,1]:.5f}]\n\n")

            f.write("--- Planner ---\n")
            f.write(f"  Algorithm     : RIT*\n")
            f.write(f"  Metric        : DiagonalAnisotropicMetric\n")
            f.write(f"  Path cost     : {cost:.6f}\n")
            f.write(f"  Waypoints     : {len(path)}\n")
            f.write(f"  Bottle collision : ENABLED (attached body)\n")

        print("[FILE] Saved results/real_setup_2_world_state.txt")

        # Save path
        with open("results/real_setup_2_path.txt", "w") as f:
            f.write("=" * 62 + "\n")
            f.write("  Real Setup 2 — Complete Path (Joint Configurations)\n")
            f.write("=" * 62 + "\n")
            f.write(f"  Waypoints : {len(path)}\n")
            f.write(f"  Path cost : {cost:.6f}\n")
            f.write(f"  DOF       : 6\n\n")
            f.write("  Each row: joint_1  joint_2  joint_3  joint_4  joint_5  joint_6  (radians)\n")
            f.write("-" * 62 + "\n")
            for i, q in enumerate(path):
                f.write(f"  {i:4d}  " + "  ".join(f"{v:+10.6f}" for v in q) + "\n")
            f.write("-" * 62 + "\n")

        print("[FILE] Saved results/real_setup_2_path.txt")
    else:
        print("\n[RESULT] No path found.")

    # ── Loop path animation until window is closed ────────────
    if path:
        path_fine = interpolate_path(path, max_step=0.02)
        print("\n[LOOP] Replaying path (close PyBullet window to exit) ...")
        try:
            while p.isConnected(physicsClientId=cid):
                env.set_joint_positions(q_start)
                time.sleep(0.3)
                env.visualize_path(path_fine, delay=0.02, trail=False)
                time.sleep(0.5)
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
