#!/usr/bin/env python3
"""
Tiago_pro_dual_grasp_box.py — Tiago Pro dual-arm reach to a tall box.

Replicates the table + tall box from UR10_grasp_cube.py, but replaces the
UR10e with a PAL Tiago Pro mobile manipulator standing on the floor in
front of the table. The cube is removed. The planning goal is a bimanual
pre-grasp: each arm's gripper sits 2 cm away from one end (short face) of
the box, with the gripper approach axis horizontal (parallel to floor).

Usage:
    python Tiago_pro_dual_grasp_box.py
"""

import os
import sys
import time
import numpy as np
import pybullet as p
import pybullet_data

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from rit_star.rit_star import RITStar
from rit_star.metric import DiagonalAnisotropicMetric
TIAGO_URDF = os.path.join(HERE, "manipulator_env", "models",
                          "tiago_pro", "tiago_pro.urdf")

# ═══════════════════════════════════════════════════════════════════════
# Scene geometry (kept identical to UR10_grasp_cube.py)
# ═══════════════════════════════════════════════════════════════════════

TABLE_SURFACE_Z = 0.75
TABLE_THK  = 0.05
TABLE_LEN  = 1.00   # x extent
TABLE_WID  = 1.50   # y extent
TABLE_CX   = -0.29
TABLE_CY   = -0.07

# Tall box on the table, placed in front of Tiago. The long axis runs
# along world y (side-to-side from the robot's point of view) so the
# two short ±y faces face the robot's left / right hand respectively.
BOX_X =  0.10
BOX_Y =  0.00
BOX_L = 0.12   # x extent (depth, along Tiago's forward direction)
BOX_W = 0.485  # y extent (long axis, side-to-side) — 15% narrower than original 0.57
BOX_H = 0.38   # z extent (height)
BOX_Z_MID = TABLE_SURFACE_Z + BOX_H / 2

# Two red obstacle boxes placed at the left / right ends of the table,
# flanking the central box. They are 40% taller than the centre box and
# sit flush with each short edge of the table.
OBS_BOX_L = BOX_W          # x extent (same as centre box y-span)
OBS_BOX_W = BOX_L / 2      # y extent — half the centre box depth
OBS_BOX_H = BOX_H * 1.4    # 40% taller than centre box
OBS_BOX_Z_MID = TABLE_SURFACE_Z + OBS_BOX_H / 2
OBS_BOX_X = BOX_X           # same x-centre as the main box
OBS_RIGHT_Y = TABLE_CY + TABLE_WID / 2 - OBS_BOX_W / 2  # flush with right table edge
OBS_LEFT_Y  = BOX_Y - (OBS_RIGHT_Y - BOX_Y)             # mirrored about BOX_Y → equal corridors

CLR_BOX   = [0.78, 0.64, 0.46, 1.0]
CLR_OBS   = [0.70, 0.25, 0.25, 1.0]   # red obstacle boxes
CLR_TABLE = [0.60, 0.60, 0.60, 1.0]
CLR_LEGS  = [0.25, 0.25, 0.28, 1.0]

# Tiago Pro stands next to the long (+y) edge of the table, facing −y so
# it looks across the table toward the box. Left arm ends up on world +x,
# right arm on world −x, so each hand naturally lines up with one end
# (short face) of the elongated box.
TIAGO_X   =  0.90       # on the +x (right) side of the table, clear of it
TIAGO_Y   =  0.00       # aligned with the middle of the long table axis
TIAGO_YAW =  np.pi      # face -x (look down the table toward the box)

# ═══════════════════════════════════════════════════════════════════════
# Arm joint names (matches the generated tiago_pro.urdf)
# ═══════════════════════════════════════════════════════════════════════

ARM_LEFT_JOINTS = [f"arm_left_{i}_joint" for i in range(1, 8)]
ARM_RIGHT_JOINTS = [f"arm_right_{i}_joint" for i in range(1, 8)]
TORSO_JOINT = "torso_lift_joint"
EE_LEFT_LINK  = "gripper_left_grasping_link"
EE_RIGHT_LINK = "gripper_right_grasping_link"

# Arms tucked at the sides (shoulder up, elbow bent behind the torso) so
# the home configuration stays clear of the table / box.
ARM_HOME = np.array([1.57, -1.50, 1.57, -1.57, 0.00, 0.00, 0.00])
TORSO_HOME = 0.30   # raise torso near its upper limit for more reach

# Per-joint inertia weights for Tiago Pro 7-DOF arm (proximal → distal)
TIAGO_ARM_WEIGHTS = [5.0, 5.0, 3.5, 3.5, 2.0, 2.0, 0.8]


# ═══════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════

def build_joint_maps(cid, rid):
    """Return (joint_name_to_idx, link_name_to_idx)."""
    jmap, lmap = {}, {}
    for i in range(p.getNumJoints(rid, physicsClientId=cid)):
        info = p.getJointInfo(rid, i, physicsClientId=cid)
        jmap[info[1].decode()] = i
        lmap[info[12].decode()] = i
    return jmap, lmap


def set_joint_values(cid, rid, joint_indices, values):
    for j, v in zip(joint_indices, values):
        p.resetJointState(rid, j, v, physicsClientId=cid)


def get_joint_limits(cid, rid, joint_indices):
    lower, upper = [], []
    for j in joint_indices:
        info = p.getJointInfo(rid, j, physicsClientId=cid)
        lower.append(info[8])
        upper.append(info[9])
    return np.array(lower), np.array(upper)


# ═══════════════════════════════════════════════════════════════════════
# RIT* planning helpers
# ═══════════════════════════════════════════════════════════════════════

def make_arm_collision_checker(cid, rid, arm_indices, fixed_joint_map,
                               obstacle_ids):
    """Return a collision-free checker closure for one Tiago arm.

    Each call sets *arm_indices* joints to query config *q*, fixes all
    joints in *fixed_joint_map* to their stored values, then checks
    for PyBullet contacts with every body in *obstacle_ids*.
    """
    def _checker(q):
        for jidx, v in zip(arm_indices, q):
            p.resetJointState(rid, jidx, float(v), physicsClientId=cid)
        for jidx, v in fixed_joint_map.items():
            p.resetJointState(rid, jidx, float(v), physicsClientId=cid)
        p.performCollisionDetection(physicsClientId=cid)
        for ob_id in obstacle_ids:
            if p.getContactPoints(bodyA=rid, bodyB=ob_id,
                                  physicsClientId=cid):
                return False
        return True
    return _checker


def make_dual_arm_collision_checker(cid, rid, arm_left_idx, arm_right_idx,
                                    fixed_joint_map, obstacle_ids):
    """Return a 14-D collision-free checker for both arms simultaneously.

    q[:7] = left-arm joints, q[7:] = right-arm joints.
    All joints in *fixed_joint_map* (e.g. torso) are held fixed.
    Checks: obstacles + arm/arm self-collision.
    """
    def _checker(q):
        q_l = q[:7]
        q_r = q[7:]
        for jidx, v in zip(arm_left_idx, q_l):
            p.resetJointState(rid, jidx, float(v), physicsClientId=cid)
        for jidx, v in zip(arm_right_idx, q_r):
            p.resetJointState(rid, jidx, float(v), physicsClientId=cid)
        for jidx, v in fixed_joint_map.items():
            p.resetJointState(rid, jidx, float(v), physicsClientId=cid)
        p.performCollisionDetection(physicsClientId=cid)
        for ob_id in obstacle_ids:
            if p.getContactPoints(bodyA=rid, bodyB=ob_id,
                                  physicsClientId=cid):
                return False
        # Self-collision between the two arms
        if p.getContactPoints(bodyA=rid, bodyB=rid, physicsClientId=cid):
            return False
        return True
    return _checker


def _interp_path(path, max_step=0.05):
    """Linearly interpolate a path so adjacent configs are ≤ max_step apart."""
    if not path:
        return path
    result = [path[0].copy()]
    for i in range(len(path) - 1):
        diff = path[i + 1] - path[i]
        dist = float(np.linalg.norm(diff))
        n_steps = max(1, int(np.ceil(dist / max_step)))
        for k in range(1, n_steps + 1):
            result.append(path[i] + (k / n_steps) * diff)
    return result


def plan_arm_rit_star(cid, rid, arm_indices, q_start, q_goal,
                     collision_checker, label="",
                     batch_size=150, max_iterations=250):
    """Plan a path for one 7-DOF Tiago arm using RIT*.

    Returns a list of (7,) configuration arrays from q_start to q_goal.
    Falls back to a straight-line path if no solution is found.
    """
    lower, upper = get_joint_limits(cid, rid, arm_indices)
    # Clamp degenerate URDF limits
    for i in range(len(lower)):
        if lower[i] >= upper[i]:
            lower[i] = -2 * np.pi
            upper[i] =  2 * np.pi
    bounds = list(zip(lower.tolist(), upper.tolist()))

    metric = DiagonalAnisotropicMetric(TIAGO_ARM_WEIGHTS)

    print(f"[RIT*] Planning {label} arm …")
    planner = RITStar(
        x_start=np.asarray(q_start, dtype=float),
        x_goal=np.asarray(q_goal, dtype=float),
        c_space_bounds=bounds,
        collision_checker=collision_checker,
        metric=metric,
        geodesic_tier='diagonal',
        batch_size=batch_size,
        max_iterations=max_iterations,
        random_seed=42,
        collision_step_size=0.1,
    )
    path, cost = planner.plan()
    if path:
        print(f"[RIT*] {label}: cost={cost:.4f}, waypoints={len(path)}")
    else:
        print(f"[RIT*] {label}: no path found — falling back to straight line")
        path = [np.asarray(q_start, dtype=float),
                np.asarray(q_goal, dtype=float)]
    return path


def plan_dual_arm_rit_star(cid, rid, arm_left_idx, arm_right_idx,
                           q_start_l, q_goal_l, q_start_r, q_goal_r,
                           collision_checker,
                           batch_size=200, max_iterations=300):
    """Plan both arms jointly in a 14-D configuration space using RIT*.

    The 14-D state is [q_left (7), q_right (7)].  A single RIT* tree
    explores both arms simultaneously, so cross-arm interactions
    (including self-collision between the two arms) are captured during
    planning rather than ignored.

    Returns
    -------
    path_left  : list of (7,) arrays
    path_right : list of (7,) arrays
    """
    # Build 14-D bounds: left-arm limits then right-arm limits
    lower_l, upper_l = get_joint_limits(cid, rid, arm_left_idx)
    lower_r, upper_r = get_joint_limits(cid, rid, arm_right_idx)
    for arr in (lower_l, upper_l, lower_r, upper_r):
        for i in range(len(arr)):
            if arr is lower_l or arr is lower_r:
                if arr[i] >= (upper_l if arr is lower_l else upper_r)[i]:
                    arr[i] = -2 * np.pi
            else:
                lo = (lower_l if arr is upper_l else lower_r)
                if arr[i] <= lo[i]:
                    arr[i] = 2 * np.pi
    bounds = (list(zip(lower_l.tolist(), upper_l.tolist())) +
              list(zip(lower_r.tolist(), upper_r.tolist())))

    # 14-D diagonal metric: left-arm weights then right-arm weights
    weights_14 = TIAGO_ARM_WEIGHTS + TIAGO_ARM_WEIGHTS
    metric = DiagonalAnisotropicMetric(weights_14)

    q_start_14 = np.concatenate([q_start_l, q_start_r])
    q_goal_14  = np.concatenate([q_goal_l,  q_goal_r])

    print("[RIT*] Planning both arms jointly in 14-D configuration space …")
    planner = RITStar(
        x_start=q_start_14,
        x_goal=q_goal_14,
        c_space_bounds=bounds,
        collision_checker=collision_checker,
        metric=metric,
        geodesic_tier='diagonal',
        batch_size=batch_size,
        max_iterations=max_iterations,
        random_seed=42,
        collision_step_size=0.1,
    )
    path_14, cost = planner.plan()

    if path_14:
        print(f"[RIT*] 14-D: cost={cost:.4f}, waypoints={len(path_14)}")
    else:
        print("[RIT*] 14-D: no path found — falling back to straight line")
        path_14 = [q_start_14.copy(), q_goal_14.copy()]

    path_left  = [q[:7]  for q in path_14]
    path_right = [q[7:]  for q in path_14]
    return path_left, path_right


def add_visual_box(cid, pos, he, color):
    vis = p.createVisualShape(p.GEOM_BOX, halfExtents=he,
                              rgbaColor=color, physicsClientId=cid)
    return p.createMultiBody(baseMass=0, baseVisualShapeIndex=vis,
                             basePosition=pos, physicsClientId=cid)


def add_box(cid, pos, he, color, collidable=True):
    vis = p.createVisualShape(p.GEOM_BOX, halfExtents=he,
                              rgbaColor=color, physicsClientId=cid)
    col = -1
    if collidable:
        col = p.createCollisionShape(p.GEOM_BOX, halfExtents=he,
                                     physicsClientId=cid)
    return p.createMultiBody(baseMass=0,
                             baseCollisionShapeIndex=col,
                             baseVisualShapeIndex=vis,
                             basePosition=pos,
                             physicsClientId=cid)


def build_scene(cid):
    """Table surface, legs, tall box on the table. Returns (table_id, box_id)."""
    # Table surface
    table_id = add_box(cid,
            pos=[TABLE_CX, TABLE_CY, TABLE_SURFACE_Z - TABLE_THK / 2],
            he=[TABLE_LEN / 2, TABLE_WID / 2, TABLE_THK / 2],
            color=CLR_TABLE)

    # Legs (visual only)
    leg_h = TABLE_SURFACE_Z - TABLE_THK
    leg_he = [0.03, 0.03, leg_h / 2]
    inset = 0.06
    for lx in (TABLE_CX - TABLE_LEN / 2 + inset,
               TABLE_CX + TABLE_LEN / 2 - inset):
        for ly in (TABLE_CY - TABLE_WID / 2 + inset,
                   TABLE_CY + TABLE_WID / 2 - inset):
            add_visual_box(cid, [lx, ly, leg_h / 2], leg_he, CLR_LEGS)

    # Tall box on the table
    box_id = add_box(cid,
                     pos=[BOX_X, BOX_Y, BOX_Z_MID],
                     he=[BOX_L / 2, BOX_W / 2, BOX_H / 2],
                     color=CLR_BOX)

    # Two red obstacle boxes at the ends of the table
    obs_left_id = add_box(cid,
                          pos=[OBS_BOX_X, OBS_LEFT_Y, OBS_BOX_Z_MID],
                          he=[OBS_BOX_L / 2, OBS_BOX_W / 2, OBS_BOX_H / 2],
                          color=CLR_OBS)
    obs_right_id = add_box(cid,
                           pos=[OBS_BOX_X, OBS_RIGHT_Y, OBS_BOX_Z_MID],
                           he=[OBS_BOX_L / 2, OBS_BOX_W / 2, OBS_BOX_H / 2],
                           color=CLR_OBS)
    return table_id, box_id, obs_left_id, obs_right_id


# ═══════════════════════════════════════════════════════════════════════
# Goal pose construction
# ═══════════════════════════════════════════════════════════════════════

def horizontal_approach_quat(toward_plus_y: bool):
    """Orientation that points the gripper's approach axis along ±y.

    Tiago Pro's `gripper_*_grasping_link` has the approach axis on its
    local +x (rpy="0 -1.57 0" offset from `gripper_*_base_link`). A
    yaw-only rotation therefore keeps the wrist level (local +z stays
    aligned with world +z) so the gripper sits parallel to the floor.
    """
    if toward_plus_y:
        rpy = [0.0, 0.0,  np.pi / 2]   # +x_local → +y_world
    else:
        rpy = [0.0, 0.0, -np.pi / 2]   # +x_local → -y_world
    return p.getQuaternionFromEuler(rpy)


def compute_goal_targets():
    """Return (pos_left, orn_left, pos_right, orn_right) for the two EEs.

    Tiago stands at +x facing −x. The box's long axis runs along world y.
    The robot's left hand (world −y side) reaches the −y end of the box;
    the right hand reaches the +y end. The gripper frames sit 2 cm
    outside each end face, approach axis horizontal and pointing into
    the box. Target z is near the top of the box so each arm can come
    in almost straight from the shoulder without diving through the
    table or the box body.
    """
    target_z = TABLE_SURFACE_Z + BOX_H - 0.06   # 6 cm below box top
    # Left hand: world -y end of box; approach along +y (into box)
    left_pos = [BOX_X, BOX_Y - BOX_W / 2 - 0.07, target_z]
    left_orn = horizontal_approach_quat(toward_plus_y=True)

    # Right hand: world +y end of box; approach along -y (into box)
    right_pos = [BOX_X, BOX_Y + BOX_W / 2 + 0.07, target_z]
    right_orn = horizontal_approach_quat(toward_plus_y=False)
    return left_pos, left_orn, right_pos, right_orn


# ═══════════════════════════════════════════════════════════════════════
# Inverse kinematics for one arm
# ═══════════════════════════════════════════════════════════════════════

def _arm_in_collision(cid, rid, obstacle_ids):
    """True iff any link of the robot is in contact with any obstacle
    (or in self-collision). Requires set_joint_positions beforehand."""
    p.performCollisionDetection(physicsClientId=cid)
    for ob in obstacle_ids:
        if p.getContactPoints(bodyA=rid, bodyB=ob, physicsClientId=cid):
            return True
    if p.getContactPoints(bodyA=rid, bodyB=rid, physicsClientId=cid):
        return True
    return False


def solve_arm_ik(cid, rid, arm_joint_indices, ee_link_idx,
                 target_pos, target_orn,
                 all_movable_indices, arm_home,
                 obstacle_ids=(),
                 label="", n_seeds=60, pos_tol=0.02,
                 orn_weight=1.0):
    """Plain damped-LS IK for a 7-DOF arm.

    Tries multiple seeds; keeps the best solution by combined position +
    orientation error. Only the 7 arm-joint values from the IK output
    are extracted and applied (the rest of the chain is held at whatever
    state the caller set up before calling).
    """
    rng = np.random.RandomState(0)
    arm_cols = [all_movable_indices.index(j) for j in arm_joint_indices]

    # Snapshot the non-arm joint values so we can restore them after
    # each IK call (PyBullet's IK perturbs every movable joint).
    non_arm_cols = [i for i in range(len(all_movable_indices))
                    if i not in arm_cols]
    non_arm_values = [
        p.getJointState(rid, all_movable_indices[i],
                        physicsClientId=cid)[0]
        for i in non_arm_cols
    ]

    best = None
    best_cost = float("inf")

    for s in range(n_seeds):
        # Seed the arm joints and keep everything else fixed
        if s == 0:
            seed = arm_home.copy()
        else:
            seed = arm_home + rng.uniform(-1.5, 1.5, size=7)
        for jidx, v in zip(arm_joint_indices, seed):
            p.resetJointState(rid, jidx, float(v), physicsClientId=cid)
        for ci, v in zip(non_arm_cols, non_arm_values):
            p.resetJointState(rid, all_movable_indices[ci], v,
                              physicsClientId=cid)

        kwargs = dict(
            bodyUniqueId=rid,
            endEffectorLinkIndex=ee_link_idx,
            targetPosition=list(target_pos),
            maxNumIterations=400,
            residualThreshold=1e-5,
            physicsClientId=cid,
        )
        if target_orn is not None:
            kwargs["targetOrientation"] = list(target_orn)
        q_all = p.calculateInverseKinematics(**kwargs)
        q_arm = np.array([q_all[c] for c in arm_cols])

        # Apply only the arm joints; restore the rest
        for jidx, v in zip(arm_joint_indices, q_arm):
            p.resetJointState(rid, jidx, float(v), physicsClientId=cid)
        for ci, v in zip(non_arm_cols, non_arm_values):
            p.resetJointState(rid, all_movable_indices[ci], v,
                              physicsClientId=cid)

        st = p.getLinkState(rid, ee_link_idx,
                            computeForwardKinematics=True,
                            physicsClientId=cid)
        ee_pos = np.array(st[4])
        ee_orn = np.array(st[5])
        pos_err = float(np.linalg.norm(ee_pos - np.array(target_pos)))

        # Orientation error: if a target orientation is given use
        # normalised quaternion angular distance; otherwise fall back to
        # penalising the vertical component of the approach axis.
        if target_orn is not None:
            t_orn = np.asarray(target_orn, dtype=float)
            dot = min(abs(float(np.dot(ee_orn, t_orn))), 1.0)
            orn_err = 2.0 * np.arccos(dot) / np.pi   # normalised [0,1]
        else:
            R = np.array(p.getMatrixFromQuaternion(ee_orn)).reshape(3, 3)
            approach_world = R[:, 0]
            orn_err = abs(float(approach_world[2]))   # 0 = perfectly level

        in_coll = _arm_in_collision(cid, rid, obstacle_ids) \
            if obstacle_ids else False
        cost = pos_err + orn_weight * orn_err + (10.0 if in_coll else 0.0)

        if cost < best_cost:
            best_cost = cost
            best = (q_arm.copy(), ee_pos, ee_orn, pos_err, orn_err, in_coll)

        if pos_err < pos_tol and orn_err < 0.15 and not in_coll:
            break

    q_arm, ee_pos, ee_orn, pos_err, orn_err, in_coll = best
    tag = " COLLISION" if in_coll else ""
    print(f"[IK] {label}: pos_err={pos_err:.4f}m  orn_err={orn_err:.3f}{tag}")
    return q_arm, ee_pos, pos_err


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

def main():
    gui = "--headless" not in sys.argv
    mode = p.GUI if gui else p.DIRECT
    cid = p.connect(mode)
    p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=cid)
    p.setGravity(0, 0, -9.81, physicsClientId=cid)

    # Ground plane
    plane_id = p.loadURDF("plane.urdf", physicsClientId=cid)
    p.changeVisualShape(plane_id, -1, rgbaColor=[0.78, 0.78, 0.78, 1.0],
                        physicsClientId=cid)

    # Load Tiago Pro on the floor in front of the table
    tiago_orn = p.getQuaternionFromEuler([0, 0, TIAGO_YAW])
    rid = p.loadURDF(TIAGO_URDF,
                     basePosition=[TIAGO_X, TIAGO_Y, 0.0],
                     baseOrientation=tiago_orn,
                     useFixedBase=True,
                     physicsClientId=cid)
    print(f"[ROBOT] Tiago Pro loaded at x={TIAGO_X:.2f}, y={TIAGO_Y:.2f}, "
          f"yaw={np.degrees(TIAGO_YAW):.0f}°")

    jmap, lmap = build_joint_maps(cid, rid)
    arm_left_idx  = [jmap[n] for n in ARM_LEFT_JOINTS]
    arm_right_idx = [jmap[n] for n in ARM_RIGHT_JOINTS]
    torso_idx     = jmap[TORSO_JOINT]
    ee_left_idx   = lmap[EE_LEFT_LINK]
    ee_right_idx  = lmap[EE_RIGHT_LINK]

    all_movable = [i for i in range(p.getNumJoints(rid, physicsClientId=cid))
                   if p.getJointInfo(rid, i, physicsClientId=cid)[2]
                   != p.JOINT_FIXED]

    # Torso: fix at midpoint of its travel range, held by a stiff motor
    # so gravity cannot move it during simulation steps.
    _torso_info = p.getJointInfo(rid, torso_idx, physicsClientId=cid)
    torso_lo, torso_hi = float(_torso_info[8]), float(_torso_info[9])
    torso_mid = 0.5 * (torso_lo + torso_hi)
    p.resetJointState(rid, torso_idx, torso_mid, physicsClientId=cid)
    p.setJointMotorControl2(rid, torso_idx, p.POSITION_CONTROL,
                            targetPosition=torso_mid, force=500,
                            physicsClientId=cid)
    print(f"[TORSO] limits=[{torso_lo:.3f}, {torso_hi:.3f}]  fixed at mid={torso_mid:.3f} m")

    # Apply home pose (arms tucked)
    set_joint_values(cid, rid, arm_left_idx, ARM_HOME)
    set_joint_values(cid, rid, arm_right_idx, ARM_HOME * np.array(
        [-1, 1, -1, 1, -1, 1, -1]))   # mirror for right arm

    # Scene
    table_id, box_id, obs_left_id, obs_right_id = build_scene(cid)
    # Wheels sit on the ground, so the plane is excluded from the
    # collision-check set used by the IK.
    obstacles = [table_id, box_id, obs_left_id, obs_right_id]

    # Goal targets — fingers point inward (approach axis horizontal, toward box).
    lpos, lorn, rpos, rorn = compute_goal_targets()

    arm_home_right = ARM_HOME * np.array([-1, 1, -1, 1, -1, 1, -1])
    q_left, ee_left_pos, err_l = solve_arm_ik(
        cid, rid, arm_left_idx, ee_left_idx, lpos, lorn,
        all_movable, ARM_HOME, obstacle_ids=obstacles,
        label="LEFT", n_seeds=300)

    # Apply the left-arm solution before solving for the right arm so
    # the right-arm IK sees the left arm in its target pose.
    set_joint_values(cid, rid, arm_left_idx, q_left)

    q_right, ee_right_pos, err_r = solve_arm_ik(
        cid, rid, arm_right_idx, ee_right_idx, rpos, rorn,
        all_movable, arm_home_right, obstacle_ids=obstacles,
        label="RIGHT", n_seeds=300)

    # Goal markers
    for pos, color in [(lpos, [0.0, 0.0, 1.0, 1.0]),   # left → blue
                       (rpos, [1.0, 0.0, 0.0, 1.0])]:  # right → red
        vis = p.createVisualShape(p.GEOM_SPHERE, radius=0.02,
                                  rgbaColor=color, physicsClientId=cid)
        p.createMultiBody(baseMass=0, baseVisualShapeIndex=vis,
                          basePosition=pos, physicsClientId=cid)

    # Camera
    p.resetDebugVisualizerCamera(
        cameraDistance=2.2, cameraYaw=50, cameraPitch=-25,
        cameraTargetPosition=[BOX_X, BOX_Y, BOX_Z_MID],
        physicsClientId=cid)
    p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 0, physicsClientId=cid)

    print("=" * 62)
    print("  Tiago Pro — Dual-Arm Reach Demo")
    print("=" * 62)
    print(f"  Box center : [{BOX_X:.2f}, {BOX_Y:.2f}, {BOX_Z_MID:.2f}]"
          f"  size {BOX_L:.2f}x{BOX_W:.2f}x{BOX_H:.2f}")
    print(f"  Tiago base : [{TIAGO_X:.2f}, {TIAGO_Y:.2f}, 0.00]  "
          f"yaw={np.degrees(TIAGO_YAW):.0f}°")
    print(f"  Left goal  : pos={[round(v,3) for v in lpos]} "
          f"  ee={ee_left_pos.round(3).tolist()}  err={err_l:.4f}m")
    print(f"  Right goal : pos={[round(v,3) for v in rpos]} "
          f"  ee={ee_right_pos.round(3).tolist()}  err={err_r:.4f}m")
    print("=" * 62)

    q_left_home  = ARM_HOME.copy()
    q_right_home = ARM_HOME * np.array([-1, 1, -1, 1, -1, 1, -1])

    # ── RIT* path planning in 14-D (both arms jointly) ─────────────────
    # Only the torso is fixed; both arms plan simultaneously so that
    # cross-arm interactions and self-collision are handled correctly.
    fixed_joints = {torso_idx: torso_mid}

    dual_checker = make_dual_arm_collision_checker(
        cid, rid, arm_left_idx, arm_right_idx, fixed_joints, obstacles)
    path_left, path_right = plan_dual_arm_rit_star(
        cid, rid, arm_left_idx, arm_right_idx,
        q_left_home, q_left, q_right_home, q_right,
        dual_checker)

    # Fine-grained interpolation for smooth animation
    path_left_fine  = _interp_path(path_left,  max_step=0.04)
    path_right_fine = _interp_path(path_right, max_step=0.04)

    # Pad both paths to the same length by repeating the last waypoint
    N_anim = max(len(path_left_fine), len(path_right_fine))

    def _pad(path, n):
        if not path:
            return path
        return path + [path[-1]] * (n - len(path))

    path_left_fine  = _pad(path_left_fine,  N_anim)
    path_right_fine = _pad(path_right_fine, N_anim)

    # Reset to home before animation
    set_joint_values(cid, rid, arm_left_idx,  q_left_home)
    set_joint_values(cid, rid, arm_right_idx, q_right_home)
    p.resetJointState(rid, torso_idx, torso_mid, physicsClientId=cid)
    p.setJointMotorControl2(rid, torso_idx, p.POSITION_CONTROL,
                            targetPosition=torso_mid, force=500,
                            physicsClientId=cid)

    if not gui:
        p.disconnect(cid)
        return

    print(f"[ANIM] Playing RIT* path ({N_anim} waypoints per arm) ...")
    try:
        while p.isConnected(physicsClientId=cid):
            # Forward: follow planned paths for both arms simultaneously
            for ql, qr in zip(path_left_fine, path_right_fine):
                set_joint_values(cid, rid, arm_left_idx,  ql)
                set_joint_values(cid, rid, arm_right_idx, qr)
                p.stepSimulation(physicsClientId=cid)
                time.sleep(0.01)
            time.sleep(0.8)
            # Reset to home
            set_joint_values(cid, rid, arm_left_idx,  q_left_home)
            set_joint_values(cid, rid, arm_right_idx, q_right_home)
            p.stepSimulation(physicsClientId=cid)
            time.sleep(0.8)
    except KeyboardInterrupt:
        pass
    finally:
        p.disconnect(cid)


if __name__ == "__main__":
    main()
