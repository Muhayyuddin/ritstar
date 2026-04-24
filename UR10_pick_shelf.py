#!/usr/bin/env python3
"""
run_real_setup_test.py — Replicate real UR10e + shelf setup in PyBullet GUI.

Setup (from real-world measurements):
  - UR10e mounted on TMC vibration-isolation table
  - 19.90 mm mounting slab between table surface and robot base
  - 2-compartment white shelf on the left side of the table
    Shelf center w.r.t. robot base: x=-671.38 mm, y=-684.03 mm, z=-15.90 mm
    External dimensions: 32 cm (W) × 24 cm (D) × 54 cm (H)
    Panel thickness: ~2 cm

Usage:
    python run_real_setup_test.py
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
_UR10E_INERTIAS = np.array([7.369, 13.051, 3.989, 2.1, 1.98, 0.615])
_UR10E_WEIGHTS = (_UR10E_INERTIAS / _UR10E_INERTIAS.min()).tolist()

# ═══════════════════════════════════════════════════════════════════════
# Physical constants (all in metres)
# ═══════════════════════════════════════════════════════════════════════

TABLE_SURFACE_Z = 0.75                          # table-top height above floor
SLAB_THICKNESS  = 0.01990                        # mounting slab thickness
ROBOT_BASE_Z    = TABLE_SURFACE_Z + SLAB_THICKNESS  # 0.76990 m

# Shelf position w.r.t. robot base (metres, from real measurement)
# X-axis flipped: shelf on +x side, robot stays at origin
SHELF_REL_X =  -0.67138
SHELF_REL_Y = -0.68403
SHELF_REL_Z = 0      # slightly below robot base (shelf sits on table)

# Shelf world position — bottom-centre of the shelf
SHELF_X = SHELF_REL_X
SHELF_Y = SHELF_REL_Y
SHELF_Z = ROBOT_BASE_Z + SHELF_REL_Z            # ≈ 0.75400 m

# Shelf external dimensions (metres)
SHELF_W = 0.30       # width  (updated per annotated measurement image)
SHELF_D = 0.24       # depth  (open face toward robot)
SHELF_H = 0.60       # height (updated per annotated measurement image)
SHELF_T = 0.02       # panel / wall thickness

# Table dimensions
TABLE_LEN = 1.00     # x extent
TABLE_WID = 1.50     # y extent
TABLE_THK = 0.054     # tabletop thickness

# Colours
CLR_TABLE  = [0.60, 0.60, 0.60, 1.0]
CLR_SLAB   = [0.35, 0.35, 0.40, 1.0]
CLR_SHELF  = [0.92, 0.92, 0.92, 1.0]
CLR_LEGS   = [0.25, 0.25, 0.28, 1.0]

# Table centre: only 16 cm of the table extends along +x from the robot base
# → right edge at x=0.16, centre at x = 0.16 − TABLE_LEN/2 = −0.34
TABLE_CX = -0.34
TABLE_CY = -0.07


# ═══════════════════════════════════════════════════════════════════════
# Build shelf obstacles (collision-checked during planning)
# ═══════════════════════════════════════════════════════════════════════

def build_shelf_obstacles():
    """Return list of obstacle dicts for a 2-compartment open-front shelf
    **plus the table surface** (so the planner avoids it).

    The shelf is axis-aligned. The open face points in the +y direction
    (toward the robot).

    Coordinate mapping (rotated 90°):
        width  (32 cm) → x-axis
        depth  (24 cm) → y-axis   (back wall at −y end, open face at +y)
        height (54 cm) → z-axis
    """
    sx, sy, sz = SHELF_X, SHELF_Y, SHELF_Z
    W, D, H, t = SHELF_W, SHELF_D, SHELF_H, SHELF_T
    c = CLR_SHELF

    obstacles = [
        # ── TABLE SURFACE (collision obstacle for the planner) ──
        {"type": "box", "color": CLR_TABLE,
         "pos": [TABLE_CX, TABLE_CY, TABLE_SURFACE_Z - TABLE_THK / 2],
         "half_extents": [TABLE_LEN / 2, TABLE_WID / 2, TABLE_THK / 2]},

        # ── Back wall (at −y end, full height) ──
        {"type": "box", "color": c,
         "pos": [sx, sy - D / 2 + t / 2, sz + H / 2],
         "half_extents": [W / 2, t / 2, H / 2]},

        # ── Left side wall (−x side) ──
        {"type": "box", "color": c,
         "pos": [sx - W / 2 + t / 2, sy, sz + H / 2],
         "half_extents": [t / 2, D / 2, H / 2]},

        # ── Right side wall (+x side) ──
        {"type": "box", "color": c,
         "pos": [sx + W / 2 - t / 2, sy, sz + H / 2],
         "half_extents": [t / 2, D / 2, H / 2]},

        # ── Bottom panel ──
        {"type": "box", "color": c,
         "pos": [sx, sy, sz + t / 2],
         "half_extents": [W / 2, D / 2, t / 2]},

        # ── Middle shelf ──
        {"type": "box", "color": c,
         "pos": [sx, sy, sz + H / 2],
         "half_extents": [W / 2, D / 2, t / 2]},

        # ── Top panel ──
        {"type": "box", "color": c,
         "pos": [sx, sy, sz + H - t / 2],
         "half_extents": [W / 2, D / 2, t / 2]},
    ]
    return obstacles


# ═══════════════════════════════════════════════════════════════════════
# Visual-only scenery (table, slab, legs — not collision-checked)
# ═══════════════════════════════════════════════════════════════════════

def _add_visual_box(cid, pos, half_extents, color):
    """Create a visual-only box body (no collision shape)."""
    vis = p.createVisualShape(p.GEOM_BOX, halfExtents=half_extents,
                              rgbaColor=color, physicsClientId=cid)
    return p.createMultiBody(baseMass=0,
                             baseVisualShapeIndex=vis,
                             basePosition=pos,
                             physicsClientId=cid)


def add_scenery(cid):
    """Add legs and mounting slab as visual-only bodies.
    (Table surface is already a collision obstacle from build_shelf_obstacles.)"""

    # ── Four table legs ──
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
        _add_visual_box(cid,
                        pos=[lx, ly, leg_h / 2],
                        half_extents=leg_he,
                        color=CLR_LEGS)

    # ── Mounting slab ──
    _add_visual_box(cid,
                    pos=[0.0, 0.0, TABLE_SURFACE_Z + SLAB_THICKNESS / 2],
                    half_extents=[0.10, 0.10, SLAB_THICKNESS / 2],
                    color=CLR_SLAB)


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

def compute_side_grasp_ik(env, bottle_pos):
    """Compute a side-grasp IK: gripper horizontal, approaching the bottle
    from the +y direction (open face of shelf, which faces +y toward robot).

    Gripper body-frame (measured from URDF):
        +x_ee : approach / finger-extension direction (EE base → finger tips)
        y_ee  : finger opening/closing axis
        z_ee  : perpendicular to fingers and approach

    For a SIDE GRASP parallel to the table, approaching in the −y direction
    (into the shelf from the +y open face), we need:
        +x_ee  →  −y_world   (approach into shelf)
        y_ee   →  ±x_world   (fingers open horizontally, within shelf compartment)
        z_ee   →  ∓z_world   (gripper body axis vertical)

    The grasp centre (midpoint of finger tips) is ~0.104 m ahead of the
    EE origin along +x_ee.  So the EE origin must be placed 0.104 m
    *behind* the bottle along the approach direction.
    """
    cid = env.physics_client
    n_movable = len(env._all_joint_indices)

    # ── Step 1: Compute the desired grasp-centre position ──────────
    # Grasp centre = bottle position (approach from +y, open face of shelf)
    grasp_centre = np.array(bottle_pos, dtype=float)
    grasp_centre[2] += 0.01   # 1 cm up along z

    # EE-to-grasp-centre offset along +x_ee (measured: 0.1045 m)
    GRIPPER_DEPTH = 0.1045

    # Approach direction in world (into shelf = −y)
    approach_dir = np.array([0.0, -1.0, 0.0])

    # ── Step 2: Desired EE position ───────────────────────────────
    # Place EE origin GRIPPER_DEPTH behind the grasp centre
    # Try several pre-grasp stand-offs (further back = easier IK)
    standoffs = [0.035, 0.05, 0.08, 0.10, 0.12, 0.15, 0.20, 0.25, 0.30]

    # ── Step 3: Desired EE orientation ────────────────────────────
    # For a side grasp with gripper PARALLEL to the table:
    #   x_ee = [0,-1,0]  approach into shelf (−y)
    #   y_ee = [±1,0,0]  fingers open along ±x (within shelf compartment)
    #   z_ee = cross(x_ee, y_ee)
    from scipy.spatial.transform import Rotation as Rot

    orns = []
    for sign in [+1, -1]:
        # Rotation matrix: columns = [x_ee, y_ee, z_ee] in world coords
        x_ee = np.array([0.0, -1.0, 0.0])           # approach = −y
        y_ee = np.array([sign * 1.0, 0.0, 0.0])     # fingers open along ±x
        z_ee = np.cross(x_ee, y_ee)
        R = np.column_stack([x_ee, y_ee, z_ee])
        q_orn = Rot.from_matrix(R).as_quat().tolist()  # [x,y,z,w]
        orns.append(q_orn)

    seeds = [
        [0.0, -np.pi / 2, np.pi / 2, -np.pi / 2, -np.pi / 2, 0.0],
        [-np.pi / 4, -np.pi / 2, np.pi / 2, -np.pi / 2, -np.pi / 2, 0.0],
        [-np.pi / 2, -np.pi / 3, np.pi / 3, -np.pi / 2, -np.pi / 2, 0.0],
        [-np.pi / 3, -np.pi / 4, np.pi / 3, -np.pi / 3, -np.pi / 2, 0.0],
        [-0.8, -0.5, 1.0, -1.0, -1.57, 0.0],
        # Elbow-down / arm-low seeds to avoid shelf side-wall collision
        [-1.2, -1.5, 1.5, -1.5, -1.57, 0.0],
        [-1.5, -1.2, 1.0, -1.0, -1.57, 0.0],
        [-1.0, -2.0, 1.8, -1.4, -1.57, 0.0],
        [-1.2, -0.8, 1.2, -0.4, -1.2, 0.0],
    ]

    best_q = None
    best_pos = None

    for so in standoffs:
        ee_target = grasp_centre - (GRIPPER_DEPTH + so) * approach_dir
        for oi, orn_q in enumerate(orns):
            for si, seed in enumerate(seeds):
                # Reset joints to seed so IK starts from different branch
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
                        x_actual = R[:, 0]  # approach / finger-extend axis
                        print(f"[IK]  Collision-free SIDE grasp found!")
                        print(f"      standoff={so:.2f}m, orn_idx={oi}, seed={si}")
                        print(f"      EE pos : [{ee_actual[0]:.3f}, {ee_actual[1]:.3f}, {ee_actual[2]:.3f}]")
                        print(f"      x_ee (approach): [{x_actual[0]:.3f}, {x_actual[1]:.3f}, {x_actual[2]:.3f}]")
                        print(f"      y_ee (open):     [{R[0,1]:.3f}, {R[1,1]:.3f}, {R[2,1]:.3f}]")
                        return q_arm, ee_target.tolist()

                if best_q is None:
                    best_q = q_arm
                    best_pos = ee_target.tolist()

    print("[IK]  WARNING: No collision-free IK found, using best fallback.")
    return best_q, best_pos


def main():
    from manipulator_env.demo_cli import parse_demo_args, append_demo_result_csv
    _args = parse_demo_args()
    # Build shelf obstacle list (includes table)
    shelf_obstacles = build_shelf_obstacles()

    # Bottle position (upper compartment)
    upper_floor_z = SHELF_Z + SHELF_H / 2 + SHELF_T / 2
    bottle_pos = [SHELF_X, SHELF_Y, upper_floor_z + 0.08]

    # ── Phase 1: Headless IK computation ──────────────────────────
    print("[IK] Loading PyBullet (headless) for IK computation ...")
    ik_env = UR10eRobotiqEnv(
        gui=False,
        obstacles=shelf_obstacles,
        base_position=[0.0, 0.0, ROBOT_BASE_Z],
        base_orientation=p.getQuaternionFromEuler([0, 0, np.pi]),
    )

    # Load mustard bottle in headless env (collision enabled with robot)
    mustard_urdf = "/home/muhayy/Documents/forsight-tamp/assets/ycb_objects/ycb_assets/006_mustard_bottle.urdf"
    ik_cid = ik_env.physics_client
    ik_bottle_id = p.loadURDF(
        mustard_urdf,
        basePosition=bottle_pos,
        baseOrientation=p.getQuaternionFromEuler([0, 0, np.deg2rad(-60)]),
        useFixedBase=True,
        globalScaling=0.1,
        physicsClientId=ik_cid,
    )
    # Add bottle to obstacle list so is_collision_free checks against it
    ik_env.obstacle_ids.append(ik_bottle_id)

    q_start = np.array([np.pi / 2, -np.pi / 2, np.pi / 2, -np.pi / 2, -np.pi / 2, 0.0])
    q_goal, grasp_pos = compute_side_grasp_ik(ik_env, bottle_pos)

    # Verify in headless env
    assert ik_env.is_collision_free(q_start), "Start config is in collision!"
    assert ik_env.is_collision_free(q_goal), "Goal config is in collision!"

    pos_s, _ = ik_env.get_ee_pose(q_start)
    pos_g, orn_g = ik_env.get_ee_pose(q_goal)
    R_goal = np.array(p.getMatrixFromQuaternion(orn_g)).reshape(3, 3)

    print(f"[IK] Goal found — q_goal: [{', '.join(f'{v:.3f}' for v in q_goal)}]")
    print(f"[IK] Goal EE pos: [{pos_g[0]:.3f}, {pos_g[1]:.3f}, {pos_g[2]:.3f}]")

    ik_env.disconnect()
    print("[IK] Headless env disconnected.\n")

    # ── Phase 2: main environment (GUI or headless) + planner ─────
    mode = 'headless' if _args.headless else 'GUI'
    print(f"[ENV] Loading PyBullet ({mode}) ...")
    env = UR10eRobotiqEnv(
        gui=not _args.headless,
        obstacles=shelf_obstacles,
        base_position=[0.0, 0.0, ROBOT_BASE_Z],
        base_orientation=p.getQuaternionFromEuler([0, 0, np.pi]),
    )

    # Add visual scenery (legs, slab)
    add_scenery(env.physics_client)

    # Load YCB mustard bottle (collision enabled with robot)
    cid = env.physics_client
    mustard_id = p.loadURDF(
        mustard_urdf,
        basePosition=bottle_pos,
        baseOrientation=p.getQuaternionFromEuler([0, 0, np.deg2rad(-60)]),
        useFixedBase=True,
        globalScaling=0.1,
        physicsClientId=cid,
    )
    # Add bottle to obstacle list so planner collision checks include it
    env.obstacle_ids.append(mustard_id)

    # Re-verify in GUI env
    pos_s, _ = env.get_ee_pose(q_start)
    pos_g, orn_g = env.get_ee_pose(q_goal)
    R = np.array(p.getMatrixFromQuaternion(orn_g)).reshape(3, 3)

    # Configure camera
    p.resetDebugVisualizerCamera(
        cameraDistance=1.20,
        cameraYaw=241.60,
        cameraPitch=-27.40,
        cameraTargetPosition=[-0.345, -0.355, 0.970],
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

    print("=" * 62)
    print("  Real Setup — RIT* Shelf Grasp Planning")
    print("=" * 62)
    print(f"  Bottle    : [{bottle_pos[0]:.3f}, {bottle_pos[1]:.3f}, {bottle_pos[2]:.3f}]")
    print(f"  Start EE  : [{pos_s[0]:.3f}, {pos_s[1]:.3f}, {pos_s[2]:.3f}]")
    print(f"  Goal  EE  : [{pos_g[0]:.3f}, {pos_g[1]:.3f}, {pos_g[2]:.3f}]")
    print(f"  x_ee (approach) : [{R[0,0]:.3f}, {R[1,0]:.3f}, {R[2,0]:.3f}]")
    print(f"  y_ee (open)     : [{R[0,1]:.3f}, {R[1,1]:.3f}, {R[2,1]:.3f}]")
    print(f"  Goal cfg  : [{', '.join(f'{v:.3f}' for v in q_goal)}]")
    print("=" * 62)
    print()

    # Run chosen planner with fast diagonal metric
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
            'demo': 'UR10_pick_shelf',
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

        # ── Save world state ──
        upper_floor_z = SHELF_Z + SHELF_H / 2 + SHELF_T / 2
        with open("results/UR10_pick_shelf_world_state.txt", "w") as f:
            f.write("=" * 62 + "\n")
            f.write("  Real Setup — World State\n")
            f.write("=" * 62 + "\n\n")

            f.write("--- Robot ---\n")
            f.write(f"  Base position (world)  : [0.000, 0.000, {ROBOT_BASE_Z:.5f}]\n")
            f.write(f"  Mounting slab thickness : {SLAB_THICKNESS:.5f} m\n\n")

            f.write("--- Table ---\n")
            f.write(f"  Surface Z     : {TABLE_SURFACE_Z:.5f} m\n")
            f.write(f"  Centre (x,y)  : [{TABLE_CX:.5f}, {TABLE_CY:.5f}]\n")
            f.write(f"  Dimensions    : {TABLE_LEN:.2f} x {TABLE_WID:.2f} x {TABLE_THK:.2f} m (L x W x Thick)\n\n")

            f.write("--- Shelf (2-compartment, open-front) ---\n")
            f.write(f"  World position (bottom-centre) : [{SHELF_X:.5f}, {SHELF_Y:.5f}, {SHELF_Z:.5f}]\n")
            f.write(f"  Relative to robot base         : [{SHELF_REL_X:.5f}, {SHELF_REL_Y:.5f}, {SHELF_REL_Z:.5f}]\n")
            f.write(f"  Dimensions (W x D x H)         : {SHELF_W:.2f} x {SHELF_D:.2f} x {SHELF_H:.2f} m\n")
            f.write(f"  Panel thickness                 : {SHELF_T:.2f} m\n")
            f.write(f"  Upper compartment floor Z       : {upper_floor_z:.5f} m\n\n")

            f.write("--- Mustard Bottle (YCB 006) ---\n")
            f.write(f"  Position (world) : [{bottle_pos[0]:.5f}, {bottle_pos[1]:.5f}, {bottle_pos[2]:.5f}]\n")
            f.write(f"  Scale            : 0.1\n\n")

            f.write("--- Obstacle list (collision-checked) ---\n")
            for i, obs in enumerate(shelf_obstacles):
                f.write(f"  [{i}] {obs['type']}  pos={[round(v,5) for v in obs['pos']]}  "
                        f"half_extents={[round(v,5) for v in obs['half_extents']]}\n")

            f.write("\n--- Start Configuration ---\n")
            f.write(f"  q_start (rad) : [{', '.join(f'{v:.6f}' for v in q_start)}]\n")
            f.write(f"  EE position   : [{pos_s[0]:.5f}, {pos_s[1]:.5f}, {pos_s[2]:.5f}]\n\n")

            f.write("--- Goal Configuration ---\n")
            f.write(f"  q_goal  (rad) : [{', '.join(f'{v:.6f}' for v in q_goal)}]\n")
            f.write(f"  EE position   : [{pos_g[0]:.5f}, {pos_g[1]:.5f}, {pos_g[2]:.5f}]\n")
            f.write(f"  x_ee (approach) : [{R[0,0]:.5f}, {R[1,0]:.5f}, {R[2,0]:.5f}]\n")
            f.write(f"  y_ee (open)     : [{R[0,1]:.5f}, {R[1,1]:.5f}, {R[2,1]:.5f}]\n\n")

            f.write("--- Planner ---\n")
            f.write(f"  Algorithm     : RIT*\n")
            f.write(f"  Metric        : DiagonalAnisotropicMetric\n")
            f.write(f"  Weights       : {[round(w,5) for w in _UR10E_WEIGHTS]}\n")
            f.write(f"  Batch size    : 200\n")
            f.write(f"  Max iters     : 300\n")
            f.write(f"  Path cost     : {cost:.6f}\n")
            f.write(f"  Waypoints     : {len(path)}\n")

        print("[FILE] Saved results/UR10_pick_shelf_world_state.txt")

        # ── Save complete path ──
        with open("results/UR10_pick_shelf_path.txt", "w") as f:
            f.write("=" * 62 + "\n")
            f.write("  Real Setup — Complete Path (Joint Configurations)\n")
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

        print("[FILE] Saved results/UR10_pick_shelf_path.txt")

    else:
        print("\n[RESULT] No path found.")

    # Path animation / GIF / loop
    if path:
        path_fine = interpolate_path(path, max_step=0.02)

        if _args.save_gif:
            from manipulator_env.demo_cli import save_path_gif
            _gif_tag = _args.planner.replace('*', '').replace(' ', '_')
            _gif_path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                'visualization', 'gifs',
                f'pybullet_UR10_pick_shelf_{_gif_tag}.gif')
            save_path_gif(
                env, path_fine, _gif_path,
                cam_yaw=241.60, cam_pitch=-27.40, cam_distance=1.20,
                cam_target=[-0.345, -0.355, 0.970],
                step=3, fps=20)
            print(f"[GIF] Saved {_gif_path}")

        if _args.headless:
            env.disconnect()
            return

        # One-time animation with red EE trail (matches old animate=True behaviour)
        print("\n[ANIM] Showing path with EE trail ...")
        env.set_joint_positions(q_start)
        p.stepSimulation(physicsClientId=cid)
        time.sleep(0.5)
        env.visualize_path(path_fine, delay=0.02, trail=True)
        time.sleep(1.0)

        print("\n  Looping path animation. Press Ctrl+C to exit.")
        try:
            while True:
                print_camera_view()
                env.set_joint_positions(q_start)
                p.stepSimulation(physicsClientId=cid)
                time.sleep(0.5)
                for q in path_fine:
                    env.set_joint_positions(q)
                    p.stepSimulation(physicsClientId=cid)
                    time.sleep(0.02)
                time.sleep(1.0)
        except KeyboardInterrupt:
            print("\nShutting down ...")
        finally:
            env.disconnect()
    else:
        print("\n  No path to animate. Press Ctrl+C to exit.")
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
