#!/usr/bin/env python3
"""Quick headless test: find good start/goal on opposite sides of the 26cm-thick wall."""
import sys, os, numpy as np, pybullet as p
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from manipulator_env.pybullet_env import UR10eRobotiqEnv

# ── constants (same as run_wall_env.py) ──
TABLE_SURFACE_Z = 0.75
SLAB_THICKNESS  = 0.01990
ROBOT_BASE_Z    = TABLE_SURFACE_Z + SLAB_THICKNESS

WALL_L = 0.54;  WALL_X = -0.45;  WALL_Y = 0.0
WALL_W = 0.26;  WALL_H = 0.50
WALL_Z_BOT = ROBOT_BASE_Z
WALL_Z_MID = WALL_Z_BOT + WALL_H / 2

CAN_HEIGHT = 0.11
GRIPPER_DEPTH = 0.1045

obstacles = [
    {"type": "box", "color": [0.6]*3+[1],
     "pos": [-0.34, -0.07, TABLE_SURFACE_Z - 0.025],
     "half_extents": [0.50, 0.75, 0.025]},
    {"type": "box", "color": [0.35, 0.45, 0.60, 0.85],
     "pos": [WALL_X, WALL_Y, WALL_Z_MID],
     "half_extents": [WALL_L/2, WALL_W/2, WALL_H/2]},
]

env = UR10eRobotiqEnv(gui=False, obstacles=obstacles,
                      base_position=[0.0, 0.0, ROBOT_BASE_Z])
cid = env.physics_client

TOP_DOWN_ORN = list(p.getQuaternionFromEuler([0, np.pi/2, 0]))

# Wall bounding box
wall_x_min = WALL_X - WALL_L/2   # -0.72
wall_x_max = WALL_X + WALL_L/2   # -0.18
wall_y_min = WALL_Y - WALL_W/2   # -0.13
wall_y_max = WALL_Y + WALL_W/2   # +0.13
wall_z_top = WALL_Z_BOT + WALL_H # ~1.27

print(f"Wall bbox: x=[{wall_x_min:.2f}, {wall_x_max:.2f}], "
      f"y=[{wall_y_min:.2f}, {wall_y_max:.2f}], z_top={wall_z_top:.2f}")
print(f"Robot base: [0, 0, {ROBOT_BASE_Z:.3f}]")
print()

def find_ik(target_pos, label="", n_random=300):
    """Find collision-free IK for target_pos with top-down orientation."""
    ee_target = np.array(target_pos, dtype=float)
    desired_orn = np.array(TOP_DOWN_ORN)
    n_movable = len(env._all_joint_indices)
    
    # Seeds
    td_seed = [-1.50, 2.37, -2.44, -1.57, -4.57]
    seeds = []
    for j0 in np.linspace(-np.pi, np.pi, 25):
        seeds.append([j0] + td_seed)
        seeds.append([j0, -1.0, 1.5, -2.0, -1.57, 0.0])
        seeds.append([j0, -0.5, 1.0, -2.0, -1.57, 0.0])
        seeds.append([j0, -1.2, 2.0, -3.94, -1.57, 0.0])
    
    base_seed = [-np.pi/2, np.pi/2, -np.pi/2, -np.pi/2, 0.0]
    for j0 in np.linspace(-np.pi, np.pi, 13):
        seeds.append([j0] + base_seed)
    
    rng = np.random.RandomState(42)
    for _ in range(n_random):
        seeds.append(rng.uniform(-np.pi, np.pi, size=6).tolist())
    
    candidates = []
    for use_orn in [True, False]:
        if candidates and not use_orn:
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
                jointRanges=[4*np.pi]*6 + [0.01]*(n_movable-6),
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
        print(f"  [{label}] FAILED — no collision-free IK")
        return None
    
    candidates.sort(key=lambda c: (-c[2], c[1]))
    best_q, best_pe, best_od, best_si = candidates[0]
    ee_pos, _ = env.get_ee_pose(best_q)
    print(f"  [{label}] OK — {len(candidates)} cands, "
          f"pos_err={best_pe:.4f}, orn_dot={best_od:.4f}, "
          f"EE=[{ee_pos[0]:.3f}, {ee_pos[1]:.3f}, {ee_pos[2]:.3f}]")
    print(f"    q = [{', '.join(f'{v:.4f}' for v in best_q)}]")
    return best_q

# ── Test multiple candidate positions ──
# The wall has an open end at x=-0.18 (closer to robot).
# Strategy: place can and start near the open end so the arm can swing around.
# Also try elevated positions (above wall top) and positions near the far end.

print("=" * 60)
print("Testing candidate start/goal positions")
print("=" * 60)

# Candidate can positions (+y side)
can_positions = [
    (-0.10, 0.40, "near open end"),
    (-0.15, 0.35, "at wall edge"),
    (-0.20, 0.45, "moderate"),
    (-0.30, 0.40, "mid-wall"),
    (-0.10, 0.50, "far +y, near open end"),
]

# Candidate start positions (−y side)
start_positions = [
    (-0.10, -0.40, TABLE_SURFACE_Z + 0.20, "near open end, low"),
    (-0.15, -0.35, TABLE_SURFACE_Z + 0.25, "at wall edge"),
    (-0.10, -0.35, TABLE_SURFACE_Z + 0.30, "near open end, mid"),
    (-0.20, -0.40, TABLE_SURFACE_Z + 0.20, "moderate"),
    (0.00, -0.40, TABLE_SURFACE_Z + 0.20, "past open end"),
    (-0.10, -0.50, TABLE_SURFACE_Z + 0.20, "far -y, near open end"),
]

# Test all combos
results = []
for ci, (cx, cy, clabel) in enumerate(can_positions):
    can_z = TABLE_SURFACE_Z + CAN_HEIGHT / 2
    grasp_z = can_z + GRIPPER_DEPTH + 0.08
    ee_grasp = [cx, cy, grasp_z]
    
    print(f"\n--- Can #{ci}: ({cx:.2f}, {cy:.2f}) [{clabel}] ---")
    print(f"    Grasp EE: [{cx:.3f}, {cy:.3f}, {grasp_z:.3f}]")
    q_goal = find_ik(ee_grasp, f"goal-can{ci}")
    if q_goal is None:
        continue
    
    for si, (sx, sy, sz, slabel) in enumerate(start_positions):
        print(f"  Start #{si}: ({sx:.2f}, {sy:.2f}, {sz:.2f}) [{slabel}]")
        q_start = find_ik([sx, sy, sz], f"start{si}")
        if q_start is None:
            continue
        
        # Check C-space distance
        dq = np.abs(q_goal - q_start)
        c_dist = np.linalg.norm(dq)
        # Check if j0 difference suggests arm swings around
        j0_diff = abs(q_goal[0] - q_start[0])
        
        results.append({
            'can': (cx, cy, clabel),
            'start': (sx, sy, sz, slabel),
            'q_start': q_start,
            'q_goal': q_goal,
            'c_dist': c_dist,
            'j0_diff': j0_diff,
        })
        print(f"    C-dist={c_dist:.3f}, j0_diff={j0_diff:.3f}")

print("\n" + "=" * 60)
print(f"Total valid combos: {len(results)}")
print("=" * 60)

# Sort by C-space distance (shorter = easier for planner)
results.sort(key=lambda r: r['c_dist'])
for i, r in enumerate(results[:5]):
    cx, cy, cl = r['can']
    sx, sy, sz, sl = r['start']
    print(f"\n#{i+1}: C-dist={r['c_dist']:.3f}, j0_diff={r['j0_diff']:.3f}")
    print(f"  Can: ({cx:.2f}, {cy:.2f}) [{cl}]")
    print(f"  Start: ({sx:.2f}, {sy:.2f}, {sz:.2f}) [{sl}]")
    print(f"  q_start = [{', '.join(f'{v:.4f}' for v in r['q_start'])}]")
    print(f"  q_goal  = [{', '.join(f'{v:.4f}' for v in r['q_goal'])}]")

env.disconnect()
