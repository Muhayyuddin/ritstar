#!/usr/bin/env python3
"""
Headless search for good start/goal configs on opposite sides of the wall.
Tests multiple EE target positions on each side, picks configs with best
reachability (low pos error, top-down orientation, collision-free).
"""
import sys, os, numpy as np, pybullet as p

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from manipulator_env.pybullet_env import UR10eRobotiqEnv

# ── Constants (same as run_wall_env.py) ──
TABLE_SURFACE_Z = 0.75
SLAB_THICKNESS  = 0.01990
ROBOT_BASE_Z    = TABLE_SURFACE_Z + SLAB_THICKNESS

TABLE_LEN = 1.00; TABLE_WID = 1.50; TABLE_THK = 0.05
TABLE_CX  = -0.34; TABLE_CY = -0.07

WALL_L = 0.54; WALL_X = -0.45; WALL_Y = 0.0
WALL_W = 0.26; WALL_H = 0.50
WALL_Z_BOT = ROBOT_BASE_Z
WALL_Z_MID = WALL_Z_BOT + WALL_H / 2

CAN_HEIGHT = 0.11
GRIPPER_DEPTH = 0.1045

CLR_WALL  = [0.35, 0.45, 0.60, 0.85]
CLR_TABLE = [0.60, 0.60, 0.60, 1.0]

obstacles = [
    {"type": "box", "color": CLR_TABLE,
     "pos": [TABLE_CX, TABLE_CY, TABLE_SURFACE_Z - TABLE_THK / 2],
     "half_extents": [TABLE_LEN / 2, TABLE_WID / 2, TABLE_THK / 2]},
    {"type": "box", "color": CLR_WALL,
     "pos": [WALL_X, WALL_Y, WALL_Z_MID],
     "half_extents": [WALL_L / 2, WALL_W / 2, WALL_H / 2]},
]

TOP_DOWN_ORN = list(p.getQuaternionFromEuler([0, np.pi / 2, 0]))

def find_ik_candidates(env, target_pos, n_random=300):
    """Find all collision-free IK solutions for target_pos."""
    cid = env.physics_client
    n_movable = len(env._all_joint_indices)
    ee_target = np.array(target_pos, dtype=float)
    desired_orn = np.array(TOP_DOWN_ORN)

    td_seed = [-1.50, 2.37, -2.44, -1.57, -4.57]
    seeds = []
    for j0 in np.linspace(-np.pi, np.pi, 25):
        seeds.append([j0] + td_seed)
        seeds.append([j0, -1.50, 2.37, -2.44, -1.57, -4.57])
        seeds.append([j0, -1.0, 1.5, -2.0, -1.57, 0.0])
        seeds.append([j0, -0.5, 1.0, -2.0, -1.57, 0.0])
        seeds.append([j0, -1.2, 2.0, -3.94, -1.57, 0.0])
        seeds.append([j0, -2.0, 1.5, -1.0, -1.57, 0.0])
        seeds.append([j0, -1.8, 2.5, -2.3, -1.57, 0.0])

    base_seed = [-np.pi / 2, np.pi / 2, -np.pi / 2, -np.pi / 2, 0.0]
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
                candidates.append((q_arm, pos_err, orn_dot, ee_actual))
    return candidates


def main():
    print("=" * 70)
    print("  Searching for good start/goal configs on opposite sides of wall")
    print("=" * 70)
    print(f"  Wall: x={WALL_X}, y={WALL_Y}, L={WALL_L}, W={WALL_W}, H={WALL_H}")
    print(f"  Wall y-range: [{WALL_Y - WALL_W/2:.2f}, {WALL_Y + WALL_W/2:.2f}]")
    print(f"  Wall x-range: [{WALL_X - WALL_L/2:.2f}, {WALL_X + WALL_L/2:.2f}]")
    print(f"  Wall top z:   {WALL_Z_BOT + WALL_H:.3f}")
    print()

    env = UR10eRobotiqEnv(gui=False, obstacles=obstacles,
                          base_position=[0.0, 0.0, ROBOT_BASE_Z])

    # ── Search grid: try multiple EE positions on each side ──
    # Wall y-edge: ±0.13.  So +y side needs y > 0.13+clearance, −y side y < -0.13-clearance
    # We want positions that are reachable (not too far from robot base at origin)

    grasp_height = TABLE_SURFACE_Z + CAN_HEIGHT / 2 + GRIPPER_DEPTH + 0.06

    # Candidate target positions — CLEARLY past wall x-end (-0.18) for arm clearance
    # +y side (can side): vary x and y
    plus_y_targets = []
    for x in [0.10, 0.05, 0.0, -0.05, -0.10]:
        for y in [0.25, 0.30, 0.35, 0.40, 0.45, 0.50]:
            for z in [grasp_height, TABLE_SURFACE_Z + 0.20, TABLE_SURFACE_Z + 0.30]:
                plus_y_targets.append(([x, y, z], f"+y x={x} y={y} z={z:.2f}"))

    # −y side (start side): vary x and y
    minus_y_targets = []
    for x in [0.10, 0.05, 0.0, -0.05, -0.10]:
        for y in [-0.25, -0.30, -0.35, -0.40, -0.45, -0.50]:
            for z in [grasp_height, TABLE_SURFACE_Z + 0.20, TABLE_SURFACE_Z + 0.30]:
                minus_y_targets.append(([x, y, z], f"−y x={x} y={y} z={z:.2f}"))

    print(f"Testing {len(plus_y_targets)} +y targets, {len(minus_y_targets)} −y targets ...\n")

    # Find best positions on each side
    best_plus_y = []
    for tgt, label in plus_y_targets:
        cands = find_ik_candidates(env, tgt)
        if cands:
            cands.sort(key=lambda c: (-c[2], c[1]))
            best = cands[0]
            best_plus_y.append((tgt, label, best, len(cands)))

    best_minus_y = []
    for tgt, label in minus_y_targets:
        cands = find_ik_candidates(env, tgt)
        if cands:
            cands.sort(key=lambda c: (-c[2], c[1]))
            best = cands[0]
            best_minus_y.append((tgt, label, best, len(cands)))

    # Sort by: most candidates (most reachable), then best orn_dot
    best_plus_y.sort(key=lambda x: (-x[3], -x[2][2], x[2][1]))
    best_minus_y.sort(key=lambda x: (-x[3], -x[2][2], x[2][1]))

    print("\n" + "=" * 70)
    print("  TOP 10 +y side positions (can/goal side)")
    print("=" * 70)
    for i, (tgt, label, (q, perr, odot, ee), ncand) in enumerate(best_plus_y[:10]):
        print(f"  [{i}] {label}")
        print(f"      EE actual: [{ee[0]:.3f}, {ee[1]:.3f}, {ee[2]:.3f}]  "
              f"pos_err={perr:.4f}  orn_dot={odot:.4f}  candidates={ncand}")
        print(f"      q = [{', '.join(f'{v:.4f}' for v in q)}]")

    print("\n" + "=" * 70)
    print("  TOP 10 −y side positions (start side)")
    print("=" * 70)
    for i, (tgt, label, (q, perr, odot, ee), ncand) in enumerate(best_minus_y[:10]):
        print(f"  [{i}] {label}")
        print(f"      EE actual: [{ee[0]:.3f}, {ee[1]:.3f}, {ee[2]:.3f}]  "
              f"pos_err={perr:.4f}  orn_dot={odot:.4f}  candidates={ncand}")
        print(f"      q = [{', '.join(f'{v:.4f}' for v in q)}]")

    # Print recommended pair
    if best_plus_y and best_minus_y:
        print("\n" + "=" * 70)
        print("  RECOMMENDED PAIR")
        print("=" * 70)
        g = best_plus_y[0]
        s = best_minus_y[0]
        print(f"  START (−y): EE=[{s[2][3][0]:.3f}, {s[2][3][1]:.3f}, {s[2][3][2]:.3f}]  "
              f"candidates={s[3]}")
        print(f"    q_start = [{', '.join(f'{v:.4f}' for v in s[2][0])}]")
        print(f"  GOAL  (+y): EE=[{g[2][3][0]:.3f}, {g[2][3][1]:.3f}, {g[2][3][2]:.3f}]  "
              f"candidates={g[3]}")
        print(f"    q_goal  = [{', '.join(f'{v:.4f}' for v in g[2][0])}]")
        print(f"  Can position: [{g[0][0]:.3f}, {g[0][1]:.3f}, {TABLE_SURFACE_Z + CAN_HEIGHT/2:.3f}]")

    env.disconnect()


if __name__ == "__main__":
    main()
