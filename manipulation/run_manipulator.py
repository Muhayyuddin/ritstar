#!/usr/bin/env python3
"""
run_manipulator.py — Run RIT* planner on UR10e + Robotiq 85 in PyBullet.

Usage:
    python run_manipulator.py              # GUI mode (default)
    python run_manipulator.py --headless   # no GUI, just plan
    python run_manipulator.py --scene 2    # use scene 2 (shelf)

Scenes:
    1 — Tabletop with box obstacles (default)
    2 — Shelf / narrow passage
    3 — Cluttered workspace with spheres
"""

import argparse
import sys
import os
import numpy as np
import time
import pybullet as p

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))

from manipulator_env.pybullet_env import UR10eRobotiqEnv
from manipulator_env.planner_interface import (
    plan_and_execute,
    interpolate_path,
    ManipulatorInertiaMetric,
)


# ═══════════════════════════════════════════════════════════════════════
# Scene definitions
# ═══════════════════════════════════════════════════════════════════════

def scene_tabletop():
    """Scene 1: Table with box obstacles.

    Robot needs to move from one side configuration to another,
    navigating around boxes on a table.
    """
    obstacles = [
        # Table surface
        {"type": "box", "pos": [0.5, 0.0, 0.4],
         "half_extents": [0.3, 0.3, 0.02],
         "color": [0.3, 0.3, 0.3, 1.0]},
        # Box obstacle 1
        {"type": "box", "pos": [0.4, 0.0, 0.55],
         "half_extents": [0.08, 0.08, 0.12],
         "color": [0.3, 0.3, 0.3, 1.0]},
        # Box obstacle 2
        {"type": "box", "pos": [0.6, 0.15, 0.50],
         "half_extents": [0.06, 0.06, 0.08],
         "color": [0.3, 0.3, 0.3, 1.0]},
    ]

    # Start: home-like configuration (EE above/behind table)
    q_start = np.array([0.0, -1.57, 1.57, -1.57, -1.57, 0.0])

    # Goal: arm rotated to the opposite side
    q_goal = np.array([1.57, -1.57, 1.57, -1.57, -1.57, 0.0])

    return obstacles, q_start, q_goal


def scene_shelf():
    """Scene 2: Shelf / narrow passage.

    Robot must navigate around a shelf-like structure in the workspace.
    """
    obstacles = [
        # Shelf back wall
        {"type": "box", "pos": [1.22, 0.0, 0.6],
         "half_extents": [0.02, 0.35, 0.30],
         "color": [0.3, 0.3, 0.3, 1.0]},
        # Shelf top
        {"type": "box", "pos": [1.05, 0.0, 0.90],
         "half_extents": [0.19, 0.35, 0.02],
         "color": [0.3, 0.3, 0.3, 1.0]},
        # Shelf bottom
        {"type": "box", "pos": [1.05, 0.0, 0.30],
         "half_extents": [0.19, 0.35, 0.02],
         "color": [0.3, 0.3, 0.3, 1.0]},
        # Shelf left wall
        {"type": "box", "pos": [1.05, 0.35, 0.6],
         "half_extents": [0.19, 0.02, 0.30],
         "color": [0.3, 0.3, 0.3, 1.0]},
        # Shelf right wall
        {"type": "box", "pos": [1.05, -0.35, 0.6],
         "half_extents": [0.19, 0.02, 0.30],
         "color": [0.3, 0.3, 0.3, 1.0]},
    ]

    # Start: arm lifted above/behind the shelf
    q_start = np.array([0.0, -2.0, 1.57, -1.1, -1.57, 0.0])

    # Goal: EE reaching inside the shelf
    q_goal = np.array([0.0, -1.0, 1.0, -1.57, -1.57, 0.0])

    return obstacles, q_start, q_goal


def scene_cluttered():
    """Scene 3: Cluttered workspace with spheres.

    Multiple spherical obstacles that force the planner to find
    a winding path through C-space.
    """
    rng = np.random.default_rng(123)
    obstacles = []

    # Scattered spheres in the workspace
    sphere_positions = [
        [0.3, 0.2, 0.5],
        [0.5, -0.1, 0.6],
        [0.4, 0.3, 0.3],
        [0.6, -0.3, 0.5],
        [0.3, -0.2, 0.7],
        [0.5, 0.15, 0.45],
        [0.7, 0.0, 0.4],
    ]
    for pos in sphere_positions:
        r = 0.06 + rng.uniform(0, 0.04)
        obstacles.append({
            "type": "sphere", "pos": pos, "radius": r,
            "color": [0.3, 0.3, 0.3, 1.0],
        })

    q_start = np.array([-1.57, -1.57, 1.57, -1.57, -1.57, 0.0])
    q_goal = np.array([0.7, -0.9, 1.2, -1.8, -1.57, -0.3])

    return obstacles, q_start, q_goal


SCENES = {
    1: ("Tabletop with boxes", scene_tabletop),
    2: ("Shelf / narrow passage", scene_shelf),
    3: ("Cluttered with spheres", scene_cluttered),
}


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="RIT* planner demo: UR10e + Robotiq 85 in PyBullet"
    )
    parser.add_argument("--headless", action="store_true",
                        help="Run without GUI")
    parser.add_argument("--scene", type=int, default=1, choices=[1, 2, 3],
                        help="Scene number (1=tabletop, 2=shelf, 3=cluttered)")
    parser.add_argument("--batch-size", type=int, default=200,
                        help="RIT* batch size (default 200)")
    parser.add_argument("--max-iter", type=int, default=300,
                        help="RIT* max iterations (default 300)")
    parser.add_argument("--no-smooth", action="store_true",
                        help="Disable path smoothing")
    args = parser.parse_args()

    scene_name, scene_fn = SCENES[args.scene]
    obstacles, q_start, q_goal = scene_fn()

    print("=" * 65)
    print(f"  RIT* Planner — UR10e + Robotiq 85 Gripper")
    print(f"  Scene {args.scene}: {scene_name}")
    print(f"  GUI: {not args.headless}")
    print("=" * 65)

    # Create environment
    print("\n[ENV] Loading PyBullet environment ...")
    env = UR10eRobotiqEnv(gui=not args.headless, obstacles=obstacles)

    cid = env.physics_client

    # ── Infinite very light grey floor ────────────────────────────
    if not args.headless:
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
    if not args.headless:
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

    # Verify start and goal are collision-free
    env.set_joint_positions(q_start)
    if not env.is_collision_free(q_start):
        print("[WARN] Start configuration is in collision! Adjusting ...")
        # Try small perturbation
        for _ in range(50):
            q_start += np.random.uniform(-0.05, 0.05, size=6)
            if env.is_collision_free(q_start):
                print("[WARN] Adjusted start found.")
                break

    if not env.is_collision_free(q_goal):
        print("[WARN] Goal configuration is in collision! Adjusting ...")
        for _ in range(50):
            q_goal += np.random.uniform(-0.05, 0.05, size=6)
            if env.is_collision_free(q_goal):
                print("[WARN] Adjusted goal found.")
                break

    # Print EE poses
    pos_s, _ = env.get_ee_pose(q_start)
    pos_g, _ = env.get_ee_pose(q_goal)
    print(f"\n[ENV] Start EE position: [{pos_s[0]:.3f}, {pos_s[1]:.3f}, {pos_s[2]:.3f}]")
    print(f"[ENV] Goal  EE position: [{pos_g[0]:.3f}, {pos_g[1]:.3f}, {pos_g[2]:.3f}]")

    # Plan and execute
    print()
    path, cost = plan_and_execute(
        env,
        q_start,
        q_goal,
        batch_size=args.batch_size,
        max_iterations=args.max_iter,
        smooth=not args.no_smooth,
        animate=not args.headless,
    )

    if path:
        print(f"\n{'=' * 65}")
        print(f"  RESULT: Path found with cost {cost:.4f}")
        print(f"  Waypoints: {len(path)}")
        print(f"{'=' * 65}")
    else:
        print("\n[RESULT] No path found. Try increasing --max-iter or --batch-size.")

    # ── Loop path animation until window is closed ────────────
    if path and not args.headless:
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
    elif not args.headless:
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
