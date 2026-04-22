#!/usr/bin/env python3
"""
test_env.py — Replicate the run_real_setup_test world in PyBullet, hold the
robot at a fixed configuration, and keep the GUI open.

No planner is called; the robot is simply teleported to ``Q_TARGET`` and the
simulation is stepped idly until the user closes the window or hits Ctrl+C.
"""

import os
import sys
import time
import numpy as np
import pybullet as p

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from manipulator_env.pybullet_env import UR10eRobotiqEnv
from UR10_pick_shelf import (
    ROBOT_BASE_Z, SHELF_X, SHELF_Y, SHELF_Z, SHELF_H, SHELF_T,
    build_shelf_obstacles, add_scenery,
)

Q_TARGET = np.array([
    0.388500,
    -1.153566,
    1.438728,
    -3.427242,
    -0.388503,
    0.000483,
])

MUSTARD_URDF = "/home/muhayy/Documents/forsight-tamp/assets/ycb_objects/ycb_assets/006_mustard_bottle.urdf"


def main():
    shelf_obstacles = build_shelf_obstacles()

    print("[ENV] Loading PyBullet GUI ...")
    env = UR10eRobotiqEnv(
        gui=True,
        obstacles=shelf_obstacles,
        base_position=[0.0, 0.0, ROBOT_BASE_Z],
        base_orientation=p.getQuaternionFromEuler([0, 0, np.pi]),
    )
    cid = env.physics_client

    add_scenery(cid)

    # Mustard bottle on the upper shelf compartment (same placement as the
    # real-setup demo)
    upper_floor_z = SHELF_Z + SHELF_H / 2 + SHELF_T / 2
    bottle_pos = [SHELF_X, SHELF_Y, upper_floor_z + 0.08]
    if os.path.isfile(MUSTARD_URDF):
        p.loadURDF(
            MUSTARD_URDF,
            basePosition=bottle_pos,
            baseOrientation=p.getQuaternionFromEuler([0, 0, np.deg2rad(-60)]),
            useFixedBase=True,
            globalScaling=0.1,
            physicsClientId=cid,
        )
    else:
        print(f"[WARN] Mustard bottle URDF not found at {MUSTARD_URDF}; skipping.")

    p.resetDebugVisualizerCamera(
        cameraDistance=1.20,
        cameraYaw=241.60,
        cameraPitch=-27.40,
        cameraTargetPosition=[-0.345, -0.355, 0.970],
        physicsClientId=cid,
    )
    p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 0, physicsClientId=cid)

    env.set_joint_positions(Q_TARGET)
    for _ in range(10):
        p.stepSimulation(physicsClientId=cid)

    ee_pos, _ = env.get_ee_pose(Q_TARGET)
    print(f"[CFG] q      = [{', '.join(f'{v:.4f}' for v in Q_TARGET)}]")
    print(f"[CFG] EE pos = [{ee_pos[0]:.4f}, {ee_pos[1]:.4f}, {ee_pos[2]:.4f}]")
    print("[GUI] Holding pose. Close the PyBullet window or press Ctrl+C to exit.")

    try:
        while p.isConnected(physicsClientId=cid):
            p.stepSimulation(physicsClientId=cid)
            time.sleep(1 / 240)
    except KeyboardInterrupt:
        print("\nShutting down ...")
    finally:
        env.disconnect()


if __name__ == "__main__":
    main()
