#!/usr/bin/env python3
"""Render PyBullet final-path execution as GIF for each scene.

Uses offscreen rendering (DIRECT mode) with p.getCameraImage() to
capture frames as the robot executes the planned path, then assembles
them into a GIF using Pillow.
"""
import sys
sys.stdout.reconfigure(line_buffering=True)

import os
import numpy as np
import pybullet as p
from PIL import Image

from output_paths import GIFS_DIR
from manipulator_env.pybullet_env import UR10eRobotiqEnv
from manipulator_env.planner_interface import (
    ManipulatorInertiaMetric,
    build_rit_star_planner,
    shortcut_smooth,
    interpolate_path,
    riemannian_edge_cost,
)

# ── Scene definitions (same as run_manipulator.py) ────────────────

def scene_tabletop():
    obstacles = [
        {"type": "box", "pos": [0.5, 0.0, 0.4],
         "half_extents": [0.3, 0.3, 0.02], "color": [0.3, 0.3, 0.3, 1.0]},
        {"type": "box", "pos": [0.4, 0.0, 0.55],
         "half_extents": [0.08, 0.08, 0.12], "color": [0.3, 0.3, 0.3, 1.0]},
        {"type": "box", "pos": [0.6, 0.15, 0.50],
         "half_extents": [0.06, 0.06, 0.08], "color": [0.3, 0.3, 0.3, 1.0]},
    ]
    q_start = np.array([0.0, -np.pi / 2, 0.0, -np.pi / 2, 0.0, 0.0])
    q_goal = np.array([1.2, -1.2, 1.0, -1.4, -1.57, 0.0])
    return obstacles, q_start, q_goal


def scene_shelf():
    obstacles = [
        {"type": "box", "pos": [1.22, 0.0, 0.6],
         "half_extents": [0.02, 0.35, 0.30], "color": [0.3, 0.3, 0.3, 1.0]},
        {"type": "box", "pos": [1.05, 0.0, 0.90],
         "half_extents": [0.19, 0.35, 0.02], "color": [0.3, 0.3, 0.3, 1.0]},
        {"type": "box", "pos": [1.05, 0.0, 0.30],
         "half_extents": [0.19, 0.35, 0.02], "color": [0.3, 0.3, 0.3, 1.0]},
        {"type": "box", "pos": [1.05, 0.35, 0.6],
         "half_extents": [0.19, 0.02, 0.30], "color": [0.3, 0.3, 0.3, 1.0]},
        {"type": "box", "pos": [1.05, -0.35, 0.6],
         "half_extents": [0.19, 0.02, 0.30], "color": [0.3, 0.3, 0.3, 1.0]},
    ]
    q_start = np.array([0.0, -2.0, 1.57, -1.1, -1.57, 0.0])
    q_goal = np.array([0.0, -1.0, 1.0, -1.57, -1.57, 0.0])
    return obstacles, q_start, q_goal


def scene_cluttered():
    rng = np.random.default_rng(123)
    obstacles = []
    for pos in [[0.3, 0.2, 0.5], [0.5, -0.1, 0.6], [0.4, 0.3, 0.3],
                [0.6, -0.3, 0.5], [0.3, -0.2, 0.7], [0.5, 0.15, 0.45],
                [0.7, 0.0, 0.4]]:
        r = 0.06 + rng.uniform(0, 0.04)
        obstacles.append({
            "type": "sphere", "pos": pos, "radius": r,
            "color": [0.3, 0.3, 0.3, 1.0],
        })
    q_start = np.array([-1.57, -1.57, 1.57, -1.57, -1.57, 0.0])
    q_goal = np.array([0.7, -0.9, 1.2, -1.8, -1.57, -0.3])
    return obstacles, q_start, q_goal


SCENES = {
    "tabletop":  scene_tabletop,
    "shelf":     scene_shelf,
    "cluttered": scene_cluttered,
}

# ── Camera / rendering ────────────────────────────────────────────

WIDTH, HEIGHT = 640, 480


def capture_frame(physics_client):
    """Capture an RGB frame from the PyBullet offscreen renderer."""
    view = p.computeViewMatrixFromYawPitchRoll(
        cameraTargetPosition=[0.4, 0.0, 0.5],
        distance=1.8,
        yaw=45, pitch=-30, roll=0,
        upAxisIndex=2,
        physicsClientId=physics_client,
    )
    proj = p.computeProjectionMatrixFOV(
        fov=60, aspect=WIDTH / HEIGHT,
        nearVal=0.1, farVal=5.0,
        physicsClientId=physics_client,
    )
    _, _, rgba, _, _ = p.getCameraImage(
        width=WIDTH, height=HEIGHT,
        viewMatrix=view,
        projectionMatrix=proj,
        renderer=p.ER_TINY_RENDERER,
        physicsClientId=physics_client,
    )
    img = np.array(rgba, dtype=np.uint8).reshape(HEIGHT, WIDTH, 4)
    return img[:, :, :3]  # RGB only


def render_path_gif(scene_name, scene_fn, batch_size=200, max_iterations=300):
    """Plan with RIT*, then render the final path execution as a GIF."""
    print(f"\n{'='*60}")
    print(f"  PyBullet GIF: {scene_name}")
    print(f"{'='*60}")

    obstacles, q_start, q_goal = scene_fn()
    env = UR10eRobotiqEnv(gui=False, obstacles=obstacles)
    metric = ManipulatorInertiaMetric(env)
    cid = env.physics_client

    # ── Plan ──
    print(f"  [PLAN] batch_size={batch_size}, max_iter={max_iterations}")
    planner = build_rit_star_planner(
        env, q_start, q_goal, metric=metric,
        batch_size=batch_size, max_iterations=max_iterations,
    )
    path, cost = planner.plan()

    if not path:
        print(f"  [WARN] No solution for {scene_name}, skipping GIF.")
        env.disconnect()
        return

    print(f"  [PLAN] Solution cost={cost:.4f}, waypoints={len(path)}")

    # ── Smooth ──
    path = shortcut_smooth(path, env, metric, max_iters=300)
    smooth_cost = sum(
        riemannian_edge_cost(path[i], path[i + 1], metric)
        for i in range(len(path) - 1)
    )
    print(f"  [SMOOTH] cost={smooth_cost:.4f}, waypoints={len(path)}")

    # ── Interpolate for smooth animation ──
    path_fine = interpolate_path(path, max_step=0.02)
    print(f"  [INTERP] {len(path_fine)} waypoints for animation")

    # ── Add start/goal markers ──
    env.visualize_config(q_start, color=[0, 0, 1, 1])
    env.visualize_config(q_goal, color=[1, 0, 0, 1])

    # ── Capture frames ──
    # Subsample to keep GIF size reasonable (~80-120 frames)
    step = max(1, len(path_fine) // 100)
    frames = []

    # Capture start pose
    env.set_joint_positions(q_start)
    p.stepSimulation(physicsClientId=cid)
    for _ in range(5):
        frames.append(Image.fromarray(capture_frame(cid)))

    # Capture path execution
    prev_pos = None
    for i in range(0, len(path_fine), step):
        q = path_fine[i]
        env.set_joint_positions(q)
        p.stepSimulation(physicsClientId=cid)

        # Draw EE trail
        pos, _ = env.get_ee_pose()
        if prev_pos is not None:
            p.addUserDebugLine(
                prev_pos.tolist(), pos.tolist(),
                lineColorRGB=[0, 1, 0], lineWidth=3, lifeTime=0,
                physicsClientId=cid,
            )
        prev_pos = pos

        frames.append(Image.fromarray(capture_frame(cid)))

    # Capture goal pose (hold)
    env.set_joint_positions(q_goal)
    p.stepSimulation(physicsClientId=cid)
    for _ in range(10):
        frames.append(Image.fromarray(capture_frame(cid)))

    # ── Save GIF ──
    gif_path = os.path.join(GIFS_DIR, f"pybullet_{scene_name}_final_path.gif")
    frames[0].save(
        gif_path,
        save_all=True,
        append_images=frames[1:],
        duration=50,   # ms per frame
        loop=0,
    )
    print(f"  -> saved {gif_path}  ({len(frames)} frames)")

    env.disconnect()


def main():
    os.makedirs(GIFS_DIR, exist_ok=True)
    for name, fn in SCENES.items():
        render_path_gif(name, fn)
    print("\nAll PyBullet GIFs saved!")


if __name__ == "__main__":
    main()
