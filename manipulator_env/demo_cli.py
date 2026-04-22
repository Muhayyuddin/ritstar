"""demo_cli.py — shared CLI for the UR10e / PyBullet 6-D demos.

Every demo script calls :func:`parse_demo_args` at the top of ``main()`` to
pick up ``--headless`` and ``--planner`` from either ``sys.argv`` or
environment variables (``RIT_HEADLESS``, ``RIT_PLANNER``). This lets
``run_from_config.py`` spawn each demo with any of the benchmarked planners
("RIT*", "Informed RRT*", "BIT*", "AIT*", "EIT*", "APT*") and without a GUI.
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import NamedTuple


class DemoArgs(NamedTuple):
    headless: bool
    planner: str
    seed: int
    max_iterations: int
    batch_size: int
    save_results: bool
    save_gif: bool


def parse_demo_args(default_planner: str = 'RIT*') -> DemoArgs:
    """Parse CLI + env overrides for a PyBullet 6-D demo script."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--headless', action='store_true',
                        help='Run without PyBullet GUI (no animation).')
    parser.add_argument('--planner', default=None,
                        help='Planner name: RIT* (default), Informed RRT*, '
                             'BIT*, AIT*, EIT*, APT* (short aliases accepted).')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for the planner.')
    parser.add_argument('--max-iter', type=int, default=300,
                        help='Planner max iterations (default 300).')
    parser.add_argument('--batch-size', type=int, default=200,
                        help='Planner batch size (default 200).')
    parser.add_argument('--save-results', action='store_true',
                        help='Write per-run CSV row to results/demo_runs.csv.')
    parser.add_argument('--save-gif', action='store_true',
                        help='Record path execution as GIF in visualization/gifs/.')
    parser.add_argument('-h', '--help', action='help')
    # parse_known_args so scripts that want their own flags can coexist
    args, _unknown = parser.parse_known_args()

    # Environment-variable fallbacks (used by the run_from_config dispatcher)
    if not args.headless and os.environ.get('RIT_HEADLESS', '').lower() in ('1', 'true', 'yes'):
        args.headless = True
    if args.planner is None:
        args.planner = os.environ.get('RIT_PLANNER') or default_planner
    if os.environ.get('RIT_SAVE_RESULTS', '').lower() in ('1', 'true', 'yes'):
        args.save_results = True
    if os.environ.get('RIT_SAVE_GIF', '').lower() in ('1', 'true', 'yes'):
        args.save_gif = True
    if os.environ.get('RIT_SEED'):
        try:
            args.seed = int(os.environ['RIT_SEED'])
        except ValueError:
            pass

    return DemoArgs(
        headless=args.headless,
        planner=args.planner,
        seed=args.seed,
        max_iterations=args.max_iter,
        batch_size=args.batch_size,
        save_results=args.save_results,
        save_gif=args.save_gif,
    )


def append_demo_result_csv(row: dict, path: str = 'results/demo_runs.csv') -> None:
    """Append one result row to a CSV, creating a header if the file is new."""
    import csv
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    write_header = not os.path.isfile(path)
    with open(path, 'a', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            w.writeheader()
        w.writerow(row)


# ── GIF capture helpers for PyBullet 6-D demos ───────────────────────

_DEFAULT_GIF_WIDTH = 640
_DEFAULT_GIF_HEIGHT = 480


def _capture_frame(cid: int,
                   cam_target, cam_distance: float,
                   cam_yaw: float, cam_pitch: float,
                   width: int = _DEFAULT_GIF_WIDTH,
                   height: int = _DEFAULT_GIF_HEIGHT):
    """Return a HxWx3 uint8 RGB frame using PyBullet's TINY renderer.

    Works in both GUI and headless modes.
    """
    import numpy as np
    import pybullet as p
    view = p.computeViewMatrixFromYawPitchRoll(
        cameraTargetPosition=list(cam_target),
        distance=float(cam_distance),
        yaw=float(cam_yaw), pitch=float(cam_pitch), roll=0.0,
        upAxisIndex=2,
        physicsClientId=cid,
    )
    proj = p.computeProjectionMatrixFOV(
        fov=60, aspect=width / height,
        nearVal=0.1, farVal=8.0,
        physicsClientId=cid,
    )
    _, _, rgba, _, _ = p.getCameraImage(
        width=width, height=height,
        viewMatrix=view, projectionMatrix=proj,
        renderer=p.ER_TINY_RENDERER,
        physicsClientId=cid,
    )
    arr = np.array(rgba, dtype=np.uint8).reshape(height, width, 4)
    return arr[:, :, :3]


def save_path_gif(env, path, gif_path: str,
                  cam_yaw: float = -45.0,
                  cam_pitch: float = -30.0,
                  cam_distance: float = 1.8,
                  cam_target=(0.0, -0.2, 1.0),
                  step: int = 1,
                  fps: int = 20,
                  pre_hold: int = 6,
                  post_hold: int = 10,
                  pre_reset=None) -> str:
    """Record a GIF of the robot executing ``path`` in joint space.

    Parameters
    ----------
    env : UR10eRobotiqEnv
    path : list of (6,) joint arrays
    gif_path : str — output file path (directory will be created if needed)
    cam_yaw, cam_pitch, cam_distance, cam_target : float
        Camera params passed to PyBullet's TINY renderer.
    step : int — capture every Nth waypoint (use >1 for long paths).
    fps : int — frames per second in the output GIF.
    pre_hold, post_hold : int — extra copies of the first / last frame.
    pre_reset : callable(env) or None
        Called once before capture (e.g. to reset attached bodies).

    Returns the final gif_path.
    """
    import pybullet as p
    from PIL import Image

    os.makedirs(os.path.dirname(gif_path) or '.', exist_ok=True)
    cid = env.physics_client

    if pre_reset is not None:
        try:
            pre_reset(env)
        except Exception as exc:
            print(f'  [WARN] pre_reset callback failed: {exc}')

    frames = []

    def _grab():
        frames.append(Image.fromarray(
            _capture_frame(cid, cam_target, cam_distance,
                           cam_yaw, cam_pitch)))

    if not path:
        # At least show the static start pose
        _grab()
    else:
        env.set_joint_positions(path[0])
        p.stepSimulation(physicsClientId=cid)
        for _ in range(max(1, pre_hold)):
            _grab()
        for i in range(0, len(path), max(1, step)):
            env.set_joint_positions(path[i])
            p.stepSimulation(physicsClientId=cid)
            _grab()
        env.set_joint_positions(path[-1])
        p.stepSimulation(physicsClientId=cid)
        for _ in range(max(1, post_hold)):
            _grab()

    duration_ms = max(1, int(1000.0 / max(1, fps)))
    frames[0].save(
        gif_path,
        save_all=True,
        append_images=frames[1:],
        duration=duration_ms,
        loop=0,
    )
    return gif_path
