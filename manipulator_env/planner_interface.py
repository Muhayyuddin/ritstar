"""
planner_interface.py — Connect PyBullet UR10e environment to the RIT* planner.

Provides:
  - build_rit_star_planner(): creates the RITStar planner from a PyBullet env
  - plan_and_execute(): full pipeline — plan → smooth → visualize
"""

from __future__ import annotations

import sys
import os
import time
import numpy as np
from typing import List, Tuple, Optional

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from rit_star.rit_star import RITStar, riemannian_edge_cost
from rit_star.baselines import InformedRRTStar, BITStar, AITStar, EITStar, APTStar
from rit_star.metric import RiemannianMetric, TaskSpaceMetric, DiagonalAnisotropicMetric
from manipulator_env.pybullet_env import UR10eRobotiqEnv


# Generic planner dispatch — any of the 6 benchmarked planners can be
# built for the 6-D UR10e demos, same signature as the 2-D/3-D comparison.
_PLANNER_CLASSES = {
    'RIT*':          RITStar,
    'Informed RRT*': InformedRRTStar,
    'BIT*':          BITStar,
    'AIT*':          AITStar,
    'EIT*':          EITStar,
    'APT*':          APTStar,
}

_PLANNER_ALIASES = {
    'rit': 'RIT*', 'rit*': 'RIT*',
    'irrt': 'Informed RRT*', 'informed_rrt': 'Informed RRT*',
    'informed-rrt': 'Informed RRT*', 'informedrrt': 'Informed RRT*',
    'informed_rrt*': 'Informed RRT*', 'informed rrt': 'Informed RRT*',
    'informed rrt*': 'Informed RRT*',
    'bit': 'BIT*', 'bit*': 'BIT*',
    'ait': 'AIT*', 'ait*': 'AIT*',
    'eit': 'EIT*', 'eit*': 'EIT*',
    'apt': 'APT*', 'apt*': 'APT*',
}


def _resolve_planner_name(name: str) -> str:
    """Map a user-supplied planner short name to its canonical key."""
    if name in _PLANNER_CLASSES:
        return name
    canonical = _PLANNER_ALIASES.get(name.strip().lower())
    if canonical is None:
        raise ValueError(
            f'Unknown planner "{name}". Known: {list(_PLANNER_CLASSES)}')
    return canonical


class ManipulatorInertiaMetric(RiemannianMetric):
    """Joint-space metric weighted by link inertias.

    G(q) = J(q)^T W J(q) + lambda * diag(m_1, ..., m_6)

    Combines task-space Jacobian-based cost with joint-inertia
    regularization.  Heavier joints cost more to move — this is
    physically motivated and produces natural-looking paths.

    Parameters
    ----------
    env : UR10eRobotiqEnv
        PyBullet environment (used for Jacobian computation).
    task_weight : float
        Scaling for the J^T W J term.
    joint_inertias : array-like of float
        Per-joint inertia weights [shoulder, shoulder_lift, elbow, w1, w2, w3].
    regularization : float
        Ridge to ensure strict positive-definiteness.
    """

    # Approximate link masses from UR10e spec (kg)
    DEFAULT_INERTIAS = np.array([7.369, 13.051, 3.989, 2.1, 1.98, 0.615])

    def __init__(self, env: UR10eRobotiqEnv,
                 task_weight: float = 1.0,
                 joint_inertias: Optional[np.ndarray] = None,
                 regularization: float = 0.05):
        super().__init__(6)
        self._env = env
        self._task_w = task_weight
        self._reg = regularization
        if joint_inertias is not None:
            self._m = np.asarray(joint_inertias, dtype=float)
        else:
            self._m = self.DEFAULT_INERTIAS.copy()
        # Normalize so the metric is O(1)
        self._m = self._m / self._m.max()
        self._diag_m = np.diag(self._m)

    def G(self, x: np.ndarray) -> np.ndarray:
        J = self._env.compute_jacobian(x)
        G = self._task_w * (J.T @ J) + self._reg * self._diag_m
        # Ensure symmetry
        return 0.5 * (G + G.T)

    def G_inv(self, x: np.ndarray) -> np.ndarray:
        return np.linalg.inv(self.G(x))

    def sqrt_det_G(self, x: np.ndarray) -> float:
        return float(np.sqrt(max(np.linalg.det(self.G(x)), 1e-30)))


def build_rit_star_planner(
    env: UR10eRobotiqEnv,
    q_start: np.ndarray,
    q_goal: np.ndarray,
    metric: Optional[RiemannianMetric] = None,
    batch_size: int = 200,
    max_iterations: int = 300,
    geodesic_tier: str = "diagonal",
    seed: int = 42,
) -> RITStar:
    """Create an RITStar planner for the UR10e manipulator.

    Parameters
    ----------
    env : UR10eRobotiqEnv
    q_start, q_goal : (6,) arrays — start/goal configurations
    metric : RiemannianMetric or None
        If None, uses ManipulatorInertiaMetric (Jacobian + inertia).
    batch_size : int
    max_iterations : int
    geodesic_tier : str — 'diagonal' is recommended for 6D
    seed : int

    Returns
    -------
    RITStar planner instance, ready to call .plan()
    """
    q_start = np.asarray(q_start, dtype=float)
    q_goal = np.asarray(q_goal, dtype=float)

    if metric is None:
        metric = ManipulatorInertiaMetric(env)

    bounds = env.get_bounds()

    def collision_checker(q):
        return env.is_collision_free(q)

    planner = RITStar(
        x_start=q_start,
        x_goal=q_goal,
        c_space_bounds=bounds,
        collision_checker=collision_checker,
        metric=metric,
        geodesic_tier=geodesic_tier,
        batch_size=batch_size,
        max_iterations=max_iterations,
        random_seed=seed,
    )
    return planner


def build_planner(
    name: str,
    env: UR10eRobotiqEnv,
    q_start: np.ndarray,
    q_goal: np.ndarray,
    metric: Optional[RiemannianMetric] = None,
    batch_size: int = 200,
    max_iterations: int = 300,
    seed: int = 42,
):
    """Build any of the 6 benchmarked planners for a UR10e env.

    Accepts the same canonical / short names as the 2-D/3-D pipeline
    (RIT*, Informed RRT*, BIT*, AIT*, EIT*, APT*).
    """
    canonical = _resolve_planner_name(name)
    cls = _PLANNER_CLASSES[canonical]
    if metric is None:
        metric = ManipulatorInertiaMetric(env)

    bounds = env.get_bounds()

    def collision_checker(q):
        return env.is_collision_free(q)

    kwargs = dict(
        x_start=np.asarray(q_start, dtype=float),
        x_goal=np.asarray(q_goal, dtype=float),
        c_space_bounds=bounds,
        collision_checker=collision_checker,
        metric=metric,
        batch_size=batch_size,
        max_iterations=max_iterations,
        random_seed=seed,
    )
    if canonical == 'RIT*':
        kwargs['geodesic_tier'] = 'diagonal'
    return cls(**kwargs)


def shortcut_smooth(path: List[np.ndarray],
                    env: UR10eRobotiqEnv,
                    metric: RiemannianMetric,
                    max_iters: int = 200,
                    rng: Optional[np.random.Generator] = None) -> List[np.ndarray]:
    """Shortcut-based path smoothing in joint space.

    Randomly picks two waypoints, checks if the shortcut is
    collision-free and cheaper, and replaces the sub-path.
    """
    if rng is None:
        rng = np.random.default_rng(0)

    path = [np.array(q) for q in path]

    for _ in range(max_iters):
        if len(path) < 3:
            break
        i = rng.integers(0, len(path) - 2)
        j = rng.integers(i + 2, len(path))
        if env.is_edge_collision_free(path[i], path[j], n_checks=15):
            old_cost = sum(
                riemannian_edge_cost(path[k], path[k + 1], metric)
                for k in range(i, j)
            )
            new_cost = riemannian_edge_cost(path[i], path[j], metric)
            if new_cost < old_cost:
                path = path[: i + 1] + [path[j]] + path[j + 1:]

    return path


def interpolate_path(path: List[np.ndarray],
                     max_step: float = 0.05) -> List[np.ndarray]:
    """Linearly interpolate a path so adjacent configs are close."""
    result = [path[0].copy()]
    for i in range(len(path) - 1):
        diff = path[i + 1] - path[i]
        dist = float(np.linalg.norm(diff))
        n_steps = max(1, int(np.ceil(dist / max_step)))
        for k in range(1, n_steps + 1):
            result.append(path[i] + (k / n_steps) * diff)
    return result


def plan_and_execute(
    env: UR10eRobotiqEnv,
    q_start: np.ndarray,
    q_goal: np.ndarray,
    metric: Optional[RiemannianMetric] = None,
    batch_size: int = 200,
    max_iterations: int = 300,
    smooth: bool = True,
    animate: bool = True,
    animate_delay: float = 0.02,
    planner_name: str = 'RIT*',
    seed: int = 42,
) -> Tuple[List[np.ndarray], float]:
    """Full pipeline: plan with the chosen planner, smooth, animate.

    Parameters
    ----------
    env : UR10eRobotiqEnv
    q_start, q_goal : (6,) arrays
    metric : RiemannianMetric or None
    batch_size, max_iterations : planner parameters
    smooth : bool — apply shortcut smoothing
    animate : bool — animate in PyBullet GUI (ignored if env is headless)
    animate_delay : float
    planner_name : str
        Any of 'RIT*', 'Informed RRT*', 'BIT*', 'AIT*', 'EIT*', 'APT*'
        (case-insensitive; short aliases like 'bit', 'irrt' accepted).
        Default 'RIT*' preserves legacy behaviour.
    seed : int — planner random seed.

    Returns
    -------
    (path, cost) — the final path and its Riemannian cost
    """
    q_start = np.asarray(q_start, dtype=float)
    q_goal = np.asarray(q_goal, dtype=float)

    if metric is None:
        metric = ManipulatorInertiaMetric(env)

    canonical = _resolve_planner_name(planner_name)
    tag = f'[{canonical}]'
    print(f"{tag} Planning from q_start to q_goal in 6-DOF C-space ...")
    print(f"       batch_size={batch_size}, max_iter={max_iterations}, seed={seed}")

    # Disable rendering so collision checks don't visually move the robot
    env.disable_rendering()

    # Set robot to start config visually before hiding
    env.set_joint_positions(q_start)

    planner = build_planner(
        canonical, env, q_start, q_goal,
        metric=metric,
        batch_size=batch_size,
        max_iterations=max_iterations,
        seed=seed,
    )

    path, cost = planner.plan()

    if not path:
        env.enable_rendering()
        print(f"{tag} No solution found!")
        return [], float("inf")

    print(f"{tag} Solution found! Cost = {cost:.4f}, waypoints = {len(path)}")

    stats = planner.get_stats()
    if stats:
        print(f"{tag} Iterations: {len(stats)}, "
              f"final tree size: {stats[-1].get('n_vertices', '?')}")

    # Smooth (still with rendering off)
    if smooth and len(path) > 2:
        print(f"{tag} Shortcut smoothing ...")
        path = shortcut_smooth(path, env, metric, max_iters=300)
        smooth_cost = sum(
            riemannian_edge_cost(path[i], path[i + 1], metric)
            for i in range(len(path) - 1)
        )
        print(f"{tag} After smoothing: cost = {smooth_cost:.4f}, "
              f"waypoints = {len(path)}")
        cost = smooth_cost

    # Re-enable rendering now that planning is done
    env.enable_rendering()

    # Interpolate for smooth animation
    path_fine = interpolate_path(path, max_step=0.02)
    print(f"{tag} Interpolated to {len(path_fine)} waypoints for animation.")

    # Show start/goal markers
    env.visualize_config(q_start, color=[0, 0, 1, 1])   # blue = start
    env.visualize_config(q_goal, color=[1, 0, 0, 1])     # red = goal

    # Animate in a loop (finite to avoid X server timeout)
    if animate:
        n_loops = 3
        print(f"{tag} Animating path ({n_loops} loops, Ctrl+C to stop) ...")
        try:
            for loop_i in range(n_loops):
                env.set_joint_positions(q_start)
                time.sleep(0.3)
                env.visualize_path(path_fine, delay=animate_delay, trail=(loop_i == 0))
                time.sleep(0.5)
            # Hold final pose for a few seconds so user can inspect
            env.set_joint_positions(q_goal)
            print(f"{tag} Animation complete. Holding final pose for 5s ...")
            time.sleep(5.0)
        except KeyboardInterrupt:
            print(f"\n{tag} Animation stopped.")
    else:
        print(f"{tag} Done (headless mode, no animation).")

    return path, cost
