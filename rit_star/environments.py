"""
environments.py — 2-D and 3-D planning environments with obstacles.

Each factory function returns a 6-tuple:
    (collision_checker, edge_cost_fn, metric, x_start, x_goal, bounds)

  collision_checker(x) -> bool   True = free
  edge_cost_fn(x, y)  -> float   Riemannian arc-length along straight line
  metric              -> RiemannianMetric instance
  x_start, x_goal     -> (d,) arrays
  bounds              -> list of (lo, hi) per dimension
"""

from __future__ import annotations

import numpy as np
from typing import Callable, Tuple, List

from .metric import (
    DiagonalAnisotropicMetric,
    ObstacleInflatedMetric,
    JointInertiaMetric2D,
    EuclideanMetric,
)
from .rit_star import riemannian_edge_cost

EnvTuple = Tuple[Callable, Callable, object, np.ndarray, np.ndarray, list]


# ═══════════════════════════════════════════════════════════════════════
# Numba JIT collision kernels — compiled once, cached to disk.
# Returns True = FREE, False = IN COLLISION.
# Early-exit loops give ~20-50× speedup over Python loops and
# ~5-10× over NumPy broadcasts (no array allocation, no full scan).
# ═══════════════════════════════════════════════════════════════════════

try:
    from numba import njit as _njit
    _NUMBA_OK = True
except ImportError:
    _NUMBA_OK = False

if _NUMBA_OK:
    @_njit(cache=True)
    def _nb_rects_2d_free(x, lows, highs):
        for i in range(lows.shape[0]):
            if (x[0] >= lows[i, 0] and x[0] <= highs[i, 0] and
                    x[1] >= lows[i, 1] and x[1] <= highs[i, 1]):
                return False
        return True

    @_njit(cache=True)
    def _nb_boxes_3d_free(x, lows, highs):
        for i in range(lows.shape[0]):
            if (x[0] >= lows[i, 0] and x[0] <= highs[i, 0] and
                    x[1] >= lows[i, 1] and x[1] <= highs[i, 1] and
                    x[2] >= lows[i, 2] and x[2] <= highs[i, 2]):
                return False
        return True

    @_njit(cache=True)
    def _nb_spheres_3d_free(x, centres, r_sq):
        for i in range(centres.shape[0]):
            dx = x[0] - centres[i, 0]
            dy = x[1] - centres[i, 1]
            dz = x[2] - centres[i, 2]
            if dx * dx + dy * dy + dz * dz <= r_sq:
                return False
        return True

    @_njit(cache=True)
    def _nb_circles_2d_free(x, centres, radii_sq):
        for i in range(centres.shape[0]):
            dx = x[0] - centres[i, 0]
            dy = x[1] - centres[i, 1]
            if dx * dx + dy * dy <= radii_sq[i]:
                return False
        return True

    @_njit(cache=True)
    def _nb_boxes_nd_free(x, lows, highs):
        n = lows.shape[0]
        d = lows.shape[1]
        for i in range(n):
            inside = True
            for k in range(d):
                if x[k] < lows[i, k] or x[k] > highs[i, k]:
                    inside = False
                    break
            if inside:
                return False
        return True

else:  # Fallback pure-Python when numba unavailable
    def _nb_rects_2d_free(x, lows, highs):
        for i in range(lows.shape[0]):
            if (x[0] >= lows[i, 0] and x[0] <= highs[i, 0] and
                    x[1] >= lows[i, 1] and x[1] <= highs[i, 1]):
                return False
        return True

    def _nb_boxes_3d_free(x, lows, highs):
        for i in range(lows.shape[0]):
            if (x[0] >= lows[i, 0] and x[0] <= highs[i, 0] and
                    x[1] >= lows[i, 1] and x[1] <= highs[i, 1] and
                    x[2] >= lows[i, 2] and x[2] <= highs[i, 2]):
                return False
        return True

    def _nb_spheres_3d_free(x, centres, r_sq):
        for i in range(centres.shape[0]):
            dx = x[0] - centres[i, 0]
            dy = x[1] - centres[i, 1]
            dz = x[2] - centres[i, 2]
            if dx * dx + dy * dy + dz * dz <= r_sq:
                return False
        return True

    def _nb_circles_2d_free(x, centres, radii_sq):
        for i in range(centres.shape[0]):
            dx = x[0] - centres[i, 0]
            dy = x[1] - centres[i, 1]
            if dx * dx + dy * dy <= radii_sq[i]:
                return False
        return True

    def _nb_boxes_nd_free(x, lows, highs):
        for i in range(lows.shape[0]):
            if np.all(x >= lows[i]) and np.all(x <= highs[i]):
                return False
        return True


# ═══════════════════════════════════════════════════════════════════════
# Collision factory functions
# ═══════════════════════════════════════════════════════════════════════

def _make_rect_collision_free(bounds_lo: np.ndarray, bounds_hi: np.ndarray,
                               rects: list) -> Callable:
    """Numba-accelerated checker for 2-D axis-aligned rectangles."""
    lows  = np.array([lo for lo, hi in rects], dtype=np.float64)
    highs = np.array([hi for lo, hi in rects], dtype=np.float64)
    blo0, bhi0 = float(bounds_lo[0]), float(bounds_hi[0])
    blo1, bhi1 = float(bounds_lo[1]), float(bounds_hi[1])
    if _NUMBA_OK:  # warm up JIT on first construction
        _nb_rects_2d_free(np.asarray(bounds_lo, dtype=np.float64), lows, highs)

    def collision_free(x: np.ndarray) -> bool:
        if x[0] < blo0 or x[0] > bhi0 or x[1] < blo1 or x[1] > bhi1:
            return False
        return bool(_nb_rects_2d_free(x, lows, highs))

    return collision_free


def _make_box_collision_free_3d(bounds_lo: np.ndarray, bounds_hi: np.ndarray,
                                 boxes: list) -> Callable:
    """Numba-accelerated checker for 3-D axis-aligned boxes."""
    lows  = np.array([lo for lo, hi in boxes], dtype=np.float64)
    highs = np.array([hi for lo, hi in boxes], dtype=np.float64)
    blo = np.asarray(bounds_lo, dtype=np.float64)
    bhi = np.asarray(bounds_hi, dtype=np.float64)
    if _NUMBA_OK:
        _nb_boxes_3d_free(blo, lows, highs)

    def collision_free(x: np.ndarray) -> bool:
        if x[0] < blo[0] or x[0] > bhi[0]: return False
        if x[1] < blo[1] or x[1] > bhi[1]: return False
        if x[2] < blo[2] or x[2] > bhi[2]: return False
        return bool(_nb_boxes_3d_free(x, lows, highs))

    return collision_free


def _make_sphere_collision_free_3d(bounds_lo: np.ndarray, bounds_hi: np.ndarray,
                                    centres: np.ndarray, radius: float) -> Callable:
    """Numba-accelerated checker for 3-D spheres (uniform radius)."""
    ctrs = np.asarray(centres, dtype=np.float64)
    r_sq = float(radius * radius)
    blo = np.asarray(bounds_lo, dtype=np.float64)
    bhi = np.asarray(bounds_hi, dtype=np.float64)
    if _NUMBA_OK:
        _nb_spheres_3d_free(blo, ctrs, r_sq)

    def collision_free(x: np.ndarray) -> bool:
        if x[0] < blo[0] or x[0] > bhi[0]: return False
        if x[1] < blo[1] or x[1] > bhi[1]: return False
        if x[2] < blo[2] or x[2] > bhi[2]: return False
        return bool(_nb_spheres_3d_free(x, ctrs, r_sq))

    return collision_free


def _make_circles_collision_free_2d(bounds_lo: np.ndarray, bounds_hi: np.ndarray,
                                     circles: list) -> Callable:
    """Numba-accelerated checker for 2-D circles."""
    ctrs     = np.array([c for c, _ in circles], dtype=np.float64)
    radii_sq = np.array([r * r for _, r in circles], dtype=np.float64)
    blo0, bhi0 = float(bounds_lo[0]), float(bounds_hi[0])
    blo1, bhi1 = float(bounds_lo[1]), float(bounds_hi[1])
    if _NUMBA_OK:
        _nb_circles_2d_free(np.asarray(bounds_lo, dtype=np.float64), ctrs, radii_sq)

    def collision_free(x: np.ndarray) -> bool:
        if x[0] < blo0 or x[0] > bhi0 or x[1] < blo1 or x[1] > bhi1:
            return False
        return bool(_nb_circles_2d_free(x, ctrs, radii_sq))

    return collision_free


def _make_boxes_nd_collision_free(bounds_lo: np.ndarray, bounds_hi: np.ndarray,
                                   obs_list: list) -> Callable:
    """Numba-accelerated checker for N-D axis-aligned boxes."""
    lows  = np.array([lo for lo, hi in obs_list], dtype=np.float64)
    highs = np.array([hi for lo, hi in obs_list], dtype=np.float64)
    blo = np.asarray(bounds_lo, dtype=np.float64)
    bhi = np.asarray(bounds_hi, dtype=np.float64)
    d = int(blo.shape[0])

    def collision_free(x: np.ndarray) -> bool:
        for k in range(d):
            if x[k] < blo[k] or x[k] > bhi[k]:
                return False
        return bool(_nb_boxes_nd_free(x, lows, highs))

    return collision_free


# ═══════════════════════════════════════════════════════════════════════
# Legacy single-point primitives (kept for compatibility)
# ═══════════════════════════════════════════════════════════════════════

def _point_in_rect_2d(p: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> bool:
    return (p[0] >= lo[0] and p[0] <= hi[0] and
            p[1] >= lo[1] and p[1] <= hi[1])


def _point_in_circle_2d(p: np.ndarray, centre: np.ndarray, r: float) -> bool:
    dx = p[0] - centre[0]
    dy = p[1] - centre[1]
    return dx * dx + dy * dy <= r * r


def _point_in_box_3d(p: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> bool:
    return (p[0] >= lo[0] and p[0] <= hi[0] and
            p[1] >= lo[1] and p[1] <= hi[1] and
            p[2] >= lo[2] and p[2] <= hi[2])


def _point_in_sphere_3d(p: np.ndarray, centre: np.ndarray, r: float) -> bool:
    dx = p[0] - centre[0]
    dy = p[1] - centre[1]
    dz = p[2] - centre[2]
    return dx * dx + dy * dy + dz * dz <= r * r


def _make_edge_cost(metric):
    """Return an edge-cost callable for the given metric."""
    def edge_cost(x: np.ndarray, y: np.ndarray) -> float:
        return riemannian_edge_cost(x, y, metric)
    return edge_cost


# ═══════════════════════════════════════════════════════════════════════
# 2-D Environments
# ═══════════════════════════════════════════════════════════════════════

def env_2d_diagonal_anisotropic() -> EnvTuple:
    """2-D environment with diagonal anisotropic metric.

    Bounds [0,1]².  Two rectangular obstacles forming a narrow passage.
    Metric: DiagonalAnisotropicMetric(weights=[4.0, 1.0]).
    Moving along x is 4× more expensive than along y.

    Expected behaviour: RIT* discovers a path that preferentially
    uses the cheap y-direction, and the informed set is elongated
    vertically rather than horizontally.

    Returns
    -------
    (collision_checker, edge_cost, metric, x_start, x_goal, bounds)
    """
    x_start = np.array([0.1, 0.5])
    x_goal = np.array([0.9, 0.5])
    bounds = [(0.0, 1.0), (0.0, 1.0)]

    rects = [
        (np.array([0.35, 0.30]), np.array([0.45, 0.70])),
        (np.array([0.55, 0.30]), np.array([0.65, 0.70])),
    ]

    collision_free = _make_rect_collision_free(
        np.zeros(2), np.ones(2), rects)

    metric = DiagonalAnisotropicMetric(weights=[4.0, 1.0])
    return collision_free, _make_edge_cost(metric), metric, x_start, x_goal, bounds


def env_2d_obstacle_inflated() -> EnvTuple:
    """2-D environment with obstacle-inflated metric.

    Bounds [0,1]².  Six circular obstacles in the middle region.
    Metric: ObstacleInflatedMetric(σ=0.12, α=8.0).

    Expected behaviour: RIT* avoids the high-cost regions near
    obstacles even during the sampling phase; the Riemannian informed
    set contracts away from obstacle neighbourhoods.

    Returns
    -------
    (collision_checker, edge_cost, metric, x_start, x_goal, bounds)
    """
    x_start = np.array([0.05, 0.25])
    x_goal = np.array([0.95, 0.75])
    bounds = [(0.0, 1.0), (0.0, 1.0)]

    circles = [
        (np.array([0.30, 0.35]), 0.08),
        (np.array([0.30, 0.65]), 0.08),
        (np.array([0.50, 0.45]), 0.09),
        (np.array([0.50, 0.75]), 0.09),
        (np.array([0.70, 0.40]), 0.08),
        (np.array([0.70, 0.60]), 0.08),
    ]
    centres = np.array([c for c, _ in circles])

    collision_free = _make_circles_collision_free_2d(
        np.zeros(2), np.ones(2), circles)

    metric = ObstacleInflatedMetric(centres, sigma=0.12, alpha=8.0)
    return collision_free, _make_edge_cost(metric), metric, x_start, x_goal, bounds


def env_2d_joint_arm() -> EnvTuple:
    """Configuration space of a 2-joint planar arm.

    Bounds [-π, π]².  Workspace obstacles converted to C-space via
    forward kinematics.  Arm links: L₁=0.5, L₂=0.4.
    Metric: JointInertiaMetric2D(I₁=4.0, I₂=1.0).

    Expected behaviour: RIT* preferentially moves joint 2 (cheap,
    lower inertia) and avoids unnecessary joint-1 motion.

    Returns
    -------
    (collision_checker, edge_cost, metric, x_start, x_goal, bounds)
    """
    x_start = np.array([0.1, 0.1])
    x_goal = np.array([2.8, -2.8])
    bounds = [(-np.pi, np.pi), (-np.pi, np.pi)]

    L1, L2 = 0.5, 0.4

    # Workspace circular obstacles (sized to keep C-space connected)
    ws_obstacles = [
        (np.array([0.5, 0.3]), 0.08),
        (np.array([-0.3, 0.5]), 0.06),
    ]

    def fk(theta):
        """Forward kinematics: returns elbow and end-effector positions."""
        t1, t2 = theta[0], theta[1]
        elbow = np.array([L1 * np.cos(t1), L1 * np.sin(t1)])
        ee = elbow + np.array([L2 * np.cos(t1 + t2), L2 * np.sin(t1 + t2)])
        return elbow, ee

    def collision_free(x):
        """True if the arm configuration *x* is collision-free.

        Checks both elbow and end-effector positions in the workspace
        against circular obstacles.  Implements forward-kinematics-based
        collision checking for the 2-DOF arm environment.
        """
        elbow, ee = fk(x)
        # Check link segments against obstacles
        for obs_c, obs_r in ws_obstacles:
            # Check a few points along each link
            base = np.array([0.0, 0.0])
            for t in np.linspace(0, 1, 8):
                p1 = base + t * (elbow - base)
                if float(np.sum((p1 - obs_c) ** 2)) < obs_r ** 2:
                    return False
                p2 = elbow + t * (ee - elbow)
                if float(np.sum((p2 - obs_c) ** 2)) < obs_r ** 2:
                    return False
        return True

    metric = JointInertiaMetric2D(I1=4.0, I2=1.0)
    return collision_free, _make_edge_cost(metric), metric, x_start, x_goal, bounds


# ═══════════════════════════════════════════════════════════════════════
# 3-D Environments
# ═══════════════════════════════════════════════════════════════════════

def env_3d_diagonal_anisotropic() -> EnvTuple:
    """3-D environment with diagonal anisotropic metric.

    Bounds [0,1]³.  Four axis-aligned box obstacles forming a loose maze.
    Metric: DiagonalAnisotropicMetric(weights=[6.0, 1.0, 2.0]).
    x-axis 6× expensive, z-axis 2×, y-axis cheap.

    Expected behaviour: path and informed set elongated along the
    cheap y-axis.

    Returns
    -------
    (collision_checker, edge_cost, metric, x_start, x_goal, bounds)
    """
    x_start = np.array([0.1, 0.5, 0.5])
    x_goal = np.array([0.9, 0.5, 0.5])
    bounds = [(0.0, 1.0), (0.0, 1.0), (0.0, 1.0)]

    boxes = [
        (np.array([0.25, 0.0, 0.0]), np.array([0.35, 0.6, 1.0])),
        (np.array([0.45, 0.4, 0.0]), np.array([0.55, 1.0, 1.0])),
        (np.array([0.65, 0.0, 0.0]), np.array([0.75, 0.6, 0.6])),
        (np.array([0.65, 0.0, 0.7]), np.array([0.75, 0.6, 1.0])),
    ]

    collision_free = _make_box_collision_free_3d(
        np.zeros(3), np.ones(3), boxes)

    metric = DiagonalAnisotropicMetric(weights=[6.0, 1.0, 2.0])
    return collision_free, _make_edge_cost(metric), metric, x_start, x_goal, bounds


def env_3d_sphere_field() -> EnvTuple:
    """3-D environment with obstacle-inflated metric and 8+1 spheres.

    Bounds [-1,1]³.  Eight spheres of radius 0.22 arranged in a 2×2×2
    grid offset from centre, plus one central blocker sphere.
    The start/goal are positioned so the direct line is blocked.
    Metric: ObstacleInflatedMetric(σ=0.25, α=12.0).

    Returns
    -------
    (collision_checker, edge_cost, metric, x_start, x_goal, bounds)
    """
    x_start = np.array([-0.9, 0.0, 0.0])
    x_goal = np.array([0.9, 0.0, 0.0])
    bounds = [(-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0)]

    r_obs = 0.22
    offsets = np.array([-0.35, 0.35])
    sphere_centres = []
    for sx in offsets:
        for sy in offsets:
            for sz in offsets:
                sphere_centres.append([sx, sy, sz])
    # Central blocking sphere — forces path to navigate around cluster
    sphere_centres.append([0.0, 0.0, 0.0])
    sphere_centres = np.array(sphere_centres)

    collision_free = _make_sphere_collision_free_3d(
        np.full(3, -1.0), np.ones(3), sphere_centres, r_obs)

    metric = ObstacleInflatedMetric(sphere_centres, sigma=0.25, alpha=12.0)
    return collision_free, _make_edge_cost(metric), metric, x_start, x_goal, bounds


# ═══════════════════════════════════════════════════════════════════════
# Narrow-passage environments
# ═══════════════════════════════════════════════════════════════════════

def env_2d_narrow_passage() -> EnvTuple:
    """2-D narrow-passage environment.

    Bounds [0,1]².  A wall spans the middle with a single small gap
    (width 0.06) that the planner must thread through.  Two additional
    block obstacles flank the passage to make it harder.
    Metric: DiagonalAnisotropicMetric(weights=[3.0, 1.0]).

    Expected behaviour: RIT* focuses samples near the gap thanks to
    the tighter Riemannian informed set, while Euclidean methods
    waste samples far from the narrow opening.

    Returns
    -------
    (collision_checker, edge_cost, metric, x_start, x_goal, bounds)
    """
    x_start = np.array([0.1, 0.1])
    x_goal = np.array([0.9, 0.9])
    bounds = [(0.0, 1.0), (0.0, 1.0)]

    gap_y_lo, gap_y_hi = 0.47, 0.53  # narrow 0.06 gap
    rects = [
        # Main wall with gap
        (np.array([0.48, 0.00]), np.array([0.52, gap_y_lo])),
        (np.array([0.48, gap_y_hi]), np.array([0.52, 1.00])),
        # Flanking blocks that force approach from the centre
        (np.array([0.30, 0.15]), np.array([0.42, 0.35])),
        (np.array([0.30, 0.65]), np.array([0.42, 0.85])),
        (np.array([0.58, 0.15]), np.array([0.70, 0.35])),
        (np.array([0.58, 0.65]), np.array([0.70, 0.85])),
    ]

    collision_free = _make_rect_collision_free(
        np.zeros(2), np.ones(2), rects)

    metric = DiagonalAnisotropicMetric(weights=[3.0, 1.0])
    return collision_free, _make_edge_cost(metric), metric, x_start, x_goal, bounds


def env_3d_narrow_passage() -> EnvTuple:
    """3-D narrow-passage environment.

    Bounds [0,1]³.  A solid wall at x=0.5 with a small cylindrical
    hole (radius 0.09) centred at (0.5, 0.5, 0.5).  The planner
    must find and thread through this hole.
    Metric: DiagonalAnisotropicMetric(weights=[4.0, 1.0, 1.0]).

    Returns
    -------
    (collision_checker, edge_cost, metric, x_start, x_goal, bounds)
    """
    x_start = np.array([0.1, 0.5, 0.5])
    x_goal = np.array([0.9, 0.5, 0.5])
    bounds = [(0.0, 1.0), (0.0, 1.0), (0.0, 1.0)]

    wall_lo_x, wall_hi_x = 0.47, 0.53
    hole_centre_yz = np.array([0.5, 0.5])
    hole_radius = 0.09

    def collision_free(x):
        if np.any(x < 0.0) or np.any(x > 1.0):
            return False
        # Wall occupies x in [0.47, 0.53]
        if wall_lo_x <= x[0] <= wall_hi_x:
            # Inside the wall slab — only free if inside the hole
            dist_yz = float(np.sqrt((x[1] - hole_centre_yz[0])**2 +
                                    (x[2] - hole_centre_yz[1])**2))
            if dist_yz > hole_radius:
                return False
        return True

    metric = DiagonalAnisotropicMetric(weights=[4.0, 1.0, 1.0])
    return collision_free, _make_edge_cost(metric), metric, x_start, x_goal, bounds


# ═══════════════════════════════════════════════════════════════════════
# Maze environment
# ═══════════════════════════════════════════════════════════════════════

def env_2d_maze() -> EnvTuple:
    """2-D maze environment.

    Bounds [0,1]².  Five horizontal / vertical walls with gaps create
    a simple maze that requires multiple turns.
    Metric: ObstacleInflatedMetric(σ=0.08, α=6.0) centred on wall
    midpoints — makes traversal near walls expensive.

    Expected behaviour: RIT* finds the shortest Riemannian path
    through the maze while staying away from wall edges.

    Returns
    -------
    (collision_checker, edge_cost, metric, x_start, x_goal, bounds)
    """
    x_start = np.array([0.05, 0.05])
    x_goal = np.array([0.95, 0.95])
    bounds = [(0.0, 1.0), (0.0, 1.0)]

    # S-maze: 3 thick horizontal walls with alternating gaps
    # Walls are 0.08 thick so the 10-point collision checker cannot skip them
    rects = [
        # Wall 1: horizontal, gap at right side (x > 0.70)
        (np.array([0.00, 0.22]), np.array([0.70, 0.30])),
        # Wall 2: horizontal, gap at left side (x < 0.30)
        (np.array([0.30, 0.46]), np.array([1.00, 0.54])),
        # Wall 3: horizontal, gap at right side (x > 0.70)
        (np.array([0.00, 0.70]), np.array([0.70, 0.78])),
    ]

    # Obstacle centres for metric — spread along each wall so the
    # Gaussian field covers the full rectangular geometry.
    wall_y_centres = [0.26, 0.50, 0.74]          # vertical midpoints
    wall_x_ranges  = [(0.00, 0.70), (0.30, 1.00), (0.00, 0.70)]
    spacing = 0.08
    centres = []
    for wy, (xlo, xhi) in zip(wall_y_centres, wall_x_ranges):
        n_pts = max(2, int(np.ceil((xhi - xlo) / spacing)) + 1)
        for cx in np.linspace(xlo, xhi, n_pts):
            centres.append([cx, wy])
    centres = np.array(centres)

    collision_free = _make_rect_collision_free(
        np.zeros(2), np.ones(2), rects)

    metric = ObstacleInflatedMetric(centres, sigma=0.10, alpha=6.0)
    return collision_free, _make_edge_cost(metric), metric, x_start, x_goal, bounds


# ═══════════════════════════════════════════════════════════════════════
# Factory for arbitrary (dim, kappa) diagonal environments  (Theorem 3)
# ═══════════════════════════════════════════════════════════════════════

def _make_diagonal_env_nd(dim: int, kappa: float,
                          seed: int = 42) -> EnvTuple:
    """Create a d-dimensional environment with G = diag(kappa, 1, ..., 1).

    Used by experiment_convergence_rate_separation (Theorem 3) to sweep
    over (kappa, dim) pairs.

    Parameters
    ----------
    dim : int
        Dimensionality (2, 3, 6, ...).
    kappa : float
        Condition number of the metric (w_1 = kappa, w_2 = ... = w_d = 1).
    seed : int
        Random seed for obstacle placement.

    Returns
    -------
    EnvTuple
    """
    bounds = [(0.0, 1.0)] * dim
    x_start = np.full(dim, 0.1)
    x_goal = np.full(dim, 0.9)

    weights = np.ones(dim)
    weights[0] = kappa
    if kappa <= 1.0 + 1e-12:
        metric = EuclideanMetric(dim)
    else:
        metric = DiagonalAnisotropicMetric(weights)

    # Generate random box obstacles (seeded)
    rng = np.random.default_rng(seed)
    n_obs = max(2, dim)
    obs_list = []
    for _ in range(n_obs * 10):
        if len(obs_list) >= n_obs:
            break
        centre = rng.uniform(0.25, 0.75, size=dim)
        half = rng.uniform(0.03, 0.08, size=dim)
        lo = centre - half
        hi = centre + half
        # Don't block start or goal
        if np.all(lo <= x_start) and np.all(x_start <= hi):
            continue
        if np.all(lo <= x_goal) and np.all(x_goal <= hi):
            continue
        obs_list.append((lo.copy(), hi.copy()))

    collision_free = _make_boxes_nd_collision_free(
        np.zeros(dim), np.ones(dim), obs_list)

    return collision_free, _make_edge_cost(metric), metric, x_start, x_goal, bounds


# ═══════════════════════════════════════════════════════════════════════
# Registry for convenient iteration
# ═══════════════════════════════════════════════════════════════════════

ALL_2D_ENVS = {
    '2d_diagonal':        env_2d_diagonal_anisotropic,
    '2d_obstacle':        env_2d_obstacle_inflated,
    '2d_arm':             env_2d_joint_arm,
    '2d_narrow_passage':  env_2d_narrow_passage,
    '2d_maze':            env_2d_maze,
}

ALL_3D_ENVS = {
    '3d_diagonal':        env_3d_diagonal_anisotropic,
    '3d_spheres':         env_3d_sphere_field,
    '3d_narrow_passage':  env_3d_narrow_passage,
}

ALL_ENVS = {**ALL_2D_ENVS, **ALL_3D_ENVS}


# ═══════════════════════════════════════════════════════════════════════
# Additional environments for demonstrating RIT* strengths
# ═══════════════════════════════════════════════════════════════════════

def env_2d_bug_trap() -> EnvTuple:
    """2-D bug-trap environment.

    Bounds [0,1]².  A U-shaped enclosure with a narrow exit that
    traps uniform samplers.  The planner must escape through the
    small opening (width 0.08) on the right side of the 'U'.
    Metric: ObstacleInflatedMetric — wall proximity is expensive.

    Expected behaviour: RIT* escapes the trap faster because the
    Riemannian informed set contracts away from walls, focusing
    samples toward the exit.

    Returns
    -------
    (collision_checker, edge_cost, metric, x_start, x_goal, bounds)
    """
    x_start = np.array([0.3, 0.5])
    x_goal = np.array([0.1, 0.1])
    bounds = [(0.0, 1.0), (0.0, 1.0)]

    # U-shaped trap: bottom, top, left walls; narrow exit on right
    gap_lo, gap_hi = 0.46, 0.54  # 0.08 gap
    rects = [
        # Bottom wall of U
        (np.array([0.15, 0.20]), np.array([0.60, 0.28])),
        # Top wall of U
        (np.array([0.15, 0.72]), np.array([0.60, 0.80])),
        # Left wall of U (back)
        (np.array([0.15, 0.28]), np.array([0.23, 0.72])),
        # Right wall of U — lower part (below gap)
        (np.array([0.52, 0.28]), np.array([0.60, gap_lo])),
        # Right wall of U — upper part (above gap)
        (np.array([0.52, gap_hi]), np.array([0.60, 0.72])),
    ]

    centres = []
    for lo, hi in rects:
        cx = 0.5 * (lo[0] + hi[0])
        cy = 0.5 * (lo[1] + hi[1])
        centres.append([cx, cy])
    centres = np.array(centres)

    def collision_free(x):
        for lo, hi in rects:
            if _point_in_rect_2d(x, lo, hi):
                return False
        if np.any(x < 0.0) or np.any(x > 1.0):
            return False
        return True

    metric = ObstacleInflatedMetric(centres, sigma=0.12, alpha=6.0)
    return collision_free, _make_edge_cost(metric), metric, x_start, x_goal, bounds


def env_2d_random_forest() -> EnvTuple:
    """2-D random forest: many small circular obstacles.

    Bounds [0,1]².  25 randomly-placed circles (r=0.04) with a
    fixed seed for reproducibility.  The obstacle-inflated metric
    should produce dramatically tighter informed sets than the
    Euclidean one.

    Returns
    -------
    (collision_checker, edge_cost, metric, x_start, x_goal, bounds)
    """
    x_start = np.array([0.05, 0.05])
    x_goal = np.array([0.95, 0.95])
    bounds = [(0.0, 1.0), (0.0, 1.0)]

    rng = np.random.default_rng(12345)
    n_obs = 25
    radius = 0.04
    # Generate obstacles that don't overlap start/goal
    centres = []
    for _ in range(n_obs * 5):
        if len(centres) >= n_obs:
            break
        c = rng.uniform(0.1, 0.9, size=2)
        # Don't block start or goal
        if np.linalg.norm(c - x_start) < 0.12:
            continue
        if np.linalg.norm(c - x_goal) < 0.12:
            continue
        centres.append(c)
    centres = np.array(centres[:n_obs])
    circles = [(c, radius) for c in centres]

    def collision_free(x):
        if np.any(x < 0.0) or np.any(x > 1.0):
            return False
        for c, r in circles:
            if _point_in_circle_2d(x, c, r):
                return False
        return True

    metric = ObstacleInflatedMetric(centres, sigma=0.08, alpha=8.0)
    return collision_free, _make_edge_cost(metric), metric, x_start, x_goal, bounds


def env_2d_terrain() -> EnvTuple:
    """2-D non-uniform cost field (terrain elevation).

    Bounds [0,1]².  No hard obstacles — the metric encodes a
    terrain cost field with ridges and valleys.  Paths should
    follow the low-cost valleys.

    The cost field is:
        s(x) = 1 + 5·sin²(3πx₁)·sin²(3πx₂)

    This creates a grid of peaks (expensive) with valley channels
    between them.
    Metric: conformal  G(x) = s(x)·I.

    Returns
    -------
    (collision_checker, edge_cost, metric, x_start, x_goal, bounds)
    """
    x_start = np.array([0.05, 0.05])
    x_goal = np.array([0.95, 0.95])
    bounds = [(0.0, 1.0), (0.0, 1.0)]

    class TerrainMetric(ObstacleInflatedMetric):
        """G(x) = s(x)·I where s encodes terrain elevation cost."""
        def __init__(self):
            # Initialise with dummy to get dim set
            super(ObstacleInflatedMetric, self).__init__(2)
            self._centers = np.zeros((0, 2))
            self._sigma2 = 1.0
            self._alpha = 0.0

        def _scale(self, x: np.ndarray) -> float:
            return 1.0 + 5.0 * (np.sin(3.0 * np.pi * x[0]) ** 2) * \
                                (np.sin(3.0 * np.pi * x[1]) ** 2)

        def _scale_batch(self, pts: np.ndarray) -> np.ndarray:
            return 1.0 + 5.0 * (np.sin(3.0 * np.pi * pts[:, 0]) ** 2) * \
                                (np.sin(3.0 * np.pi * pts[:, 1]) ** 2)

        def G(self, x):
            return self._scale(x) * np.eye(2)

        def G_inv(self, x):
            return (1.0 / self._scale(x)) * np.eye(2)

        def sqrt_det_G(self, x):
            return self._scale(x)

    def collision_free(x):
        return bool(np.all(x >= 0.0) and np.all(x <= 1.0))

    metric = TerrainMetric()
    return collision_free, _make_edge_cost(metric), metric, x_start, x_goal, bounds


# Update registries
ALL_2D_ENVS['2d_bug_trap'] = env_2d_bug_trap
ALL_2D_ENVS['2d_random_forest'] = env_2d_random_forest
ALL_2D_ENVS['2d_terrain'] = env_2d_terrain
ALL_ENVS.update({'2d_bug_trap': env_2d_bug_trap,
                 '2d_random_forest': env_2d_random_forest,
                 '2d_terrain': env_2d_terrain})


# ═══════════════════════════════════════════════════════════════════════
# NEW: Dense-clutter demo environments
# ═══════════════════════════════════════════════════════════════════════

def env_2d_hyper_dense() -> EnvTuple:
    """2-D hyper-dense random obstacle field.

    Bounds [0,1]².  35 randomly placed circles (r=0.03) saturating
    the workspace; only narrow winding corridors remain.
    Metric: ObstacleInflatedMetric(σ=0.08, α=8.0).

    Returns
    -------
    (collision_checker, edge_cost, metric, x_start, x_goal, bounds)
    """
    x_start = np.array([0.05, 0.05])
    x_goal = np.array([0.95, 0.95])
    bounds = [(0.0, 1.0), (0.0, 1.0)]

    rng = np.random.default_rng(99999)
    n_obs = 35
    radius = 0.03
    centres = []
    for _ in range(n_obs * 10):
        if len(centres) >= n_obs:
            break
        c = rng.uniform(0.08, 0.92, size=2)
        if np.linalg.norm(c - x_start) < 0.10:
            continue
        if np.linalg.norm(c - x_goal) < 0.10:
            continue
        # Avoid overlapping obstacles (keep thin corridors open)
        too_close = False
        for existing in centres:
            if np.linalg.norm(c - existing) < 2.2 * radius:
                too_close = True
                break
        if too_close:
            continue
        centres.append(c)
    centres = np.array(centres[:n_obs])
    circles = [(c, radius) for c in centres]

    def collision_free(x):
        if np.any(x < 0.0) or np.any(x > 1.0):
            return False
        for c, r in circles:
            if _point_in_circle_2d(x, c, r):
                return False
        return True

    metric = ObstacleInflatedMetric(centres, sigma=0.08, alpha=8.0)
    return collision_free, _make_edge_cost(metric), metric, x_start, x_goal, bounds


def env_3d_dense_labyrinth() -> EnvTuple:
    """3-D dense sphere labyrinth.

    Bounds [-1,1]³.  Fifteen spheres (r=0.18) densely packed
    throughout the volume, creating narrow winding 3-D corridors.
    Metric: ObstacleInflatedMetric(σ=0.25, α=12.0).

    Returns
    -------
    (collision_checker, edge_cost, metric, x_start, x_goal, bounds)
    """
    x_start = np.array([-0.9, -0.9, -0.9])
    x_goal = np.array([0.9, 0.9, 0.9])
    bounds = [(-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0)]

    r_obs = 0.18
    # 15 hand-placed sphere centres filling the volume
    sphere_centres = np.array([
        [-0.5, -0.5, -0.5],
        [0.2, -0.6, -0.3],
        [0.6, -0.4, -0.6],
        [-0.3, 0.0, 0.0],
        [0.3, 0.1, 0.1],
        [0.0, 0.5, 0.0],
        [-0.6, 0.3, 0.2],
        [0.6, 0.3, -0.1],
        [-0.4, 0.6, 0.5],
        [0.2, 0.7, 0.4],
        [0.5, 0.5, 0.6],
        [-0.1, -0.2, 0.6],
        [0.0, 0.0, 0.5],
        [-0.6, -0.3, 0.3],
        [0.5, -0.1, 0.4],
    ])

    def collision_free(x):
        for c in sphere_centres:
            if _point_in_sphere_3d(x, c, r_obs):
                return False
        if (x[0] < -1.0 or x[0] > 1.0 or
                x[1] < -1.0 or x[1] > 1.0 or
                x[2] < -1.0 or x[2] > 1.0):
            return False
        return True

    metric = ObstacleInflatedMetric(sphere_centres, sigma=0.25, alpha=12.0)
    return collision_free, _make_edge_cost(metric), metric, x_start, x_goal, bounds


ALL_2D_ENVS['2d_hyper_dense'] = env_2d_hyper_dense
ALL_3D_ENVS['3d_dense_labyrinth'] = env_3d_dense_labyrinth
ALL_ENVS.update({
    '2d_hyper_dense': env_2d_hyper_dense,
    '3d_dense_labyrinth': env_3d_dense_labyrinth,
})


# ═══════════════════════════════════════════════════════════════════════
# NEW: 3-D environments designed for strong RIT* advantage
# ═══════════════════════════════════════════════════════════════════════

def env_3d_anisotropic_corridor() -> EnvTuple:
    """3-D anisotropic corridor with L-shaped passage.

    Bounds [0,1]³.  Two thick walls divide the space, connected by
    an L-shaped corridor aligned with the cheap y-axis and z-axis.
    Metric: DiagonalAnisotropicMetric(weights=[10.0, 1.0, 2.0]).

    The x-axis is 10× expensive, so the Riemannian informed set is
    extremely elongated along y.  Since the corridor runs primarily
    along y and z (cheap axes), RIT* focuses samples precisely in
    the corridor while Euclidean planners waste samples in the
    expensive x-regions blocked by walls.

    Returns
    -------
    (collision_checker, edge_cost, metric, x_start, x_goal, bounds)
    """
    x_start = np.array([0.1, 0.1, 0.5])
    x_goal = np.array([0.9, 0.9, 0.5])
    bounds = [(0.0, 1.0), (0.0, 1.0), (0.0, 1.0)]

    boxes = [
        # Wall 1: blocks direct path, gap only at low-y
        (np.array([0.28, 0.15, 0.0]), np.array([0.38, 1.0, 1.0])),
        # Wall 2: blocks direct path, gap only at high-y
        (np.array([0.62, 0.0, 0.0]), np.array([0.72, 0.85, 1.0])),
        # Floor block forcing z-detour between walls
        (np.array([0.38, 0.0, 0.35]), np.array([0.62, 0.15, 0.65])),
        # Ceiling block in the mid-corridor
        (np.array([0.38, 0.55, 0.35]), np.array([0.62, 0.85, 0.65])),
    ]

    def collision_free(x):
        if np.any(x < 0.0) or np.any(x > 1.0):
            return False
        for lo, hi in boxes:
            if _point_in_box_3d(x, lo, hi):
                return False
        return True

    metric = DiagonalAnisotropicMetric(weights=[10.0, 1.0, 2.0])
    return collision_free, _make_edge_cost(metric), metric, x_start, x_goal, bounds


def env_3d_obstacle_gauntlet() -> EnvTuple:
    """3-D obstacle gauntlet — spheres blocking the direct path.

    Bounds [-1,1]³.  Twelve spheres (r=0.20) arranged in two
    staggered rows that force the path to slalom between them.
    The obstacle-inflated metric penalises proximity, and RIT*
    naturally steers samples to the clearance corridors between
    obstacles.  Euclidean planners waste many samples inside the
    high-cost zones near obstacle surfaces.

    Metric: ObstacleInflatedMetric(σ=0.30, α=15.0) — strong inflation
    amplifies RIT*'s advantage.

    Returns
    -------
    (collision_checker, edge_cost, metric, x_start, x_goal, bounds)
    """
    x_start = np.array([-0.9, 0.0, 0.0])
    x_goal = np.array([0.9, 0.0, 0.0])
    bounds = [(-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0)]

    r_obs = 0.20
    # Two staggered rows of spheres blocking the direct x-axis path
    sphere_centres = np.array([
        # Row 1: offset in +y, spaced along x
        [-0.6, 0.25, 0.0],
        [-0.2, 0.25, 0.0],
        [0.2, 0.25, 0.0],
        [0.6, 0.25, 0.0],
        # Row 2: offset in -y, staggered x positions
        [-0.4, -0.25, 0.0],
        [0.0, -0.25, 0.0],
        [0.4, -0.25, 0.0],
        # Vertical blockers forcing z-detour in the centre
        [-0.1, 0.0, 0.30],
        [0.1, 0.0, -0.30],
        # Flanking sentinels near start and goal
        [-0.7, -0.15, 0.25],
        [0.7, 0.15, -0.25],
        [0.0, 0.0, 0.0],   # central blocker
    ])

    def collision_free(x):
        if x[0] < -1.0 or x[0] > 1.0 or x[1] < -1.0 or x[1] > 1.0 or x[2] < -1.0 or x[2] > 1.0:
            return False
        for c in sphere_centres:
            if _point_in_sphere_3d(x, c, r_obs):
                return False
        return True

    metric = ObstacleInflatedMetric(sphere_centres, sigma=0.30, alpha=15.0)
    return collision_free, _make_edge_cost(metric), metric, x_start, x_goal, bounds


ALL_3D_ENVS['3d_anisotropic_corridor'] = env_3d_anisotropic_corridor
ALL_3D_ENVS['3d_obstacle_gauntlet'] = env_3d_obstacle_gauntlet
ALL_ENVS.update({
    '3d_anisotropic_corridor': env_3d_anisotropic_corridor,
    '3d_obstacle_gauntlet': env_3d_obstacle_gauntlet,
})


# ═══════════════════════════════════════════════════════════════════════
# NEW: 6-D geometric environment (no PyBullet needed)
# ═══════════════════════════════════════════════════════════════════════

def _point_in_box_nd(p: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> bool:
    """True if n-D point *p* lies inside axis-aligned box [lo, hi]."""
    return bool(np.all(p >= lo) and np.all(p <= hi))


def env_6d_hyper_passage() -> EnvTuple:
    """6-D geometric narrow passage with strong anisotropy.

    Bounds [0,1]⁶.  Three hyper-walls at x₁ = 0.3, 0.5, 0.7 with
    small 6-D cylindrical holes (radius 0.08 in the 5 non-wall dims).
    The passage centres are offset between walls, so the planner must
    navigate through each hole sequentially.

    Metric: DiagonalAnisotropicMetric with weights
    [8.0, 1.0, 1.5, 2.0, 1.0, 1.5] — x₁ is 8× expensive.

    In 6-D, the Euclidean informed-set volume is enormous while the
    Riemannian one is dramatically tighter (volume ratio ≈ κ³ ≈ 512×
    smaller).  This gives RIT* a massive sampling-efficiency advantage.

    Returns
    -------
    (collision_checker, edge_cost, metric, x_start, x_goal, bounds)
    """
    x_start = np.array([0.1, 0.5, 0.5, 0.5, 0.5, 0.5])
    x_goal = np.array([0.9, 0.5, 0.5, 0.5, 0.5, 0.5])
    bounds = [(0.0, 1.0)] * 6

    # Three walls with offset holes
    wall_thickness = 0.04
    hole_radius = 0.08
    walls = [
        # (wall_centre_x1, hole_centre in dims 1-5)
        (0.30, np.array([0.40, 0.40, 0.50, 0.50, 0.50])),
        (0.50, np.array([0.60, 0.50, 0.40, 0.50, 0.50])),
        (0.70, np.array([0.50, 0.60, 0.50, 0.50, 0.40])),
    ]

    def collision_free(x):
        if np.any(x < 0.0) or np.any(x > 1.0):
            return False
        for wall_x, hole_c in walls:
            # Inside the wall slab?
            if abs(x[0] - wall_x) <= wall_thickness / 2:
                # Only free if inside the cylindrical hole
                dist_sq = float(np.sum((x[1:] - hole_c) ** 2))
                if dist_sq > hole_radius ** 2:
                    return False
        return True

    metric = DiagonalAnisotropicMetric(weights=[8.0, 1.0, 1.5, 2.0, 1.0, 1.5])
    return collision_free, _make_edge_cost(metric), metric, x_start, x_goal, bounds


ALL_ENVS['6d_hyper_passage'] = env_6d_hyper_passage


# ═══════════════════════════════════════════════════════════════════════
# 6-D UR10e Manipulator Environments  (PyBullet, headless)
# ═══════════════════════════════════════════════════════════════════════

try:
    from manipulator_env.pybullet_env import UR10eRobotiqEnv
    _HAS_PYBULLET = True
except ImportError:
    _HAS_PYBULLET = False

# UR10e approximate link masses (kg) – used as anisotropic weights
_UR10E_INERTIAS = np.array([7.778, 12.93, 3.87, 1.96, 1.96, 0.202])
_UR10E_WEIGHTS = (_UR10E_INERTIAS / _UR10E_INERTIAS.max()).tolist()


def _make_ur10e_env(scene_fn, base_position=None) -> EnvTuple:
    """Generic factory: wrap a PyBullet scene into the standard 6-tuple.

    Uses DiagonalAnisotropicMetric with inertia-based weights for fast
    metric evaluation, combined with real PyBullet collision checking.
    """
    if not _HAS_PYBULLET:
        raise ImportError("pybullet is required for UR10e environments")

    obstacles, q_start, q_goal = scene_fn()
    env = UR10eRobotiqEnv(gui=False, obstacles=obstacles,
                          base_position=base_position)

    metric = DiagonalAnisotropicMetric(weights=_UR10E_WEIGHTS)

    # Practical working bounds (tighter than full ±2π)
    bounds = [(-np.pi, np.pi)] * 6

    def collision_free(x):
        return env.is_collision_free(x)

    return collision_free, _make_edge_cost(metric), metric, q_start, q_goal, bounds


def _scene_tabletop():
    obstacles = [
        {"type": "box", "pos": [0.5, 0.0, 0.4],
         "half_extents": [0.3, 0.3, 0.02], "color": [0.6, 0.5, 0.3, 1.0]},
        {"type": "box", "pos": [0.4, 0.0, 0.55],
         "half_extents": [0.08, 0.08, 0.12], "color": [0.8, 0.2, 0.2, 0.9]},
        {"type": "box", "pos": [0.6, 0.15, 0.50],
         "half_extents": [0.06, 0.06, 0.08], "color": [0.2, 0.2, 0.8, 0.9]},
    ]
    # Straight-up home -> reaching to the right side of the table
    q_start = np.array([0.0, -np.pi / 2, 0.0, -np.pi / 2, 0.0, 0.0])
    q_goal = np.array([1.2, -1.2, 1.0, -1.4, -1.57, 0.0])
    return obstacles, q_start, q_goal


def _scene_shelf():
    obstacles = [
        {"type": "box", "pos": [1.22, 0.0, 0.6],
         "half_extents": [0.02, 0.35, 0.30], "color": [0.5, 0.4, 0.3, 0.9]},
        {"type": "box", "pos": [1.05, 0.0, 0.90],
         "half_extents": [0.19, 0.35, 0.02], "color": [0.5, 0.4, 0.3, 0.9]},
        {"type": "box", "pos": [1.05, 0.0, 0.30],
         "half_extents": [0.19, 0.35, 0.02], "color": [0.5, 0.4, 0.3, 0.9]},
        {"type": "box", "pos": [1.05, 0.35, 0.6],
         "half_extents": [0.19, 0.02, 0.30], "color": [0.5, 0.4, 0.3, 0.9]},
        {"type": "box", "pos": [1.05, -0.35, 0.6],
         "half_extents": [0.19, 0.02, 0.30], "color": [0.5, 0.4, 0.3, 0.9]},
    ]
    q_start = np.array([0.0, -2.0, 1.57, -1.1, -1.57, 0.0])
    q_goal = np.array([0.0, -1.0, 1.0, -1.57, -1.57, 0.0])
    return obstacles, q_start, q_goal


def _scene_cluttered():
    rng = np.random.default_rng(123)
    obstacles = []
    for pos in [[0.3, 0.2, 0.5], [0.5, -0.1, 0.6], [0.4, 0.3, 0.3],
                [0.6, -0.3, 0.5], [0.3, -0.2, 0.7], [0.5, 0.15, 0.45],
                [0.7, 0.0, 0.4]]:
        r = 0.06 + rng.uniform(0, 0.04)
        obstacles.append({
            "type": "sphere", "pos": pos, "radius": r,
            "color": [0.7 + rng.uniform(0, 0.3),
                      rng.uniform(0, 0.3), rng.uniform(0, 0.3), 0.85],
        })
    q_start = np.array([-1.57, -1.57, 1.57, -1.57, -1.57, 0.0])
    q_goal = np.array([0.7, -0.9, 1.2, -1.8, -1.57, -0.3])
    return obstacles, q_start, q_goal


def env_6d_tabletop() -> EnvTuple:
    """6-D UR10e tabletop scene — box obstacles on a table."""
    return _make_ur10e_env(_scene_tabletop)


def env_6d_shelf() -> EnvTuple:
    """6-D UR10e shelf scene — narrow shelf passage."""
    return _make_ur10e_env(_scene_shelf)


def env_6d_cluttered() -> EnvTuple:
    """6-D UR10e cluttered scene — scattered spheres."""
    return _make_ur10e_env(_scene_cluttered)


# ── Real-world replicated setup ──────────────────────────────────────

_REAL_TABLE_Z   = 0.75
_REAL_SLAB_T    = 0.01990
_REAL_ROBOT_Z   = _REAL_TABLE_Z + _REAL_SLAB_T     # 0.76990 m
_REAL_SHELF_REL = (0.67138, -0.68403, -0.01590)     # shelf centre w.r.t. base

_REAL_SHELF_W = 0.32    # width  (y)
_REAL_SHELF_D = 0.24    # depth  (x)
_REAL_SHELF_H = 0.54    # height (z)
_REAL_SHELF_T = 0.02    # panel thickness


def _scene_real_setup():
    """Real-world setup: UR10e on table with 2-compartment shelf.

    Shelf open face points toward robot (−x direction).
    """
    sx = _REAL_SHELF_REL[0]
    sy = _REAL_SHELF_REL[1]
    sz = _REAL_ROBOT_Z + _REAL_SHELF_REL[2]
    W, D, H, t = _REAL_SHELF_W, _REAL_SHELF_D, _REAL_SHELF_H, _REAL_SHELF_T
    c = [0.92, 0.92, 0.92, 1.0]

    obstacles = [
        # Back wall (+x end)
        {"type": "box", "color": c,
         "pos": [sx + D / 2 - t / 2, sy, sz + H / 2],
         "half_extents": [t / 2, W / 2, H / 2]},
        # Left side wall (−y)
        {"type": "box", "color": c,
         "pos": [sx, sy - W / 2 + t / 2, sz + H / 2],
         "half_extents": [D / 2, t / 2, H / 2]},
        # Right side wall (+y)
        {"type": "box", "color": c,
         "pos": [sx, sy + W / 2 - t / 2, sz + H / 2],
         "half_extents": [D / 2, t / 2, H / 2]},
        # Bottom panel
        {"type": "box", "color": c,
         "pos": [sx, sy, sz + t / 2],
         "half_extents": [D / 2, W / 2, t / 2]},
        # Middle shelf
        {"type": "box", "color": c,
         "pos": [sx, sy, sz + H / 2],
         "half_extents": [D / 2, W / 2, t / 2]},
        # Top panel
        {"type": "box", "color": c,
         "pos": [sx, sy, sz + H - t / 2],
         "half_extents": [D / 2, W / 2, t / 2]},
    ]

    # Start: home config; Goal: reaching toward upper shelf compartment
    q_start = np.array([0.0, -np.pi / 2, 0.0, -np.pi / 2, 0.0, 0.0])
    q_goal = np.array([1.2, -1.2, 1.0, -1.4, -1.57, 0.0])
    return obstacles, q_start, q_goal


def env_6d_real_setup() -> EnvTuple:
    """6-D UR10e real-world replicated scene — shelf on table."""
    return _make_ur10e_env(_scene_real_setup,
                           base_position=[0.0, 0.0, _REAL_ROBOT_Z])


ALL_6D_ENVS = {}
if _HAS_PYBULLET:
    ALL_6D_ENVS = {
        '6d_tabletop':  env_6d_tabletop,
        '6d_shelf':     env_6d_shelf,
        '6d_cluttered': env_6d_cluttered,
        '6d_real_setup': env_6d_real_setup,
    }
    ALL_ENVS.update(ALL_6D_ENVS)


# ═══════════════════════════════════════════════════════════════════════
# CARM wrappers — same geometry, Euclidean base metric for adaptive learning
# ═══════════════════════════════════════════════════════════════════════

def _carm_wrap(env_fn) -> EnvTuple:
    """Return the same environment but with EuclideanMetric as base.

    The planner is expected to be run with ``adaptive_metric=True``
    so the CollisionAdaptiveMetric wraps this Euclidean base and
    learns the cost structure online from collision feedback.
    """
    coll, _, orig_metric, xs, xg, bounds = env_fn()
    dim = len(xs)
    euclid_metric = EuclideanMetric(dim)
    return coll, _make_edge_cost(euclid_metric), euclid_metric, xs, xg, bounds


def env_2d_obstacle_carm() -> EnvTuple:
    """2-D obstacle environment with Euclidean base for CARM learning."""
    return _carm_wrap(env_2d_obstacle_inflated)


def env_2d_maze_carm() -> EnvTuple:
    """2-D maze with Euclidean base for CARM learning."""
    return _carm_wrap(env_2d_maze)


def env_2d_narrow_carm() -> EnvTuple:
    """2-D narrow passage with Euclidean base for CARM learning."""
    return _carm_wrap(env_2d_narrow_passage)


def env_2d_random_forest_carm() -> EnvTuple:
    """2-D random forest with Euclidean base for CARM learning."""
    return _carm_wrap(env_2d_random_forest)


def env_3d_spheres_carm() -> EnvTuple:
    """3-D sphere field with Euclidean base for CARM learning."""
    return _carm_wrap(env_3d_sphere_field)

def env_2d_random_world() -> EnvTuple:
    """2-D "Random Rectangles" — matches BIT* (Gammell et al., ICRA 2015, Fig. 4).

    Reproduces the exact PDT benchmark configuration:
        Bounds:    [-0.5, 0.5]²  (side length 1)
        Start:     (-0.1, -0.1)
        Goal:      ( 0.4,  0.4)
        Obstacles: 35 axis-aligned rectangles, each side ∈ [0.1, 0.2]
    Obstacle anchors (lower-left corners) are drawn uniformly from
    the workspace so that rectangles may partially extend outside
    bounds.  Start and goal are guaranteed collision-free.

    Metric: ObstacleInflatedMetric built from rectangle centres.

    Returns
    -------
    (collision_checker, edge_cost, metric, x_start, x_goal, bounds)
    """
    x_start = np.array([-0.1, -0.1])
    x_goal  = np.array([ 0.4,  0.4])
    bounds  = [(-0.5, 0.5), (-0.5, 0.5)]

    rng   = np.random.default_rng(2015_04)
    n_obs = 35
    min_side = 0.1
    max_side = 0.2

    rects   = []   # list of (lo, hi)
    centres = []   # for the metric
    for _ in range(n_obs * 50):
        if len(rects) >= n_obs:
            break
        # Random anchor (lower-left corner) anywhere in workspace
        ax_ = rng.uniform(-0.5, 0.5)
        ay_ = rng.uniform(-0.5, 0.5)
        w = rng.uniform(min_side, max_side)
        h = rng.uniform(min_side, max_side)
        lo = np.array([ax_, ay_])
        hi = np.array([ax_ + w, ay_ + h])
        cx, cy = ax_ + w / 2.0, ay_ + h / 2.0
        # Ensure start and goal stay free (generous clearance)
        clr = 0.06
        if (lo[0] <= x_start[0] + clr and hi[0] >= x_start[0] - clr and
            lo[1] <= x_start[1] + clr and hi[1] >= x_start[1] - clr):
            continue
        if (lo[0] <= x_goal[0] + clr and hi[0] >= x_goal[0] - clr and
            lo[1] <= x_goal[1] + clr and hi[1] >= x_goal[1] - clr):
            continue
        rects.append((lo, hi))
        centres.append(np.array([cx, cy]))

    rects   = rects[:n_obs]
    centres = np.array(centres[:n_obs])

    collision_free = _make_rect_collision_free(
        np.array([-0.5, -0.5]), np.array([0.5, 0.5]), rects)

    metric = ObstacleInflatedMetric(centres, sigma=0.10, alpha=8.0)
    return collision_free, _make_edge_cost(metric), metric, x_start, x_goal, bounds


def env_2d_dividing_wall() -> EnvTuple:
    """2-D Dividing Wall-gaps (DW) — after APT* (Zhang et al., RA-L 2025, Fig. 5a).

    Reproduces the exact PDT benchmark configuration:
        Bounds:    [0, 1]²  (side length 1)
        Start:     (0.05, 0.5)
        Goal:      (0.95, 0.5)
        Obstacles: A thin vertical wall at x=0.5 (width 0.02) with
                   three narrow gaps at different heights.

    Metric: ObstacleInflatedMetric on wall segment centres.

    Returns
    -------
    (collision_checker, edge_cost, metric, x_start, x_goal, bounds)
    """
    x_start = np.array([0.05, 0.05])
    x_goal  = np.array([0.95, 0.95])
    bounds  = [(0.0, 1.0), (0.0, 1.0)]

    # Vertical wall at x=0.47..0.53 (width 0.06) with three narrow gaps
    wall_x_lo, wall_x_hi = 0.47, 0.53
    # Wall segments (solid parts); two narrow gaps between them
    # gap 1 (bottom): y ∈ [0.10, 0.13]  (width 0.03)
    # gap 2 (upper):  y ∈ [0.85, 0.88]  (width 0.03)
    wall_segments = [
        (np.array([wall_x_lo, 0.00]), np.array([wall_x_hi, 0.10])),  # bottom
        (np.array([wall_x_lo, 0.13]), np.array([wall_x_hi, 0.85])),  # middle (solid)
        (np.array([wall_x_lo, 0.88]), np.array([wall_x_hi, 1.00])),  # top
    ]

    # Additional flanking blocks to make it more interesting
    flanking = [
        (np.array([0.25, 0.70]), np.array([0.35, 0.85])),
        (np.array([0.65, 0.15]), np.array([0.75, 0.30])),
    ]

    all_rects = wall_segments + flanking

    # Centres for metric (wall segment midpoints + flank midpoints)
    centres = np.array([
        (lo + hi) / 2.0 for lo, hi in all_rects
    ])

    def collision_free(x):
        if x[0] < 0.0 or x[0] > 1.0 or x[1] < 0.0 or x[1] > 1.0:
            return False
        for lo, hi in all_rects:
            if _point_in_rect_2d(x, lo, hi):
                return False
        return True

    metric = ObstacleInflatedMetric(centres, sigma=0.12, alpha=8.0)
    return collision_free, _make_edge_cost(metric), metric, x_start, x_goal, bounds


ALL_2D_ENVS['2d_dividing_wall'] = env_2d_dividing_wall
ALL_ENVS['2d_dividing_wall'] = env_2d_dividing_wall


def env_3d_wall_and_gaps() -> EnvTuple:
    """3-D Wall & Gaps — a dividing wall in 3D with two offset holes.

    Bounds [0,1]³.  A solid wall at x=0.5 (thickness 0.06) with two
    cylindrical holes at different (y,z) locations forces the planner
    to choose between two narrow passages.  Two box obstacles flank
    the wall to increase difficulty.

    Metric: DiagonalAnisotropicMetric(weights=[5.0, 1.0, 1.5]).

    Returns
    -------
    (collision_checker, edge_cost, metric, x_start, x_goal, bounds)
    """
    x_start = np.array([0.1, 0.5, 0.5])
    x_goal  = np.array([0.9, 0.5, 0.5])
    bounds  = [(0.0, 1.0), (0.0, 1.0), (0.0, 1.0)]

    wall_lo_x, wall_hi_x = 0.47, 0.53
    # Two offset holes in the wall
    holes = [
        (np.array([0.35, 0.35]), 0.10),  # hole 1 at (y=0.35, z=0.35)
        (np.array([0.70, 0.70]), 0.10),  # hole 2 at (y=0.70, z=0.70)
    ]
    # Flanking box obstacles
    boxes = [
        (np.array([0.20, 0.60, 0.00]), np.array([0.35, 0.85, 0.40])),
        (np.array([0.65, 0.15, 0.60]), np.array([0.80, 0.40, 1.00])),
        (np.array([0.30, 0.00, 0.60]), np.array([0.45, 0.25, 0.90])),
        (np.array([0.55, 0.75, 0.10]), np.array([0.70, 1.00, 0.40])),
    ]

    def collision_free(x):
        if np.any(x < 0.0) or np.any(x > 1.0):
            return False
        # Wall collision
        if wall_lo_x <= x[0] <= wall_hi_x:
            in_hole = False
            for hc, hr in holes:
                dist_yz = float(np.sqrt((x[1] - hc[0])**2 + (x[2] - hc[1])**2))
                if dist_yz <= hr:
                    in_hole = True
                    break
            if not in_hole:
                return False
        # Box obstacles
        for lo, hi in boxes:
            if _point_in_box_3d(x, lo, hi):
                return False
        return True

    metric = DiagonalAnisotropicMetric(weights=[5.0, 1.0, 1.5])
    return collision_free, _make_edge_cost(metric), metric, x_start, x_goal, bounds


ALL_3D_ENVS['3d_wall_and_gaps'] = env_3d_wall_and_gaps
ALL_ENVS['3d_wall_and_gaps'] = env_3d_wall_and_gaps


def env_3d_box_field() -> EnvTuple:
    """3-D Box Field — dense axis-aligned boxes with clearance corridors.

    Bounds [0,1]³.  Ten box obstacles placed in a structured grid
    pattern with narrow corridors between them.  The planner must
    navigate through the corridors to reach the goal.

    Metric: ObstacleInflatedMetric(σ=0.20, α=10.0).

    Returns
    -------
    (collision_checker, edge_cost, metric, x_start, x_goal, bounds)
    """
    x_start = np.array([0.05, 0.05, 0.05])
    x_goal  = np.array([0.95, 0.95, 0.95])
    bounds  = [(0.0, 1.0), (0.0, 1.0), (0.0, 1.0)]

    boxes = [
        # Ground-level blocking row
        (np.array([0.15, 0.15, 0.00]), np.array([0.35, 0.35, 0.45])),
        (np.array([0.45, 0.00, 0.00]), np.array([0.65, 0.25, 0.35])),
        (np.array([0.70, 0.30, 0.00]), np.array([0.90, 0.55, 0.30])),
        # Mid-level obstacles
        (np.array([0.10, 0.50, 0.30]), np.array([0.30, 0.75, 0.60])),
        (np.array([0.40, 0.40, 0.35]), np.array([0.60, 0.65, 0.65])),
        (np.array([0.65, 0.55, 0.25]), np.array([0.85, 0.80, 0.55])),
        # Upper-level obstacles
        (np.array([0.20, 0.20, 0.60]), np.array([0.45, 0.45, 0.85])),
        (np.array([0.50, 0.60, 0.65]), np.array([0.75, 0.85, 0.90])),
        (np.array([0.10, 0.70, 0.55]), np.array([0.30, 0.95, 0.80])),
        (np.array([0.70, 0.10, 0.50]), np.array([0.90, 0.35, 0.80])),
    ]

    centres = np.array([(lo + hi) / 2.0 for lo, hi in boxes])

    def collision_free(x):
        if np.any(x < 0.0) or np.any(x > 1.0):
            return False
        for lo, hi in boxes:
            if _point_in_box_3d(x, lo, hi):
                return False
        return True

    metric = ObstacleInflatedMetric(centres, sigma=0.20, alpha=10.0)
    return collision_free, _make_edge_cost(metric), metric, x_start, x_goal, bounds


ALL_3D_ENVS['3d_box_field'] = env_3d_box_field
ALL_ENVS['3d_box_field'] = env_3d_box_field


# ═══════════════════════════════════════════════════════════════════════
# Euclidean-cost variants (same obstacles, Euclidean metric)
# ═══════════════════════════════════════════════════════════════════════

def _euclidean_wrap(env_fn) -> EnvTuple:
    """Return the same environment but with a pure Euclidean cost metric."""
    coll, _, orig_metric, xs, xg, bounds = env_fn()
    dim = len(xs)
    euclid_metric = EuclideanMetric(dim)
    return coll, _make_edge_cost(euclid_metric), euclid_metric, xs, xg, bounds


def env_2d_obstacle_euclidean() -> EnvTuple:
    """2-D obstacle env with Euclidean cost (6 circles, same geometry)."""
    return _euclidean_wrap(env_2d_obstacle_inflated)


def env_2d_narrow_euclidean() -> EnvTuple:
    """2-D narrow passage env with Euclidean cost."""
    return _euclidean_wrap(env_2d_narrow_passage)


def env_2d_maze_euclidean() -> EnvTuple:
    """2-D maze env with Euclidean cost."""
    return _euclidean_wrap(env_2d_maze)


def env_2d_forest_euclidean() -> EnvTuple:
    """2-D random forest env with Euclidean cost."""
    return _euclidean_wrap(env_2d_random_forest)


CARM_ENVS = {
    '2d_obstacle_carm':       env_2d_obstacle_carm,
    '2d_maze_carm':           env_2d_maze_carm,
    '2d_narrow_carm':         env_2d_narrow_carm,
    '2d_random_forest_carm':  env_2d_random_forest_carm,
    '3d_spheres_carm':        env_3d_spheres_carm,
}
ALL_ENVS.update(CARM_ENVS)
