"""
rit_star.py — Core RIT* (Riemannian Informed Trees) planner.

RIT* is a BIT*-style batch-informed planner that replaces the Euclidean
informed set with a Riemannian geodesic ball.  In each iteration a batch
of samples is drawn from I_R, neighbours are connected within a shrinking
radius, and the tree is rewired toward lower-cost paths.  Because
Vol(I_R) < Vol(I_euclid) when G ≠ I, fewer samples are wasted and the
planner converges faster (Theorem 3).

Public interface
----------------
    planner = RITStar(x_start, x_goal, ...)
    path, cost = planner.plan()
    stats = planner.get_stats()
"""

from __future__ import annotations

import time
import heapq
from collections import deque
import numpy as np
from scipy.spatial import KDTree
from typing import Callable, List, Optional, Tuple

from .metric import (RiemannianMetric, DiagonalAnisotropicMetric,
                     EuclideanMetric)
from .metric import CollisionAdaptiveMetric
from .geodesic import GeodesicComputer, diagonal_geodesic, midpoint_geodesic_distance
from .informed_set import RiemannianInformedSet, EuclideanInformedSet, volume_ratio_bound
from .metric_cache import MetricFieldCache


# ═══════════════════════════════════════════════════════════════════════
# Utility: midpoint-based geodesic edge cost (arXiv:2602.00992)
# ═══════════════════════════════════════════════════════════════════════


def riemannian_edge_cost(x: np.ndarray, y: np.ndarray,
                         metric: RiemannianMetric,
                         n_quad: int = 10) -> float:
    """Riemannian geodesic distance between x and y.

    Uses the midpoint-based approximation from Kyaw & Kelly (arXiv:2602.00992):

        d̂(x, y) = √( (y−x)ᵀ G((x+y)/2) (y−x) )

    This matches the true geodesic distance with third-order accuracy O(h³)
    and requires only ONE metric evaluation per call, replacing the previous
    10-point Gauss–Legendre quadrature along the straight line (which was
    inconsistent with the heuristic and 10× slower for CARM).

    The ``n_quad`` parameter is kept for API compatibility but ignored.

    Parameters
    ----------
    x, y : (d,) arrays
    metric : RiemannianMetric
    n_quad : int
        Unused — kept for backward compatibility.

    Returns
    -------
    float
    """
    return midpoint_geodesic_distance(x, y, metric)


def _fast_edge_cost(x: np.ndarray, y: np.ndarray,
                    metric: RiemannianMetric) -> float:
    """Fast midpoint edge cost — same as riemannian_edge_cost."""
    return midpoint_geodesic_distance(x, y, metric)


# ═══════════════════════════════════════════════════════════════════════
# Collision checking helpers
# ═══════════════════════════════════════════════════════════════════════

def check_edge_collision(x: np.ndarray, y: np.ndarray,
                         collision_free: Callable, n_checks: int = 20,
                         step_size: float = 0.0) -> bool:
    """Return True if the straight-line edge [x, y] is collision-free.

    Parameters
    ----------
    x, y : (d,) arrays
    collision_free : callable  x -> bool  (True = free)
    n_checks : int
        Minimum number of interpolation points.
    step_size : float
        Maximum distance between consecutive checks. If 0 (default),
        auto-selects based on edge length: n_checks for short edges,
        more for long edges.

    Returns
    -------
    bool
        True if all interpolation points are free.
    """
    diff = y - x
    length = float(np.sqrt(diff @ diff))
    if length < 1e-12:
        return collision_free(x)
    if step_size > 0:
        n = max(n_checks, int(np.ceil(length / step_size)))
    else:
        # Adaptive: scale checks with edge length, capped to avoid
        # excessive PyBullet calls in high-D
        n = max(n_checks, min(n_checks * 5, int(np.ceil(length / 0.01))))
    inv_n = 1.0 / n
    for i in range(n + 1):
        pt = x + (i * inv_n) * diff
        if not collision_free(pt):
            return False
    return True


def check_edge_collision_adaptive(x: np.ndarray, y: np.ndarray,
                                  collision_free: Callable,
                                  coarse_step: float = 0.05,
                                  fine_step: float = 0.01,
                                  n_min: int = 3) -> bool:
    """Adaptive-resolution collision check: coarse pass then local refinement.

    First checks at coarse resolution.  If the coarse pass finds a
    collision, returns False immediately.  If it passes, refines only
    the sub-segments whose midpoints are close to obstacle boundaries
    (detected via a free→occupied transition in the coarse grid).
    """
    diff = y - x
    length = float(np.linalg.norm(diff))
    if length < 1e-12:
        return collision_free(x)
    n_coarse = max(n_min, int(np.ceil(length / coarse_step)))
    # Coarse pass
    prev_free = collision_free(x)
    if not prev_free:
        return False
    for i in range(1, n_coarse + 1):
        t = i / n_coarse
        pt = x + t * diff
        cur_free = collision_free(pt)
        if not cur_free:
            return False
        prev_free = cur_free
    # All coarse points free — do fine refinement on full edge
    n_fine = max(n_min, int(np.ceil(length / fine_step)))
    if n_fine <= n_coarse:
        return True
    for i in range(1, n_fine):
        t = i / n_fine
        pt = x + t * diff
        if not collision_free(pt):
            return False
    return True


def check_edge_collision_with_feedback(
        x: np.ndarray, y: np.ndarray,
        collision_free: Callable,
    n_checks: int = 20,
        step_size: float = 0.0) -> Tuple[bool, Optional[np.ndarray]]:
    """Edge collision check that returns the collision point.

    Same logic as ``check_edge_collision`` but additionally returns
    the first interpolation point found in collision so it can be
    fed back to an adaptive metric.

    Returns
    -------
    (is_free, collision_point)
        is_free : bool — True if all interpolation points are free.
        collision_point : (d,) array or None — first point in collision.
    """
    diff = y - x
    length = float(np.sqrt(diff @ diff))
    if length < 1e-12:
        if not collision_free(x):
            return False, x.copy()
        return True, None
    if step_size > 0:
        n = max(n_checks, int(np.ceil(length / step_size)))
    else:
        n = max(n_checks, min(n_checks * 5, int(np.ceil(length / 0.01))))
    inv_n = 1.0 / n
    for i in range(n + 1):
        pt = x + (i * inv_n) * diff
        if not collision_free(pt):
            return False, pt.copy()
    return True, None


# ═══════════════════════════════════════════════════════════════════════
# Node
# ═══════════════════════════════════════════════════════════════════════

class Node:
    """A vertex in the search tree.

    Attributes
    ----------
    x : (d,) ndarray — configuration
    cost : float — cost-to-come from start
    parent : Node or None
    children : list[Node]
    heuristic : float — precomputed h_R(x, x_goal)
    f_value : float — cost + heuristic (for priority ordering)
    """

    __slots__ = ('x', 'cost', 'parent', 'children', 'heuristic', 'f_value')

    def __init__(self, x: np.ndarray, cost: float = np.inf):
        self.x = np.asarray(x, dtype=float)
        self.cost = cost
        self.parent: Optional[Node] = None
        self.children: List[Node] = []
        self.heuristic: float = 0.0
        self.f_value: float = np.inf

    def __lt__(self, other):
        return self.f_value < other.f_value


# ═══════════════════════════════════════════════════════════════════════
# RIT* planner
# ═══════════════════════════════════════════════════════════════════════

class RITStar:
    """Riemannian Informed Trees (RIT*) planner.

    Parameters
    ----------
    x_start, x_goal : (d,) arrays
    c_space_bounds : list of (min, max) per dimension
    collision_checker : callable  x -> bool  (True = free)
    metric : RiemannianMetric
    geodesic_tier : str
    batch_size : int
    max_iterations : int
    connection_radius_factor : float
    prune_threshold : float
    random_seed : int
    adaptive_metric : bool
        Enable Collision-Adaptive Riemannian Metric (CARM).
        When True, the planner wraps the given metric in a
        CollisionAdaptiveMetric and feeds collision feedback
        to it during planning, periodically rebuilding the
        metric cache.
    carm_sigma : float
        Gaussian kernel bandwidth for CARM (default 0.1).
    carm_alpha : float
        Penalty strength for CARM (default 5.0).
    carm_rebuild_interval : int
        Rebuild the metric cache every N iterations when metric
        has changed (default 20).
    """

    def __init__(self,
                 x_start: np.ndarray,
                 x_goal: np.ndarray,
                 c_space_bounds: list,
                 collision_checker: Callable,
                 metric: RiemannianMetric,
                 geodesic_tier: str = 'diagonal',
                 batch_size: int = 100,
                 max_iterations: int = 200,
                 connection_radius_factor: float = 1.1,
                 prune_threshold: float = 1.05,
                 random_seed: int = 0,
                 adaptive_metric: bool = False,
                 carm_sigma: float = 0.1,
                 carm_alpha: float = 5.0,
                 carm_rebuild_interval: int = 20,
                 collision_step_size: float = 0.0):

        self.x_start = np.asarray(x_start, dtype=float)
        self.x_goal = np.asarray(x_goal, dtype=float)
        self.dim = len(self.x_start)
        self.bounds = [(float(lo), float(hi)) for lo, hi in c_space_bounds]
        self.collision_free = collision_checker
        self.batch_size = batch_size
        self.max_iterations = max_iterations
        self.r_factor = connection_radius_factor
        self.prune_thresh = prune_threshold
        self.rng = np.random.default_rng(random_seed)

        # ── CARM: Collision-Adaptive Riemannian Metric ────────────────
        self._adaptive_mode = adaptive_metric
        self._carm_rebuild_interval = carm_rebuild_interval
        self._carm_last_update_count = 0
        if adaptive_metric:
            self._carm = CollisionAdaptiveMetric(
                metric, len(self.x_start),
                sigma=carm_sigma, alpha=carm_alpha)
            self.metric = self._carm
        else:
            self._carm = None
            self.metric = metric

        # ── Contribution 1: Metric Tensor Field Cache ────────────────
        # Adaptive resolution: 32 for 2D/3D, but scale down for higher
        # dimensions to avoid memory explosion (res^d grid points).
        cache_res = max(3, min(32, int(5e5 ** (1.0 / self.dim))))
        # Collision step size: 0 = auto (0.01 for <=3D, 0.05 for >=4D)
        if collision_step_size <= 0:
            self._collision_step_size = 0.01 if self.dim <= 3 else 0.05
        else:
            self._collision_step_size = collision_step_size
        # min checks: 20 for all dims
        _min_checks = 20
        self._mc = MetricFieldCache(self.metric, self.bounds, resolution=cache_res,
                                    collision_step_size=self._collision_step_size,
                                    min_collision_checks=_min_checks)
        self._cache_res = cache_res

        # Create GeodesicComputer AFTER cache for accurate conformal geodesics
        self.gc = GeodesicComputer(self.metric, tier=geodesic_tier,
                                    bounds=self.bounds, metric_cache=self._mc)

        # Tree
        self.start_node = Node(self.x_start, cost=0.0)
        self.start_node.heuristic = self._mc.heuristic(self.x_start, self.x_goal)
        self.start_node.f_value = self.start_node.heuristic

        self.goal_node: Optional[Node] = None
        self.vertices: List[Node] = [self.start_node]
        self.c_best = np.inf

        # Stats
        self._stats: list = []
        self._t0 = 0.0

        # Informed set (initialized to None; built when c_best < inf)
        self._informed_set: Optional[RiemannianInformedSet] = None

        # ── Contribution 2: Whitened Informed Sampling ────────────────
        # For conformal metrics (G = s(x)·I), whitening is a uniform
        # scale that can make the ellipsoid degenerate — use standard
        # Euclidean ellipsoid instead.  Whitening only helps anisotropic
        # metrics where G_avg has different eigenvalues per dimension.
        self._use_whitening = not self._mc._is_conformal and not self._mc._is_euclidean
        if self._use_whitening:
            G_avg = self._mc.avg_metric_along(self.x_start, self.x_goal, n=5)
            self._L_whiten = np.linalg.cholesky(G_avg)
            self._L_inv_whiten = np.linalg.inv(self._L_whiten)
            self._ws = self._L_whiten @ self.x_start
            self._wg = self._L_whiten @ self.x_goal
        else:
            self._L_whiten = None
            self._L_inv_whiten = None
            self._ws = None
            self._wg = None
        self._whitened_eis: Optional[EuclideanInformedSet] = None
        self._euclid_eis: Optional[EuclideanInformedSet] = None
        lo = np.array([b[0] for b in self.bounds])
        hi = np.array([b[1] for b in self.bounds])
        self._bounds_lo = lo
        self._bounds_hi = hi

        # Precompute volume of unit ball in R^d
        from scipy.special import gamma as gamma_fn
        self._zeta_d = (np.pi ** (self.dim / 2.0)) / gamma_fn(self.dim / 2.0 + 1.0)

        # KD-tree transform for diagonal metrics
        self._use_weighted_kd = isinstance(metric, DiagonalAnisotropicMetric)
        if self._use_weighted_kd:
            self._sqrt_w = np.sqrt(metric.weights)
            self._w = np.asarray(metric.weights, dtype=float)
        else:
            self._sqrt_w = np.ones(self.dim)
            self._w = None

        # Bridge sampling state for narrow-passage detection
        self._stall_count = 0
        self._last_c_best = np.inf
        self._stall_threshold = 5 + self.dim  # more patience in high-D
        self._bridge_fraction = 0.3  # fraction of batch for bridge samples

        # ── Contribution 4: Dimension-Adaptive Early Stopping ─────────
        # Higher dimensions need more iterations to converge, so we
        # scale the convergence window and tighten the tolerance.
        self._early_stop_window = 25 + 8 * max(0, self.dim - 2)
        self._best_cost_window: deque = deque(maxlen=self._early_stop_window)
        self._early_stop_tol = max(0.0005, 0.002 / self.dim)

        # ── Contribution 5: Adaptive Oversampling Factor ──────────────
        # Track whitened-sampling acceptance rate and dynamically adjust
        # the oversampling multiplier so bounds-clipping losses in
        # high-D are compensated.
        self._whiten_accept_rate = 1.0
        self._whiten_oversample = 1.0 + 0.15 * self.dim  # initial guess



    # ── public API ───────────────────────────────────────────────────

    def plan(self) -> Tuple[List[np.ndarray], float]:
        """Run the RIT* planning loop.

        Returns
        -------
        path : list of (d,) arrays from start to goal, or empty if no
               solution found.
        cost : float — final best cost (inf if unsolved).

        Notes
        -----
        Implements Algorithm 1 of the RIT* paper following BIT*-style
        batch processing: prune → sample → edge-queue processing →
        rewire, repeating until max_iterations.
        """
        self._t0 = time.time()
        for it in range(self.max_iterations):
            # BIT*-style: prune and update informed set BEFORE sampling
            if self.c_best < np.inf:
                self._prune()
                self._update_informed_set()
            samples = self._sample_batch(it)
            self._extend_tree(samples, it)
            if self.c_best < np.inf:
                self._update_stall_counter()
            # CARM: rebuild metric cache when enough new collision data
            if self._adaptive_mode:
                self._maybe_rebuild_carm_cache(it)
            elapsed = time.time() - self._t0
            self._record_stats(it, elapsed)
            # Early termination check
            if self._should_stop_early():
                break

        path = self._extract_path()
        # Path shortcutting post-processing (dimension-scaled attempts)
        if path and len(path) > 2:
            n_sc = 80 + 50 * max(0, self.dim - 2)  # more attempts in high-D
            path = self._shortcut_path(path, n_attempts=n_sc)
        # Recompute exact final cost using original metric (not cache)
        if path and len(path) > 1:
            exact = sum(riemannian_edge_cost(path[i], path[i+1], self.metric)
                        for i in range(len(path) - 1))
            self.c_best = exact
            if self._stats:
                self._stats[-1]['c_best'] = exact
        return path, self.c_best

    def plan_stepwise(self):
        """Generator that yields tree state after each iteration.

        Yields
        ------
        dict with keys: iteration, vertices (list of Node), edges
        (list of (parent_x, child_x) tuples), path (list of arrays or []),
        c_best (float).
        """
        self._t0 = time.time()
        for it in range(self.max_iterations):
            # BIT*-style: prune and update informed set BEFORE sampling
            if self.c_best < np.inf:
                self._prune()
                self._update_informed_set()
            samples = self._sample_batch(it)
            self._extend_tree(samples, it)
            if self.c_best < np.inf:
                self._update_stall_counter()
            # CARM: rebuild metric cache when enough new collision data
            if self._adaptive_mode:
                self._maybe_rebuild_carm_cache(it)
            elapsed = time.time() - self._t0
            self._record_stats(it, elapsed)

            edges = []
            for v in self.vertices:
                if v.parent is not None:
                    edges.append((v.parent.x.copy(), v.x.copy()))
            path = self._extract_path()
            yield {
                'iteration': it,
                'vertices': [v.x.copy() for v in self.vertices],
                'edges': edges,
                'path': path,
                'c_best': self.c_best,
            }

        # Recompute exact final cost
        path = self._extract_path()
        if path and len(path) > 1:
            exact = sum(riemannian_edge_cost(path[i], path[i+1], self.metric)
                        for i in range(len(path) - 1))
            self.c_best = exact
            if self._stats:
                self._stats[-1]['c_best'] = exact

    def get_stats(self) -> list:
        """Per-iteration statistics for plotting.

        Returns
        -------
        list of dict with keys: iteration, n_vertices, c_best,
        informed_set_volume, euclidean_set_volume, volume_ratio,
        acceptance_rate, time_elapsed, n_samples_total.
        """
        return self._stats

    @property
    def base_metric(self) -> 'RiemannianMetric':
        """The underlying base metric, stripping CARM if active.

        Use this to evaluate final path cost consistently regardless of
        whether CARM is enabled (useful for ablation comparisons).
        """
        if self._carm is not None:
            return self._carm.base
        return self.metric

    def disable_riemannian_sampling(self) -> None:
        """Switch to Euclidean (non-whitened) informed-set sampling.

        Disables the anisotropic whitening transformation so that
        samples are drawn from the standard Euclidean ellipsoid.
        Isolates the contribution of Riemannian-aware sampling in
        ablation studies.  Must be called before ``plan()``.
        """
        self._use_whitening = False
        self._whitened_eis = None

    def disable_cascading(self) -> None:
        """Disable L1/L2 cascading edge-cost filters.

        Forces every candidate edge through the full metric evaluation
        instead of the cached L1/L2 upper-bound filters.  This is a
        pure speed optimisation — disabling it should not change the
        final path cost.  Must be called before ``plan()``.
        """
        self._mc._no_cascading = True

    def disable_smoothing(self) -> None:
        """Disable post-processing path shortcutting.

        Skips the shortcut-smoothing pass after the tree search
        completes.  Must be called before ``plan()``.
        """
        self._shortcut_path = lambda path, n_attempts=0: path

    def _fast_heuristic(self, x: np.ndarray) -> float:
        """Admissible Riemannian heuristic to goal via cached metric.

        Uses the L1 midpoint estimate from the metric field cache,
        which is tighter than Euclidean distance for non-identity
        metrics and produces better pruning and f-value ordering.
        """
        return self._mc.heuristic(x, self.x_goal)

    # ── internal: sampling ───────────────────────────────────────────

    def _sample_batch(self, iteration: int = 0) -> np.ndarray:
        """Draw a batch of samples (Contribution 2: Whitened Informed Sampling).

        For anisotropic metrics: sample from a Euclidean ellipsoid in the
        whitened (metric-adapted) space, then transform back — rejection-free.
        For conformal/Euclidean metrics: sample from standard Euclidean
        ellipsoid directly (whitening offers no benefit for isotropic G).
        Before c_best is found: sample uniformly from C_space.
        Always include x_goal in every batch.

        When the planner stalls (no improvement for K iterations),
        bridge sampling is mixed in to bias towards narrow passages.

        Returns
        -------
        (batch_size, d) array
        """
        n = self.batch_size - 1  # reserve one slot for goal

        # Determine how many bridge samples to use
        # In high-D, bridge sampling has near-zero acceptance rate and
        # wastes collision checks — reduce fraction substantially.
        n_bridge = 0
        if self._stall_count >= self._stall_threshold and self.c_best < np.inf:
            frac = self._bridge_fraction
            if self.dim >= 4:
                frac *= 0.15  # ~4.5% in 6D instead of 30%
            elif self.dim >= 3:
                frac *= 0.5   # ~15% in 3D
            n_bridge = max(1, int(n * frac))

        n_informed = n - n_bridge

        if self.c_best < np.inf and self._use_whitening and self._whitened_eis is not None:
            # Anisotropic: whitened ellipsoid sampling with adaptive
            # oversampling (Contribution 5).  Track acceptance rate and
            # increase the oversampling factor when bounds clipping
            # discards many proposals.
            n_draw = int(n_informed * self._whiten_oversample)
            pts_w = self._whitened_eis.sample(n_draw, rng=self.rng)
            pts = (self._L_inv_whiten @ pts_w.T).T
            in_bounds = np.all(
                (pts >= self._bounds_lo) & (pts <= self._bounds_hi),
                axis=1)
            pts = pts[in_bounds]
            # Update adaptive oversampling factor
            rate = len(pts) / max(n_draw, 1)
            self._whiten_accept_rate = 0.8 * self._whiten_accept_rate + 0.2 * rate
            if self._whiten_accept_rate > 0.05:
                self._whiten_oversample = min(
                    1.0 / self._whiten_accept_rate + 0.2, 5.0)
            if len(pts) < n_informed:
                extra = self.rng.uniform(
                    self._bounds_lo, self._bounds_hi,
                    size=(n_informed - len(pts), self.dim))
                pts = np.vstack([pts, extra]) if len(pts) > 0 else extra
            else:
                pts = pts[:n_informed]
        elif self.c_best < np.inf and self._euclid_eis is not None:
            # Conformal / Euclidean: standard ellipsoid sampling
            pts = self._euclid_eis.sample(n_informed, rng=self.rng)
        else:
            pts = self.rng.uniform(
                self._bounds_lo, self._bounds_hi, size=(n_informed, self.dim))

        # Mix in bridge samples when stalling
        if n_bridge > 0:
            bridge_pts = self._generate_bridge_samples(n_bridge)
            pts = np.vstack([pts, bridge_pts])

        return np.vstack([pts, self.x_goal.reshape(1, -1)])

    def _generate_bridge_samples(self, n: int) -> np.ndarray:
        """Generate stall-recovery samples (Contribution 6: Path-Guided
        Stall Recovery).

        When the planner stalls, instead of purely random bridge sampling
        (which has vanishing acceptance in high dimensions), bias samples
        toward the current best path.  This exploits the fact that
        improvements are most likely *near* the existing solution.

        Strategy:
          - 60% path-guided: Gaussian noise around random waypoints on
            the current best path (exploits solution topology).
          - 40% bridge: classical collision-midpoint sampling (explores
            narrow passages not on the current path).

        In d >= 4 the bridge fraction is reduced further because its
        acceptance rate drops exponentially.
        """
        path = self._extract_path()
        n_path_guided = int(n * 0.6) if (path and len(path) >= 2) else 0
        # In very high dimensions, bias more heavily toward path-guided
        if self.dim >= 4 and path and len(path) >= 2:
            n_path_guided = int(n * 0.8)
        n_bridge = n - n_path_guided

        collected = []

        # ── Path-guided samples ──────────────────────────────────────
        if n_path_guided > 0 and path:
            path_arr = np.array(path)
            # Adaptive sigma: shrinks as c_best approaches c* estimate
            base_sigma = 0.05 * np.mean(self._bounds_hi - self._bounds_lo)
            for _ in range(n_path_guided * 3):
                if len(collected) >= n_path_guided:
                    break
                # Pick random point along path (interpolated)
                seg = self.rng.integers(0, len(path_arr) - 1)
                t = self.rng.uniform()
                base = (1 - t) * path_arr[seg] + t * path_arr[seg + 1]
                pt = base + self.rng.normal(0, base_sigma, size=self.dim)
                pt = np.clip(pt, self._bounds_lo, self._bounds_hi)
                if self.collision_free(pt):
                    collected.append(pt)

        # ── Bridge samples (classical) ───────────────────────────────
        sigma = 0.1 * np.mean(self._bounds_hi - self._bounds_lo)
        attempts = 0
        max_attempts = n_bridge * 20
        while len(collected) < n_path_guided + n_bridge and attempts < max_attempts:
            p1 = self.rng.uniform(self._bounds_lo, self._bounds_hi)
            p2 = p1 + self.rng.normal(0, sigma, size=self.dim)
            p2 = np.clip(p2, self._bounds_lo, self._bounds_hi)
            mid = 0.5 * (p1 + p2)
            if self.collision_free(mid):
                if not self.collision_free(p1) or not self.collision_free(p2):
                    collected.append(mid)
                elif attempts > max_attempts // 2:
                    collected.append(mid)
            attempts += 1

        if len(collected) < n:
            extra = self.rng.uniform(
                self._bounds_lo, self._bounds_hi,
                size=(n - len(collected), self.dim))
            if collected:
                return np.vstack([np.array(collected), extra])
            return extra
        return np.array(collected[:n])

    def _update_stall_counter(self):
        """Track whether c_best is improving."""
        if self.c_best < self._last_c_best - 1e-8:
            self._stall_count = 0
            self._last_c_best = self.c_best
        else:
            self._stall_count += 1

    def _should_stop_early(self) -> bool:
        """Return True if cost has converged (no meaningful improvement)."""
        self._best_cost_window.append(self.c_best)
        if len(self._best_cost_window) < self._early_stop_window:
            return False
        oldest = self._best_cost_window[0]
        newest = self._best_cost_window[-1]
        if oldest == np.inf or newest == np.inf:
            return False
        rel_improvement = (oldest - newest) / max(abs(oldest), 1e-12)
        return rel_improvement < self._early_stop_tol

    def _compute_connection_radius(self, n_vertices: int) -> float:
        """Metric-adapted connection radius (Theorem 2).

        r_n^R = gamma_R * (log(n)/n)^(1/d)

        where gamma_R uses the Riemannian volume mu_R(I_R) instead of
        Euclidean volume.  For the KD-tree query, the returned radius
        is in the KD-tree's coordinate system (weighted for diagonal,
        Euclidean otherwise).

        This is the critical formula for the AO proof: too small ->
        disconnected graph (not complete), too large -> O(n^2) edges.

        Parameters
        ----------
        n_vertices : int

        Returns
        -------
        float
        """
        n = max(n_vertices, 2)
        d = self.dim

        # Riemannian volume of informed set (or workspace if no solution)
        if self.c_best < np.inf:
            mu_R = self._riemannian_informed_volume()
        else:
            mu_R = self._riemannian_workspace_volume()

        mu_R = max(mu_R, 1e-12)

        # gamma_R from Theorem 2 — matches Eq. (radius) in the paper
        gamma_R = 2.0 * ((1.0 / d) ** (1.0 / d)) * \
                  ((mu_R / self._zeta_d) ** (1.0 / d))

        r_R = gamma_R * ((np.log(n) / n) ** (1.0 / d))

        # Dimension-adaptive radius boost: in high-D the theoretical
        # radius is often too small for practical finite-sample
        # performance.  A mild multiplicative boost improves connectivity
        # without breaking the AO guarantee (only affects finite time).
        if d >= 4:
            r_R *= 1.0 + 0.15 * (d - 3)

        # Apply user-configured radius multiplier.
        # This was intended to tune finite-sample connectivity but was
        # accidentally not used, making the planner overly conservative.
        r_R *= self.r_factor

        # Scale factor: if KD-tree uses weighted coords, radius is already
        # in weighted space. Otherwise, convert Riemannian radius to Euclidean.
        if self._use_weighted_kd:
            # In weighted space, Riemannian distance ~ Euclidean distance
            pass
        else:
            # For spatially-varying metrics, use min eigenvalue scaling
            lam_min = self._mc.min_eigenvalue()
            r_R = r_R / max(np.sqrt(lam_min), 1e-6)

        # Clamp
        diag = np.sqrt(sum((hi - lo) ** 2 for lo, hi in self.bounds))
        return min(r_R, diag * 0.5)

    def _riemannian_informed_volume(self) -> float:
        """Compute Vol(I_R) — the Lebesgue measure of the Riemannian informed set.

        Theorem 1: Vol(I_R) = volume_ratio_bound * Vol(I_E), where
        Vol(I_R) <= Vol(I_E) whenever G succeq I.
        This is the quantity that drives the connection radius (Eq. radius).
        """
        euclid_vol = self._quick_volume()
        # G = I: Riemannian set is identical to Euclidean set
        if isinstance(self.metric, EuclideanMetric):
            return euclid_vol
        # All other metrics: use analytical ratio from Theorem 1
        # Pass c_best so the eccentricity correction is included
        c = self.c_best if self.c_best < np.inf else None
        vr = volume_ratio_bound(self.metric, self.x_start,
                                self.x_goal, self.dim, c_best=c)
        return vr * euclid_vol

    def _riemannian_workspace_volume(self) -> float:
        """Riemannian volume of the workspace.

        For constant: sqrt(det(G)) * prod(hi-lo).
        For varying: integrate sqrt(det(G)) over workspace (cached from grid).
        """
        return self._mc.riemannian_volume()

    def _quick_volume(self) -> float:
        """Euclidean informed-set volume estimate for radius computation."""
        if self.c_best == np.inf:
            return float(np.prod([hi - lo for lo, hi in self.bounds]))
        c_euclid = float(np.linalg.norm(self.x_goal - self.x_start))
        if c_euclid < 1e-12:
            return 1e-12
        # Use c_best as an upper bound on Euclidean path cost too
        # (Riemannian cost >= Euclidean cost for w >= 1 metrics)
        r1 = self.c_best / 2.0
        r2 = np.sqrt(max(self.c_best ** 2 - c_euclid ** 2, 0.0)) / 2.0
        return self._zeta_d * r1 * r2 ** (self.dim - 1)

    # ── internal: tree extension (BIT*-style edge-queue processing) ──

    def _extend_tree(self, samples: np.ndarray, iteration: int = 0):
        """BIT*-style edge-queue batch processing with Riemannian metrics.

        Replaces sequential per-sample processing with a priority queue
        of candidate edges ordered by f(e) = g(v) + ĉ_R(v,x) + ĥ_R(x),
        following the BIT* batch paradigm but using Riemannian costs.

        Steps:
          1. Filter collision-free samples, pre-filter by Riemannian f-value
          2. Build edge queue from tree vertices to samples (L1 estimates)
          3. Process edges in f-value order (L2 cost + collision check)
          4. Rewire newly connected vertices through the tree

        Uses Contribution 3: Cascading Lazy Edge Evaluation.
          - Level 1 (midpoint) for edge-queue ranking & filtering
          - Level 2 (Simpson's 3-pt) for additional filtering
          - Level 3 (Gauss-Legendre 10-pt + collision) for the ~5%
            surviving edges — provides actual tree edge costs

        Parameters
        ----------
        samples : (batch_size, d) array
        iteration : int
        """
        mc = self._mc
        c_best = self.c_best

        # ── Step 1: Filter free samples and pre-filter by f-value ────
        unconnected = []  # list of (sample_array, heuristic_to_goal)
        for s in samples:
            if c_best < np.inf:
                h_s = mc.heuristic(s, self.x_goal)
                h_start = mc.heuristic(self.x_start, s)
                if h_start + h_s >= c_best:
                    continue
            if not self.collision_free(s):
                if self._adaptive_mode:
                    self._carm.add_collision_point(s)
                continue
            if c_best == np.inf:
                h_s = mc.heuristic(s, self.x_goal)
            unconnected.append((s, h_s))

        if not unconnected:
            return

        # ── Step 2: Build KD-tree over tree vertices and edge queue ──
        n_tree = len(self.vertices)
        tree_coords = np.stack([v.x for v in self.vertices])
        if self._use_weighted_kd:
            kd = KDTree(tree_coords * self._sqrt_w)
        else:
            kd = KDTree(tree_coords)

        r = self._compute_connection_radius(n_tree + len(unconnected))
        max_neighbours = min(
            15 + iteration // 15 + 3 * max(0, self.dim - 3),
            20 + 5 * self.dim)

        # Build edge queue: tree vertex → unconnected sample
        # Ordered by f(e) = g(v) + ĉ_R(v,x) + ĥ_R(x) using L1 estimates
        edge_queue = []
        _cnt = 0
        _is_diag = self._use_weighted_kd  # True when DiagonalAnisotropicMetric
        _w = self._w if _is_diag else None

        for si, (s, h_s) in enumerate(unconnected):
            q = s * self._sqrt_w if _is_diag else s
            idxs = kd.query_ball_point(q, r)
            if not idxs:
                _, idx = kd.query(q)
                idxs = [int(idx)]
            elif len(idxs) > max_neighbours:
                ovr = np.array(idxs)
                diffs = tree_coords[ovr] - s
                dists = np.einsum('ij,ij->i', diffs, diffs)
                order = np.argpartition(dists, max_neighbours)[:max_neighbours]
                idxs = ovr[order].tolist()

            # Batch L1 cost for all neighbours at once (avoids per-edge
            # Python call overhead for constant-metric fast paths)
            nbr_arr = np.array(idxs, dtype=np.intp)
            nbr_coords = tree_coords[nbr_arr]   # (k, d)
            nbr_diffs  = nbr_coords - s          # (k, d)
            if _is_diag:
                c_hats = np.sqrt(np.einsum('ij,j,ij->i', nbr_diffs, _w, nbr_diffs))
            else:
                c_hats = mc.edge_cost_l1(nbr_coords[0], s) if len(idxs) == 1 else \
                         np.array([mc.edge_cost_l1(nbr_coords[k], s)
                                   for k in range(len(idxs))])

            nbr_costs = np.array([self.vertices[i].cost for i in idxs])
            f_es = nbr_costs + c_hats + h_s

            for k_local, (idx, f_e) in enumerate(zip(idxs, f_es)):
                if c_best == np.inf or f_e < c_best:
                    heapq.heappush(edge_queue, (float(f_e), _cnt, idx, si))
                    _cnt += 1

        # ── Step 3: Process edges in f-value order ───────────────────
        connected = {}  # si → Node (tracks which samples are connected)

        while edge_queue:
            f_e, _, vidx, si = heapq.heappop(edge_queue)

            if f_e >= self.c_best:
                break

            s, h_s = unconnected[si]
            v = self.vertices[vidx]

            # Skip if sample already connected with a cost that this
            # edge cannot beat (L1 lower bound)
            if si in connected:
                existing = connected[si]
                if existing.cost <= v.cost + mc.edge_cost_l1(v.x, s):
                    continue

            # L2 filter (Simpson's 3-pt) — cheap filter before L3
            ec_l2 = mc.edge_cost_l2(v.x, s)
            if v.cost + ec_l2 + h_s >= self.c_best:
                continue

            # L3: 10-point Gauss-Legendre + collision checking
            # Only the ~5% of edges surviving L1+L2 reach here
            if self._adaptive_mode:
                ec, is_free, coll_pt = mc.edge_cost_l3_with_collision_feedback(
                    v.x, s, self.collision_free)
                if not is_free and coll_pt is not None:
                    self._carm.add_collision_point(coll_pt)
            else:
                ec, is_free = mc.edge_cost_l3_with_collision(
                    v.x, s, self.collision_free)

            if not is_free:
                continue

            new_cost = v.cost + ec

            # Re-check with L3 exact cost (may differ from L2 estimate)
            if new_cost + h_s >= self.c_best:
                continue

            if not is_free:
                continue

            is_goal = np.allclose(s, self.x_goal, atol=1e-8)

            # Goal reconnection
            if is_goal and self.goal_node is not None:
                if new_cost < self.goal_node.cost:
                    if self.goal_node.parent is not None:
                        old_p = self.goal_node.parent
                        if self.goal_node in old_p.children:
                            old_p.children.remove(self.goal_node)
                    self.goal_node.parent = v
                    self.goal_node.cost = new_cost
                    self.goal_node.f_value = new_cost
                    v.children.append(self.goal_node)
                    self.c_best = new_cost
                continue

            # Improve an already-connected sample's parent
            if si in connected:
                existing = connected[si]
                if new_cost < existing.cost:
                    if existing.parent is not None and existing in existing.parent.children:
                        existing.parent.children.remove(existing)
                    existing.parent = v
                    existing.cost = new_cost
                    existing.f_value = new_cost + existing.heuristic
                    v.children.append(existing)
                    self._propagate_cost(existing)
                continue

            # Add new node to tree
            new_node = Node(s.copy(), cost=new_cost)
            new_node.parent = v
            new_node.heuristic = 0.0 if is_goal else self._fast_heuristic(s)
            new_node.f_value = new_cost + new_node.heuristic
            v.children.append(new_node)
            self.vertices.append(new_node)
            connected[si] = new_node

            if is_goal:
                self.goal_node = new_node
                self.c_best = new_cost

        # ── Step 4: Rewire newly connected vertices ──────────────────
        if connected:
            self._rewire_vertices(list(connected.values()), r)

        if self.goal_node is not None:
            self.c_best = self.goal_node.cost

    def _rewire_vertices(self, new_nodes: list, r: float):
        """Rewire existing tree vertices through newly added nodes.

        For each new node, check if it provides a cheaper route to
        nearby existing vertices than their current parent.  Uses the
        same L1→L2→collision cascade as edge processing.

        Parameters
        ----------
        new_nodes : list of Node
            Recently connected vertices to attempt rewiring from.
        r : float
            Connection radius (in KD-tree coordinate space).
        """
        if not new_nodes:
            return
        mc = self._mc
        c_best = self.c_best

        # Build KD-tree of the full current tree (including nodes added
        # during _extend_tree this iteration) so rewiring can route
        # through any new node from the same batch.
        coords = np.stack([v.x for v in self.vertices])
        if self._use_weighted_kd:
            kd = KDTree(coords * self._sqrt_w)
        else:
            kd = KDTree(coords)
        for nn in new_nodes:
            q = nn.x * self._sqrt_w if self._use_weighted_kd else nn.x
            idxs = kd.query_ball_point(q, r)

            for idx in idxs:
                v = self.vertices[idx]
                if v is nn or v is self.start_node or v is nn.parent:
                    continue
                if v.f_value > c_best:
                    continue
                # L1 fast filter
                ec_l1 = mc.edge_cost_l1(nn.x, v.x)
                if nn.cost + ec_l1 >= v.cost:
                    continue
                # L2 filter (Simpson's 3-pt)
                ec_l2 = mc.edge_cost_l2(nn.x, v.x)
                if nn.cost + ec_l2 >= v.cost:
                    continue
                if nn.cost + ec_l2 + v.heuristic >= c_best:
                    continue
                # L3: 10-point Gauss-Legendre + collision checking
                if self._adaptive_mode:
                    ec, is_free, coll_pt = mc.edge_cost_l3_with_collision_feedback(
                        nn.x, v.x, self.collision_free)
                    if not is_free and coll_pt is not None:
                        self._carm.add_collision_point(coll_pt)
                else:
                    ec, is_free = mc.edge_cost_l3_with_collision(
                        nn.x, v.x, self.collision_free)
                if not is_free:
                    continue
                new_cost_v = nn.cost + ec
                if new_cost_v < v.cost:
                    if v.parent is not None and v in v.parent.children:
                        v.parent.children.remove(v)
                    v.parent = nn
                    v.cost = new_cost_v
                    v.f_value = new_cost_v + v.heuristic
                    nn.children.append(v)
                    self._propagate_cost(v)

        if self.goal_node is not None:
            self.c_best = self.goal_node.cost

    def _maybe_rebuild_carm_cache(self, iteration: int) -> None:
        """Rebuild metric cache when CARM has accumulated enough new data.

        Called every iteration in adaptive mode.  Only rebuilds when:
          1. Enough iterations have passed (rebuild_interval), AND
          2. The collision count has grown since the last rebuild.

        After rebuilding, vertex heuristics and f-values are updated
        to reflect the new metric, and the KD-tree is invalidated.
        """
        if self._carm is None:
            return
        new_count = self._carm.update_count
        if new_count <= self._carm_last_update_count:
            return
        if iteration > 0 and iteration % self._carm_rebuild_interval != 0:
            return

        self._carm_last_update_count = new_count
        # Rebuild the metric field cache with the updated adaptive metric
        _min_checks = 20
        self._mc = MetricFieldCache(self.metric, self.bounds,
                                    resolution=self._cache_res,
                                    collision_step_size=self._collision_step_size,
                                    min_collision_checks=_min_checks)
        # Re-propagate costs from root so that g-values reflect the
        # updated CARM metric.  Edges that now pass through inflated
        # (near-obstacle) regions become more expensive, raising f-values
        # above c_best and enabling tighter Riemannian-set pruning.
        self.start_node.cost = 0.0
        self.start_node.heuristic = self._mc.heuristic(self.x_start, self.x_goal)
        self.start_node.f_value = self.start_node.heuristic
        self._propagate_cost(self.start_node)
        # Update heuristics for any vertices not yet reached by propagation
        # (disconnected subtrees, goal node)
        for v in self.vertices:
            v.heuristic = self._mc.heuristic(v.x, self.x_goal)
            v.f_value = v.cost + v.heuristic

    def _propagate_cost(self, node: Node):
        """Iteratively update cost-to-come for descendants after rewire.

        Uses a stack instead of recursion to avoid hitting Python's
        recursion limit on deep trees.
        """
        mc = self._mc
        stack = [node]
        while stack:
            current = stack.pop()
            for child in current.children:
                ec = mc.edge_cost_exact(current.x, child.x)
                child.cost = current.cost + ec
                child.f_value = child.cost + child.heuristic
                stack.append(child)

    # ── internal: pruning ────────────────────────────────────────────

    def _prune(self):
        """Remove vertices whose f-value exceeds prune_threshold * c_best.

        Notes
        -----
        Implements the pruning step of Algorithm 1 — vertices that
        cannot possibly improve the solution are discarded.
        """
        if self.c_best == np.inf:
            return
        threshold = self.prune_thresh * self.c_best
        kept = []
        for v in self.vertices:
            if v is self.start_node:
                kept.append(v)
                continue
            if v is self.goal_node:
                kept.append(v)
                continue
            if v.f_value <= threshold:
                kept.append(v)
            else:
                # Detach from parent
                if v.parent is not None and v in v.parent.children:
                    v.parent.children.remove(v)
        self.vertices = kept

    # ── internal: informed set management ────────────────────────────

    def _update_informed_set(self):
        """Rebuild informed set after c_best improves.

        For anisotropic metrics: builds a whitened EuclideanInformedSet.
        For conformal/Euclidean: builds a standard EuclideanInformedSet.
        """
        if self.c_best < np.inf:
            if self._use_whitening:
                d_w = float(np.linalg.norm(self._wg - self._ws))
                c_eff = max(self.c_best, d_w + 1e-6)
                self._whitened_eis = EuclideanInformedSet(
                    self._ws, self._wg, c_eff, bounds=None)
            else:
                self._euclid_eis = EuclideanInformedSet(
                    self.x_start, self.x_goal, self.c_best,
                    bounds=self.bounds)

    # ── internal: path extraction ────────────────────────────────────

    def _extract_path(self) -> List[np.ndarray]:
        """Trace the path from x_goal back to x_start via parent pointers.

        Returns
        -------
        list of (d,) arrays, start to goal (or empty if unsolved).
        """
        if self.goal_node is None:
            return []
        path = []
        node = self.goal_node
        while node is not None:
            path.append(node.x.copy())
            node = node.parent
        path.reverse()
        return path

    # ── internal: path shortcutting ───────────────────────────────────

    def _shortcut_path(self, path: List[np.ndarray],
                       n_attempts: int = 50) -> List[np.ndarray]:
        """Multi-strategy post-processing for path smoothing.

        Strategy 1: Greedy forward shortcutting — try to skip as many
                    waypoints as possible from each vertex.
        Strategy 2: Greedy backward shortcutting — same in reverse.
        Strategy 3: Random shortcutting — pick random pairs.
        Strategy 4 (high-D): Local perturbation — move interior
                    waypoints toward the midpoint of their neighbours.

        Multiple rounds are applied for high-dimensional spaces where
        a single pass is insufficient.
        """
        if len(path) <= 2:
            return path
        mc = self._mc
        improved = list(path)

        n_rounds = 1 + max(0, self.dim - 3)  # more rounds in high-D

        for _round in range(n_rounds):
            # ── Strategy 1: greedy forward shortcutting ──
            improved = self._greedy_shortcut_forward(improved)

            # ── Strategy 2: greedy backward shortcutting ──
            improved = self._greedy_shortcut_backward(improved)

            # ── Strategy 3: random shortcutting ──
            n_rand = n_attempts // n_rounds
            for _ in range(n_rand):
                if len(improved) <= 2:
                    break
                i = self.rng.integers(0, len(improved) - 2)
                j = self.rng.integers(i + 2, len(improved))
                # Use cumulative costs to avoid O(j-i) recomputation
                seg_costs = [mc.edge_cost_exact(improved[k], improved[k + 1])
                             for k in range(i, j)]
                old_cost = sum(seg_costs)
                new_cost = mc.edge_cost_exact(improved[i], improved[j])
                if new_cost < old_cost:
                    if check_edge_collision(improved[i], improved[j],
                                            self.collision_free):
                        improved = improved[:i + 1] + improved[j:]

        # ── Strategy 4: local perturbation for high-D ──
        if self.dim >= 4 and len(improved) > 2:
            improved = self._local_perturbation(improved,
                                                n_attempts=20 * self.dim)

        return improved

    def _greedy_shortcut_forward(self, path: List[np.ndarray]) -> List[np.ndarray]:
        """Greedily skip waypoints from start to goal."""
        if len(path) <= 2:
            return path
        mc = self._mc
        # Pre-compute cumulative edge costs to avoid redundant recalculation
        n = len(path)
        edge_costs = [mc.edge_cost_exact(path[k], path[k + 1]) for k in range(n - 1)]
        cum_cost = [0.0] * n
        for k in range(1, n):
            cum_cost[k] = cum_cost[k - 1] + edge_costs[k - 1]

        result = [path[0]]
        i = 0
        while i < n - 1:
            best_j = i + 1
            for j in range(n - 1, i + 1, -1):
                old_cost = cum_cost[j] - cum_cost[i]
                new_cost = mc.edge_cost_exact(path[i], path[j])
                if new_cost < old_cost:
                    if check_edge_collision(path[i], path[j],
                                            self.collision_free):
                        best_j = j
                        break
            result.append(path[best_j])
            i = best_j
        return result

    def _greedy_shortcut_backward(self, path: List[np.ndarray]) -> List[np.ndarray]:
        """Greedily skip waypoints from goal to start (reversed)."""
        rev = list(reversed(path))
        smoothed = self._greedy_shortcut_forward(rev)
        return list(reversed(smoothed))

    def _local_perturbation(self, path: List[np.ndarray],
                            n_attempts: int = 100) -> List[np.ndarray]:
        """Move interior waypoints to reduce local cost.

        For each interior waypoint, try moving it toward the midpoint
        of its neighbors.  If the new position is collision-free and
        the local cost (prev→new→next) decreases, keep the move.
        """
        mc = self._mc
        improved = list(path)
        for _ in range(n_attempts):
            if len(improved) <= 2:
                break
            i = self.rng.integers(1, len(improved) - 1)
            prev_pt = improved[i - 1]
            next_pt = improved[i + 1]
            cur_pt = improved[i]
            mid = 0.5 * (prev_pt + next_pt)
            # Try moving partway toward midpoint
            alpha = self.rng.uniform(0.3, 0.9)
            new_pt = cur_pt + alpha * (mid - cur_pt)
            new_pt = np.clip(new_pt, self._bounds_lo, self._bounds_hi)
            if not self.collision_free(new_pt):
                continue
            old_cost = (mc.edge_cost_exact(prev_pt, cur_pt) +
                        mc.edge_cost_exact(cur_pt, next_pt))
            new_cost = (mc.edge_cost_exact(prev_pt, new_pt) +
                        mc.edge_cost_exact(new_pt, next_pt))
            if new_cost < old_cost:
                if (check_edge_collision(prev_pt, new_pt, self.collision_free) and
                        check_edge_collision(new_pt, next_pt, self.collision_free)):
                    improved[i] = new_pt
        return improved

    # ── internal: statistics ─────────────────────────────────────────

    def _record_stats(self, iteration: int, elapsed: float):
        """Append one row of per-iteration statistics."""
        n_samples_total = (iteration + 1) * self.batch_size

        informed_vol = 0.0
        euclid_vol = 0.0
        vol_ratio = 1.0
        analytical_vr = 1.0
        acc_rate = 1.0  # whitened sampling ≈ 100% acceptance

        if self.c_best < np.inf:
            # Euclidean informed-set volume (used by baselines)
            euclid_vol = self._zeta_d * (self.c_best / 2.0) * \
                (np.sqrt(max(self.c_best**2 -
                 float(np.linalg.norm(self.x_goal - self.x_start))**2, 0.0))
                 / 2.0) ** (self.dim - 1)
            # Theorem 1: Vol(I_R)/Vol(I_E) — same call used by radius formula
            analytical_vr = volume_ratio_bound(
                self.metric, self.x_start, self.x_goal, self.dim,
                c_best=self.c_best)
            # True Riemannian informed-set volume = ratio × Euclidean volume
            informed_vol = analytical_vr * euclid_vol
            vol_ratio = analytical_vr

        self._stats.append({
            'iteration': iteration,
            'n_vertices': len(self.vertices),
            'c_best': self.c_best,
            'informed_set_volume': informed_vol,
            'euclidean_set_volume': euclid_vol,
            'volume_ratio': vol_ratio,
            'analytical_volume_ratio': analytical_vr,
            'acceptance_rate': acc_rate,
            'time_elapsed': elapsed,
            'n_samples_total': n_samples_total,
            'carm_collision_points': (self._carm.n_collision_points
                                      if self._carm else 0),
        })
