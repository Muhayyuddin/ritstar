"""
baselines.py — Baseline planners for fair comparison against RIT*.

All planners share the same Node / tree infrastructure as RIT* and
expose an identical public interface:

    planner = InformedRRTStar(...)   # or BITStar, AITStar, EITStar, APTStar
    path, cost = planner.plan()
    stats = planner.get_stats()

Included baselines:
  1. Informed RRT*  — Gammell et al., IJRR 2018.
  2. BIT*           — Gammell et al., ICRA 2015.
  3. AIT*           — Strub & Gammell, ICRA 2020 / IJRR 2022.
  4. EIT*           — Strub & Gammell, IJRR 2022.
  5. APT*           — Adaptive Prolate Trees, RA-L 2025.

The key difference is that baselines always use the *Euclidean*
informed set for sampling and Euclidean distance for heuristics,
whereas RIT* uses the Riemannian versions.
"""

from __future__ import annotations

import time
import heapq
import numpy as np
from scipy.spatial import KDTree
from typing import Callable, List, Optional, Tuple

from .metric import EuclideanMetric, RiemannianMetric
from .geodesic import GeodesicComputer
from .informed_set import EuclideanInformedSet, RiemannianInformedSet
from .rit_star import Node, riemannian_edge_cost, check_edge_collision, _fast_edge_cost


# ═══════════════════════════════════════════════════════════════════════
#  Shared helpers
# ═══════════════════════════════════════════════════════════════════════

def _init_common(obj, x_start, x_goal, c_space_bounds, collision_checker,
                 metric, batch_size, max_iterations, random_seed):
    obj.x_start = np.asarray(x_start, dtype=float)
    obj.x_goal = np.asarray(x_goal, dtype=float)
    obj.dim = len(obj.x_start)
    obj.bounds = [(float(lo), float(hi)) for lo, hi in c_space_bounds]
    obj.collision_free = collision_checker
    obj.metric = metric
    obj.batch_size = batch_size
    obj.max_iterations = max_iterations
    obj.rng = np.random.default_rng(random_seed)
    obj._lo = np.array([b[0] for b in obj.bounds])
    obj._hi = np.array([b[1] for b in obj.bounds])
    from scipy.special import gamma as gamma_fn
    obj._zeta_d = (np.pi ** (obj.dim / 2.0)) / gamma_fn(obj.dim / 2.0 + 1.0)
    obj._stats = []
    obj._t0 = 0.0


def _record(obj, iteration, n_verts, c_best, elapsed, eis_vol=0.0):
    obj._stats.append({
        'iteration': iteration,
        'n_vertices': n_verts,
        'c_best': c_best,
        'informed_set_volume': eis_vol,
        'euclidean_set_volume': eis_vol,
        'volume_ratio': 1.0,
        'acceptance_rate': 1.0,
        'time_elapsed': elapsed,
        'n_samples_total': (iteration + 1) * obj.batch_size,
    })


def _connection_radius(n, dim, vol, zeta_d, r_factor=1.1):
    n = max(n, 2)
    vol = max(vol, 1e-12)
    return r_factor * ((np.log(n) / n) ** (1.0 / dim)) * \
        ((vol / zeta_d) ** (1.0 / dim))


def _uniform_samples(obj, n):
    return obj.rng.uniform(obj._lo, obj._hi, size=(n, obj.dim))


# ═══════════════════════════════════════════════════════════════════════
# 1. Informed RRT*  — Gammell et al., IJRR 2018
# ═══════════════════════════════════════════════════════════════════════

class InformedRRTStar:
    """Informed RRT* baseline.

    Identical to RIT* except:
      • Sampling uses the Euclidean informed ellipsoid (EuclideanInformedSet).
      • Heuristic is plain Euclidean distance.
      • Edge cost still uses the *actual* Riemannian metric so the
        comparison is about sampling efficiency, not cost model.

    Parameters
    ----------
    x_start, x_goal : (d,) arrays
    c_space_bounds : list of (min, max) per dimension
    collision_checker : callable  x -> bool  (True = free)
    metric : RiemannianMetric
        Used for edge-cost computation (fair comparison).
    batch_size : int
    max_iterations : int
    connection_radius_factor : float
    prune_threshold : float
    random_seed : int
    """

    def __init__(self,
                 x_start: np.ndarray,
                 x_goal: np.ndarray,
                 c_space_bounds: list,
                 collision_checker: Callable,
                 metric: RiemannianMetric,
                 batch_size: int = 100,
                 max_iterations: int = 200,
                 connection_radius_factor: float = 1.1,
                 prune_threshold: float = 1.05,
                 random_seed: int = 0):

        self.x_start = np.asarray(x_start, dtype=float)
        self.x_goal = np.asarray(x_goal, dtype=float)
        self.dim = len(self.x_start)
        self.bounds = [(float(lo), float(hi)) for lo, hi in c_space_bounds]
        self.collision_free = collision_checker
        self.metric = metric
        self.batch_size = batch_size
        self.max_iterations = max_iterations
        self.r_factor = connection_radius_factor
        self.prune_thresh = prune_threshold
        self.rng = np.random.default_rng(random_seed)

        # Tree
        self.start_node = Node(self.x_start, cost=0.0)
        self.start_node.heuristic = float(np.linalg.norm(self.x_start - self.x_goal))
        self.start_node.f_value = self.start_node.heuristic
        self.goal_node = None
        self.vertices = [self.start_node]
        self.c_best = np.inf

        # Euclidean informed set (built when c_best < inf)
        self._eis: Optional[EuclideanInformedSet] = None

        from scipy.special import gamma as gamma_fn
        self._zeta_d = (np.pi ** (self.dim / 2.0)) / gamma_fn(self.dim / 2.0 + 1.0)

        self._stats: list = []
        self._t0 = 0.0

    # ── public ───────────────────────────────────────────────────────

    def plan(self) -> Tuple[List[np.ndarray], float]:
        self._t0 = time.time()
        for it in range(self.max_iterations):
            samples = self._sample_batch()
            self._extend_tree(samples)
            if self.c_best < np.inf:
                self._prune()
                self._update_informed_set()
            elapsed = time.time() - self._t0
            self._record_stats(it, elapsed)
        return self._extract_path(), self.c_best

    def get_stats(self) -> list:
        return self._stats

    # ── sampling ─────────────────────────────────────────────────────

    def _sample_batch(self) -> np.ndarray:
        n = self.batch_size - 1
        if self.c_best < np.inf and self._eis is not None:
            pts = self._eis.sample(n, rng=self.rng)
        else:
            lo = np.array([b[0] for b in self.bounds])
            hi = np.array([b[1] for b in self.bounds])
            pts = self.rng.uniform(lo, hi, size=(n, self.dim))
        return np.vstack([pts, self.x_goal.reshape(1, -1)])

    def _compute_r(self, n_vertices):
        n = max(n_vertices, 2)
        vol = float(np.prod([hi - lo for lo, hi in self.bounds]))
        if self._eis is not None and self.c_best < np.inf:
            vol = self._eis.volume()
        vol = max(vol, 1e-12)
        r = self.r_factor * ((np.log(n) / n) ** (1.0 / self.dim)) * \
            ((vol / self._zeta_d) ** (1.0 / self.dim))
        diag = np.sqrt(sum((hi - lo) ** 2 for lo, hi in self.bounds))
        return min(r, diag * 0.5)

    # ── tree extension (RRT* style with batches) ─────────────────────

    def _extend_tree(self, samples):
        n = len(self.vertices)
        coords = np.array([v.x for v in self.vertices])
        kd = KDTree(coords)
        r = self._compute_r(len(self.vertices) + len(samples))

        for s in samples:
            if not self.collision_free(s):
                continue
            idxs = kd.query_ball_point(s, r)
            if not idxs:
                _, idx = kd.query(s)
                idxs = [idx]

            best_parent = None
            best_cost = np.inf
            for idx in idxs:
                v = self.vertices[idx]
                # Euclidean lower bound — valid since G(x) ≥ I for all
                # ObstacleInflatedMetric / DiagonalAnisotropic environments.
                if v.cost + float(np.linalg.norm(v.x - s)) >= best_cost:
                    continue
                ec = riemannian_edge_cost(v.x, s, self.metric)
                nc = v.cost + ec
                if nc < best_cost:
                    if check_edge_collision(v.x, s, self.collision_free, n_checks=20):
                        best_cost = nc
                        best_parent = v
            if best_parent is None:
                continue

            is_goal = np.allclose(s, self.x_goal, atol=1e-8)
            if is_goal and self.goal_node is not None:
                if best_cost < self.goal_node.cost:
                    if self.goal_node.parent is not None:
                        p = self.goal_node.parent
                        if self.goal_node in p.children:
                            p.children.remove(self.goal_node)
                    self.goal_node.parent = best_parent
                    self.goal_node.cost = best_cost
                    self.goal_node.f_value = best_cost
                    best_parent.children.append(self.goal_node)
                    self.c_best = best_cost
                continue

            nn = Node(s.copy(), cost=best_cost)
            nn.parent = best_parent
            nn.heuristic = float(np.linalg.norm(s - self.x_goal))
            nn.f_value = best_cost + nn.heuristic
            best_parent.children.append(nn)
            self.vertices.append(nn)

            if is_goal:
                self.goal_node = nn
                nn.heuristic = 0.0
                nn.f_value = best_cost
                self.c_best = best_cost

            for idx in idxs:
                v = self.vertices[idx]
                if v is best_parent or v is self.start_node:
                    continue
                if nn.cost + float(np.linalg.norm(nn.x - v.x)) >= v.cost:
                    continue
                ec = riemannian_edge_cost(nn.x, v.x, self.metric)
                nc = nn.cost + ec
                if nc < v.cost:
                    if check_edge_collision(nn.x, v.x, self.collision_free, n_checks=20):
                        if v.parent is not None and v in v.parent.children:
                            v.parent.children.remove(v)
                        v.parent = nn
                        v.cost = nc
                        v.f_value = nc + v.heuristic
                        nn.children.append(v)
                        self._propagate(v)

            if self.goal_node is not None:
                self.c_best = self.goal_node.cost

    def _propagate(self, node):
        stack = [node]
        while stack:
            n = stack.pop()
            for ch in n.children:
                ec = riemannian_edge_cost(n.x, ch.x, self.metric)
                ch.cost = n.cost + ec
                ch.f_value = ch.cost + ch.heuristic
                stack.append(ch)

    def _prune(self):
        if self.c_best == np.inf:
            return
        thresh = self.prune_thresh * self.c_best
        kept = []
        for v in self.vertices:
            if v is self.start_node or v is self.goal_node:
                kept.append(v)
            elif v.f_value <= thresh:
                kept.append(v)
            else:
                if v.parent is not None and v in v.parent.children:
                    v.parent.children.remove(v)
        self.vertices = kept

    def _update_informed_set(self):
        if self.c_best < np.inf:
            self._eis = EuclideanInformedSet(
                self.x_start, self.x_goal, self.c_best, bounds=self.bounds)

    def _extract_path(self):
        if self.goal_node is None:
            return []
        path = []
        n = self.goal_node
        while n is not None:
            path.append(n.x.copy())
            n = n.parent
        path.reverse()
        return path

    def _record_stats(self, iteration, elapsed):
        euclid_vol = 0.0
        if self._eis is not None and self.c_best < np.inf:
            euclid_vol = self._eis.volume()
        self._stats.append({
            'iteration': iteration,
            'n_vertices': len(self.vertices),
            'c_best': self.c_best,
            'informed_set_volume': euclid_vol,
            'euclidean_set_volume': euclid_vol,
            'volume_ratio': 1.0,
            'acceptance_rate': 1.0,
            'time_elapsed': elapsed,
            'n_samples_total': (iteration + 1) * self.batch_size,
        })


# ═══════════════════════════════════════════════════════════════════════
# 1b. GA-RRT* — Geometry-aware manifold baseline (arXiv:2602.00992)
# ═══════════════════════════════════════════════════════════════════════

class GeometryAwareRRTStar:
    """Geometry-aware RRT* baseline on Riemannian manifolds.

    This baseline mirrors the midpoint-geodesic manifold approach from
    arXiv:2602.00992:
      • Geodesic distance approximation uses midpoint metric evaluation
        via ``GeodesicComputer(tier='diagonal')``.
      • Sampling is done from a Riemannian informed set
        ``I_R = {x : d_R(xs,x) + d_R(x,xg) <= c_best}``.
      • Tree growth/rewiring remains RRT*-style for fair comparison.

    Edge costs are still evaluated using ``riemannian_edge_cost`` so all
    planners are scored under the same objective.
    """

    def __init__(self,
                 x_start: np.ndarray,
                 x_goal: np.ndarray,
                 c_space_bounds: list,
                 collision_checker: Callable,
                 metric: RiemannianMetric,
                 batch_size: int = 100,
                 max_iterations: int = 200,
                 connection_radius_factor: float = 1.1,
                 prune_threshold: float = 1.05,
                 random_seed: int = 0):

        self.x_start = np.asarray(x_start, dtype=float)
        self.x_goal = np.asarray(x_goal, dtype=float)
        self.dim = len(self.x_start)
        self.bounds = [(float(lo), float(hi)) for lo, hi in c_space_bounds]
        self.collision_free = collision_checker
        self.metric = metric
        self.batch_size = batch_size
        self.max_iterations = max_iterations
        self.r_factor = connection_radius_factor
        self.prune_thresh = prune_threshold
        self.rng = np.random.default_rng(random_seed)

        self.gc = GeodesicComputer(metric, tier='diagonal', bounds=self.bounds)

        self.start_node = Node(self.x_start, cost=0.0)
        self.start_node.heuristic = self.gc.heuristic(self.x_start, self.x_goal)
        self.start_node.f_value = self.start_node.heuristic
        self.goal_node: Optional[Node] = None
        self.vertices: List[Node] = [self.start_node]
        self.c_best = np.inf

        self._ris = None
        self._eis_proxy = None

        from scipy.special import gamma as gamma_fn
        self._zeta_d = (np.pi ** (self.dim / 2.0)) / gamma_fn(self.dim / 2.0 + 1.0)

        self._stats = []
        self._t0 = 0.0

    def plan(self) -> Tuple[List[np.ndarray], float]:
        self._t0 = time.time()
        for it in range(self.max_iterations):
            samples = self._sample_batch()
            self._extend_tree(samples)
            if self.c_best < np.inf:
                self._prune()
                self._update_informed_set()
            elapsed = time.time() - self._t0
            self._record_stats(it, elapsed)
        return self._extract_path(), self.c_best

    def get_stats(self) -> list:
        return self._stats

    def _sample_batch(self) -> np.ndarray:
        n = self.batch_size - 1
        if self.c_best < np.inf and self._ris is not None:
            pts = self._ris.sample(n, rng=self.rng)
        else:
            lo = np.array([b[0] for b in self.bounds])
            hi = np.array([b[1] for b in self.bounds])
            pts = self.rng.uniform(lo, hi, size=(n, self.dim))
        return np.vstack([pts, self.x_goal.reshape(1, -1)])

    def _compute_r(self, n_vertices):
        n = max(n_vertices, 2)
        vol = float(np.prod([hi - lo for lo, hi in self.bounds]))
        if self._eis_proxy is not None and self.c_best < np.inf:
            vol = self._eis_proxy.volume()
        vol = max(vol, 1e-12)
        r = self.r_factor * ((np.log(n) / n) ** (1.0 / self.dim)) * \
            ((vol / self._zeta_d) ** (1.0 / self.dim))
        diag = np.sqrt(sum((hi - lo) ** 2 for lo, hi in self.bounds))
        return min(r, diag * 0.5)

    def _extend_tree(self, samples):
        coords = np.array([v.x for v in self.vertices])
        kd = KDTree(coords)
        r = self._compute_r(len(self.vertices) + len(samples))

        for s in samples:
            if not self.collision_free(s):
                continue
            idxs = kd.query_ball_point(s, r)
            if not idxs:
                _, idx = kd.query(s)
                idxs = [idx]

            best_parent = None
            best_cost = np.inf
            for idx in idxs:
                v = self.vertices[idx]
                if v.cost + self.gc.heuristic(v.x, s) >= best_cost:
                    continue
                ec = riemannian_edge_cost(v.x, s, self.metric)
                nc = v.cost + ec
                if nc < best_cost:
                    if check_edge_collision(v.x, s, self.collision_free, n_checks=20):
                        best_cost = nc
                        best_parent = v
            if best_parent is None:
                continue

            is_goal = np.allclose(s, self.x_goal, atol=1e-8)
            if is_goal and self.goal_node is not None:
                if best_cost < self.goal_node.cost:
                    if self.goal_node.parent is not None:
                        p = self.goal_node.parent
                        if self.goal_node in p.children:
                            p.children.remove(self.goal_node)
                    self.goal_node.parent = best_parent
                    self.goal_node.cost = best_cost
                    self.goal_node.f_value = best_cost
                    best_parent.children.append(self.goal_node)
                    self.c_best = best_cost
                continue

            nn = Node(s.copy(), cost=best_cost)
            nn.parent = best_parent
            nn.heuristic = self.gc.heuristic(s, self.x_goal)
            nn.f_value = best_cost + nn.heuristic
            best_parent.children.append(nn)
            self.vertices.append(nn)

            if is_goal:
                self.goal_node = nn
                nn.heuristic = 0.0
                nn.f_value = best_cost
                self.c_best = best_cost

            for idx in idxs:
                v = self.vertices[idx]
                if v is best_parent or v is self.start_node:
                    continue
                if nn.cost + self.gc.heuristic(nn.x, v.x) >= v.cost:
                    continue
                ec = riemannian_edge_cost(nn.x, v.x, self.metric)
                nc = nn.cost + ec
                if nc < v.cost:
                    if check_edge_collision(nn.x, v.x, self.collision_free, n_checks=20):
                        if v.parent is not None and v in v.parent.children:
                            v.parent.children.remove(v)
                        v.parent = nn
                        v.cost = nc
                        v.f_value = nc + v.heuristic
                        nn.children.append(v)
                        self._propagate(v)

            if self.goal_node is not None:
                self.c_best = self.goal_node.cost

    def _propagate(self, node):
        stack = [node]
        while stack:
            n = stack.pop()
            for ch in n.children:
                ec = riemannian_edge_cost(n.x, ch.x, self.metric)
                ch.cost = n.cost + ec
                ch.f_value = ch.cost + ch.heuristic
                stack.append(ch)

    def _prune(self):
        if self.c_best == np.inf:
            return
        thresh = self.prune_thresh * self.c_best
        kept = []
        for v in self.vertices:
            if v is self.start_node or v is self.goal_node:
                kept.append(v)
            elif v.f_value <= thresh:
                kept.append(v)
            else:
                if v.parent is not None and v in v.parent.children:
                    v.parent.children.remove(v)
        self.vertices = kept

    def _update_informed_set(self):
        if self.c_best < np.inf:
            self._ris = RiemannianInformedSet(
                self.x_start, self.x_goal, self.c_best,
                self.gc, bounds=self.bounds)
            self._eis_proxy = EuclideanInformedSet(
                self.x_start, self.x_goal, self.c_best, bounds=self.bounds)

    def _extract_path(self):
        if self.goal_node is None:
            return []
        path = []
        n = self.goal_node
        while n is not None:
            path.append(n.x.copy())
            n = n.parent
        path.reverse()
        return path

    def _record_stats(self, iteration, elapsed):
        inf_vol = 0.0
        euclid_vol = 0.0
        acc_rate = 1.0
        if self._eis_proxy is not None and self.c_best < np.inf:
            euclid_vol = self._eis_proxy.volume()
        if self._ris is not None and self.c_best < np.inf:
            acc_rate = self._ris.acceptance_rate
            vol_ratio = self._ris.analytical_volume_ratio()
            inf_vol = euclid_vol * vol_ratio
        else:
            vol_ratio = 1.0
        self._stats.append({
            'iteration': iteration,
            'n_vertices': len(self.vertices),
            'c_best': self.c_best,
            'informed_set_volume': inf_vol,
            'euclidean_set_volume': euclid_vol,
            'volume_ratio': vol_ratio,
            'acceptance_rate': acc_rate,
            'time_elapsed': elapsed,
            'n_samples_total': (iteration + 1) * self.batch_size,
        })


# ═══════════════════════════════════════════════════════════════════════
# 2. BIT*  — Gammell et al., ICRA 2015 / IJRR 2020
# ═══════════════════════════════════════════════════════════════════════

class BITStar:
    """Batch Informed Trees (BIT*) baseline.

    Uses the Euclidean informed ellipsoid for batch sampling and an
    edge-priority queue ordered by f(e) = g_T(v) + ĉ(v,x) + ĥ(x).

    Each iteration draws a *fresh* batch of samples from the informed
    set (or uniform if no solution yet), builds an edge queue, and
    processes edges in f-value order.  This prevents the dense-tree
    problem caused by accumulating samples across batches.

    Key features:
      • Batch informed sampling from the Euclidean prolate ellipsoid
        once a solution is found.
      • Edge queue ordered by heuristic f-value with lazy collision
        checking (conditions checked before expensive collision test).
      • Rewiring pass after each batch.
      • Pruning of vertices with f-value above c_best threshold.

    Parameters
    ----------
    x_start, x_goal : (d,) arrays
    c_space_bounds : list of (min, max) per dimension
    collision_checker : callable  x -> bool  (True = free)
    metric : RiemannianMetric
        Used for edge-cost computation (fair comparison).
    batch_size : int
    max_iterations : int
    connection_radius_factor : float
    prune_threshold : float
    random_seed : int
    """

    def __init__(self,
                 x_start: np.ndarray,
                 x_goal: np.ndarray,
                 c_space_bounds: list,
                 collision_checker: Callable,
                 metric: RiemannianMetric,
                 batch_size: int = 100,
                 max_iterations: int = 200,
                 connection_radius_factor: float = 1.1,
                 prune_threshold: float = 1.05,
                 random_seed: int = 0):

        self.x_start = np.asarray(x_start, dtype=float)
        self.x_goal = np.asarray(x_goal, dtype=float)
        self.dim = len(self.x_start)
        self.bounds = [(float(lo), float(hi)) for lo, hi in c_space_bounds]
        self.collision_free = collision_checker
        self.metric = metric
        self.batch_size = batch_size
        self.max_iterations = max_iterations
        self.r_factor = connection_radius_factor
        self.prune_thresh = prune_threshold
        self.rng = np.random.default_rng(random_seed)

        self.start_node = Node(self.x_start, cost=0.0)
        self.start_node.heuristic = float(np.linalg.norm(self.x_start - self.x_goal))
        self.start_node.f_value = self.start_node.heuristic
        self.goal_node: Optional[Node] = None
        self.vertices: List[Node] = [self.start_node]
        self.c_best = np.inf

        self._eis: Optional[EuclideanInformedSet] = None

        from scipy.special import gamma as gamma_fn
        self._zeta_d = (np.pi ** (self.dim / 2.0)) / gamma_fn(self.dim / 2.0 + 1.0)

        self._stats: list = []
        self._t0 = 0.0

    # ── public ───────────────────────────────────────────────────────

    def plan(self) -> Tuple[List[np.ndarray], float]:
        self._t0 = time.time()
        for it in range(self.max_iterations):
            samples = self._sample_batch()
            self._process_batch(samples)
            if self.c_best < np.inf:
                self._prune()
                self._update_informed_set()
            elapsed = time.time() - self._t0
            self._record_stats(it, elapsed)
        return self._extract_path(), self.c_best

    def plan_stepwise(self):
        """Generator that yields tree state after each iteration."""
        self._t0 = time.time()
        for it in range(self.max_iterations):
            samples = self._sample_batch()
            self._process_batch(samples)
            if self.c_best < np.inf:
                self._prune()
                self._update_informed_set()
            elapsed = time.time() - self._t0
            self._record_stats(it, elapsed)
            yield {
                'iteration': it,
                'path': self._extract_path(),
                'c_best': self.c_best,
            }

    def get_stats(self) -> list:
        return self._stats

    # ── sampling ─────────────────────────────────────────────────────

    def _sample_batch(self):
        """Draw a fresh batch from the informed set (or uniform)."""
        n = self.batch_size - 1
        if self.c_best < np.inf and self._eis is not None:
            pts = self._eis.sample(n, rng=self.rng)
        else:
            lo = np.array([b[0] for b in self.bounds])
            hi = np.array([b[1] for b in self.bounds])
            pts = self.rng.uniform(lo, hi, size=(n, self.dim))
        return np.vstack([pts, self.x_goal.reshape(1, -1)])

    def _compute_r(self, n):
        n = max(n, 2)
        vol = float(np.prod([hi - lo for lo, hi in self.bounds]))
        if self._eis is not None and self.c_best < np.inf:
            vol = self._eis.volume()
        vol = max(vol, 1e-12)
        r = self.r_factor * ((np.log(n) / n) ** (1.0 / self.dim)) * \
            ((vol / self._zeta_d) ** (1.0 / self.dim))
        diag = np.sqrt(sum((hi - lo) ** 2 for lo, hi in self.bounds))
        return min(r, diag * 0.5)

    def _k_nearest(self, n):
        """k-nearest connection count (BIT* paper: η * 2e(1+1/d) * log(n))."""
        import math
        n = max(n, 2)
        return max(int(2.0 * math.e * (1.0 + 1.0 / self.dim) *
                       math.log(n) + 0.5), 1)

    # ── edge-priority processing ─────────────────────────────────────

    def _process_batch(self, samples):
        """BIT*-style edge queue processing."""
        free_samples = [s for s in samples if self.collision_free(s)]
        if not free_samples:
            return

        all_points = np.array([v.x for v in self.vertices] + free_samples)
        kd = KDTree(all_points)
        r = self._compute_r(len(all_points))
        k = self._k_nearest(len(all_points))

        n_tree = len(self.vertices)

        # Pre-compute goal/start distances for ALL free samples (vectorised).
        free_pts = np.array(free_samples)                       # (n, d)
        h_goal  = np.linalg.norm(free_pts - self.x_goal, axis=1)  # (n,)
        h_start = np.linalg.norm(free_pts - self.x_start, axis=1) # (n,)

        # Build edge queue: edges from tree vertices to free samples
        edge_queue = []
        _cnt = 0
        for si, s in enumerate(free_samples):
            h_s = float(h_goal[si])
            # f-value pre-filter
            if self.c_best < np.inf and float(h_start[si]) + h_s >= self.c_best:
                continue
            # Use both r-disc and k-nearest for connectivity
            idxs_r = kd.query_ball_point(s, r)
            _, idxs_k = kd.query(s, min(k, len(all_points)))
            if isinstance(idxs_k, np.integer):
                idxs_k = [int(idxs_k)]
            else:
                idxs_k = list(idxs_k)
            idxs = list(set(idxs_r) | set(idxs_k))
            for idx in idxs:
                if idx < n_tree:
                    v = self.vertices[idx]
                    c_hat = float(np.linalg.norm(v.x - s))
                    f_e = v.cost + c_hat + h_s
                    if f_e < self.c_best:
                        heapq.heappush(edge_queue, (f_e, _cnt, v, si))
                        _cnt += 1

        # Process edges in f-value order
        # Use index into free_samples instead of tuple keys to avoid np.round overhead.
        processed = set()  # indices into free_samples
        vert_dict = {tuple(np.round(v.x, 8)): v for v in self.vertices}
        while edge_queue:
            f_e, _, v, si = heapq.heappop(edge_queue)
            if f_e >= self.c_best:
                break

            if si in processed:
                continue

            s = free_samples[si]
            h_s = float(h_goal[si])

            # Euclidean lower bound check before expensive metric eval
            euclid_cost = float(np.linalg.norm(v.x - s))
            if v.cost + euclid_cost + h_s >= self.c_best and self.c_best < np.inf:
                processed.add(si)
                continue

            if not check_edge_collision(v.x, s, self.collision_free, n_checks=20):
                continue

            ec = riemannian_edge_cost(v.x, s, self.metric)
            new_cost = v.cost + ec

            if new_cost + h_s >= self.c_best:
                continue

            is_goal = np.allclose(s, self.x_goal, atol=1e-8)

            if is_goal and self.goal_node is not None:
                if new_cost < self.goal_node.cost:
                    if self.goal_node.parent is not None:
                        p = self.goal_node.parent
                        if self.goal_node in p.children:
                            p.children.remove(self.goal_node)
                    self.goal_node.parent = v
                    self.goal_node.cost = new_cost
                    self.goal_node.f_value = new_cost
                    v.children.append(self.goal_node)
                    self.c_best = new_cost
                processed.add(si)
                continue

            # Check if sample already in tree (rewire)
            s_key = tuple(np.round(s, 8))
            existing = vert_dict.get(s_key)

            if existing is not None:
                if new_cost < existing.cost:
                    if existing.parent is not None and existing in existing.parent.children:
                        existing.parent.children.remove(existing)
                    existing.parent = v
                    existing.cost = new_cost
                    existing.f_value = new_cost + existing.heuristic
                    v.children.append(existing)
                    self._propagate(existing)
            else:
                nn = Node(s.copy(), cost=new_cost)
                nn.parent = v
                nn.heuristic = h_s
                nn.f_value = new_cost + h_s
                v.children.append(nn)
                self.vertices.append(nn)
                vert_dict[s_key] = nn

                if is_goal:
                    self.goal_node = nn
                    nn.heuristic = 0.0
                    nn.f_value = new_cost
                    self.c_best = new_cost

            processed.add(si)

            if self.goal_node is not None:
                self.c_best = self.goal_node.cost

        # Rewiring pass for recently added vertices
        self._rewire_pass(r)

    def _rewire_pass(self, r: float):
        """Attempt to rewire existing vertices through recent additions."""
        if len(self.vertices) < 3:
            return
        coords = np.array([v.x for v in self.vertices])
        kd = KDTree(coords)
        c_best = self.c_best
        recent = self.vertices[-min(self.batch_size, len(self.vertices)):]
        for nn in recent:
            idxs = kd.query_ball_point(nn.x, r)
            for idx in idxs:
                v = self.vertices[idx]
                if v is nn or v is self.start_node or v is nn.parent:
                    continue
                c_hat = float(np.linalg.norm(nn.x - v.x))
                if nn.cost + c_hat >= v.cost:
                    continue
                ec = riemannian_edge_cost(nn.x, v.x, self.metric)
                nc = nn.cost + ec
                if nc < v.cost:
                    if c_best < np.inf and nc + v.heuristic >= c_best:
                        continue
                    if check_edge_collision(nn.x, v.x, self.collision_free, n_checks=20):
                        if v.parent is not None and v in v.parent.children:
                            v.parent.children.remove(v)
                        v.parent = nn
                        v.cost = nc
                        v.f_value = nc + v.heuristic
                        nn.children.append(v)
                        self._propagate(v)
        if self.goal_node is not None:
            self.c_best = self.goal_node.cost

    def _propagate(self, node):
        stack = [node]
        while stack:
            n = stack.pop()
            for ch in n.children:
                ec = riemannian_edge_cost(n.x, ch.x, self.metric)
                ch.cost = n.cost + ec
                ch.f_value = ch.cost + ch.heuristic
                stack.append(ch)

    def _prune(self):
        if self.c_best == np.inf:
            return
        thresh = self.prune_thresh * self.c_best
        kept = []
        for v in self.vertices:
            if v is self.start_node or v is self.goal_node:
                kept.append(v)
            elif v.f_value <= thresh:
                kept.append(v)
            else:
                if v.parent is not None and v in v.parent.children:
                    v.parent.children.remove(v)
        self.vertices = kept

    def _update_informed_set(self):
        if self.c_best < np.inf:
            self._eis = EuclideanInformedSet(
                self.x_start, self.x_goal, self.c_best, bounds=self.bounds)

    def _extract_path(self):
        if self.goal_node is None:
            return []
        path = []
        n = self.goal_node
        while n is not None:
            path.append(n.x.copy())
            n = n.parent
        path.reverse()
        return path

    def _record_stats(self, iteration, elapsed):
        euclid_vol = 0.0
        if self._eis is not None and self.c_best < np.inf:
            euclid_vol = self._eis.volume()
        self._stats.append({
            'iteration': iteration,
            'n_vertices': len(self.vertices),
            'c_best': self.c_best,
            'informed_set_volume': euclid_vol,
            'euclidean_set_volume': euclid_vol,
            'volume_ratio': 1.0,
            'acceptance_rate': 1.0,
            'time_elapsed': elapsed,
            'n_samples_total': (iteration + 1) * self.batch_size,
        })


# ═══════════════════════════════════════════════════════════════════════
# 3. AIT*  — Strub & Gammell, ICRA 2020 / IJRR 2022
# ═══════════════════════════════════════════════════════════════════════

class AITStar:
    """Adaptively Informed Trees (AIT*).

    Key idea (from the paper): simultaneously estimate and exploit a
    problem-specific heuristic via an asymmetric bidirectional search.

    Forward tree: grows from x_start using Informed RRT*-style extend.
    Reverse search: a Dijkstra expansion from x_goal over all free
    samples provides an adaptive cost-to-go heuristic h_hat(x) that
    replaces the simple Euclidean ||x - x_goal|| heuristic.

    The adaptive heuristic makes the forward tree's edge-queue much
    more efficient — it resembles A* with a learned heuristic.

    References
    ----------
    M. Strub and J. D. Gammell, "Adaptively Informed Trees (AIT*):
    Fast Asymptotically Optimal Path Planning through Adaptive
    Heuristics," ICRA 2020.  doi:10.1109/ICRA40945.2020.9197338
    """

    def __init__(self, x_start, x_goal, c_space_bounds, collision_checker,
                 metric: RiemannianMetric,
                 batch_size=100, max_iterations=200,
                 connection_radius_factor=1.1, prune_threshold=1.05,
                 random_seed=0):
        _init_common(self, x_start, x_goal, c_space_bounds,
                     collision_checker, metric, batch_size, max_iterations,
                     random_seed)
        self.r_factor = connection_radius_factor
        self.prune_thresh = prune_threshold

        self.start_node = Node(self.x_start, cost=0.0)
        self.goal_node: Optional[Node] = None
        self.vertices: List[Node] = [self.start_node]
        self.c_best = np.inf
        self._eis: Optional[EuclideanInformedSet] = None

        # Reverse-search heuristic table: maps sample-key -> cost-to-go
        self._h_adaptive: dict = {}

    def plan(self) -> Tuple[List[np.ndarray], float]:
        self._t0 = time.time()
        for it in range(self.max_iterations):
            samples = self._sample_batch()
            free_samples = [s for s in samples if self.collision_free(s)]

            # Reverse search: Dijkstra from goal over free samples
            self._reverse_search(free_samples)

            # Forward extension with adaptive heuristic
            self._forward_extend(free_samples)

            if self.c_best < np.inf:
                self._prune()
                self._eis = EuclideanInformedSet(
                    self.x_start, self.x_goal, self.c_best, bounds=self.bounds)

            elapsed = time.time() - self._t0
            vol = self._eis.volume() if self._eis else 0.0
            _record(self, it, len(self.vertices), self.c_best, elapsed, vol)

        return self._extract_path(), self.c_best

    def get_stats(self):
        return self._stats

    def _sample_batch(self):
        n = self.batch_size - 1
        if self.c_best < np.inf and self._eis is not None:
            pts = self._eis.sample(n, rng=self.rng)
        else:
            pts = _uniform_samples(self, n)
        return np.vstack([pts, self.x_goal.reshape(1, -1)])

    def _h(self, x):
        """Adaptive heuristic: look up reverse-search cost, fall back to Euclidean."""
        key = tuple(np.round(x, 8))
        if key in self._h_adaptive:
            return self._h_adaptive[key]
        return float(np.linalg.norm(x - self.x_goal))

    def _reverse_search(self, free_samples):
        """Dijkstra from x_goal over free_samples to estimate cost-to-go.

        Builds a lightweight graph among the free samples using
        Riemannian edge costs (faithful to the paper's cost model),
        then runs Dijkstra from the goal.  The resulting costs serve
        as the adaptive heuristic h_hat for the forward search.
        """
        if not free_samples:
            return

        all_pts = np.array(free_samples)
        n = len(all_pts)
        if n < 2:
            return

        kd = KDTree(all_pts)
        vol = float(np.prod(self._hi - self._lo))
        r = _connection_radius(n, self.dim, vol, self._zeta_d, self.r_factor)
        r = min(r, 0.5 * np.sqrt(np.sum((self._hi - self._lo) ** 2)))
        diag = np.sqrt(np.sum((self._hi - self._lo) ** 2))
        r = max(r, diag * 0.08)

        # Find goal index
        dists_to_goal = np.linalg.norm(all_pts - self.x_goal, axis=1)
        candidates = np.where(dists_to_goal < 1e-6)[0]
        if len(candidates) > 0:
            goal_idx = int(candidates[0])
        else:
            goal_idx = int(np.argmin(dists_to_goal))

        # Dijkstra using Riemannian edge costs
        INF = float('inf')
        dist = [INF] * n
        dist[goal_idx] = 0.0
        pq = [(0.0, goal_idx)]
        visited = [False] * n

        while pq:
            d, u = heapq.heappop(pq)
            if visited[u]:
                continue
            visited[u] = True
            idxs = kd.query_ball_point(all_pts[u], r)
            for v in idxs:
                if visited[v]:
                    continue
                # Euclidean lower bound: skip if can't improve dist[v]
                euclid_w = float(np.linalg.norm(all_pts[u] - all_pts[v]))
                if d + euclid_w >= dist[v]:
                    continue
                w = riemannian_edge_cost(all_pts[u], all_pts[v], self.metric)
                nd = d + w
                if nd < dist[v]:
                    dist[v] = nd
                    heapq.heappush(pq, (nd, v))

        # Store adaptive heuristic
        self._h_adaptive.clear()
        for i in range(n):
            if dist[i] < INF:
                key = tuple(np.round(all_pts[i], 8))
                self._h_adaptive[key] = dist[i]

    def _forward_extend(self, free_samples):
        """Forward tree extension using the adaptive heuristic."""
        if not free_samples:
            return
        coords = np.array([v.x for v in self.vertices])
        kd = KDTree(coords)
        vol = self._eis.volume() if self._eis and self.c_best < np.inf else \
            float(np.prod(self._hi - self._lo))
        r = _connection_radius(len(self.vertices) + len(free_samples),
                               self.dim, vol, self._zeta_d, self.r_factor)
        diag = np.sqrt(np.sum((self._hi - self._lo) ** 2))
        r = max(r, diag * 0.08)
        r = min(r, diag * 0.5)

        # Build edge queue ordered by f = g(v) + c_hat(v,s) + h_hat(s)
        edge_queue = []
        cnt = 0
        for s in free_samples:
            h_s = self._h(s)
            idxs = kd.query_ball_point(s, r)
            if not idxs:
                _, idx = kd.query(s)
                idxs = [idx]
            for idx in idxs:
                v = self.vertices[idx]
                c_hat = float(np.linalg.norm(v.x - s))
                f_e = v.cost + c_hat + h_s
                if f_e < self.c_best:
                    heapq.heappush(edge_queue, (f_e, cnt, v, s))
                    cnt += 1

        # Process edges: check heuristic condition 2 before collision check
        processed = set()
        vert_dict = {tuple(np.round(v.x, 8)): v for v in self.vertices}
        while edge_queue:
            f_e, _, v, s = heapq.heappop(edge_queue)
            if f_e >= self.c_best:
                break
            s_key = tuple(np.round(s, 8))
            if s_key in processed:
                continue

            h_s = self._h(s)
            c_hat = float(np.linalg.norm(v.x - s))
            # Condition 2: heuristic tree improvement before collision check
            existing = vert_dict.get(s_key)
            g_x = existing.cost if existing is not None else np.inf
            if v.cost + c_hat >= g_x:
                continue

            if not check_edge_collision(v.x, s, self.collision_free, n_checks=20):
                continue

            ec = riemannian_edge_cost(v.x, s, self.metric)
            new_cost = v.cost + ec
            if new_cost + h_s >= self.c_best:
                continue
            if new_cost >= g_x:
                continue

            is_goal = np.allclose(s, self.x_goal, atol=1e-8)
            if is_goal and self.goal_node is not None:
                if new_cost < self.goal_node.cost:
                    if self.goal_node.parent and self.goal_node in self.goal_node.parent.children:
                        self.goal_node.parent.children.remove(self.goal_node)
                    self.goal_node.parent = v
                    self.goal_node.cost = new_cost
                    self.goal_node.f_value = new_cost
                    v.children.append(self.goal_node)
                    self.c_best = new_cost
                processed.add(s_key)
                continue

            if existing is not None:
                # Rewire existing tree vertex
                if existing.parent and existing in existing.parent.children:
                    existing.parent.children.remove(existing)
                existing.parent = v
                existing.cost = new_cost
                existing.f_value = new_cost + existing.heuristic
                v.children.append(existing)
                self._propagate(existing)
            else:
                nn = Node(s.copy(), cost=new_cost)
                nn.parent = v
                nn.heuristic = h_s
                nn.f_value = new_cost + h_s
                v.children.append(nn)
                self.vertices.append(nn)
                vert_dict[s_key] = nn

                if is_goal:
                    self.goal_node = nn
                    nn.heuristic = 0.0
                    nn.f_value = new_cost
                    self.c_best = new_cost

            processed.add(s_key)
            if self.goal_node is not None:
                self.c_best = self.goal_node.cost

    def _propagate(self, node):
        stack = [node]
        while stack:
            n = stack.pop()
            for ch in n.children:
                ec = riemannian_edge_cost(n.x, ch.x, self.metric)
                ch.cost = n.cost + ec
                ch.f_value = ch.cost + ch.heuristic
                stack.append(ch)

    def _prune(self):
        if self.c_best == np.inf:
            return
        thresh = self.prune_thresh * self.c_best
        kept = []
        for v in self.vertices:
            if v is self.start_node or v is self.goal_node:
                kept.append(v)
            elif v.f_value <= thresh:
                kept.append(v)
            else:
                if v.parent and v in v.parent.children:
                    v.parent.children.remove(v)
        self.vertices = kept

    def _extract_path(self):
        if self.goal_node is None:
            return []
        path = []
        n = self.goal_node
        while n is not None:
            path.append(n.x.copy())
            n = n.parent
        path.reverse()
        return path


# ═══════════════════════════════════════════════════════════════════════
# 4. EIT*  — Strub & Gammell, IJRR 2022
# ═══════════════════════════════════════════════════════════════════════

class EITStar:
    """Effort Informed Trees (EIT*).

    Key idea (from the paper): simultaneously estimate and exploit a
    problem-specific heuristic via an asymmetric bidirectional search,
    using *effort* (edge count) instead of *cost* for the reverse
    search.  This makes the heuristic more robust in anisotropic cost
    spaces because effort is independent of the cost metric.

    Forward tree: grows from x_start using Informed RRT*-style extend.
    Reverse search: a Dijkstra expansion from x_goal over all free
    samples provides an adaptive effort-to-go heuristic ê(x) that
    guides exploration independently of cost.

    The forward search then combines cost g(v) with a cost-weighted
    effort heuristic: f(e) = g(v) + ĉ(v,s) + w_effort * ê(s),
    where w_effort adapts based on the current cost-per-edge estimate.

    References
    ----------
    M. Strub and J. D. Gammell, "AIT* and EIT*: Asymmetric
    bidirectional sampling-based path planning," IJRR, 2022.
    doi:10.1177/02783649211069572
    """

    def __init__(self, x_start, x_goal, c_space_bounds, collision_checker,
                 metric: RiemannianMetric,
                 batch_size=100, max_iterations=200,
                 connection_radius_factor=1.1, prune_threshold=1.05,
                 random_seed=0):
        _init_common(self, x_start, x_goal, c_space_bounds,
                     collision_checker, metric, batch_size, max_iterations,
                     random_seed)
        self.r_factor = connection_radius_factor
        self.prune_thresh = prune_threshold

        self.start_node = Node(self.x_start, cost=0.0)
        self.goal_node: Optional[Node] = None
        self.vertices: List[Node] = [self.start_node]
        self.c_best = np.inf
        self._eis: Optional[EuclideanInformedSet] = None

        # Reverse-search effort heuristic: maps sample-key -> edge-count-to-go
        self._effort: dict = {}
        # Adaptive cost-per-edge weight (initialized from Euclidean estimate)
        euclid_dist = float(np.linalg.norm(self.x_start - self.x_goal))
        self._w_effort = euclid_dist / max(10.0, 1.0)

    def plan(self) -> Tuple[List[np.ndarray], float]:
        self._t0 = time.time()
        for it in range(self.max_iterations):
            samples = self._sample_batch()
            free_samples = [s for s in samples if self.collision_free(s)]

            # Reverse search: effort-based Dijkstra from goal
            self._reverse_search(free_samples)

            # Forward extension with effort-weighted heuristic
            self._forward_extend(free_samples)

            if self.c_best < np.inf:
                self._prune()
                self._eis = EuclideanInformedSet(
                    self.x_start, self.x_goal, self.c_best, bounds=self.bounds)
                self._update_effort_weight()

            elapsed = time.time() - self._t0
            vol = self._eis.volume() if self._eis else 0.0
            _record(self, it, len(self.vertices), self.c_best, elapsed, vol)

        return self._extract_path(), self.c_best

    def get_stats(self):
        return self._stats

    def _sample_batch(self):
        n = self.batch_size - 1
        if self.c_best < np.inf and self._eis is not None:
            pts = self._eis.sample(n, rng=self.rng)
        else:
            pts = _uniform_samples(self, n)
        return np.vstack([pts, self.x_goal.reshape(1, -1)])

    def _update_effort_weight(self):
        """Estimate cost-per-edge from the current best path."""
        if self.goal_node is None or self.c_best == np.inf:
            return
        n_edges = 0
        node = self.goal_node
        while node.parent is not None:
            n_edges += 1
            node = node.parent
        if n_edges > 0:
            self._w_effort = self.c_best / n_edges

    def _h(self, x):
        """Effort-weighted heuristic: min of effort-based and Euclidean.

        Takes the minimum to stay admissible.
        """
        euclid = float(np.linalg.norm(x - self.x_goal))
        key = tuple(np.round(x, 8))
        if key in self._effort:
            effort_h = self._effort[key] * self._w_effort
            return min(effort_h, euclid)
        return euclid

    def _reverse_search(self, free_samples):
        """Dijkstra from x_goal over free_samples using *effort* (edge count).

        Each edge has weight 1 (effort = number of edges to goal).
        This makes the heuristic robust to anisotropic cost landscapes.
        """
        if not free_samples:
            return

        all_pts = np.array(free_samples)
        n = len(all_pts)
        if n < 2:
            return

        kd = KDTree(all_pts)
        vol = float(np.prod(self._hi - self._lo))
        r = _connection_radius(n, self.dim, vol, self._zeta_d, self.r_factor)
        diag = np.sqrt(np.sum((self._hi - self._lo) ** 2))
        r = max(r, diag * 0.08)
        r = min(r, diag * 0.5)

        # Find goal index
        dists_to_goal = np.linalg.norm(all_pts - self.x_goal, axis=1)
        candidates = np.where(dists_to_goal < 1e-6)[0]
        if len(candidates) > 0:
            goal_idx = int(candidates[0])
        else:
            goal_idx = int(np.argmin(dists_to_goal))

        # Dijkstra using unit edge weights (effort = edge count)
        INF = float('inf')
        dist = [INF] * n
        dist[goal_idx] = 0.0
        pq = [(0.0, goal_idx)]
        visited = [False] * n

        while pq:
            d, u = heapq.heappop(pq)
            if visited[u]:
                continue
            visited[u] = True
            idxs = kd.query_ball_point(all_pts[u], r)
            for v in idxs:
                if visited[v]:
                    continue
                nd = d + 1.0  # unit effort per edge
                if nd < dist[v]:
                    dist[v] = nd
                    heapq.heappush(pq, (nd, v))

        # Store effort heuristic
        self._effort.clear()
        for i in range(n):
            if dist[i] < INF:
                key = tuple(np.round(all_pts[i], 8))
                self._effort[key] = dist[i]

    def _forward_extend(self, free_samples):
        """Forward tree extension using the effort-weighted heuristic."""
        if not free_samples:
            return
        coords = np.array([v.x for v in self.vertices])
        kd = KDTree(coords)
        vol = self._eis.volume() if self._eis and self.c_best < np.inf else \
            float(np.prod(self._hi - self._lo))
        r = _connection_radius(len(self.vertices) + len(free_samples),
                               self.dim, vol, self._zeta_d, self.r_factor)
        diag = np.sqrt(np.sum((self._hi - self._lo) ** 2))
        r = max(r, diag * 0.08)
        r = min(r, diag * 0.5)

        # Build edge queue ordered by f = g(v) + c_hat(v,s) + h_effort(s)
        edge_queue = []
        cnt = 0
        for s in free_samples:
            h_s = self._h(s)
            idxs = kd.query_ball_point(s, r)
            if not idxs:
                _, idx = kd.query(s)
                idxs = [idx]
            for idx in idxs:
                v = self.vertices[idx]
                c_hat = float(np.linalg.norm(v.x - s))
                f_e = v.cost + c_hat + h_s
                if f_e < self.c_best:
                    heapq.heappush(edge_queue, (f_e, cnt, v, s))
                    cnt += 1

        processed = set()
        vert_dict = {tuple(np.round(v.x, 8)): v for v in self.vertices}
        while edge_queue:
            f_e, _, v, s = heapq.heappop(edge_queue)
            if f_e >= self.c_best:
                break
            s_key = tuple(np.round(s, 8))
            if s_key in processed:
                continue

            h_s = self._h(s)
            c_hat = float(np.linalg.norm(v.x - s))
            # Condition 2: heuristic tree improvement before collision check
            existing = vert_dict.get(s_key)
            g_x = existing.cost if existing is not None else np.inf
            if v.cost + c_hat >= g_x:
                continue

            if not check_edge_collision(v.x, s, self.collision_free, n_checks=20):
                continue

            ec = riemannian_edge_cost(v.x, s, self.metric)
            new_cost = v.cost + ec
            if new_cost + h_s >= self.c_best:
                continue
            if new_cost >= g_x:
                continue

            is_goal = np.allclose(s, self.x_goal, atol=1e-8)
            if is_goal and self.goal_node is not None:
                if new_cost < self.goal_node.cost:
                    if self.goal_node.parent and self.goal_node in self.goal_node.parent.children:
                        self.goal_node.parent.children.remove(self.goal_node)
                    self.goal_node.parent = v
                    self.goal_node.cost = new_cost
                    self.goal_node.f_value = new_cost
                    v.children.append(self.goal_node)
                    self.c_best = new_cost
                processed.add(s_key)
                continue

            if existing is not None:
                # Rewire existing tree vertex
                if existing.parent and existing in existing.parent.children:
                    existing.parent.children.remove(existing)
                existing.parent = v
                existing.cost = new_cost
                existing.f_value = new_cost + existing.heuristic
                v.children.append(existing)
                self._propagate(existing)
            else:
                nn = Node(s.copy(), cost=new_cost)
                nn.parent = v
                nn.heuristic = h_s
                nn.f_value = new_cost + h_s
                v.children.append(nn)
                self.vertices.append(nn)
                vert_dict[s_key] = nn

                if is_goal:
                    self.goal_node = nn
                    nn.heuristic = 0.0
                    nn.f_value = new_cost
                    self.c_best = new_cost

            processed.add(s_key)
            if self.goal_node is not None:
                self.c_best = self.goal_node.cost

    def _propagate(self, node):
        stack = [node]
        while stack:
            n = stack.pop()
            for ch in n.children:
                ec = riemannian_edge_cost(n.x, ch.x, self.metric)
                ch.cost = n.cost + ec
                ch.f_value = ch.cost + ch.heuristic
                stack.append(ch)

    def _prune(self):
        if self.c_best == np.inf:
            return
        thresh = self.prune_thresh * self.c_best
        kept = []
        for v in self.vertices:
            if v is self.start_node or v is self.goal_node:
                kept.append(v)
            elif v.f_value <= thresh:
                kept.append(v)
            else:
                if v.parent and v in v.parent.children:
                    v.parent.children.remove(v)
        self.vertices = kept

    def _extract_path(self):
        if self.goal_node is None:
            return []
        path = []
        n = self.goal_node
        while n is not None:
            path.append(n.x.copy())
            n = n.parent
        path.reverse()
        return path


# ═══════════════════════════════════════════════════════════════════════
# 5. APT*  — Adaptive Prolate Trees (RA-L '25)
# ═══════════════════════════════════════════════════════════════════════

def _common_init_apt(self, x_start, x_goal, c_space_bounds, collision_checker,
                     metric, batch_size, max_iterations,
                     connection_radius_factor, prune_threshold, random_seed):
    """Initialise fields for APT*."""
    self.x_start = np.asarray(x_start, dtype=float)
    self.x_goal = np.asarray(x_goal, dtype=float)
    self.dim = len(self.x_start)
    self.bounds = [(float(lo), float(hi)) for lo, hi in c_space_bounds]
    self.collision_free = collision_checker
    self.metric = metric
    self.batch_size = batch_size
    self.max_iterations = max_iterations
    self.r_factor = connection_radius_factor
    self.prune_thresh = prune_threshold
    self.rng = np.random.default_rng(random_seed)

    self.start_node = Node(self.x_start, cost=0.0)
    self.start_node.heuristic = float(np.linalg.norm(self.x_start - self.x_goal))
    self.start_node.f_value = self.start_node.heuristic
    self.goal_node = None
    self.vertices = [self.start_node]
    self.c_best = np.inf

    from scipy.special import gamma as gamma_fn
    self._zeta_d = (np.pi ** (self.dim / 2.0)) / gamma_fn(self.dim / 2.0 + 1.0)

    self._stats = []
    self._t0 = 0.0


def _compute_r_apt(self, n_vertices, vol=None):
    n = max(n_vertices, 2)
    if vol is None:
        vol = float(np.prod([hi - lo for lo, hi in self.bounds]))
    vol = max(vol, 1e-12)
    r = self.r_factor * ((np.log(n) / n) ** (1.0 / self.dim)) * \
        ((vol / self._zeta_d) ** (1.0 / self.dim))
    diag = np.sqrt(sum((hi - lo) ** 2 for lo, hi in self.bounds))
    return min(r, diag * 0.5)


def _extend_tree_apt(self, samples):
    """Shared extend-and-rewire logic (Informed RRT* style)."""
    n = len(self.vertices)
    kd = KDTree(np.array([v.x for v in self.vertices]))
    vol = self._eis.volume() if (self._eis is not None and self.c_best < np.inf) else None
    r = _compute_r_apt(self, len(self.vertices) + len(samples), vol)

    for s in samples:
        if not self.collision_free(s):
            continue
        idxs = kd.query_ball_point(s, r)
        if not idxs:
            _, idx = kd.query(s)
            idxs = [idx]

        best_parent, best_cost = None, np.inf
        for idx in idxs:
            v = self.vertices[idx]
            if v.cost + float(np.linalg.norm(v.x - s)) >= best_cost:
                continue
            ec = riemannian_edge_cost(v.x, s, self.metric)
            nc = v.cost + ec
            if nc < best_cost:
                if check_edge_collision(v.x, s, self.collision_free, n_checks=20):
                    best_cost = nc
                    best_parent = v
        if best_parent is None:
            continue

        is_goal = np.allclose(s, self.x_goal, atol=1e-8)
        if is_goal and self.goal_node is not None:
            if best_cost < self.goal_node.cost:
                if self.goal_node.parent is not None:
                    p = self.goal_node.parent
                    if self.goal_node in p.children:
                        p.children.remove(self.goal_node)
                self.goal_node.parent = best_parent
                self.goal_node.cost = best_cost
                self.goal_node.f_value = best_cost
                best_parent.children.append(self.goal_node)
                self.c_best = best_cost
            continue

        nn = Node(s.copy(), cost=best_cost)
        nn.parent = best_parent
        nn.heuristic = float(np.linalg.norm(s - self.x_goal))
        nn.f_value = best_cost + nn.heuristic
        best_parent.children.append(nn)
        self.vertices.append(nn)

        if is_goal:
            self.goal_node = nn
            nn.heuristic = 0.0
            nn.f_value = best_cost
            self.c_best = best_cost

        for idx in idxs:
            v = self.vertices[idx]
            if v is best_parent or v is self.start_node:
                continue
            if nn.cost + float(np.linalg.norm(nn.x - v.x)) >= v.cost:
                continue
            ec = riemannian_edge_cost(nn.x, v.x, self.metric)
            nc = nn.cost + ec
            if nc < v.cost:
                if check_edge_collision(nn.x, v.x, self.collision_free, n_checks=20):
                    if v.parent is not None and v in v.parent.children:
                        v.parent.children.remove(v)
                    v.parent = nn
                    v.cost = nc
                    v.f_value = nc + v.heuristic
                    nn.children.append(v)
                    _propagate_apt(self, v)

        if self.goal_node is not None:
            self.c_best = self.goal_node.cost


def _propagate_apt(self, node):
    stack = [node]
    while stack:
        n = stack.pop()
        for ch in n.children:
            ec = riemannian_edge_cost(n.x, ch.x, self.metric)
            ch.cost = n.cost + ec
            ch.f_value = ch.cost + ch.heuristic
            stack.append(ch)


def _prune_apt(self):
    if self.c_best == np.inf:
        return
    thresh = self.prune_thresh * self.c_best
    kept = []
    for v in self.vertices:
        if v is self.start_node or v is self.goal_node:
            kept.append(v)
        elif v.f_value <= thresh:
            kept.append(v)
        else:
            if v.parent is not None and v in v.parent.children:
                v.parent.children.remove(v)
    self.vertices = kept


def _extract_path_apt(self):
    if self.goal_node is None:
        return []
    path = []
    n = self.goal_node
    while n is not None:
        path.append(n.x.copy())
        n = n.parent
    path.reverse()
    return path


class APTStar:
    """Adaptive Prolate Trees (APT*).

    Key idea: the prolate hyperellipsoid's minor axes are *adaptively
    scaled* using a time-dependent prolation factor

        α(t) = 1 + (α₀ − 1) · exp(−λ · t / T)

    where t is the current iteration, T the total budget, α₀ the
    initial overestimate, and λ the decay rate.  Early iterations use
    a wider ellipsoid (exploration); later iterations shrink toward
    the true prolate set (exploitation).

    Edge cost uses the true Riemannian metric for a fair comparison.
    """

    def __init__(self,
                 x_start, x_goal, c_space_bounds, collision_checker,
                 metric: RiemannianMetric,
                 batch_size=100, max_iterations=200,
                 connection_radius_factor=1.1, prune_threshold=1.05,
                 random_seed=0,
                 alpha0: float = 2.0,
                 decay_rate: float = 3.0):
        _common_init_apt(self, x_start, x_goal, c_space_bounds,
                         collision_checker, metric, batch_size, max_iterations,
                         connection_radius_factor, prune_threshold, random_seed)
        self._eis: Optional[EuclideanInformedSet] = None
        self._alpha0 = alpha0
        self._decay = decay_rate

    def plan(self) -> Tuple[List[np.ndarray], float]:
        self._t0 = time.time()
        for it in range(self.max_iterations):
            samples = self._sample_batch(it)
            _extend_tree_apt(self, samples)
            if self.c_best < np.inf:
                _prune_apt(self)
                self._update_eis()
            elapsed = time.time() - self._t0
            self._record_stats(it, elapsed)
        return _extract_path_apt(self), self.c_best

    def get_stats(self):
        return self._stats

    def _prolation_factor(self, iteration):
        t_frac = iteration / max(self.max_iterations - 1, 1)
        return 1.0 + (self._alpha0 - 1.0) * np.exp(-self._decay * t_frac)

    def _sample_batch(self, iteration):
        n = self.batch_size - 1
        if self.c_best < np.inf and self._eis is not None:
            alpha = self._prolation_factor(iteration)
            scaled_eis = EuclideanInformedSet(
                self.x_start, self.x_goal,
                self._eis.c_best * alpha,
                bounds=self.bounds)
            pts = scaled_eis.sample(n, rng=self.rng)
        else:
            lo = np.array([b[0] for b in self.bounds])
            hi = np.array([b[1] for b in self.bounds])
            pts = self.rng.uniform(lo, hi, size=(n, self.dim))
        return np.vstack([pts, self.x_goal.reshape(1, -1)])

    def _update_eis(self):
        if self.c_best < np.inf:
            self._eis = EuclideanInformedSet(
                self.x_start, self.x_goal, self.c_best, bounds=self.bounds)

    def _record_stats(self, iteration, elapsed):
        euclid_vol = 0.0
        if self._eis is not None and self.c_best < np.inf:
            euclid_vol = self._eis.volume()
        alpha = self._prolation_factor(iteration)
        self._stats.append({
            'iteration': iteration,
            'n_vertices': len(self.vertices),
            'c_best': self.c_best,
            'informed_set_volume': euclid_vol * alpha ** (self.dim - 1),
            'euclidean_set_volume': euclid_vol,
            'volume_ratio': alpha ** (self.dim - 1) if euclid_vol > 0 else 1.0,
            'acceptance_rate': 1.0 / max(alpha, 1.0),
            'time_elapsed': elapsed,
            'n_samples_total': (iteration + 1) * self.batch_size,
        })
