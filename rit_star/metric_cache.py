"""
metric_cache.py — Metric Tensor Field Cache (MTF-Cache).

Pre-computes the metric tensor G(x) on a regular grid at
initialization and provides O(1) lookups via multilinear
interpolation during planning.

Key contributions (Section III-B of the MAIT* paper):
  - Amortizes expensive metric evaluations (e.g. sum-of-Gaussians
    in obstacle-inflated metrics) across the entire planning session.
  - Grid resolution is adaptive: fine enough for O(h²) interpolation
    error, coarse enough for negligible memory/init cost.
  - Provides multi-level edge cost evaluation:
      Level 1: midpoint estimate      (1 cached lookup)
      Level 2: Simpson's rule         (3 cached lookups, O(h⁴))
      Level 3: Gauss-Legendre 10-pt   (10 cached lookups, exact)

For spatially-constant metrics (Euclidean, DiagonalAnisotropic),
the cache degenerates to a single stored matrix with zero overhead.
"""

from __future__ import annotations

import numpy as np
from numpy.polynomial.legendre import leggauss
from .metric import (
    RiemannianMetric,
    EuclideanMetric,
    DiagonalAnisotropicMetric,
)


# Pre-compute 10-point Gauss-Legendre nodes once at module load
_GL_NODES_10, _GL_WEIGHTS_10 = leggauss(10)
_GL_TS_10 = 0.5 * (_GL_NODES_10 + 1.0)
_GL_WS_10 = 0.5 * _GL_WEIGHTS_10


class MetricFieldCache:
    """Cached metric tensor field with fast interpolated lookups.

    Parameters
    ----------
    metric : RiemannianMetric
        The underlying metric whose G(x) is expensive to evaluate.
    bounds : list of (lo, hi) per dimension
        Workspace bounds for the grid.
    resolution : int
        Grid points per axis (default 32).  Memory: res^d × d² floats.
    """

    __slots__ = (
        'metric', 'dim', '_lo', '_hi', '_range', '_res',
        '_is_diagonal', '_is_euclidean', '_is_constant',
        '_w', '_sqrt_w', '_G0', '_I',
        '_G_grid', '_scale_grid', '_inv_step',
        '_is_conformal',
        '_s_min', '_lambda_min',
        '_riemannian_vol_cache',
    )

    def __init__(self, metric: RiemannianMetric, bounds: list,
                 resolution: int = 32):
        self.metric = metric
        self.dim = len(bounds)
        self._lo = np.array([b[0] for b in bounds], dtype=float)
        self._hi = np.array([b[1] for b in bounds], dtype=float)
        self._range = self._hi - self._lo
        self._res = resolution

        self._is_diagonal = isinstance(metric, DiagonalAnisotropicMetric)
        self._is_euclidean = isinstance(metric, EuclideanMetric)
        self._is_constant = self._is_diagonal or self._is_euclidean
        self._is_conformal = isinstance(metric, ObstacleInflatedMetric)
        self._riemannian_vol_cache = None

        # Constant-metric fast paths
        self._s_min = 1.0
        self._lambda_min = 1.0
        if self._is_diagonal:
            self._w = np.asarray(metric._weights, dtype=float)
            self._sqrt_w = np.sqrt(self._w)
            self._G0 = np.diag(self._w)
            self._I = None
            self._G_grid = None
            self._scale_grid = None
            self._lambda_min = float(np.min(self._w))
        elif self._is_euclidean:
            self._w = None
            self._sqrt_w = None
            self._G0 = np.eye(self.dim)
            self._I = self._G0
            self._G_grid = None
            self._scale_grid = None
        else:
            self._w = None
            self._sqrt_w = None
            self._G0 = None
            self._I = np.eye(self.dim)
            self._build_grid()

    # ── grid construction ─────────────────────────────────────────

    def _build_grid(self):
        """Pre-compute metric field on regular grid (vectorized)."""
        res = self._res
        dim = self.dim
        axes = [np.linspace(self._lo[d], self._hi[d], res)
                for d in range(dim)]

        if self._is_conformal:
            # Optimisation: for G = s(x)·I we only store the scalar
            grid_shape = tuple([res] * dim)
            # Vectorized: build all grid points then batch-evaluate
            grids = np.meshgrid(*axes, indexing='ij')
            pts = np.column_stack([g.ravel() for g in grids])  # (res^d, d)
            metric = self.metric
            if hasattr(metric, '_scale_batch'):
                scales = metric._scale_batch(pts)
            else:
                scales = np.array([metric._scale(p) for p in pts])
            scale_grid = scales.reshape(grid_shape)
            self._scale_grid = scale_grid
            self._G_grid = None
            # Track minimum scale for admissible heuristic
            self._s_min = float(np.min(scales))
            self._lambda_min = self._s_min
        else:
            # General: store full d×d matrices
            grid_shape = tuple([res] * dim + [dim, dim])
            grids = np.meshgrid(*axes, indexing='ij')
            pts = np.column_stack([g.ravel() for g in grids])
            n_pts = pts.shape[0]
            G_flat = np.empty((n_pts, dim, dim))
            metric = self.metric
            for idx in range(n_pts):
                G_flat[idx] = metric.G(pts[idx])
            G_grid = G_flat.reshape(grid_shape)
            self._G_grid = G_grid
            self._scale_grid = None
            # Track minimum eigenvalue for admissible heuristic
            all_eigs = np.linalg.eigvalsh(G_flat)  # (n_pts, dim)
            self._lambda_min = float(np.min(all_eigs))

        self._inv_step = np.where(
            self._range > 0, (res - 1) / self._range, 0.0)

    # ── interpolated lookup ───────────────────────────────────────

    def G(self, x: np.ndarray) -> np.ndarray:
        """Return interpolated metric tensor G(x).  O(1)."""
        if self._is_diagonal:
            return self._G0
        if self._is_euclidean:
            return self._G0
        if self._is_conformal:
            return self._interp_scale(x) * self._I
        return self._interp_G(x)

    def scale(self, x: np.ndarray) -> float:
        """For conformal metrics, return the scalar factor s(x)."""
        if self._is_conformal:
            return self._interp_scale(x)
        return 1.0

    def _grid_coords(self, x):
        gc = (x - self._lo) * self._inv_step
        # Inline clip+floor for 2D/3D to avoid numpy overhead
        res_max = self._res - 2
        upper = self._res - 1.0 - 1e-10
        if self.dim == 2:
            g0 = gc[0]; g1 = gc[1]
            if g0 < 0.0: g0 = 0.0
            elif g0 > upper: g0 = upper
            if g1 < 0.0: g1 = 0.0
            elif g1 > upper: g1 = upper
            i0 = int(g0); i1 = int(g1)
            if i0 > res_max: i0 = res_max
            if i1 > res_max: i1 = res_max
            return (i0, i1), (g0 - i0, g1 - i1)
        elif self.dim == 3:
            g0 = gc[0]; g1 = gc[1]; g2 = gc[2]
            if g0 < 0.0: g0 = 0.0
            elif g0 > upper: g0 = upper
            if g1 < 0.0: g1 = 0.0
            elif g1 > upper: g1 = upper
            if g2 < 0.0: g2 = 0.0
            elif g2 > upper: g2 = upper
            i0 = int(g0); i1 = int(g1); i2 = int(g2)
            if i0 > res_max: i0 = res_max
            if i1 > res_max: i1 = res_max
            if i2 > res_max: i2 = res_max
            return (i0, i1, i2), (g0 - i0, g1 - i1, g2 - i2)
        gc = np.clip(gc, 0.0, upper)
        ix = np.minimum(gc.astype(int), res_max)
        return ix, gc - ix

    def _interp_scale(self, x) -> float:
        """Interpolate scalar conformal factor."""
        ix, frac = self._grid_coords(x)
        S = self._scale_grid
        if self.dim == 2:
            i, j = ix[0], ix[1]
            fx, fy = frac[0], frac[1]
            return float(
                (1-fx)*(1-fy) * S[i, j]   + fx*(1-fy) * S[i+1, j] +
                (1-fx)*fy     * S[i, j+1] + fx*fy     * S[i+1, j+1])
        elif self.dim == 3:
            i, j, k = ix[0], ix[1], ix[2]
            fx, fy, fz = frac[0], frac[1], frac[2]
            c00 = (1-fx)*S[i,j,k]     + fx*S[i+1,j,k]
            c01 = (1-fx)*S[i,j,k+1]   + fx*S[i+1,j,k+1]
            c10 = (1-fx)*S[i,j+1,k]   + fx*S[i+1,j+1,k]
            c11 = (1-fx)*S[i,j+1,k+1] + fx*S[i+1,j+1,k+1]
            c0 = (1-fy)*c00 + fy*c10
            c1 = (1-fy)*c01 + fy*c11
            return float((1-fz)*c0 + fz*c1)
        # fallback
        return float(self.metric._scale(x))

    def _interp_G(self, x) -> np.ndarray:
        """Interpolate full G matrix."""
        ix, frac = self._grid_coords(x)
        G = self._G_grid
        if self.dim == 2:
            i, j = ix[0], ix[1]
            fx, fy = frac[0], frac[1]
            return ((1-fx)*(1-fy) * G[i,j]   + fx*(1-fy) * G[i+1,j] +
                    (1-fx)*fy     * G[i,j+1] + fx*fy     * G[i+1,j+1])
        elif self.dim == 3:
            i, j, k = ix[0], ix[1], ix[2]
            fx, fy, fz = frac[0], frac[1], frac[2]
            c00 = (1-fx)*G[i,j,k]     + fx*G[i+1,j,k]
            c01 = (1-fx)*G[i,j,k+1]   + fx*G[i+1,j,k+1]
            c10 = (1-fx)*G[i,j+1,k]   + fx*G[i+1,j+1,k]
            c11 = (1-fx)*G[i,j+1,k+1] + fx*G[i+1,j+1,k+1]
            c0 = (1-fy)*c00 + fy*c10
            c1 = (1-fy)*c01 + fy*c11
            return (1-fz)*c0 + fz*c1
        # fallback
        return self.metric.G(x)

    # ── multi-level edge cost ─────────────────────────────────────

    def edge_cost_l1(self, x: np.ndarray, y: np.ndarray) -> float:
        """Level 1: midpoint estimate (1 cached G lookup).

        Used for candidate ranking and fast filtering.
        """
        diff = y - x
        dd = float(diff @ diff)
        if self._is_euclidean:
            return float(np.sqrt(dd))
        if self._is_diagonal:
            return float(np.sqrt(diff @ (self._w * diff)))
        if self._is_conformal:
            mid = 0.5 * (x + y)
            s = self._interp_scale(mid)
            return float(np.sqrt(s * dd))
        Gm = self._interp_G(0.5 * (x + y))
        return float(np.sqrt(max(float(diff @ Gm @ diff), 0.0)))

    def edge_cost_l2(self, x: np.ndarray, y: np.ndarray) -> float:
        """Level 2: Simpson's rule (3 cached G lookups), O(h⁴) accurate.

        Used as a filter in the cascading edge evaluation.  Edges that
        pass L2 filtering proceed to L3 for exact cost + collision check.
        """
        diff = y - x
        if self._is_euclidean:
            return float(np.sqrt(diff @ diff))
        if self._is_diagonal:
            return float(np.sqrt(diff @ (self._w * diff)))
        if self._is_conformal:
            dd = float(diff @ diff)
            s0 = self._interp_scale(x)
            sm = self._interp_scale(0.5 * (x + y))
            s1 = self._interp_scale(y)
            f0 = np.sqrt(s0 * dd)
            fm = np.sqrt(sm * dd)
            f1 = np.sqrt(s1 * dd)
            return float((f0 + 4.0*fm + f1) / 6.0)
        G0 = self.G(x)
        Gm = self.G(0.5 * (x + y))
        G1 = self.G(y)
        f0 = np.sqrt(max(float(diff @ G0 @ diff), 0.0))
        fm = np.sqrt(max(float(diff @ Gm @ diff), 0.0))
        f1 = np.sqrt(max(float(diff @ G1 @ diff), 0.0))
        return float((f0 + 4.0*fm + f1) / 6.0)

    def edge_cost_exact(self, x: np.ndarray, y: np.ndarray) -> float:
        """Level 3: 10-point Gauss-Legendre (cached G lookups).

        Used for actual tree edge costs and final path cost reporting.
        Only called for the ~5%% of edges that survive L1 and L2 filtering.
        """
        diff = y - x
        if self._is_euclidean:
            return float(np.sqrt(diff @ diff))
        if self._is_diagonal:
            return float(np.sqrt(diff @ (self._w * diff)))
        if self._is_conformal:
            dd = float(diff @ diff)
            total = 0.0
            for t_i, w_i in zip(_GL_TS_10, _GL_WS_10):
                s = self._interp_scale(x + t_i * diff)
                total += w_i * np.sqrt(s * dd)
            return float(total)
        total = 0.0
        for t_i, w_i in zip(_GL_TS_10, _GL_WS_10):
            Gi = self.G(x + t_i * diff)
            total += w_i * np.sqrt(max(float(diff @ Gi @ diff), 0.0))
        return float(total)

    def edge_cost_l3_with_collision(self, x: np.ndarray, y: np.ndarray,
                                     collision_free, n_checks: int = 20):
        """Level 3: 10-point Gauss-Legendre + collision checking.

        Combines accurate cost computation with collision checking for
        the ~5% of edges that survive L1 and L2 filtering (Section III-B).
        Collision is checked at the 10 quadrature points plus additional
        evenly-spaced points to ensure adequate coverage.

        Parameters
        ----------
        x, y : (d,) arrays
        collision_free : callable  x -> bool  (True = free)
        n_checks : int
            Minimum number of evenly-spaced collision checks (default 20).

        Returns
        -------
        (cost, is_free) : (float, bool)
            cost is the 10-point GL edge cost, is_free indicates collision status.
        """
        diff = y - x
        length = float(np.sqrt(diff @ diff))

        # Check endpoints first (fast reject)
        if length < 1e-12:
            return (0.0, True) if collision_free(x) else (np.inf, False)

        # For constant metrics, cost is trivial — just do collision checks
        if self._is_euclidean or self._is_diagonal:
            n = max(n_checks, min(n_checks * 5, int(np.ceil(length / 0.02))))
            inv_n = 1.0 / n
            for i in range(n + 1):
                if not collision_free(x + (i * inv_n) * diff):
                    return np.inf, False
            if self._is_euclidean:
                return float(np.sqrt(diff @ diff)), True
            return float(np.sqrt(diff @ (self._w * diff))), True

        # Varying metrics: check collision at 10 GL quadrature points
        # while computing cost, then fill in extra collision checks
        total = 0.0
        checked_ts = set()

        if self._is_conformal:
            dd = float(diff @ diff)
            for t_i, w_i in zip(_GL_TS_10, _GL_WS_10):
                pt = x + t_i * diff
                if not collision_free(pt):
                    return np.inf, False
                checked_ts.add(t_i)
                s = self._interp_scale(pt)
                total += w_i * np.sqrt(s * dd)
        else:
            for t_i, w_i in zip(_GL_TS_10, _GL_WS_10):
                pt = x + t_i * diff
                if not collision_free(pt):
                    return np.inf, False
                checked_ts.add(t_i)
                Gi = self.G(pt)
                total += w_i * np.sqrt(max(float(diff @ Gi @ diff), 0.0))

        # Additional evenly-spaced collision checks for coverage
        # (GL points cluster near endpoints, so uniform checks fill gaps)
        n_extra = max(n_checks, min(n_checks * 5, int(np.ceil(length / 0.02))))
        inv_n = 1.0 / n_extra
        for i in range(n_extra + 1):
            t = i * inv_n
            # Skip if close to an already-checked GL point
            skip = False
            for t_gl in checked_ts:
                if abs(t - t_gl) < 0.02:
                    skip = True
                    break
            if skip:
                continue
            if not collision_free(x + t * diff):
                return np.inf, False

        return float(total), True

    def edge_cost_l3_with_collision_feedback(self, x: np.ndarray, y: np.ndarray,
                                              collision_free, n_checks: int = 20):
        """Level 3 with collision feedback for CARM.

        Same as edge_cost_l3_with_collision but returns the first
        collision point for adaptive metric updates.

        Returns
        -------
        (cost, is_free, collision_point) : (float, bool, ndarray or None)
        """
        diff = y - x
        length = float(np.sqrt(diff @ diff))

        if length < 1e-12:
            if not collision_free(x):
                return np.inf, False, x.copy()
            return 0.0, True, None

        if self._is_euclidean or self._is_diagonal:
            n = max(n_checks, min(n_checks * 5, int(np.ceil(length / 0.02))))
            inv_n = 1.0 / n
            for i in range(n + 1):
                pt = x + (i * inv_n) * diff
                if not collision_free(pt):
                    return np.inf, False, pt.copy()
            if self._is_euclidean:
                return float(np.sqrt(diff @ diff)), True, None
            return float(np.sqrt(diff @ (self._w * diff))), True, None

        total = 0.0
        checked_ts = set()

        if self._is_conformal:
            dd = float(diff @ diff)
            for t_i, w_i in zip(_GL_TS_10, _GL_WS_10):
                pt = x + t_i * diff
                if not collision_free(pt):
                    return np.inf, False, pt.copy()
                checked_ts.add(t_i)
                s = self._interp_scale(pt)
                total += w_i * np.sqrt(s * dd)
        else:
            for t_i, w_i in zip(_GL_TS_10, _GL_WS_10):
                pt = x + t_i * diff
                if not collision_free(pt):
                    return np.inf, False, pt.copy()
                checked_ts.add(t_i)
                Gi = self.G(pt)
                total += w_i * np.sqrt(max(float(diff @ Gi @ diff), 0.0))

        n_extra = max(n_checks, min(n_checks * 5, int(np.ceil(length / 0.02))))
        inv_n = 1.0 / n_extra
        for i in range(n_extra + 1):
            t = i * inv_n
            skip = False
            for t_gl in checked_ts:
                if abs(t - t_gl) < 0.02:
                    skip = True
                    break
            if skip:
                continue
            pt = x + t * diff
            if not collision_free(pt):
                return np.inf, False, pt.copy()

        return float(total), True, None

    def heuristic(self, x: np.ndarray, goal: np.ndarray) -> float:
        """Admissible heuristic: lower bound on Riemannian distance to goal.

        For constant metrics (Euclidean, DiagonalAnisotropic) the L1
        midpoint estimate is exact.  For spatially-varying metrics the
        midpoint estimate can overestimate (inadmissible) when the
        straight line to goal passes through high-cost regions while
        the true geodesic goes around them.  We use
        sqrt(lambda_min) * ||x - goal|| as a guaranteed lower bound.
        """
        if self._is_constant:
            return self.edge_cost_l1(x, goal)
        # Admissible lower bound: min metric eigenvalue × Euclidean dist
        diff = goal - x
        return float(np.sqrt(self._lambda_min * (diff @ diff)))

    # ── batch scale interpolation (for whitened sampling) ─────────

    def batch_scale(self, pts: np.ndarray) -> np.ndarray:
        """Interpolate scalar conformal factor for batch of points.

        Parameters
        ----------
        pts : (N, d) array

        Returns
        -------
        (N,) array of scale factors (1.0 for non-conformal metrics)
        """
        if self._is_constant:
            return np.ones(len(pts))
        if not self._is_conformal:
            return np.ones(len(pts))
        # Vectorized grid interpolation for batch of points
        S = self._scale_grid
        gc = (pts - self._lo) * self._inv_step  # (N, d)
        upper = self._res - 1.0 - 1e-10
        gc = np.clip(gc, 0.0, upper)
        ix = np.minimum(gc.astype(int), self._res - 2)
        frac = gc - ix
        if self.dim == 2:
            i0, i1 = ix[:, 0], ix[:, 1]
            f0, f1 = frac[:, 0], frac[:, 1]
            return ((1-f0)*(1-f1)*S[i0, i1] + f0*(1-f1)*S[i0+1, i1] +
                    (1-f0)*f1*S[i0, i1+1] + f0*f1*S[i0+1, i1+1])
        elif self.dim == 3:
            i0, i1, i2 = ix[:, 0], ix[:, 1], ix[:, 2]
            f0, f1, f2 = frac[:, 0], frac[:, 1], frac[:, 2]
            c00 = (1-f0)*S[i0, i1, i2]     + f0*S[i0+1, i1, i2]
            c01 = (1-f0)*S[i0, i1, i2+1]   + f0*S[i0+1, i1, i2+1]
            c10 = (1-f0)*S[i0, i1+1, i2]   + f0*S[i0+1, i1+1, i2]
            c11 = (1-f0)*S[i0, i1+1, i2+1] + f0*S[i0+1, i1+1, i2+1]
            c0 = (1-f1)*c00 + f1*c10
            c1 = (1-f1)*c01 + f1*c11
            return (1-f2)*c0 + f2*c1
        return np.array([self._interp_scale(p) for p in pts])

    def avg_metric_along(self, x_start: np.ndarray, x_goal: np.ndarray,
                         n: int = 5) -> np.ndarray:
        """Average G along the straight line from start to goal.

        Used by WhitenedInformedSampling to compute the whitening
        transform.

        Parameters
        ----------
        n : int
            Number of sample points along the line.

        Returns
        -------
        (d, d) averaged metric tensor.
        """
        if self._is_diagonal:
            return self._G0.copy()
        if self._is_euclidean:
            return self._G0.copy()
        G_avg = np.zeros((self.dim, self.dim))
        for i in range(n):
            t = i / max(n - 1, 1)
            pt = x_start + t * (x_goal - x_start)
            G_avg += self.G(pt)
        return G_avg / n

    # ── Theorem 2: Riemannian volume integration ─────────────────

    def riemannian_volume(self) -> float:
        """Integrate sqrt(det(G(x))) over the workspace grid.

        Returns the total Riemannian volume mu_R(workspace).
        Cached after first call.

        For constant metrics: sqrt(det(G)) * prod(hi - lo).
        For conformal: integrates s(x)^(d/2) over the grid.
        For general: integrates sqrt(det(G(x))) over the grid.
        """
        if self._riemannian_vol_cache is not None:
            return self._riemannian_vol_cache

        workspace_vol = float(np.prod(self._range))

        if self._is_euclidean:
            self._riemannian_vol_cache = workspace_vol
            return workspace_vol

        if self._is_diagonal:
            sqrt_det = float(np.sqrt(np.prod(self._w)))
            self._riemannian_vol_cache = sqrt_det * workspace_vol
            return self._riemannian_vol_cache

        if self._is_conformal and self._scale_grid is not None:
            # Integrate s(x)^(d/2) over the grid using trapezoidal rule
            d = self.dim
            cell_vol = workspace_vol / max(self._scale_grid.size, 1)
            sqrt_det_vals = self._scale_grid ** (d / 2.0)
            self._riemannian_vol_cache = float(np.sum(sqrt_det_vals) * cell_vol)
            return self._riemannian_vol_cache

        if self._G_grid is not None:
            # General: compute det of each stored G matrix
            shape = self._G_grid.shape
            n_pts = 1
            for s in shape[:-2]:
                n_pts *= s
            G_flat = self._G_grid.reshape(n_pts, self.dim, self.dim)
            dets = np.linalg.det(G_flat)
            sqrt_dets = np.sqrt(np.maximum(dets, 0.0))
            cell_vol = workspace_vol / max(n_pts, 1)
            self._riemannian_vol_cache = float(np.sum(sqrt_dets) * cell_vol)
            return self._riemannian_vol_cache

        # Fallback for metrics without a grid
        self._riemannian_vol_cache = workspace_vol
        return workspace_vol

    def min_eigenvalue(self) -> float:
        """Return the global minimum eigenvalue lambda_min across the grid.

        Used by Theorem 2 for converting Riemannian radius to Euclidean
        radius in KD-tree queries.
        """
        return self._lambda_min


# Need the import here to avoid circular dependency issues
from .metric import ObstacleInflatedMetric  # noqa: E402
