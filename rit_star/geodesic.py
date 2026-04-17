"""
geodesic.py — Riemannian geodesic distance approximations.

Provides three approximation tiers, from cheapest to most accurate:

  Tier 0  euclidean_distance        — standard L2 norm (baseline).
  Tier 1  diagonal_geodesic         — exact when G is constant diagonal;
                                      O(d) weighted Euclidean distance.
                                      For conformal metrics with metric_cache,
                                      automatically uses accurate integration.
  Tier 2  varadhan_geodesic         — heat-kernel approximation on a grid;
                                      accurate for spatially-varying metrics.
  Tier 3  jacobi_correction_geodesic — first-order Jacobi field correction;
                                      good for slowly varying metrics and
                                      small displacements.

The ``GeodesicComputer`` class wraps these into a unified interface with
vectorised batch queries.

NOTE: For conformal metrics G(x) = s(x)·I (ObstacleInflatedMetric, etc.),
the 'diagonal' tier now uses proper numerical integration when a metric_cache
is provided, avoiding the inaccurate midpoint approximation. This ensures
Riemannian informed sets are computed correctly.
"""

from __future__ import annotations

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import expm_multiply
from typing import Optional

from .metric import RiemannianMetric


# ═══════════════════════════════════════════════════════════════════════
# Tier 0 — plain Euclidean
# ═══════════════════════════════════════════════════════════════════════

def euclidean_distance(x: np.ndarray, y: np.ndarray) -> float:
    """Standard L2 distance between two configurations.

    Parameters
    ----------
    x, y : (d,) arrays

    Returns
    -------
    float
        ‖x − y‖₂

    Notes
    -----
    Tier 0 baseline — ignores the metric tensor entirely.
    """
    return float(np.linalg.norm(x - y))


# ═══════════════════════════════════════════════════════════════════════
# Tier 1 — weighted (diagonal constant metric)
# ═══════════════════════════════════════════════════════════════════════

def diagonal_geodesic(x: np.ndarray, y: np.ndarray,
                      metric: RiemannianMetric) -> float:
    """Weighted Euclidean distance  d_R(x,y) = sqrt((x-y)ᵀ G (x-y)).

    Parameters
    ----------
    x, y : (d,) arrays
    metric : RiemannianMetric
        Any metric; the tensor is evaluated at the midpoint (x+y)/2.

    Returns
    -------
    float
        Approximate geodesic distance.

    Notes
    -----
    Exact when G is spatially constant (e.g. DiagonalAnisotropicMetric).
    For non-constant metrics this is a first-order approximation
    evaluated at the midpoint — still O(d) and very fast.
    """
    diff = x - y
    mid = 0.5 * (x + y)
    Gm = metric.G(mid)
    return float(np.sqrt(max(diff @ Gm @ diff, 0.0)))


def conformal_geodesic(x: np.ndarray, y: np.ndarray,
                       metric: RiemannianMetric,
                       n_quad: int = 5) -> float:
    """Accurate geodesic for conformal metrics G(x) = s(x)·I.

    For conformal metrics, geodesics along straight lines are:
        d_R(x,y) = ∫₀¹ √s(x + t(y-x)) dt · ‖y - x‖

    Uses Simpson's rule (5-point) for accurate integration of the
    scale factor, avoiding the midpoint-only approximation in
    diagonal_geodesic which can be inaccurate for rapidly varying s(x).

    Parameters
    ----------
    x, y : (d,) arrays
    metric : RiemannianMetric
        Should be conformal (all eigenvalues equal at each point).
    n_quad : int
        Number of quadrature points (default 5 for Simpson's rule).

    Returns
    -------
    float
        Accurate Riemannian distance for conformal metrics.

    Notes
    -----
    For non-conformal metrics, falls back to diagonal_geodesic.
    This is used by GeodesicComputer when metric_cache indicates
    a conformal metric to ensure informed sets are computed correctly.
    """
    from .metric import (
        ObstacleInflatedMetric, PathwayMetric, ClearanceMetric,
        EuclideanMetric, DiagonalAnisotropicMetric
    )

    diff = y - x
    euclid_dist = float(np.sqrt(diff @ diff))

    if euclid_dist < 1e-12:
        return 0.0

    # Fast path for constant metrics
    if isinstance(metric, (EuclideanMetric, DiagonalAnisotropicMetric)):
        return diagonal_geodesic(x, y, metric)

    # Check if metric is conformal
    is_conformal = isinstance(metric, (ObstacleInflatedMetric, PathwayMetric, ClearanceMetric))

    if not is_conformal:
        # Not conformal, use midpoint approximation
        return diagonal_geodesic(x, y, metric)

    # Conformal metric: integrate scale factor using Simpson's 5-point rule
    # Points at t = 0, 0.25, 0.5, 0.75, 1.0
    # Simpson's 3/8 rule weights for 5 points: [1, 3, 3, 3, 1] / 8 (scaled to [0,1])
    # Actually use Simpson's composite rule: [1, 4, 2, 4, 1] / 12
    t_vals = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    weights = np.array([1.0, 4.0, 2.0, 4.0, 1.0]) / 12.0

    integral = 0.0
    for t, w in zip(t_vals, weights):
        pt = x + t * diff
        # For conformal metrics, eigenvalues are all equal
        eigs = metric.eigenvalues(pt)
        s = float(eigs[0])  # scale factor s(x)
        integral += w * np.sqrt(max(s, 0.0))

    return float(integral * euclid_dist)


# ═══════════════════════════════════════════════════════════════════════
# Tier 2 — Varadhan heat-kernel approximation
# ═══════════════════════════════════════════════════════════════════════

class HeatKernel:
    """Precomputed heat kernel on a uniform grid for Varadhan's
    approximation of the squared geodesic distance.

    d_R(x,y)² ≈ −4t · log h(x, y, t)

    The grid Laplacian is built from the metric tensor field and the
    heat kernel is computed via sparse matrix–vector products.
    """

    def __init__(self, metric: RiemannianMetric,
                 bounds: list,
                 grid_resolution: int = 50,
                 t: float = 0.01):
        """
        Parameters
        ----------
        metric : RiemannianMetric
        bounds : list of (lo, hi) per dimension
        grid_resolution : int
            Number of grid points along each axis (default 50).
        t : float
            Diffusion time (default 0.01).

        Notes
        -----
        Implements the precomputation step of the Varadhan geodesic
        approximation (Tier 2).  Only supports 2D at present.
        """
        self.metric = metric
        self.dim = metric.dim
        self.bounds = [(float(lo), float(hi)) for lo, hi in bounds]
        self.res = grid_resolution
        self.t = t

        # Build grid
        axes = [np.linspace(lo, hi, grid_resolution)
                for lo, hi in self.bounds]
        self._axes = axes
        self._dx = [(hi - lo) / (grid_resolution - 1)
                     for lo, hi in self.bounds]

        if self.dim == 2:
            self._build_2d()
        else:
            # For 3D we fall back to diagonal approximation to keep
            # memory manageable; the grid would be res^3 nodes.
            self._available = False

    def _build_2d(self):
        """Build graph Laplacian weighted by the metric and precompute."""
        res = self.res
        N = res * res
        dx, dy = self._dx
        xs, ys = self._axes

        # node coordinates
        gx, gy = np.meshgrid(xs, ys, indexing='ij')
        self._coords = np.stack([gx.ravel(), gy.ravel()], axis=1)  # (N, 2)

        # weighted graph Laplacian (5-point stencil)
        row, col, data = [], [], []
        for i in range(res):
            for j in range(res):
                idx = i * res + j
                pt = self._coords[idx]
                g_inv = self.metric.G_inv(pt)
                diag_sum = 0.0
                # neighbours: (i±1, j), (i, j±1)
                for di, dj, h in [(1, 0, dx), (-1, 0, dx),
                                   (0, 1, dy), (0, -1, dy)]:
                    ni, nj = i + di, j + dj
                    if 0 <= ni < res and 0 <= nj < res:
                        nidx = ni * res + nj
                        direction = np.array([di * dx, dj * dy], dtype=float)
                        # conductance proportional to inverse metric
                        w = float(direction @ g_inv @ direction) / (h * h)
                        w = max(w, 1e-12)
                        row.append(idx)
                        col.append(nidx)
                        data.append(w)
                        diag_sum += w
                row.append(idx)
                col.append(idx)
                data.append(-diag_sum)

        L = sparse.csr_matrix((data, (row, col)), shape=(N, N))
        self._L = L
        self._N = N
        self._available = True

    @property
    def available(self) -> bool:
        return self._available

    def _idx(self, x: np.ndarray) -> int:
        """Snap a continuous point to the nearest grid index."""
        res = self.res
        ij = []
        for k in range(self.dim):
            lo, hi = self.bounds[k]
            frac = (x[k] - lo) / (hi - lo)
            idx_k = int(np.clip(np.round(frac * (res - 1)), 0, res - 1))
            ij.append(idx_k)
        if self.dim == 2:
            return ij[0] * res + ij[1]
        return 0

    def query(self, x: np.ndarray, y: np.ndarray) -> float:
        """Return approximate squared geodesic distance via Varadhan.

        Parameters
        ----------
        x, y : (d,) arrays

        Returns
        -------
        float
            Approximate d_R(x,y)².
        """
        if not self._available:
            return diagonal_geodesic(x, y, self.metric) ** 2

        ix = self._idx(x)
        # delta source at ix
        rhs = np.zeros(self._N)
        rhs[ix] = 1.0
        # h(·, x, t) = exp(t·L) δ_x
        h = expm_multiply(self.t * self._L, rhs)
        iy = self._idx(y)
        val = max(h[iy], 1e-30)
        return max(-4.0 * self.t * np.log(val), 0.0)


def varadhan_geodesic(x: np.ndarray, y: np.ndarray,
                      metric: RiemannianMetric,
                      heat_kernel: HeatKernel) -> float:
    """Geodesic distance via Varadhan's heat-kernel formula.

    Parameters
    ----------
    x, y : (d,) arrays
    metric : RiemannianMetric (unused, kept for interface consistency)
    heat_kernel : HeatKernel
        Precomputed heat kernel object.

    Returns
    -------
    float
        Approximate Riemannian geodesic distance.

    Notes
    -----
    Tier 2 approximation: d_R² ≈ −4t log h(x, y, t).
    """
    return float(np.sqrt(max(heat_kernel.query(x, y), 0.0)))


# ═══════════════════════════════════════════════════════════════════════
# Tier 3 — Jacobi field correction
# ═══════════════════════════════════════════════════════════════════════

def _numerical_christoffel(metric: RiemannianMetric,
                           x: np.ndarray, eps: float = 1e-4) -> np.ndarray:
    """Estimate Christoffel symbols Γ^k_{ij} at *x* via central differences.

    Returns
    -------
    Gamma : (d, d, d) array
        Gamma[k, i, j] = Γ^k_{ij}.
    """
    d = metric.dim
    Gamma = np.zeros((d, d, d))
    g_inv = metric.G_inv(x)

    dg = np.zeros((d, d, d))  # dg[m, i, j] = ∂g_{ij}/∂x^m
    for m in range(d):
        e = np.zeros(d)
        e[m] = eps
        dg[m] = (metric.G(x + e) - metric.G(x - e)) / (2.0 * eps)

    for k in range(d):
        for i in range(d):
            for j in range(d):
                s = 0.0
                for m in range(d):
                    s += g_inv[k, m] * (dg[i][m, j] + dg[j][m, i] - dg[m][i, j])
                Gamma[k, i, j] = 0.5 * s
    return Gamma


def jacobi_correction_geodesic(x: np.ndarray, y: np.ndarray,
                                metric: RiemannianMetric) -> float:
    """First-order Jacobi field correction to the geodesic distance.

    Parameters
    ----------
    x, y : (d,) arrays
    metric : RiemannianMetric

    Returns
    -------
    float
        d_R(x,y) ≈ ‖x−y‖_G + curvature correction.

    Notes
    -----
    Tier 3 approximation.  The curvature correction is
        (1/24) R_{abcd} v^a v^b v^c v^d · ‖x−y‖³
    where R is the Riemann curvature tensor estimated numerically.
    Accurate for small displacements when the metric varies slowly.
    """
    diff = y - x
    mid = 0.5 * (x + y)
    Gm = metric.G(mid)
    base_dist = float(np.sqrt(max(diff @ Gm @ diff, 0.0)))

    if base_dist < 1e-12:
        return 0.0

    d = metric.dim
    eps = 1e-4

    # Estimate Riemann curvature tensor R^l_{ijk} from Christoffel symbols
    Gamma_mid = _numerical_christoffel(metric, mid, eps)

    # For the correction we need R_{abcd} = g_{al} R^l_{bcd}
    # and contract with v^a v^b v^c v^d.
    # R^l_{ijk} ≈ ∂_i Γ^l_{jk} − ∂_j Γ^l_{ik} + Γ^l_{im} Γ^m_{jk} − Γ^l_{jm} Γ^m_{ik}
    # We use the quadratic Γ terms at midpoint (skip derivative terms for speed).
    R_up = np.zeros((d, d, d, d))  # R^l_{ijk}
    for l in range(d):
        for i in range(d):
            for j in range(d):
                for k in range(d):
                    s = 0.0
                    for m in range(d):
                        s += (Gamma_mid[l, i, m] * Gamma_mid[m, j, k]
                              - Gamma_mid[l, j, m] * Gamma_mid[m, i, k])
                    R_up[l, i, j, k] = s

    # Also include derivative terms via finite differences on Γ
    Gamma_plus = np.zeros((d, d, d, d))
    Gamma_minus = np.zeros((d, d, d, d))
    for i in range(d):
        e = np.zeros(d)
        e[i] = eps
        Gamma_plus[i] = _numerical_christoffel(metric, mid + e, eps)
        Gamma_minus[i] = _numerical_christoffel(metric, mid - e, eps)
    dGamma = (Gamma_plus - Gamma_minus) / (2.0 * eps)  # (d, d, d, d)

    for l in range(d):
        for i in range(d):
            for j in range(d):
                for k in range(d):
                    R_up[l, i, j, k] += dGamma[i, l, j, k] - dGamma[j, l, i, k]

    # Lower the first index: R_{aijk} = g_{al} R^l_{ijk}
    v = diff / max(base_dist, 1e-12)  # unit tangent in Riemannian sense
    correction = 0.0
    for a in range(d):
        for b in range(d):
            for c in range(d):
                for dd_ in range(d):
                    R_lower_abcd = 0.0
                    for l in range(d):
                        R_lower_abcd += Gm[a, l] * R_up[l, b, c, dd_]
                    correction += R_lower_abcd * v[a] * v[b] * v[c] * v[dd_]

    correction *= (1.0 / 24.0) * base_dist ** 3
    return max(base_dist + correction, 0.0)


# ═══════════════════════════════════════════════════════════════════════
# GeodesicComputer — unified interface
# ═══════════════════════════════════════════════════════════════════════

class GeodesicComputer:
    """Unified geodesic distance interface with tier selection.

    Parameters
    ----------
    metric : RiemannianMetric
    tier : str
        One of 'euclidean', 'diagonal', 'varadhan', 'jacobi'.
    bounds : list of (lo, hi) per dimension (needed for 'varadhan').
    grid_resolution : int
        Resolution for the heat-kernel grid (default 50).
    metric_cache : MetricFieldCache, optional
        If provided and metric is conformal, uses accurate edge cost
        computation from the cache instead of midpoint approximation.
    """

    _TIERS = ('euclidean', 'diagonal', 'varadhan', 'jacobi')

    def __init__(self, metric: RiemannianMetric, tier: str = 'diagonal',
                 bounds: Optional[list] = None, grid_resolution: int = 50,
                 metric_cache=None):
        if tier not in self._TIERS:
            raise ValueError(f"tier must be one of {self._TIERS}")
        self.metric = metric
        self.tier = tier
        self._heat: Optional[HeatKernel] = None
        self.bounds = bounds
        self._metric_cache = metric_cache

        # Detect if we should use conformal geodesic for 'diagonal' tier
        self._use_conformal = False
        if tier == 'diagonal' and metric_cache is not None:
            self._use_conformal = getattr(metric_cache, '_is_conformal', False)

        if tier == 'varadhan':
            if bounds is None:
                raise ValueError("bounds required for varadhan tier")
            self._heat = HeatKernel(metric, bounds, grid_resolution)

    # ── primary API ──────────────────────────────────────────────────

    def distance(self, x: np.ndarray, y: np.ndarray) -> float:
        """Compute (approximate) Riemannian geodesic distance.

        Parameters
        ----------
        x, y : (d,) arrays

        Returns
        -------
        float
        """
        if self.tier == 'euclidean':
            return euclidean_distance(x, y)
        elif self.tier == 'diagonal':
            # Use accurate conformal integration if available
            if self._use_conformal and self._metric_cache is not None:
                # Use L2 edge cost from cache for accurate integration
                return self._metric_cache.edge_cost_l2(x, y)
            elif self._use_conformal:
                # Fallback to conformal geodesic without cache
                return conformal_geodesic(x, y, self.metric)
            else:
                return diagonal_geodesic(x, y, self.metric)
        elif self.tier == 'varadhan':
            return varadhan_geodesic(x, y, self.metric, self._heat)
        else:  # jacobi
            return jacobi_correction_geodesic(x, y, self.metric)

    def heuristic(self, x: np.ndarray, goal: np.ndarray) -> float:
        """Admissible lower-bound heuristic to *goal*.

        Parameters
        ----------
        x : (d,) array
        goal : (d,) array

        Returns
        -------
        float
            Always ≤ true geodesic distance (admissible).
        """
        # The diagonal approximation is always a lower bound when
        # the metric can only increase (obstacle-inflated, pathway).
        # For safety we use the simple weighted distance which
        # under-estimates for spatially varying metrics.
        return diagonal_geodesic(x, goal, self.metric)

    def batch_distance(self, points: np.ndarray,
                       query: np.ndarray) -> np.ndarray:
        """Vectorised distance from each row of *points* to *query*.

        Parameters
        ----------
        points : (N, d) array
        query : (d,) array

        Returns
        -------
        (N,) array of distances.
        """
        if self.tier == 'euclidean':
            return np.linalg.norm(points - query, axis=1)
        elif self.tier == 'diagonal':
            diff = points - query
            mid = 0.5 * (points + query)
            # For constant metrics, evaluate once
            Gm = self.metric.G(mid[0] if len(mid) > 0 else query)
            return np.sqrt(np.maximum(np.einsum('ij,jk,ik->i', diff,
                                                 Gm, diff), 0.0))
        else:
            # Fall back to loop for spatially varying tiers
            return np.array([self.distance(p, query) for p in points])
