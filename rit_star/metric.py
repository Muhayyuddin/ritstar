"""
metric.py — Riemannian metric tensor definitions for RIT*.

Each metric class encodes a positive-definite matrix field G(x) over
configuration space.  Directions where G has large eigenvalues are expensive
to traverse; small eigenvalues are cheap.  The planner uses G to build
Riemannian informed sets that are tighter than the standard Euclidean
ellipsoid, accelerating convergence to the optimal path.

All classes expose:
    G(x)          -> (d,d) positive-definite matrix
    G_inv(x)      -> (d,d) inverse of G(x)
    sqrt_det_G(x) -> scalar, sqrt(det(G(x)))  (for volume computations)
"""

from __future__ import annotations

import numpy as np
from abc import ABC, abstractmethod
from typing import Optional, Sequence


class RiemannianMetric(ABC):
    """Abstract base class for Riemannian metric tensors."""

    def __init__(self, dim: int):
        """
        Parameters
        ----------
        dim : int
            Dimensionality of the configuration space.
        """
        self.dim = dim

    @abstractmethod
    def G(self, x: np.ndarray) -> np.ndarray:
        """Return the (d,d) metric tensor at configuration *x*."""

    @abstractmethod
    def G_inv(self, x: np.ndarray) -> np.ndarray:
        """Return the inverse of G(x)."""

    @abstractmethod
    def sqrt_det_G(self, x: np.ndarray) -> float:
        """Return sqrt(det(G(x))), used for volume element computation."""

    def eigenvalues(self, x: np.ndarray) -> np.ndarray:
        """Return sorted eigenvalues of G(x) in ascending order.

        Used by Theorem 1 (volume ratio bound) and Theorem 2
        (metric-adapted connection radius).
        """
        return np.sort(np.linalg.eigvalsh(self.G(x)))

    def condition_number(self, x: np.ndarray) -> float:
        """κ(G(x)) = λ_max / λ_min of G at configuration x.

        Used by Theorem 3 (convergence rate separation).
        """
        eigs = self.eigenvalues(x)
        return float(eigs[-1] / max(eigs[0], 1e-30))


# ──────────────────────────────────────────────────────────────────────
class EuclideanMetric(RiemannianMetric):
    """G(x) = I for all x.  Baseline metric — RIT* with this metric
    is equivalent to standard Informed RRT*."""

    def __init__(self, dim: int):
        """
        Parameters
        ----------
        dim : int
            Dimensionality of the space.

        Notes
        -----
        Implements the trivial identity metric (flat Euclidean space).
        """
        super().__init__(dim)
        self._I = np.eye(dim)

    def G(self, x: np.ndarray) -> np.ndarray:
        """Return identity matrix (constant everywhere)."""
        return self._I.copy()

    def G_inv(self, x: np.ndarray) -> np.ndarray:
        """Return identity matrix (self-inverse)."""
        return self._I.copy()

    def sqrt_det_G(self, x: np.ndarray) -> float:
        """Return 1.0 (det(I) = 1)."""
        return 1.0

    def eigenvalues(self, x: np.ndarray) -> np.ndarray:
        return np.ones(self.dim)

    def condition_number(self, x: np.ndarray) -> float:
        return 1.0


# ──────────────────────────────────────────────────────────────────────
class DiagonalAnisotropicMetric(RiemannianMetric):
    """G(x) = diag(w_1, w_2, ..., w_d),  spatially constant.

    Per-dimension cost weights make some axes more expensive than others.
    For 2D: w = [3.0, 1.0] means the x-axis is 3× more expensive.
    For 3D: w = [5.0, 1.0, 2.0].

    This is the simplest non-trivial metric and the primary test case.
    """

    def __init__(self, weights: Sequence[float]):
        """
        Parameters
        ----------
        weights : array-like of float
            Per-dimension cost weights (all must be > 0).

        Raises
        ------
        ValueError
            If any weight is non-positive.
        """
        weights = np.asarray(weights, dtype=float)
        if np.any(weights <= 0):
            raise ValueError("All weights must be positive.")
        super().__init__(len(weights))
        self._weights = weights
        self._G = np.diag(weights)
        self._G_inv = np.diag(1.0 / weights)
        self._sqrt_det = float(np.sqrt(np.prod(weights)))

    @property
    def weights(self) -> np.ndarray:
        """Weight vector (read-only copy)."""
        return self._weights.copy()

    def G(self, x: np.ndarray) -> np.ndarray:
        """Return diag(w).  Constant — *x* is ignored."""
        return self._G.copy()

    def G_inv(self, x: np.ndarray) -> np.ndarray:
        """Return diag(1/w)."""
        return self._G_inv.copy()

    def sqrt_det_G(self, x: np.ndarray) -> float:
        """Return sqrt(prod(w))."""
        return self._sqrt_det

    def eigenvalues(self, x: np.ndarray) -> np.ndarray:
        return np.sort(self._weights)

    def condition_number(self, x: np.ndarray) -> float:
        return float(np.max(self._weights) / max(np.min(self._weights), 1e-30))


# ──────────────────────────────────────────────────────────────────────
class ObstacleInflatedMetric(RiemannianMetric):
    """G(x) = (1 + alpha * Σᵢ exp(-‖x - oᵢ‖² / σ²)) · I

    Models safety-aware cost: configurations near obstacle centres oᵢ
    are expensive to traverse.  The metric is isotropic but spatially
    varying — it inflates uniformly near each obstacle centre.
    """

    def __init__(self, obstacle_centers: np.ndarray,
                 sigma: float = 0.3, alpha: float = 10.0):
        """
        Parameters
        ----------
        obstacle_centers : (N, d) array
            Positions of obstacle centres.
        sigma : float
            Gaussian inflation radius (default 0.3).
        alpha : float
            Inflation strength (default 10.0).
        """
        obstacle_centers = np.atleast_2d(obstacle_centers)
        dim = obstacle_centers.shape[1]
        super().__init__(dim)
        self._centers = obstacle_centers.astype(float)
        self._sigma2 = sigma * sigma
        self._alpha = alpha

    def _scale(self, x: np.ndarray) -> float:
        """Scalar inflation factor at *x*."""
        diff = self._centers - x  # (N, d)
        sq_dists = np.sum(diff * diff, axis=1)  # (N,)
        return 1.0 + self._alpha * np.sum(np.exp(-sq_dists / self._sigma2))

    def _scale_batch(self, pts: np.ndarray) -> np.ndarray:
        """Vectorized scalar inflation factor for (M, d) array of points."""
        # pts: (M, d), centers: (N, d)
        # Use cdist for efficient pairwise distance computation
        from scipy.spatial.distance import cdist
        sq_dists = cdist(pts, self._centers, 'sqeuclidean')  # (M, N)
        return 1.0 + self._alpha * np.sum(
            np.exp(-sq_dists / self._sigma2), axis=1)  # (M,)

    def G(self, x: np.ndarray) -> np.ndarray:
        """Return s(x)·I where s encodes obstacle proximity.

        Implements the obstacle-inflated metric from the RIT* formulation.
        """
        return self._scale(x) * np.eye(self.dim)

    def G_inv(self, x: np.ndarray) -> np.ndarray:
        """Return (1/s(x))·I."""
        return (1.0 / self._scale(x)) * np.eye(self.dim)

    def sqrt_det_G(self, x: np.ndarray) -> float:
        """Return s(x)^(d/2)."""
        return self._scale(x) ** (self.dim / 2.0)

    def eigenvalues(self, x: np.ndarray) -> np.ndarray:
        s = self._scale(x)
        return np.full(self.dim, s)

    def condition_number(self, x: np.ndarray) -> float:
        return 1.0  # conformal: all eigenvalues equal


# ──────────────────────────────────────────────────────────────────────
class JointInertiaMetric2D(RiemannianMetric):
    """Spatially-varying metric for a 2-joint planar arm.

    G(θ) = R(θ)ᵀ · diag(I₁, I₂) · R(θ)

    where R(θ) is a rotation matrix that depends on joint angles
    and I₁, I₂ are the joint inertias.  Joint 1 has higher inertia
    (more expensive to move).  This is the most interesting metric
    because it varies across configuration space.
    """

    def __init__(self, I1: float = 3.0, I2: float = 1.0):
        """
        Parameters
        ----------
        I1 : float
            Inertia of joint 1 (proximal, default 3.0).
        I2 : float
            Inertia of joint 2 (distal, default 1.0).
        """
        super().__init__(2)
        self._I1 = I1
        self._I2 = I2
        self._D = np.diag([I1, I2])

    def _rotation(self, theta: np.ndarray) -> np.ndarray:
        """2×2 rotation matrix parameterised by the first joint angle."""
        c, s = np.cos(theta[0]), np.sin(theta[0])
        return np.array([[c, -s],
                         [s,  c]])

    def G(self, x: np.ndarray) -> np.ndarray:
        """Return R(θ)ᵀ · diag(I₁,I₂) · R(θ).

        Implements the joint-inertia metric for the 2-DOF planar arm.
        """
        R = self._rotation(x)
        return R.T @ self._D @ R

    def G_inv(self, x: np.ndarray) -> np.ndarray:
        """Return R(θ)ᵀ · diag(1/I₁, 1/I₂) · R(θ)."""
        R = self._rotation(x)
        D_inv = np.diag([1.0 / self._I1, 1.0 / self._I2])
        return R.T @ D_inv @ R

    def sqrt_det_G(self, x: np.ndarray) -> float:
        """Return sqrt(I₁ · I₂) — determinant independent of θ."""
        return np.sqrt(self._I1 * self._I2)

    def eigenvalues(self, x: np.ndarray) -> np.ndarray:
        return np.sort([self._I1, self._I2]).astype(float)

    def condition_number(self, x: np.ndarray) -> float:
        return float(max(self._I1, self._I2) / min(self._I1, self._I2))


# ──────────────────────────────────────────────────────────────────────
class PathwayMetric(RiemannianMetric):
    """G(x) = (base + corridor · (1 − exp(−d²/w²))) · I

    Creates a cheap corridor through the space connecting start to
    goal.  Points close to the corridor line segment have low cost;
    points far away are penalised.

    Used to test whether RIT* discovers and exploits natural cost
    corridors induced by the metric.
    """

    def __init__(self, x_start: np.ndarray, x_goal: np.ndarray,
                 base_cost: float = 1.0, corridor_cost: float = 8.0,
                 width: float = 0.15):
        """
        Parameters
        ----------
        x_start : array
            Start configuration (defines one endpoint of the corridor).
        x_goal : array
            Goal configuration (defines the other endpoint).
        base_cost : float
            Metric scale inside the corridor (default 1.0).
        corridor_cost : float
            Additional penalty scale outside the corridor (default 8.0).
        width : float
            Gaussian half-width of the corridor (default 0.15).
        """
        x_start = np.asarray(x_start, dtype=float)
        x_goal = np.asarray(x_goal, dtype=float)
        super().__init__(len(x_start))
        self._p0 = x_start
        self._dir = x_goal - x_start
        self._len2 = float(np.dot(self._dir, self._dir))
        self._base = base_cost
        self._corr = corridor_cost
        self._w2 = width * width

    def _dist_to_segment_sq(self, x: np.ndarray) -> float:
        """Squared distance from *x* to the start–goal line segment."""
        v = x - self._p0
        t = np.clip(np.dot(v, self._dir) / max(self._len2, 1e-12), 0.0, 1.0)
        proj = self._p0 + t * self._dir
        diff = x - proj
        return float(np.dot(diff, diff))

    def _scale(self, x: np.ndarray) -> float:
        d2 = self._dist_to_segment_sq(x)
        return self._base + self._corr * (1.0 - np.exp(-d2 / self._w2))

    def G(self, x: np.ndarray) -> np.ndarray:
        """Return s(x)·I where s encodes distance from the corridor.

        Implements the pathway/corridor metric for RIT*.
        """
        return self._scale(x) * np.eye(self.dim)

    def G_inv(self, x: np.ndarray) -> np.ndarray:
        """Return (1/s(x))·I — PathwayMetric."""
        return (1.0 / self._scale(x)) * np.eye(self.dim)

    def sqrt_det_G(self, x: np.ndarray) -> float:
        """Return s(x)^(d/2) — PathwayMetric."""
        return self._scale(x) ** (self.dim / 2.0)

    def eigenvalues(self, x: np.ndarray) -> np.ndarray:
        s = self._scale(x)
        return np.full(self.dim, s)

    def condition_number(self, x: np.ndarray) -> float:
        return 1.0  # conformal: all eigenvalues equal


# ──────────────────────────────────────────────────────────────────────
class ClearanceMetric(RiemannianMetric):
    """G(x) = (1 / max(clearance(x), ε)²) · I

    Metric cost is inversely proportional to distance from nearest
    obstacle.  Incentivises maximum clearance — useful for
    safety-critical applications.

    Parameters
    ----------
    obstacle_centers : (N, d) array
        Positions of obstacle centres.
    obstacle_radii : (N,) array or float
        Radii of each obstacle (or uniform radius).
    epsilon : float
        Minimum clearance to avoid singularity (default 0.05).
    """

    def __init__(self, obstacle_centers: np.ndarray,
                 obstacle_radii=0.1, epsilon: float = 0.05):
        obstacle_centers = np.atleast_2d(obstacle_centers)
        dim = obstacle_centers.shape[1]
        super().__init__(dim)
        self._centers = obstacle_centers.astype(float)
        if np.isscalar(obstacle_radii):
            self._radii = np.full(len(obstacle_centers), float(obstacle_radii))
        else:
            self._radii = np.asarray(obstacle_radii, dtype=float)
        self._eps = epsilon

    def _clearance(self, x: np.ndarray) -> float:
        """Minimum distance from x to the nearest obstacle surface."""
        dists = np.linalg.norm(self._centers - x, axis=1) - self._radii
        return max(float(np.min(dists)), self._eps)

    def _scale(self, x: np.ndarray) -> float:
        c = self._clearance(x)
        return 1.0 / (c * c)

    def G(self, x: np.ndarray) -> np.ndarray:
        return self._scale(x) * np.eye(self.dim)

    def G_inv(self, x: np.ndarray) -> np.ndarray:
        c = self._clearance(x)
        return (c * c) * np.eye(self.dim)

    def sqrt_det_G(self, x: np.ndarray) -> float:
        return self._scale(x) ** (self.dim / 2.0)

    def eigenvalues(self, x: np.ndarray) -> np.ndarray:
        s = self._scale(x)
        return np.full(self.dim, s)

    def condition_number(self, x: np.ndarray) -> float:
        return 1.0  # conformal

    def _scale_batch(self, pts: np.ndarray) -> np.ndarray:
        """Vectorized scale for (M, d) array."""
        from scipy.spatial.distance import cdist
        dists = cdist(pts, self._centers)  # (M, N)
        clearances = np.maximum(
            np.min(dists - self._radii, axis=1), self._eps)
        return 1.0 / (clearances * clearances)


# ──────────────────────────────────────────────────────────────────────
class TaskSpaceMetric(RiemannianMetric):
    """G(q) = J(q)ᵀ W J(q)  — task-space velocity metric.

    Ties Riemannian cost to task-space velocity via the manipulator
    Jacobian.  Movements that produce large task-space displacement
    are cheap; those that barely move the end-effector are expensive.

    Parameters
    ----------
    jacobian_fn : callable  q -> (m, d) array
        Maps joint configuration q to the (m × d) Jacobian matrix.
    task_weights : (m, m) array or None
        Positive definite weight matrix in task space (default I_m).
    dim : int
        Dimensionality of C-space.
    regularization : float
        Small ridge added to G to ensure strict positive-definiteness.
    """

    def __init__(self, jacobian_fn, dim: int,
                 task_weights=None, regularization: float = 0.01):
        super().__init__(dim)
        self._jacobian = jacobian_fn
        self._reg = regularization
        if task_weights is not None:
            self._W = np.asarray(task_weights, dtype=float)
        else:
            self._W = None  # determined at first call

    def G(self, x: np.ndarray) -> np.ndarray:
        J = self._jacobian(x)
        if self._W is None:
            self._W = np.eye(J.shape[0])
        return J.T @ self._W @ J + self._reg * np.eye(self.dim)

    def G_inv(self, x: np.ndarray) -> np.ndarray:
        return np.linalg.inv(self.G(x))

    def sqrt_det_G(self, x: np.ndarray) -> float:
        return float(np.sqrt(max(np.linalg.det(self.G(x)), 1e-30)))


# ──────────────────────────────────────────────────────────────────────
class LearnedMetric(RiemannianMetric):
    """G(x) predicted by a neural network: x → L(x), G = LLᵀ.

    Loads a pretrained model that maps configurations to the lower-
    triangular Cholesky factor L, ensuring G is always SPD.

    Parameters
    ----------
    model_fn : callable  x -> (d, d) or (d*(d+1)/2,) array
        A function (e.g. a small MLP wrapped in a lambda) that returns
        either the full Cholesky factor L(x) or its flattened lower-
        triangular entries.
    dim : int
        Dimensionality of the configuration space.
    """

    def __init__(self, model_fn, dim: int):
        super().__init__(dim)
        self._model = model_fn

    def _cholesky(self, x: np.ndarray) -> np.ndarray:
        """Get the Cholesky factor L at x from the model."""
        out = self._model(x)
        out = np.asarray(out, dtype=float)
        if out.ndim == 1:
            # Unpack flattened lower-triangular
            L = np.zeros((self.dim, self.dim))
            idx = 0
            for i in range(self.dim):
                for j in range(i + 1):
                    L[i, j] = out[idx]
                    idx += 1
            # Ensure positive diagonal
            np.fill_diagonal(L, np.abs(np.diag(L)) + 1e-4)
            return L
        # Full matrix — ensure lower-triangular with positive diagonal
        L = np.tril(out)
        np.fill_diagonal(L, np.abs(np.diag(L)) + 1e-4)
        return L

    def G(self, x: np.ndarray) -> np.ndarray:
        L = self._cholesky(x)
        return L @ L.T

    def G_inv(self, x: np.ndarray) -> np.ndarray:
        L = self._cholesky(x)
        L_inv = np.linalg.inv(L)
        return L_inv.T @ L_inv

    def sqrt_det_G(self, x: np.ndarray) -> float:
        L = self._cholesky(x)
        return float(np.abs(np.prod(np.diag(L))))


# ──────────────────────────────────────────────────────────────────────
class CollisionAdaptiveMetric(RiemannianMetric):
    """Collision-Adaptive Riemannian Metric (CARM).

    Learns a Riemannian metric field online from collision feedback
    during planning.  No a-priori obstacle model is required — only
    a binary collision checker.

    The metric is a conformal scaling of a base metric:

        G_CARM(x) = s(x) · G_base(x)

    where the adaptive scale factor is

        s(x) = 1 + α · (1/N) Σ_i K_σ(x, c_i)

    with c_i the collision points discovered so far and K_σ a
    Gaussian kernel with bandwidth σ.

    As the planner explores and discovers obstacle boundaries,
    the metric tensor inflates near collisions, causing the
    informed set to contract away from obstacles.  This yields:

      1.  Tighter informed sets without requiring obstacle geometry.
      2.  Self-improving sampling: later batches are more efficient.
      3.  Principled bridging of trajectory-optimization awareness
          (obstacle gradients) into sampling-based planning.

    Parameters
    ----------
    base_metric : RiemannianMetric
        Starting metric (e.g. EuclideanMetric).  CARM refines it.
    dim : int
        Configuration-space dimensionality.
    sigma : float
        Gaussian kernel bandwidth — controls the spatial influence
        radius of each collision point (default 0.1).
    alpha : float
        Penalty strength — higher means more aggressive inflation
        near collisions (default 5.0).
    max_points : int
        Maximum stored collision points.  Older points are
        downsampled when the limit is exceeded (default 500).
    """

    def __init__(self, base_metric: RiemannianMetric, dim: int,
                 sigma: float = 0.1, alpha: float = 5.0,
                 max_points: int = 500):
        super().__init__(dim)
        self.base = base_metric
        self.sigma = float(sigma)
        self._sigma2 = self.sigma * self.sigma
        self.alpha = float(alpha)
        self.max_points = int(max_points)

        self._collision_points: list = []
        self._collision_array: Optional[np.ndarray] = None
        self._dirty = True          # array needs rebuild
        self._update_count = 0      # total collision samples received
        self._frozen = False         # stop accepting new points

    # ── collision feedback interface ─────────────────────────────────

    def add_collision_point(self, point: np.ndarray) -> None:
        """Record a collision point for metric adaptation.

        Parameters
        ----------
        point : (d,) array
            Configuration that was found to be in collision.
        """
        if self._frozen:
            return
        self._collision_points.append(np.asarray(point, dtype=float))
        self._dirty = True
        self._update_count += 1
        # Downsample when exceeding capacity
        if len(self._collision_points) > self.max_points:
            self._downsample()

    def add_collision_points_batch(self, points: np.ndarray) -> None:
        """Record multiple collision points at once.

        Parameters
        ----------
        points : (N, d) array
        """
        if self._frozen:
            return
        for p in points:
            self._collision_points.append(np.asarray(p, dtype=float))
        self._dirty = True
        self._update_count += len(points)
        if len(self._collision_points) > self.max_points:
            self._downsample()

    def freeze(self) -> None:
        """Stop accepting new collision points (metric is stable)."""
        self._frozen = True

    @property
    def n_collision_points(self) -> int:
        return len(self._collision_points)

    @property
    def update_count(self) -> int:
        return self._update_count

    # ── metric evaluation ────────────────────────────────────────────

    def _ensure_array(self) -> None:
        if self._dirty and self._collision_points:
            self._collision_array = np.array(self._collision_points)
            self._dirty = False

    def _collision_scale(self, x: np.ndarray) -> float:
        """Adaptive conformal scale factor at point x.

        s(x) = 1 + alpha * mean_i(K_sigma(x, c_i))

        Returns 1.0 when no collision data is available.
        """
        if not self._collision_points:
            return 1.0
        self._ensure_array()
        diffs = self._collision_array - x
        dists_sq = np.sum(diffs * diffs, axis=1)
        kernel_vals = np.exp(-dists_sq / (2.0 * self._sigma2))
        return 1.0 + self.alpha * float(np.mean(kernel_vals))

    def _collision_scale_batch(self, pts: np.ndarray) -> np.ndarray:
        """Vectorized scale for (M, d) array of query points."""
        M = pts.shape[0]
        if not self._collision_points:
            return np.ones(M)
        self._ensure_array()
        # (M, N) pairwise squared distances
        # Efficient: ||p - c||^2 = ||p||^2 - 2 p·c + ||c||^2
        pp = np.sum(pts * pts, axis=1, keepdims=True)    # (M, 1)
        cc = np.sum(self._collision_array * self._collision_array,
                     axis=1, keepdims=True).T              # (1, N)
        pc = pts @ self._collision_array.T                 # (M, N)
        dists_sq = pp - 2.0 * pc + cc
        np.maximum(dists_sq, 0.0, out=dists_sq)
        kernel_vals = np.exp(-dists_sq / (2.0 * self._sigma2))
        return 1.0 + self.alpha * np.mean(kernel_vals, axis=1)

    def G(self, x: np.ndarray) -> np.ndarray:
        return self._collision_scale(x) * self.base.G(x)

    def G_inv(self, x: np.ndarray) -> np.ndarray:
        return (1.0 / self._collision_scale(x)) * self.base.G_inv(x)

    def sqrt_det_G(self, x: np.ndarray) -> float:
        s = self._collision_scale(x)
        return (s ** (self.dim / 2.0)) * self.base.sqrt_det_G(x)

    def eigenvalues(self, x: np.ndarray) -> np.ndarray:
        s = self._collision_scale(x)
        return s * self.base.eigenvalues(x)

    def condition_number(self, x: np.ndarray) -> float:
        # Conformal scaling preserves the base condition number
        return self.base.condition_number(x)

    # ── batch interface (used by MetricFieldCache) ───────────────────

    def _scale_batch(self, pts: np.ndarray) -> np.ndarray:
        """Batch conformal scale factor (for cache compatibility)."""
        base_scale = np.ones(len(pts))
        if hasattr(self.base, '_scale_batch'):
            base_scale = self.base._scale_batch(pts)
        elif hasattr(self.base, '_scale'):
            base_scale = np.array([self.base._scale(p) for p in pts])
        return self._collision_scale_batch(pts) * base_scale

    # ── internal ─────────────────────────────────────────────────────

    def _downsample(self) -> None:
        """Reduce stored collision points via farthest-point subsampling."""
        n_keep = self.max_points // 2
        pts = np.array(self._collision_points)
        N = len(pts)
        if N <= n_keep:
            return
        # Greedy farthest-point sampling for diverse coverage
        kept_idx = [0]
        min_dists = np.full(N, np.inf)
        for _ in range(n_keep - 1):
            last = pts[kept_idx[-1]]
            d2 = np.sum((pts - last) ** 2, axis=1)
            np.minimum(min_dists, d2, out=min_dists)
            next_idx = int(np.argmax(min_dists))
            kept_idx.append(next_idx)
        self._collision_points = [pts[i].copy() for i in kept_idx]
        self._dirty = True
