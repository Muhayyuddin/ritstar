"""
informed_set.py — Riemannian and Euclidean informed sets for sampling.

The *informed set* constrains where the planner draws new samples.

  I_R       = { x : d_R(x_s, x) + d_R(x, x_g) ≤ c_best }
  I_euclid  = { x : ‖x_s − x‖ + ‖x − x_g‖  ≤ c_best }

Because the Riemannian metric inflates distances in expensive
directions, I_R ⊂ I_euclid (Theorem 1), yielding fewer wasted
samples and faster convergence.
"""

from __future__ import annotations

import numpy as np
from typing import Optional

from .geodesic import GeodesicComputer, diagonal_geodesic
from .metric import (
    RiemannianMetric,
    DiagonalAnisotropicMetric,
    EuclideanMetric,
    ObstacleInflatedMetric,
)


# ═══════════════════════════════════════════════════════════════════════
# Riemannian informed set
# ═══════════════════════════════════════════════════════════════════════

class RiemannianInformedSet:
    """Sample from I_R = { x : d_R(x_s,x) + d_R(x,x_g) ≤ c_best }.

    Uses rejection sampling with the Euclidean ellipsoid as the
    proposal bounding region.

    Parameters
    ----------
    x_start, x_goal : (d,) arrays
    c_best : float
        Current best solution cost (Riemannian).
    geodesic_computer : GeodesicComputer
    bounds : list of (lo, hi) per dimension
        Hard workspace limits for clipping proposals.
    """

    def __init__(self, x_start: np.ndarray, x_goal: np.ndarray,
                 c_best: float, geodesic_computer: GeodesicComputer,
                 bounds: Optional[list] = None):
        self.x_start = np.asarray(x_start, dtype=float)
        self.x_goal = np.asarray(x_goal, dtype=float)
        self.c_best = float(c_best)
        self.gc = geodesic_computer
        self.dim = len(self.x_start)
        self.bounds = bounds

        # Euclidean informed ellipsoid used as proposal region
        self._c_min_euclid = float(np.linalg.norm(self.x_goal - self.x_start))
        self._center = 0.5 * (self.x_start + self.x_goal)

        # Build Euclidean ellipsoid proposal for efficient sampling
        self._eis_proposal = EuclideanInformedSet(
            self.x_start, self.x_goal, self.c_best, bounds=bounds)

        # Precompute for fast diagonal membership (Fix 4)
        self._metric = geodesic_computer.metric
        self._is_diagonal = isinstance(self._metric, DiagonalAnisotropicMetric)
        if self._is_diagonal:
            self._sqrt_w = np.sqrt(self._metric._weights)

        # acceptance bookkeeping
        self._total_proposed = 0
        self._total_accepted = 0

    @property
    def acceptance_rate(self) -> float:
        """Fraction of proposals that fell inside I_R."""
        if self._total_proposed == 0:
            return 0.0
        return self._total_accepted / self._total_proposed

    # ── membership ───────────────────────────────────────────────────

    def is_member(self, x: np.ndarray) -> bool:
        """Check x ∈ I_R.

        Parameters
        ----------
        x : (d,) array

        Returns
        -------
        bool
            True iff d_R(x_s, x) + d_R(x, x_g) ≤ c_best.

        Notes
        -----
        Implements the membership predicate of Definition 2 (Riemannian
        informed set).  Uses a cheap Euclidean pre-filter (Fix 3).
        """
        # Cheap Euclidean pre-filter: if Euclidean sum > c_best, skip
        d1e = float(np.linalg.norm(x - self.x_start))
        d2e = float(np.linalg.norm(x - self.x_goal))
        if (d1e + d2e) > self.c_best:
            return False
        d1 = self.gc.distance(self.x_start, x)
        d2 = self.gc.distance(x, self.x_goal)
        return (d1 + d2) <= self.c_best

    def batch_is_member(self, pts: np.ndarray) -> np.ndarray:
        """Vectorized membership test for (N, d) array of points.

        Returns boolean mask of length N.
        Uses Euclidean pre-filter + fast diagonal path when available.
        """
        N = pts.shape[0]
        mask = np.ones(N, dtype=bool)

        # Stage 1: vectorized Euclidean pre-filter (Fix 3)
        d1e = np.linalg.norm(pts - self.x_start, axis=1)
        d2e = np.linalg.norm(pts - self.x_goal, axis=1)
        euclid_sum = d1e + d2e
        mask &= (euclid_sum <= self.c_best)

        # Stage 2: Riemannian check on survivors
        survivors = np.where(mask)[0]
        if len(survivors) == 0:
            return mask

        if self._is_diagonal:
            # Fast vectorized diagonal geodesic (Fix 4)
            w = self._metric._weights
            diff_s = pts[survivors] - self.x_start
            diff_g = pts[survivors] - self.x_goal
            d1r = np.sqrt(np.maximum(np.sum(diff_s * diff_s * w, axis=1), 0.0))
            d2r = np.sqrt(np.maximum(np.sum(diff_g * diff_g * w, axis=1), 0.0))
            riem_sum = d1r + d2r
            mask[survivors] = riem_sum <= self.c_best
        else:
            # Semi-vectorized: compute midpoint metric once per point,
            # then use vectorized quadratic form (avoids Python loop
            # over quadrature points inside gc.distance)
            surv_pts = pts[survivors]
            n_surv = len(survivors)
            xs = self.x_start
            xg = self.x_goal

            # Batch compute d_R(xs, pt) ≈ sqrt(diff^T G(mid) diff)
            diff_s = surv_pts - xs
            mid_s = 0.5 * (surv_pts + xs)
            diff_g = surv_pts - xg
            mid_g = 0.5 * (surv_pts + xg)

            d1r = np.empty(n_surv)
            d2r = np.empty(n_surv)
            for j in range(n_surv):
                Gm1 = self._metric.G(mid_s[j])
                d1r[j] = np.sqrt(max(float(diff_s[j] @ Gm1 @ diff_s[j]), 0.0))
                Gm2 = self._metric.G(mid_g[j])
                d2r[j] = np.sqrt(max(float(diff_g[j] @ Gm2 @ diff_g[j]), 0.0))

            riem_sum = d1r + d2r
            mask[survivors] = riem_sum <= self.c_best

        return mask

    # ── sampling ─────────────────────────────────────────────────────

    def sample(self, n_samples: int = 1,
               rng: Optional[np.random.Generator] = None) -> np.ndarray:
        """Draw *n_samples* uniformly from I_R via rejection sampling.

        Parameters
        ----------
        n_samples : int
        rng : numpy random Generator (optional).

        Returns
        -------
        (n_samples, d) array of configurations inside I_R.

        Notes
        -----
        Uses the Euclidean ellipsoid as proposal (Fix 2) with vectorized
        batch membership testing (Fix 1) and Euclidean pre-filter (Fix 3).
        """
        if rng is None:
            rng = np.random.default_rng()

        # Direct sampling for diagonal metrics (no rejection needed)
        if self._is_diagonal:
            return self._sample_diagonal_direct(n_samples, rng)

        collected = []
        max_rounds = 50
        # Adaptive oversampling: increase multiplier when acceptance is low
        base_mult = 3
        if self.acceptance_rate > 0 and self.acceptance_rate < 0.3:
            base_mult = max(3, int(1.0 / self.acceptance_rate) + 1)
        batch_draw = max(n_samples * base_mult, 200)

        for _ in range(max_rounds):
            # Draw proposals from Euclidean ellipsoid (Fix 2)
            proposals = self._eis_proposal.sample(batch_draw, rng=rng)
            self._total_proposed += len(proposals)

            # Vectorized membership test (Fix 1 + 3 + 4)
            mask = self.batch_is_member(proposals)
            accepted = proposals[mask]
            self._total_accepted += len(accepted)

            if len(accepted) > 0:
                collected.append(accepted)
                total = sum(a.shape[0] for a in collected)
                if total >= n_samples:
                    break
            # Increase batch size if acceptance is very low
            if self._total_proposed > 0:
                rate = self._total_accepted / self._total_proposed
                if rate < 0.1:
                    batch_draw = min(batch_draw * 2, n_samples * 20)

        if not collected:
            # Fallback: return ellipsoid samples directly
            return self._eis_proposal.sample(n_samples, rng=rng)

        result = np.vstack(collected)
        return result[:n_samples]

    def _sample_diagonal_direct(self, n_samples: int,
                                rng: np.random.Generator) -> np.ndarray:
        """Direct sampling for diagonal metrics without rejection.

        For G = diag(w), the Riemannian informed set is an ellipsoid
        in weighted space.  Sample from a Euclidean ellipsoid in the
        whitened space and transform back.
        """
        w = self._metric._weights
        sqrt_w = self._sqrt_w
        # Transform start/goal to weighted space
        ws = sqrt_w * self.x_start
        wg = sqrt_w * self.x_goal
        # Build Euclidean ellipsoid in weighted space
        eis_w = EuclideanInformedSet(ws, wg, self.c_best, bounds=None)
        pts_w = eis_w.sample(int(n_samples * 1.3), rng=rng)
        # Transform back
        pts = pts_w / sqrt_w
        # Clip to bounds
        if self.bounds is not None:
            for k in range(self.dim):
                pts[:, k] = np.clip(pts[:, k],
                                    self.bounds[k][0], self.bounds[k][1])
        # Filter points that ended up outside after clipping
        if len(pts) > n_samples:
            pts = pts[:n_samples]
        elif len(pts) < n_samples:
            extra = eis_w.sample(n_samples - len(pts), rng=rng) / sqrt_w
            if self.bounds is not None:
                for k in range(self.dim):
                    extra[:, k] = np.clip(extra[:, k],
                                          self.bounds[k][0], self.bounds[k][1])
            pts = np.vstack([pts, extra])
        return pts[:n_samples]

    # ── volume estimation ────────────────────────────────────────────

    def volume_estimate(self, n_monte_carlo: int = 10000,
                        rng: Optional[np.random.Generator] = None) -> float:
        """Monte Carlo volume estimate of I_R.

        Parameters
        ----------
        n_monte_carlo : int
        rng : random Generator.

        Returns
        -------
        float
            Estimated volume (fraction of bounding box × box volume).

        Notes
        -----
        Used to experimentally validate Theorem 1: Vol(I_R) < Vol(I_euclid).
        """
        if rng is None:
            rng = np.random.default_rng(42)

        r1 = self.c_best / 2.0
        lo = self._center - r1
        hi = self._center + r1
        if self.bounds is not None:
            for k in range(self.dim):
                lo[k] = max(lo[k], self.bounds[k][0])
                hi[k] = min(hi[k], self.bounds[k][1])

        box_vol = float(np.prod(hi - lo))
        pts = rng.uniform(lo, hi, size=(n_monte_carlo, self.dim))
        inside = sum(1 for pt in pts if self.is_member(pt))
        return box_vol * inside / n_monte_carlo

    # ── Theorem 1: analytical volume ratio ───────────────────────────

    def analytical_volume_ratio(self) -> float:
        """Closed-form Vol(I_R)/Vol(I_E) from Theorem 1.

        For constant diagonal metrics: exact.
        For conformal metrics: upper bound using (s_min/s_max)^(d/2).
        For general metrics: first-order approximation using the
        average metric along the start-goal segment.

        Returns
        -------
        float in (0, 1]  (1.0 when G = I).
        """
        return volume_ratio_bound(
            self._metric, self.x_start, self.x_goal, self.dim)

    # ── Theorem 3: sample efficiency & convergence rate ──────────────

    def sample_efficiency_ratio(self) -> float:
        """Predicted speedup factor: n_E/n_R samples needed for same gap.

        Returns prod(sqrt(lambda_i / lambda_min)) >= 1 — higher means
        RIT* needs fewer samples.

        Theorem 3: for diagonal G = diag(kappa, 1, ..., 1) this equals
        sqrt(kappa).
        """
        vr = self.analytical_volume_ratio()
        return 1.0 / max(vr, 1e-30)

    def convergence_rate_ratio(self) -> float:
        """Predicted gap ratio at same n: E[gap_R]/E[gap_E].

        Returns (Vol(I_R)/Vol(I_E))^(2/d) <= 1 — lower means faster
        convergence.

        Theorem 3: for G = diag(kappa, 1, ..., 1) this equals
        kappa^(-1/d).
        """
        vr = self.analytical_volume_ratio()
        return float(vr ** (2.0 / max(self.dim, 1)))

    # ── visualisation helper ─────────────────────────────────────────

    def visualize_2d(self, ax, resolution: int = 200,
                     color: str = 'purple', alpha: float = 0.15):
        """Fill the 2-D informed set region on a matplotlib axis.

        Parameters
        ----------
        ax : matplotlib Axes
        resolution : int
            Grid resolution for the membership mask.
        color : str
        alpha : float

        Notes
        -----
        Uses meshgrid + ``is_member`` → boolean mask → ``contourf``.
        """
        if self.dim != 2:
            return
        r1 = self.c_best / 2.0
        lo = self._center - r1
        hi = self._center + r1
        if self.bounds is not None:
            for k in range(2):
                lo[k] = max(lo[k], self.bounds[k][0])
                hi[k] = min(hi[k], self.bounds[k][1])
                # Add larger margin to keep visualization inside bounds
                margin = 0.1 * (self.bounds[k][1] - self.bounds[k][0])
                lo[k] = max(lo[k], self.bounds[k][0] + margin)
                hi[k] = min(hi[k], self.bounds[k][1] - margin)

        xs = np.linspace(lo[0], hi[0], resolution)
        ys = np.linspace(lo[1], hi[1], resolution)
        XX, YY = np.meshgrid(xs, ys)
        # Vectorized membership test
        pts = np.column_stack([XX.ravel(), YY.ravel()])
        member_mask = self.batch_is_member(pts)
        
        # Also check bounds for visualization
        if self.bounds is not None:
            bounds_mask = np.ones(len(pts), dtype=bool)
            for k in range(2):
                bounds_mask &= (pts[:, k] >= self.bounds[k][0]) & (pts[:, k] <= self.bounds[k][1])
            member_mask &= bounds_mask
        
        mask = member_mask.reshape(XX.shape).astype(float)

        ax.contourf(XX, YY, mask, levels=[0.5, 1.5],
                    colors=[color], alpha=alpha)


# ═══════════════════════════════════════════════════════════════════════
# Standalone volume-ratio bound (Theorem 1)
# ═══════════════════════════════════════════════════════════════════════

def volume_ratio_bound(metric: RiemannianMetric,
                       x_start: np.ndarray, x_goal: np.ndarray,
                       dim: int,
                       c_best: float = None) -> float:
    """Compute the analytical volume ratio Vol(I_R)/Vol(I_E).

    Theorem 1: for constant diagonal G with eigenvalues lambda_i,

      Vol(I_R)     1              (c²_best - c²_{min,R})^{(d-1)/2}
      -------- = --------- · ─────────────────────────────────────
      Vol(I_E)   √det(G)      (c²_best - c²_{min,E})^{(d-1)/2}

    The first factor is the Jacobian from the whitening transform;
    the second captures the different eccentricities of the two
    ellipsoids (the Riemannian one is thinner because c_{min,R} >
    c_{min,E} for anisotropic G).

    When c_best is not provided, only the Jacobian factor is returned
    (a valid upper bound).

    For conformal G(x) = s(x)*I:
      Upper bound: (s_min / s_max)^(d/2).

    For general spatially-varying G:
      First-order approximation using the average metric along the
      start-goal line segment.

    Parameters
    ----------
    metric : RiemannianMetric
    x_start, x_goal : (d,) arrays
    dim : int
    c_best : float, optional
        Current best cost.  When supplied, the exact formula (Jacobian
        × eccentricity) is used instead of the Jacobian-only bound.

    Returns
    -------
    float in (0, 1]
    """
    if isinstance(metric, EuclideanMetric):
        return 1.0

    if isinstance(metric, DiagonalAnisotropicMetric):
        w = metric._weights
        # Jacobian factor: 1 / sqrt(det(G)) = prod(1/sqrt(w_i))
        jacobian = float(np.prod(1.0 / np.sqrt(w)))
        if c_best is not None:
            xs = np.asarray(x_start, dtype=float)
            xg = np.asarray(x_goal, dtype=float)
            diff = xg - xs
            c_min_R = float(np.sqrt(np.sum(w * diff * diff)))
            c_min_E = float(np.linalg.norm(diff))
            cb2 = c_best * c_best
            num = cb2 - c_min_R * c_min_R
            den = cb2 - c_min_E * c_min_E
            if den > 0 and num > 0:
                shape_factor = (num / den) ** ((dim - 1) / 2.0)
                return float(jacobian * shape_factor)
        return float(jacobian)

    # Sample eigenvalues along the start-goal line for varying metrics
    mid = 0.5 * (np.asarray(x_start) + np.asarray(x_goal))
    eigs = metric.eigenvalues(mid)
    eigs = np.maximum(eigs, 1e-30)
    lam_min = eigs[0]

    if isinstance(metric, ObstacleInflatedMetric):
        # Conformal: eigenvalues are all equal at each point.
        # Use a bound based on scale variation along the segment.
        n_pts = 11
        scales = []
        xs = np.asarray(x_start, dtype=float)
        xg = np.asarray(x_goal, dtype=float)
        for i in range(n_pts):
            t = i / (n_pts - 1)
            pt = xs + t * (xg - xs)
            scales.append(metric._scale(pt))
        s_min = min(scales)
        s_max = max(scales)
        return float((s_min / max(s_max, 1e-30)) ** (dim / 2.0))

    # General case: use average metric along the segment
    n_pts = 11
    xs = np.asarray(x_start, dtype=float)
    xg = np.asarray(x_goal, dtype=float)
    G_avg = np.zeros((dim, dim))
    for i in range(n_pts):
        t = i / (n_pts - 1)
        pt = xs + t * (xg - xs)
        G_avg += metric.G(pt)
    G_avg /= n_pts

    eigs_avg = np.sort(np.linalg.eigvalsh(G_avg))
    eigs_avg = np.maximum(eigs_avg, 1e-30)
    lam_min_avg = eigs_avg[0]
    return float(np.prod(np.sqrt(lam_min_avg / eigs_avg)))


# ═══════════════════════════════════════════════════════════════════════
# Euclidean informed set (baseline — prolate hyperellipsoid)
# ═══════════════════════════════════════════════════════════════════════

class EuclideanInformedSet:
    """Standard Euclidean prolate hyperellipsoid from Informed RRT*.

    I_euclid = { x : ‖x_s − x‖ + ‖x − x_g‖ ≤ c_best }

    Parameters
    ----------
    x_start, x_goal : (d,) arrays
    c_best : float
    bounds : list of (lo, hi) per dimension (optional clip region).
    """

    def __init__(self, x_start: np.ndarray, x_goal: np.ndarray,
                 c_best: float, bounds: Optional[list] = None):
        self.x_start = np.asarray(x_start, dtype=float)
        self.x_goal = np.asarray(x_goal, dtype=float)
        self.c_best = float(c_best)
        self.dim = len(self.x_start)
        self.bounds = bounds

        self._center = 0.5 * (self.x_start + self.x_goal)
        diff = self.x_goal - self.x_start
        self.c_min = float(np.linalg.norm(diff))

        # Semi-axis lengths
        self._r1 = self.c_best / 2.0
        r2_sq = max(self.c_best ** 2 - self.c_min ** 2, 0.0) / 4.0
        self._r2 = np.sqrt(r2_sq)

        # Rotation: first axis along start→goal direction
        if self.c_min > 1e-12:
            a1 = diff / self.c_min
        else:
            a1 = np.zeros(self.dim)
            a1[0] = 1.0

        # Build rotation matrix via Gram–Schmidt
        M = np.eye(self.dim)
        M[:, 0] = a1
        Q, _ = np.linalg.qr(M)
        # Ensure first column is a1
        if np.dot(Q[:, 0], a1) < 0:
            Q = -Q
        self._C = Q  # rotation matrix
        # Diagonal of radii
        self._L = np.diag(
            [self._r1] + [self._r2] * (self.dim - 1)
        )
        self._L_inv = np.diag(
            [1.0 / max(self._r1, 1e-12)] +
            [1.0 / max(self._r2, 1e-12)] * (self.dim - 1)
        )
        self._CL = self._C @ self._L  # for sampling transform

    # ── membership ───────────────────────────────────────────────────

    def is_member(self, x: np.ndarray) -> bool:
        """Check x ∈ I_euclid (sum of distances ≤ c_best).

        Parameters
        ----------
        x : (d,) array

        Returns
        -------
        bool
        """
        d1 = float(np.linalg.norm(x - self.x_start))
        d2 = float(np.linalg.norm(x - self.x_goal))
        return (d1 + d2) <= self.c_best

    # ── sampling ─────────────────────────────────────────────────────

    def sample(self, n_samples: int = 1,
               rng: Optional[np.random.Generator] = None) -> np.ndarray:
        """Uniform samples from the ellipsoid via unit-ball transform.

        Parameters
        ----------
        n_samples : int
        rng : numpy random Generator.

        Returns
        -------
        (n_samples, d) array

        Notes
        -----
        Samples u ~ Uniform(B^d), then applies x = C L u + centre.
        """
        if rng is None:
            rng = np.random.default_rng()

        d = self.dim
        # Sample uniformly in unit ball
        gauss = rng.standard_normal((n_samples, d))
        norms = np.linalg.norm(gauss, axis=1, keepdims=True)
        on_sphere = gauss / np.maximum(norms, 1e-12)
        radii = rng.uniform(0, 1, size=(n_samples, 1)) ** (1.0 / d)
        unit_ball = on_sphere * radii  # (n_samples, d)

        # Transform to ellipsoid
        samples = (self._CL @ unit_ball.T).T + self._center  # (n, d)

        # Clip to bounds
        if self.bounds is not None:
            for k in range(d):
                samples[:, k] = np.clip(samples[:, k],
                                        self.bounds[k][0], self.bounds[k][1])
        return samples

    # ── volume ───────────────────────────────────────────────────────

    def volume(self) -> float:
        """Analytical volume of the d-dimensional ellipsoid.

        Returns
        -------
        float
            V = (π^(d/2) / Γ(d/2+1)) · r₁ · r₂^(d−1)

        Notes
        -----
        Implements the closed-form volume of the Euclidean informed set.
        """
        from scipy.special import gamma as gamma_fn
        d = self.dim
        unit_ball_vol = (np.pi ** (d / 2.0)) / gamma_fn(d / 2.0 + 1.0)
        return unit_ball_vol * self._r1 * self._r2 ** (d - 1)

    def volume_estimate(self, n_monte_carlo: int = 10000,
                        rng: Optional[np.random.Generator] = None) -> float:
        """Monte Carlo volume estimate (for comparison with I_R).

        Parameters
        ----------
        n_monte_carlo : int
        rng : random Generator.

        Returns
        -------
        float
        """
        if rng is None:
            rng = np.random.default_rng(42)
        r1 = self.c_best / 2.0
        lo = self._center - r1
        hi = self._center + r1
        if self.bounds is not None:
            for k in range(self.dim):
                lo[k] = max(lo[k], self.bounds[k][0])
                hi[k] = min(hi[k], self.bounds[k][1])

        box_vol = float(np.prod(hi - lo))
        pts = rng.uniform(lo, hi, size=(n_monte_carlo, self.dim))
        inside = np.sum([self.is_member(p) for p in pts])
        return box_vol * inside / n_monte_carlo

    # ── visualisation ────────────────────────────────────────────────

    def visualize_2d(self, ax, resolution: int = 200,
                     color: str = 'gray', alpha: float = 0.15):
        """Fill the 2-D ellipsoid on a matplotlib axis.

        Parameters
        ----------
        ax : matplotlib Axes
        resolution : int
        color, alpha : appearance.
        """
        if self.dim != 2:
            return
        r1 = self.c_best / 2.0
        lo = self._center - r1
        hi = self._center + r1
        if self.bounds is not None:
            for k in range(2):
                lo[k] = max(lo[k], self.bounds[k][0])
                hi[k] = min(hi[k], self.bounds[k][1])
                # Add larger margin to keep visualization inside bounds
                margin = 0.1 * (self.bounds[k][1] - self.bounds[k][0])
                lo[k] = max(lo[k], self.bounds[k][0] + margin)
                hi[k] = min(hi[k], self.bounds[k][1] - margin)

        xs = np.linspace(lo[0], hi[0], resolution)
        ys = np.linspace(lo[1], hi[1], resolution)
        XX, YY = np.meshgrid(xs, ys)
        mask = np.zeros_like(XX, dtype=float)
        for i in range(resolution):
            for j in range(resolution):
                pt = np.array([XX[i, j], YY[i, j]])
                if self.is_member(pt):
                    # Also check bounds for visualization
                    in_bounds = True
                    if self.bounds is not None:
                        for k in range(2):
                            if pt[k] < self.bounds[k][0] or pt[k] > self.bounds[k][1]:
                                in_bounds = False
                                break
                    if in_bounds:
                        mask[i, j] = 1.0
        ax.contourf(XX, YY, mask, levels=[0.5, 1.5],
                    colors=[color], alpha=alpha)
