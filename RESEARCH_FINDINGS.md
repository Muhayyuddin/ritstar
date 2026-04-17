# Deep Research: Riemannian Ellipse Computation in RIT*

## Executive Summary

**CRITICAL ISSUE FOUND**: Riemannian informed sets (ellipses) are computed using an inaccurate first-order approximation for spatially-varying conformal metrics, while edge costs use proper numerical integration. This inconsistency affects the 2D obstacle demo and any environment using ObstacleInflatedMetric or other spatially-varying conformal metrics.

## Technical Analysis

### Background: What is the Riemannian Informed Set?

The Riemannian informed set is defined as:
```
I_R = {x ∈ C : d_R(x_s, x) + d_R(x, x_g) ≤ c_best}
```

where d_R is the **true Riemannian geodesic distance**.

### The Problem

For conformal metrics G(x) = s(x)·I with spatially-varying scale s(x):

**TRUE Riemannian distance** (along straight line):
```
d_R(x, y) = ∫₀¹ √s(γ(t)) ‖γ'(t)‖ dt
          = ∫₀¹ √s(x + t(y-x)) dt · ‖y - x‖
```

**CURRENT implementation** (geodesic_tier='diagonal'):
```
d_approx(x, y) = √s((x+y)/2) · ‖y - x‖
```

This evaluates s() at the MIDPOINT ONLY, not the integral!

### Error Analysis

For a metric that varies linearly from s₀ to s₁ along a segment:
- Midpoint approximation: √s_mid = √((s₀ + s₁)/2)
- True integral: ∫₀¹ √(s₀ + t(s₁-s₀)) dt ≠ √((s₀ + s₁)/2)

**Example**: s₀ = 1.0, s₁ = 9.0
- Midpoint: √5.0 ≈ 2.236
- True integral: (2/3)(9^(3/2) - 1^(3/2))/(9-1) ≈ 2.25
- Relative error: ~0.6%

But for more complex variations (Gaussian bumps in ObstacleInflatedMetric), errors can be much larger!

### Code Locations

**CORRECT implementation (for edge costs)**:
- File: `rit_star/metric_cache.py`
- Function: `edge_cost_l2()` [Lines 264-284]
  - Uses Simpson's rule (3-point integration) for conformal metrics
- Function: `edge_cost_exact()` [Lines 286-307]
  - Uses 10-point Gauss-Legendre quadrature for conformal metrics

**INCORRECT implementation (for informed sets)**:
- File: `rit_star/geodesic.py`
- Function: `diagonal_geodesic()` [Lines 55-73]
  - Uses MIDPOINT ONLY for all metrics
- Used by: `GeodesicComputer.distance()` when tier='diagonal'
- Used in: `RiemannianInformedSet.is_member()` for membership testing

**Affected files**:
- `run_2d_obstacle_demo.py` [Line 32]: uses geodesic_tier='diagonal'
- Any planner using ObstacleInflatedMetric with tier='diagonal'

### Why This Matters

1. **Incorrect membership tests**: Points may be wrongly included/excluded from I_R
2. **Wrong volume estimates**: Monte Carlo volume estimation uses is_member()
3. **Sampling inefficiency**: Rejection sampling draws from wrong region
4. **Theorem validation**: Volume ratio calculations (Theorem 1) may show incorrect results
5. **Paper claims**: Convergence improvements may be overstated

### The Inconsistency Pattern

```
Edge costs (tree building):     ✓ Proper integration
Geodesic distance (heuristic):  ✓ Okay (admissible lower bound)
Informed set membership:        ✗ Midpoint approximation
Volume calculations:            ✗ Based on incorrect membership
```

## Recommended Fixes

### Option 1: Add Conformal-Aware Geodesic Tier (RECOMMENDED)

Create a new geodesic computation that properly integrates conformal metrics:

```python
def conformal_geodesic(x: np.ndarray, y: np.ndarray, 
                       metric: RiemannianMetric,
                       n_quad: int = 5) -> float:
    """Accurate geodesic for conformal metrics G(x) = s(x)·I.
    
    For conformal metrics, geodesics are straight lines, so:
    d_R(x,y) = ∫₀¹ √s(x + t(y-x)) dt · ‖y - x‖
    """
    from .metric import ObstacleInflatedMetric, PathwayMetric, ClearanceMetric
    
    diff = y - x
    euclid_dist = float(np.sqrt(diff @ diff))
    
    if euclid_dist < 1e-12:
        return 0.0
    
    # Check if metric is conformal (all eigenvalues equal)
    if not (isinstance(metric, (ObstacleInflatedMetric, PathwayMetric, ClearanceMetric)) 
            or metric.condition_number(0.5*(x+y)) < 1.01):
        # Not conformal, fall back to diagonal approximation
        mid = 0.5 * (x + y)
        Gm = metric.G(mid)
        return float(np.sqrt(max(diff @ Gm @ diff, 0.0)))
    
    # Conformal metric: integrate scale factor
    # Use Gauss-Legendre quadrature
    nodes_5 = np.array([0.0, 0.5 - np.sqrt(5+2*np.sqrt(10/7))/6,
                        0.5, 0.5 + np.sqrt(5+2*np.sqrt(10/7))/6, 1.0])
    weights_5 = np.array([0.5 * w for w in [0.2369268850561891, 0.4786286704993665,
                          0.5688888888888889, 0.4786286704993665, 0.2369268850561891]])
    
    integral = 0.0
    for t, w in zip(nodes_5, weights_5):
        pt = x + t * diff
        # For conformal G = s(x)·I, eigenvalues are all s(x)
        eigs = metric.eigenvalues(pt)
        s = eigs[0]  # all eigenvalues equal for conformal
        integral += w * np.sqrt(s)
    
    return float(integral * euclid_dist)
```

### Option 2: Use MetricFieldCache for Geodesic Distance

Leverage the existing accurate edge cost computation:

```python
class GeodesicComputer:
    def __init__(self, metric, tier='diagonal', bounds=None, 
                 metric_cache=None):
        self.metric = metric
        self.tier = tier
        self._mc = metric_cache  # Pass in the metric cache
        
    def distance(self, x, y):
        if self.tier == 'euclidean':
            return euclidean_distance(x, y)
        elif self.tier == 'diagonal':
            # Use cached L2 if available for conformal metrics
            if self._mc is not None and self._mc._is_conformal:
                return self._mc.edge_cost_l2(x, y)
            return diagonal_geodesic(x, y, self.metric)
        # ... rest of tiers
```

### Option 3: Document the Approximation

At minimum, add clear documentation that:
1. `geodesic_tier='diagonal'` is only accurate for constant metrics
2. For conformal metrics, it's a first-order approximation
3. Recommend which tier to use for each metric type

## Testing the Fix

### Validation Test

Create a test to measure the error:

```python
def test_conformal_geodesic_accuracy():
    """Compare midpoint vs integrated geodesic for conformal metrics."""
    from rit_star.metric import ObstacleInflatedMetric
    import numpy as np
    
    # Create metric with significant spatial variation
    obstacles = np.array([[0.5, 0.5]])
    metric = ObstacleInflatedMetric(obstacles, sigma=0.12, alpha=8.0)
    
    # Test points: one far from obstacle, one near
    x = np.array([0.1, 0.1])
    y = np.array([0.45, 0.45])  # Approaching obstacle
    
    # Midpoint approximation (current)
    from rit_star.geodesic import diagonal_geodesic
    d_approx = diagonal_geodesic(x, y, metric)
    
    # Accurate integration (10-point Gauss-Legendre)
    from rit_star.rit_star import riemannian_edge_cost
    d_true = riemannian_edge_cost(x, y, metric, n_quad=10)
    
    error = abs(d_true - d_approx) / d_true * 100
    print(f"Midpoint approx: {d_approx:.6f}")
    print(f"True integral:   {d_true:.6f}")
    print(f"Relative error:  {error:.2f}%")
    
    assert error < 5.0, f"Error {error}% exceeds 5% threshold"
```

### Demo Comparison

Compare informed set shapes with corrected vs incorrect geodesic:

```python
# Run with both methods and visualize difference
gc_approx = GeodesicComputer(metric, tier='diagonal')
gc_accurate = GeodesicComputer(metric, tier='conformal')  # NEW

ris_approx = RiemannianInformedSet(xs, xg, c_best, gc_approx, bounds)
ris_accurate = RiemannianInformedSet(xs, xg, c_best, gc_accurate, bounds)

# Compare volumes
vol_approx = ris_approx.volume_estimate(10000)
vol_accurate = ris_accurate.volume_estimate(10000)
print(f"Volume difference: {abs(vol_approx - vol_accurate)/vol_accurate * 100:.1f}%")
```

## Impact on Paper Results

### Potentially Affected Claims

1. **Volume ratio reduction** (Theorem 1 validation)
   - Current measurements use incorrect I_R
   - True reduction might be different

2. **Sample efficiency** (Theorem 3)
   - Based on volume ratios
   - Needs recalculation with correct geodesics

3. **Convergence rate comparisons**
   - If baseline uses correct Euclidean ellipse but RIT* uses incorrect Riemannian ellipse

### What Needs Verification

1. Re-run all experiments with corrected geodesic computation
2. Verify Theorem 1 bound still holds
3. Check if convergence improvements are maintained
4. Validate that I_R ⊂ I_E still holds empirically

## Next Steps

1. **Implement** conformal geodesic integration (Option 1 or 2)
2. **Test** accuracy improvement with validation script
3. **Re-run** 2D obstacle demo with corrected implementation
4. **Compare** results (volume ratios, sampling efficiency, etc.)
5. **Update** other environments if they use conformal metrics
6. **Document** which geodesic tier to use for each metric type

## References

### Key Code Sections

- `rit_star/geodesic.py` [Lines 55-73]: diagonal_geodesic (needs fix)
- `rit_star/informed_set.py` [Lines 98-116]: RiemannianInformedSet.is_member
- `rit_star/metric_cache.py` [Lines 274-284]: edge_cost_l2 (correct pattern)
- `rit_star/rit_star.py` [Lines 44-92]: riemannian_edge_cost (correct pattern)
- `run_2d_obstacle_demo.py` [Line 32]: Uses geodesic_tier='diagonal'

### Related Memory Files

- `/memories/repo/rit_star_code_mapping.md` [Lines 44-68]: Geodesic tier documentation
- `/memories/repo/rit_star_code_mapping.md` [Lines 194-237]: Informed set implementation

---

**Conclusion**: The Riemannian ellipse (informed set) computation is **mathematically inconsistent** for spatially-varying conformal metrics. While edge costs use proper numerical integration, geodesic distances for informed sets use a crude midpoint approximation. This needs to be fixed for correct implementation of the RIT* algorithm.
