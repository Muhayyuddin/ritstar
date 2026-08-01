"""RIT* — Riemannian Informed Trees for Cost-Adaptive Optimal Motion Planning.

Reference implementation accompanying the IEEE RA-L paper:

    M. Ud Din, A. Nadar, J. Rosell, I. Hussain,
    "RIT*: Riemannian Informed Trees for Cost-Adaptive Optimal Motion Planning",
    IEEE Robotics and Automation Letters, 2026.

Public API
----------
Planners
    RITStar, InformedRRTStar, GeometryAwareRRTStar,
    BITStar, AITStar, EITStar, APTStar

Metrics
    RiemannianMetric, EuclideanMetric, DiagonalAnisotropicMetric,
    ObstacleInflatedMetric, JointInertiaMetric2D, CollisionAdaptiveMetric

Informed sets
    RiemannianInformedSet, EuclideanInformedSet, volume_ratio_bound

Geodesics
    GeodesicComputer, diagonal_geodesic, midpoint_geodesic_distance,
    riemannian_edge_cost

Caching
    MetricFieldCache
"""

from __future__ import annotations

__version__ = "1.0.0"

from .rit_star import RITStar, riemannian_edge_cost
from .baselines import (
    InformedRRTStar,
    GeometryAwareRRTStar,
    BITStar,
    AITStar,
    EITStar,
    APTStar,
)
from .metric import (
    RiemannianMetric,
    EuclideanMetric,
    DiagonalAnisotropicMetric,
    ObstacleInflatedMetric,
    JointInertiaMetric2D,
    CollisionAdaptiveMetric,
)
from .informed_set import (
    RiemannianInformedSet,
    EuclideanInformedSet,
    volume_ratio_bound,
)
from .geodesic import (
    GeodesicComputer,
    diagonal_geodesic,
    midpoint_geodesic_distance,
)
from .metric_cache import MetricFieldCache

__all__ = [
    "__version__",
    # Planners
    "RITStar",
    "InformedRRTStar",
    "GeometryAwareRRTStar",
    "BITStar",
    "AITStar",
    "EITStar",
    "APTStar",
    # Metrics
    "RiemannianMetric",
    "EuclideanMetric",
    "DiagonalAnisotropicMetric",
    "ObstacleInflatedMetric",
    "JointInertiaMetric2D",
    "CollisionAdaptiveMetric",
    # Informed sets
    "RiemannianInformedSet",
    "EuclideanInformedSet",
    "volume_ratio_bound",
    # Geodesics
    "GeodesicComputer",
    "diagonal_geodesic",
    "midpoint_geodesic_distance",
    "riemannian_edge_cost",
    # Cache
    "MetricFieldCache",
]
