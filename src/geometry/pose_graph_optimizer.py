"""Backward-compatible re-export.

Implementation moved to the :mod:`src.geometry.pose_graph` package
(see docs/IMPROVEMENT_PLAN.md, item 1.2). Import paths are preserved:
``from src.geometry.pose_graph_optimizer import PoseGraphOptimizer`` still works.
"""

from src.geometry.pose_graph.model_5dof import (
    GraphEdge,
    _affine_to_state,
    _predict_forward,
    _predict_inverse,
    _state_to_affine,
)
from src.geometry.pose_graph.optimizer import (
    PoseGraphOptimizer,
    affine_fit_residual,
    homography_to_affine,
    homography_to_similarity,
)

__all__ = [
    "PoseGraphOptimizer",
    "GraphEdge",
    "homography_to_affine",
    "affine_fit_residual",
    "homography_to_similarity",
    "_affine_to_state",
    "_state_to_affine",
    "_predict_forward",
    "_predict_inverse",
]
