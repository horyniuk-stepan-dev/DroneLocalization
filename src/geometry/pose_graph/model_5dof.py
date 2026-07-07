"""5-DoF model: edge dataclass, state<->affine, forward/inverse prediction.

Single source of the pure geometric transforms used by the optimizer.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.geometry.affine_utils import decompose_affine_5dof


@dataclass
class GraphEdge:
    """Ребро графу між кадрами з відносним перетворенням."""

    from_id: int
    to_id: int
    dtx: float
    dty: float
    log_dsx: float
    log_dsy: float
    dtheta: float
    weight: float
    edge_type: str
    inliers: int = 0
    rmse: float = 0.0


def _affine_to_state(affine_2x3: np.ndarray, cx: float, cy: float) -> np.ndarray:
    tx, ty, sx, sy, angle = decompose_affine_5dof(affine_2x3)
    c_x = affine_2x3[0, 0] * cx + affine_2x3[0, 1] * cy + tx
    c_y = affine_2x3[1, 0] * cx + affine_2x3[1, 1] * cy + ty
    return np.array(
        [c_x, c_y, np.log(max(sx, 1e-9)), np.log(max(sy, 1e-9)), angle], dtype=np.float64
    )


def _state_to_affine(state: np.ndarray, cx: float, cy: float, sign: float = 1.0) -> np.ndarray:
    c_x, c_y, log_sx, log_sy, theta = state
    sx, sy = float(np.clip(np.exp(log_sx), 1e-6, 1e6)), float(np.clip(np.exp(log_sy), 1e-6, 1e6))
    c, s = np.cos(theta), np.sin(theta)

    M00, M01 = c * sx, -s * sign * sy
    M10, M11 = s * sx, c * sign * sy
    tx = c_x - (M00 * cx + M01 * cy)
    ty = c_y - (M10 * cx + M11 * cy)
    return np.array([[M00, M01, tx], [M10, M11, ty]], dtype=np.float64)


def _predicted_translation(tx_i, ty_i, sx_i, sy_i, cos_i, sin_i, dtx, dty, sign):
    """Predicted child-frame centre (tx_j, ty_j) under the 5-DoF model.

    Elementwise (scalars or numpy arrays). Single source of the
    rotate-scale-translate step shared by edge_residual and predict_forward/inverse.
    """
    pred_tx = tx_i + cos_i * sx_i * dtx - sign * sin_i * sy_i * dty
    pred_ty = ty_i + sin_i * sx_i * dtx + sign * cos_i * sy_i * dty
    return pred_tx, pred_ty


def edge_residual(state_i, state_j, dtx, dty, log_dsx, log_dsy, dtheta, weight, cx, sign):
    """Weighted 5-vector edge residual - THE single source of the residual formula.

    Broadcasts along the last axis, so the vectorized optimizer path
    (state_i/state_j shape (N, 5)) and the per-edge diagnostics path (shape (5,))
    share exactly one formula. Returns shape (..., 5).
    """
    tx_i, ty_i = state_i[..., 0], state_i[..., 1]
    log_sx_i, log_sy_i, theta_i = state_i[..., 2], state_i[..., 3], state_i[..., 4]
    tx_j, ty_j = state_j[..., 0], state_j[..., 1]
    log_sx_j, log_sy_j, theta_j = state_j[..., 2], state_j[..., 3], state_j[..., 4]

    sx_i, sy_i = np.exp(log_sx_i), np.exp(log_sy_i)
    cos_i, sin_i = np.cos(theta_i), np.sin(theta_i)
    pred_tx, pred_ty = _predicted_translation(
        tx_i, ty_i, sx_i, sy_i, cos_i, sin_i, dtx, dty, sign
    )

    angle_diff = theta_j - theta_i - sign * dtheta
    r0 = (weight / sx_i) * (tx_j - pred_tx)
    r1 = (weight / sy_i) * (ty_j - pred_ty)
    r2 = (weight * cx) * (log_sx_j - log_sx_i - log_dsx)
    r3 = (weight * cx) * (log_sy_j - log_sy_i - log_dsy)
    r4 = (weight * cx) * np.arctan2(np.sin(angle_diff), np.cos(angle_diff))
    return np.stack([r0, r1, r2, r3, r4], axis=-1)


def _predict_forward(state_i: np.ndarray, edge: GraphEdge, sign: float) -> np.ndarray:
    tx_i, ty_i, log_sx_i, log_sy_i, theta_i = state_i
    sx_i, sy_i = np.exp(log_sx_i), np.exp(log_sy_i)
    c_i, s_i = np.cos(theta_i), np.sin(theta_i)
    pred_tx, pred_ty = _predicted_translation(
        tx_i, ty_i, sx_i, sy_i, c_i, s_i, edge.dtx, edge.dty, sign
    )
    return np.array(
        [
            pred_tx,
            pred_ty,
            log_sx_i + edge.log_dsx,
            log_sy_i + edge.log_dsy,
            theta_i + sign * edge.dtheta,
        ],
        dtype=np.float64,
    )


def _predict_inverse(state_j: np.ndarray, edge: GraphEdge, sign: float) -> np.ndarray:
    tx_j, ty_j, log_sx_j, log_sy_j, theta_j = state_j
    inv_dsx, inv_dsy = 1.0 / np.exp(edge.log_dsx), 1.0 / np.exp(edge.log_dsy)
    inv_dtheta = -edge.dtheta
    cos_inv, sin_inv = np.cos(inv_dtheta), np.sin(inv_dtheta)

    inv_dtx = inv_dsx * (cos_inv * (-edge.dtx) - sin_inv * (-edge.dty))
    inv_dty = inv_dsy * (sin_inv * (-edge.dtx) + cos_inv * (-edge.dty))

    sx_j, sy_j = np.exp(log_sx_j), np.exp(log_sy_j)
    c_j, s_j = np.cos(theta_j), np.sin(theta_j)
    pred_tx, pred_ty = _predicted_translation(
        tx_j, ty_j, sx_j, sy_j, c_j, s_j, inv_dtx, inv_dty, sign
    )
    return np.array(
        [
            pred_tx,
            pred_ty,
            log_sx_j + np.log(inv_dsx),
            log_sy_j + np.log(inv_dsy),
            theta_j + sign * inv_dtheta,
        ],
        dtype=np.float64,
    )
