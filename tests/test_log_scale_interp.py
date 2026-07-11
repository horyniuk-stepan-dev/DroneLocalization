"""Тести log-scale інтерполяції масштабу (RESEARCH_INTEGRATION_PLAN 1.3).

Флаг graph_optimization.log_scale_interp (дефолт off). Перевіряємо:
- обидва режими точні у вузлах;
- log-режим дає геометричну (не арифметичну) середину масштабу;
- дефолт off не змінює поведінки.
"""

import numpy as np
import pytest

from src.geometry.affine_utils import (
    build_5dof_pchip,
    compose_affine_5dof,
    sample_5dof_pchip,
)

REF = (960.0, 540.0)


def _affine(scale: float, angle: float = 0.0, tx: float = 0.0, ty: float = 0.0):
    return compose_affine_5dof(tx, ty, scale, scale, angle, sign=-1.0)


@pytest.mark.parametrize("log_scale", [False, True])
def test_exact_at_nodes(log_scale):
    ids = [0.0, 10.0]
    affines = [_affine(1.0), _affine(4.0)]
    interp, sign, rng = build_5dof_pchip(ids, affines, REF, log_scale=log_scale)
    for fid, a in zip(ids, affines):
        M = sample_5dof_pchip(interp, sign, rng, REF, fid, log_scale=log_scale)
        np.testing.assert_allclose(M, a, atol=1e-9)


def test_midpoint_geometric_vs_arithmetic():
    """s1=1, s2=4: лінійна середина = 2.5, геодезична (log) = 2.0."""
    ids = [0.0, 10.0]
    affines = [_affine(1.0), _affine(4.0)]

    def mid_scale(log_scale):
        interp, sign, rng = build_5dof_pchip(ids, affines, REF, log_scale=log_scale)
        M = sample_5dof_pchip(interp, sign, rng, REF, 5.0, log_scale=log_scale)
        return float(np.linalg.norm(M[:2, 0]))

    assert mid_scale(False) == pytest.approx(2.5, abs=1e-6)
    assert mid_scale(True) == pytest.approx(2.0, abs=1e-6)


def test_log_scale_preserves_sign_and_angle():
    ids = [0.0, 10.0]
    affines = [_affine(1.0, angle=0.1), _affine(2.0, angle=0.3)]
    interp, sign, rng = build_5dof_pchip(ids, affines, REF, log_scale=True)
    M = sample_5dof_pchip(interp, sign, rng, REF, 5.0, log_scale=True)
    assert np.linalg.det(M[:2, :2]) < 0  # Y-flip збережено
    assert float(np.arctan2(M[1, 0], M[0, 0])) == pytest.approx(0.2, abs=1e-6)


def test_default_flag_off():
    from config import APP_CONFIG, get_cfg

    assert get_cfg(APP_CONFIG, "graph_optimization.log_scale_interp", None) is False
