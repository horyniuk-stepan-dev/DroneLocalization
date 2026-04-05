import numpy as np
import pytest

from src.geometry.affine_utils import (
    compose_affine_5dof,
    decompose_affine,
    decompose_affine_5dof,
    unwrap_angles,
)


def test_unwrap_angles():
    # Test wrapping around Pi
    angles = np.array([np.pi - 0.1, -np.pi + 0.1, np.pi - 0.2])
    unwrapped = unwrap_angles(angles)
    # The jump from pi - 0.1 to -pi + 0.1 is ~ -2*pi. Unwrapping should add 2*pi to the second to make it strictly continuous
    assert len(unwrapped) == 3
    assert abs(unwrapped[1] - (angles[1] + 2 * np.pi)) < 1e-5


def test_compose_decompose_affine_5dof():
    scale_x = 1.2
    scale_y = 0.8
    angle = np.pi / 4  # 45 degrees
    tx = 100.0
    ty = -50.0

    # Compose
    matrix = compose_affine_5dof(scale_x, scale_y, angle, tx, ty)
    assert matrix.shape == (2, 3)

    # Decompose
    tx_dec, ty_dec, sx_dec, sy_dec, ang_dec = decompose_affine_5dof(matrix)

    assert pytest.approx(scale_x, rel=1e-4) == sx_dec
    assert pytest.approx(scale_y, rel=1e-4) == sy_dec
    assert pytest.approx(angle, rel=1e-4) == ang_dec
    assert pytest.approx(tx, rel=1e-4) == tx_dec
    assert pytest.approx(ty, rel=1e-4) == ty_dec


def test_decompose_affine():
    # General affine matrix
    # A = [ [s_x * cos(a), -s_y * sin(a + shear)], ... ] etc., it's just linear algebra
    # We will test decompose_affine directly with an identity matrix
    matrix = np.array([[1.0, 0.0, 50.0], [0.0, 1.0, -50.0]])
    tx, ty, scale, angle = decompose_affine(matrix)

    assert scale == 1.0
    assert angle == 0.0
    assert tx == 50.0
    assert ty == -50.0
