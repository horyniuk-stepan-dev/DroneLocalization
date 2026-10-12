"""Full image-to-map transforms and explicit affine compatibility views."""

import numpy as np


def as_homography(matrix):
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.shape == (2, 3):
        matrix = np.vstack([matrix, [0.0, 0.0, 1.0]])
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
        raise ValueError("Expected a finite affine or homography")
    return matrix


def project_points(matrix, points):
    points = np.asarray(points, dtype=np.float64)
    H = as_homography(matrix)
    scale = np.max(np.abs(H[2]))
    if scale == 0:
        raise ValueError("Singular projective transform")
    projected = np.column_stack([points, np.ones(len(points))]) @ (H / scale).T
    if np.any(np.abs(projected[:, 2]) < 1e-12):
        raise ValueError("Projective horizon at an image point")
    result = projected[:, :2] / projected[:, 2:3]
    if not np.isfinite(result).all():
        raise ValueError("Non-finite projected points")
    return result


def project_frame_points(matrix, points, width, height, *, min_denominator_ratio=0.2):
    """Project within the reference frame's finite projective chart.

    Query centers/FOV can lie outside the reference image. Reject points near or
    across its horizon instead of returning mirrored coordinates or inventing a
    geographic measurement from an affine approximation.
    """
    H = as_homography(matrix)
    pts = np.asarray(points, dtype=np.float64)
    reference = H[2] @ [width / 2, height / 2, 1.0]
    if reference == 0 or not np.isfinite(reference):
        raise ValueError("Invalid reference projective chart")
    ratio = (np.column_stack([pts, np.ones(len(pts))]) @ H[2]) / reference
    if not np.isfinite(ratio).all() or np.any(ratio < min_denominator_ratio):
        raise ValueError("Point outside safe reference projective chart")
    return project_points(H, pts)


def valid_footprint(matrix, width, height, *, orientation=None):
    """No horizon, collapse or reflection relative to the requested orientation."""
    try:
        H = as_homography(matrix)
        row_scale = np.max(np.abs(H[2]))
        if row_scale == 0 or min(width, height) <= 0:
            return False
        H = H / row_scale
        corners = np.array([[0, 0, 1], [width, 0, 1], [width, height, 1], [0, height, 1]])
        denominators = corners @ H[2]
        scale = np.max(np.abs(denominators))
        if scale <= 1e-12 or not (
            np.all(denominators > scale * 1e-6) or np.all(denominators < -scale * 1e-6)
        ):
            return False
        H = H / scale
        det = np.linalg.det(H)
        if not np.isfinite(det) or abs(det) < 1e-15:
            return False
        return orientation is None or np.sign(det) * np.sign(denominators[0]) == orientation
    except (ValueError, np.linalg.LinAlgError):
        return False


def affine_at_center(matrix, width, height):
    """Tangent affine: exact center and local Jacobian, not a replacement for H."""
    H = as_homography(matrix)
    center = np.array([width / 2, height / 2])
    value = project_points(H, center[None])[0]
    denominator = H[2, :2] @ center + H[2, 2]
    jacobian = (H[:2, :2] - value[:, None] * H[2, :2]) / denominator
    return np.column_stack([jacobian, value - jacobian @ center])


def get_frame_transform(database, frame_id):
    """Use full geometry when present; support old databases and lightweight adapters."""
    getter = getattr(database, "get_frame_affine", None)
    if callable(getter):
        affine = getter(frame_id)
    else:
        matrices = getattr(database, "frame_affine", None)
        valid = getattr(database, "frame_valid", None)
        affine = (
            matrices[frame_id]
            if matrices is not None
            and 0 <= frame_id < len(matrices)
            and (valid is None or valid[frame_id])
            else None
        )
    if affine is None:
        return None
    matrices = getattr(database, "frame_homography", None)
    if isinstance(matrices, np.ndarray) and 0 <= frame_id < len(matrices):
        return matrices[frame_id]
    return affine
