"""Continuous in-plane rotation of the query frame (localization.rotation_mode).

Why
---
The legacy path only tries the four quarter turns (np.rot90). A query whose
heading differs from the reference frames by anything else reaches ALIKED and
LightGlue with up to ±45° residual rotation. Neither model is rotation
invariant. Measured on FlightSimulator/test_frame.png (same frame, rotated, so
the best case): 1638 inliers at 0°, 702 at 30°, 362 at 45°, 6 at 135°.
np.rot90 also swaps width and height, so a 90° query reached DINO squashed
along the other axis than the 16:9 reference frames.

What this module does
---------------------
* ``rotate_view`` rotates the frame about its centre into a canvas of the SAME
  size (cv2.warpAffine). The view keeps the reference frame geometry, so the
  DINO input is squashed exactly like the database frames for every angle. The
  pixels that fall outside the original frame are marked invalid. ALIKED
  keypoints there are dropped, and VLAD ignores the tokens there.
* ``rotation_matrix`` is the same mapping as a 3×3 transform from original-frame
  pixels to view pixels. The Localizer folds it into the query→reference
  homography, so everything downstream (centre, FOV, optical flow, object
  projection) works in ORIGINAL frame coordinates, whatever the angle.
* ``next_prior_deg`` turns the residual rotation of a verified homography into
  the angle to apply to the next keyframe. In steady flight the next query is
  then aligned to within the heading change between two keyframes.

Angle convention: cv2.getRotationMatrix2D, positive = counter-clockwise on
screen (the same sense as np.rot90 with k > 0), degrees in [0, 360).
"""

from __future__ import annotations

import cv2
import numpy as np

QUARTER = "quarter"
CONTINUOUS = "continuous"


def is_continuous(config) -> bool:
    from config import get_cfg

    return str(get_cfg(config, "localization.rotation_mode", QUARTER)) == CONTINUOUS


def norm_deg(angle: float) -> float:
    """Wrap to [0, 360)."""
    a = float(angle) % 360.0
    return 0.0 if abs(a - 360.0) < 1e-9 else a


def scan_angles(step_deg: float) -> list[float]:
    """Bootstrap / recovery angles: 0, step, 2*step, ... < 360."""
    step = float(step_deg)
    if not np.isfinite(step) or step <= 0 or step >= 360:
        return [0.0]
    n = max(1, int(round(360.0 / step)))
    return [norm_deg(i * 360.0 / n) for i in range(n)]


def rotation_matrix(angle_deg: float, width: int, height: int) -> np.ndarray:
    """3×3 transform original-frame pixels → same-size rotated view pixels."""
    m = cv2.getRotationMatrix2D((width / 2.0, height / 2.0), float(angle_deg), 1.0)
    return np.vstack([m, [0.0, 0.0, 1.0]]).astype(np.float64)


def rotate_view(
    image: np.ndarray,
    angle_deg: float,
    mask: np.ndarray | None = None,
    erode_px: int = 8,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray]:
    """Rotate ``image`` (and an optional 0/255 ``mask``) about the centre.

    Returns ``(view, view_mask, valid)``. ``valid`` is a uint8 0/255 map of the
    view pixels that come from the original frame, eroded by ``erode_px`` so
    keypoints on the artificial black border are rejected. ``view_mask`` is the
    rotated ``mask`` AND ``valid`` (or just ``valid`` when ``mask`` is None).
    Angle 0 returns the inputs unchanged (no resampling).
    """
    h, w = image.shape[:2]
    a = norm_deg(angle_deg)
    if a == 0.0:
        valid = np.full((h, w), 255, dtype=np.uint8)
        return image, (mask if mask is not None else None), valid
    m = rotation_matrix(a, w, h)[:2]
    view = cv2.warpAffine(
        image, m, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0
    )
    ones = np.full((h, w), 255, dtype=np.uint8)
    valid = cv2.warpAffine(
        ones, m, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0
    )
    if erode_px > 0:
        k = 2 * int(erode_px) + 1
        valid = cv2.erode(valid, np.ones((k, k), dtype=np.uint8))
    if mask is not None:
        rotated_mask = cv2.warpAffine(
            mask, m, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0
        )
        view_mask = np.where((rotated_mask > 128) & (valid > 128), 255, 0).astype(np.uint8)
    else:
        view_mask = valid
    return view, view_mask, valid


def residual_angle_deg(H: np.ndarray) -> float:
    """In-plane rotation of the linear part of ``H`` (view → reference), degrees.

    Uses the closest rotation of the 2×2 block (atan2(a10 - a01, a00 + a11)),
    so moderate anisotropic scale and mild perspective barely bias it. The
    sign is that of the rotation's [1, 0] entry in y-down pixel coordinates.
    """
    a = np.asarray(H, dtype=np.float64)[:2, :2]
    # closest rotation: atan2(a10 - a01, a00 + a11)
    return float(np.degrees(np.arctan2(a[1, 0] - a[0, 1], a[0, 0] + a[1, 1])))


def next_prior_deg(applied_deg: float, H_view_to_ref: np.ndarray) -> float:
    """Angle to apply next so the view would have matched with zero residual.

    View = R(applied)·Q with cv2's matrix = Rot(-applied) in pixel coordinates;
    Ref ≈ Rot(phi)·View ⇒ Ref ≈ Rot(phi - applied)·Q ⇒ apply (applied - phi).
    """
    return norm_deg(float(applied_deg) - residual_angle_deg(H_view_to_ref))


def angle_distance_deg(a: float, b: float) -> float:
    d = abs(norm_deg(a) - norm_deg(b))
    return min(d, 360.0 - d)


def view_mask(
    shape: tuple[int, int],
    angle_deg: float,
    mask: np.ndarray | None = None,
    erode_px: int = 8,
) -> np.ndarray | None:
    """Keypoint mask of a rotated view without re-warping the image.

    ``shape`` is (height, width) of the ORIGINAL frame. Returns the rotated
    ``mask`` AND the eroded valid area (0/255); None when angle is 0 and there
    is no mask (nothing to filter).
    """
    h, w = int(shape[0]), int(shape[1])
    a = norm_deg(angle_deg)
    if a == 0.0:
        return mask
    m = rotation_matrix(a, w, h)[:2]
    valid = cv2.warpAffine(
        np.full((h, w), 255, dtype=np.uint8),
        m,
        (w, h),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    if erode_px > 0:
        k = 2 * int(erode_px) + 1
        valid = cv2.erode(valid, np.ones((k, k), dtype=np.uint8))
    if mask is None:
        return valid
    rotated = cv2.warpAffine(
        mask, m, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0
    )
    return np.where((rotated > 128) & (valid > 128), 255, 0).astype(np.uint8)
