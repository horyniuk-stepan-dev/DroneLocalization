"""ScaleManager — GSD-ratio estimation & tracking (query altitude / DB altitude).

Mirrors the angular prior logic (A2/A3 in IMPROVEMENT_PLAN) for scale:
- **Temporal prior**: after each successful localization, extract scale from
  the homography → EMA-smoothed → reuse on the next keyframe (1 forward pass).
- **Pyramid scan**: when no prior exists (bootstrap, out-of-coverage, score
  gate failure) → scan a discrete set of scales [0.5, 0.7, 1.0, 1.4, 2.0].
- **Depth hint**: if DepthAnythingV2 scales are available for both query and
  DB frames, use the ratio to narrow the pyramid search.
- **Frame normalization**: r > 1 (higher than DB) → center-crop + upscale;
  r < 1 (lower) → downscale.

Reference: docs/SCALE_INVARIANCE.md §1–4.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from config import get_cfg
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

# Default octave grid around the DB altitude — covers ~50–200 m if DB is ~100 m
_DEFAULT_PYRAMID = (0.5, 0.7, 1.0, 1.4, 2.0)


@dataclass
class CropInfo:
    """Metadata needed to reverse-map coordinates after normalize()."""

    scale_r: float
    """The GSD ratio that was applied (query_altitude / db_altitude)."""

    crop_x: int
    crop_y: int
    """Top-left corner of the crop in the *original* (resolution-normalised) frame."""

    crop_w: int
    crop_h: int
    """Size of the crop before upscale (0 if no crop was applied)."""

    resize_scale: float
    """The actual resize factor applied to the crop (may differ from r due to rounding)."""


def crop_to_affine(
    crop_info: CropInfo, norm_w: int, norm_h: int, inverse: bool = False
) -> np.ndarray:
    """3x3 affine mapping pre-normalize frame coords -> normalized frame coords.

    ``normalize()`` applies (center-crop -> resize); for a point ``p_o`` in the
    pre-normalize (rotated) frame the normalized coords are ``p_n = S @ (p_o - c)``
    with ``S = diag(norm_w / crop_w, norm_h / crop_h)`` and ``c = (crop_x, crop_y)``.
    The per-axis scales are derived from the *actual* crop/output sizes, so the
    integer-rounding aspect drift of ``resize_scale`` is not accumulated.

    Args:
        crop_info: CropInfo returned by ``normalize()``.
        norm_w: width of the normalized frame (``frame.shape[1]`` after normalize).
        norm_h: height of the normalized frame.
        inverse: if True, return the exact analytic inverse (normalized -> original).

    Returns:
        (3, 3) float64 matrix. Degenerate crop sizes (<= 0) yield the identity.
    """
    if crop_info.crop_w <= 0 or crop_info.crop_h <= 0:
        return np.eye(3, dtype=np.float64)
    sx = norm_w / crop_info.crop_w
    sy = norm_h / crop_info.crop_h
    if inverse:
        return np.array(
            [
                [1.0 / sx, 0.0, float(crop_info.crop_x)],
                [0.0, 1.0 / sy, float(crop_info.crop_y)],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
    return np.array(
        [
            [sx, 0.0, -crop_info.crop_x * sx],
            [0.0, sy, -crop_info.crop_y * sy],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


class ScaleManager:
    """GSD-ratio estimation & tracking (query altitude / DB altitude) without telemetry.

    Designed as a stateless-ish collaborator (owns only the EMA prior);
    the caller (Localizer) passes frames and receives CropInfo back.
    """

    def __init__(self, config: dict | None = None):
        cfg = config or {}
        self._pyramid: tuple[float, ...] = tuple(
            get_cfg(cfg, "localization.scale_pyramid", list(_DEFAULT_PYRAMID))
        )
        self._ema_alpha: float = get_cfg(cfg, "localization.scale_prior_ema", 0.7)
        self._rescan_min: float = get_cfg(
            cfg, "localization.scale_rescan_min_score", 0.65
        )
        self._use_depth_hint: bool = get_cfg(
            cfg, "localization.scale_use_depth_hint", True
        )
        # Clipping range for the prior (prevents runaway EMA)
        self._min_r: float = 0.3
        self._max_r: float = 3.5

        # ── mutable state ──
        self._prior: float | None = None
        self._depth_hint: float | None = None  # set externally per-keyframe

        logger.info(
            f"ScaleManager initialized | pyramid={self._pyramid}, "
            f"ema_alpha={self._ema_alpha}, rescan_min={self._rescan_min}"
        )

    # ── public API ──────────────────────────────────────────────────────────

    @property
    def prior(self) -> float | None:
        """Current scale prior (EMA-smoothed r), or None if unknown."""
        return self._prior

    @property
    def rescan_min_score(self) -> float:
        return self._rescan_min

    def candidates(self) -> list[float]:
        """Scale levels to scan on the next keyframe.

        If prior is valid → [prior] (1 level, combined with rotation prior
        = 1 forward pass).  Otherwise → full pyramid (5 levels × N rotations
        = batched forward).

        When a depth hint is available and no prior, the pyramid is reordered
        so the closest level to the hint comes first (early-stop friendly).
        """
        if self._prior is not None:
            return [self._prior]

        pyramid = list(self._pyramid)

        if self._depth_hint is not None and self._use_depth_hint:
            # Sort pyramid by distance to depth hint — closest first
            hint = self._depth_hint
            pyramid.sort(key=lambda r: abs(r - hint))
            logger.debug(
                f"Scale pyramid reordered by depth hint {hint:.2f}: {pyramid}"
            )

        return pyramid

    def normalize(self, frame: np.ndarray, r: float) -> tuple[np.ndarray, CropInfo]:
        """Normalize *frame* to approximate the DB's GSD given scale ratio *r*.

        Args:
            frame: (H, W, 3) uint8 — the resolution-normalised query frame.
            r: estimated GSD ratio (query_altitude / db_altitude).
               r > 1 → flying HIGHER than DB → crop center + upscale.
               r < 1 → flying LOWER  than DB → downscale.
               r ≈ 1 → no-op.

        Returns:
            (normalised_frame, CropInfo) — CropInfo allows the caller to
            reverse-map pixel coordinates back to the original frame.
        """
        h, w = frame.shape[:2]

        # Tolerance band — no transform needed
        if 0.85 <= r <= 1.18:
            return frame, CropInfo(
                scale_r=r, crop_x=0, crop_y=0,
                crop_w=w, crop_h=h, resize_scale=1.0,
            )

        if r > 1.0:
            # Flying HIGHER: center-crop (side = 1/r) + upscale to original size
            # This simulates what the camera would see at the DB altitude.
            crop_side_ratio = 1.0 / r
            crop_w = max(32, int(w * crop_side_ratio))
            crop_h = max(32, int(h * crop_side_ratio))

            cx, cy = w // 2, h // 2
            x1 = max(0, cx - crop_w // 2)
            y1 = max(0, cy - crop_h // 2)
            x2 = min(w, x1 + crop_w)
            y2 = min(h, y1 + crop_h)

            cropped = frame[y1:y2, x1:x2]
            # Upscale back to original resolution
            normalised = cv2.resize(cropped, (w, h), interpolation=cv2.INTER_CUBIC)
            actual_scale = w / (x2 - x1)

            return normalised, CropInfo(
                scale_r=r, crop_x=x1, crop_y=y1,
                crop_w=x2 - x1, crop_h=y2 - y1,
                resize_scale=actual_scale,
            )
        else:
            # Flying LOWER: downscale the frame to match DB's GSD
            # The query covers a smaller area — fewer pixels is correct.
            new_w = max(32, int(w * r))
            new_h = max(32, int(h * r))
            downscaled = cv2.resize(
                frame, (new_w, new_h), interpolation=cv2.INTER_AREA
            )
            actual_scale = new_w / w

            return downscaled, CropInfo(
                scale_r=r, crop_x=0, crop_y=0,
                crop_w=w, crop_h=h,
                resize_scale=actual_scale,
            )

    def reverse_center(
        self, center_norm: np.ndarray, crop_info: CropInfo
    ) -> np.ndarray:
        """Map a point from the normalised frame back to the original frame coords.

        Args:
            center_norm: (1, 2) point in the normalised (cropped/resized) frame.
            crop_info: CropInfo from the normalize() call.

        Returns:
            (1, 2) point in the original (pre-normalize) resolution-normalised frame.
        """
        x, y = float(center_norm[0, 0]), float(center_norm[0, 1])

        if crop_info.resize_scale == 1.0:
            return center_norm

        if crop_info.scale_r > 1.0:
            # Undo upscale → undo crop offset
            x_crop = x / crop_info.resize_scale
            y_crop = y / crop_info.resize_scale
            x_orig = x_crop + crop_info.crop_x
            y_orig = y_crop + crop_info.crop_y
        else:
            # Undo downscale
            x_orig = x / crop_info.resize_scale
            y_orig = y / crop_info.resize_scale

        return np.array([[x_orig, y_orig]], dtype=np.float64)

    def update_from_homography(
        self, H: np.ndarray, frame_w: int, frame_h: int
    ) -> None:
        """Extract scale from a successful homography and update the EMA prior.

        Uses homography_to_affine → decompose_affine_5dof → sqrt(sx * sy).
        """
        try:
            from src.geometry.affine_utils import decompose_affine_5dof
            from src.geometry.pose_graph.optimizer import homography_to_affine

            M = homography_to_affine(H, frame_w, frame_h)
            if M is None:
                return

            _tx, _ty, sx, sy, _angle = decompose_affine_5dof(M)
            r_measured = float(np.sqrt(abs(sx) * abs(sy)))

            # Sanity check
            if not (self._min_r <= r_measured <= self._max_r):
                logger.debug(
                    f"ScaleManager: measured r={r_measured:.3f} out of "
                    f"[{self._min_r}, {self._max_r}] — ignoring"
                )
                return

            if self._prior is None:
                self._prior = r_measured
            else:
                self._prior = (
                    self._ema_alpha * r_measured
                    + (1.0 - self._ema_alpha) * self._prior
                )
                self._prior = float(
                    np.clip(self._prior, self._min_r, self._max_r)
                )

            logger.debug(
                f"ScaleManager: r_measured={r_measured:.3f}, "
                f"prior={self._prior:.3f}"
            )
        except Exception as e:
            logger.warning(f"ScaleManager.update_from_homography failed: {e}")

    def set_depth_hint(
        self, query_depth_scale: float, db_depth_scale: float
    ) -> None:
        """Set a depth-based scale hint for pyramid reordering.

        Args:
            query_depth_scale: 1/median_depth for the query frame.
            db_depth_scale: median 1/median_depth across DB frames.
        """
        if db_depth_scale < 1e-8:
            return
        ratio = query_depth_scale / db_depth_scale
        self._depth_hint = float(np.clip(ratio, self._min_r, self._max_r))
        logger.debug(
            f"ScaleManager depth hint set: query={query_depth_scale:.4f}, "
            f"db={db_depth_scale:.4f}, ratio={self._depth_hint:.3f}"
        )

    def invalidate(self) -> None:
        """Reset the prior (out-of-coverage, too many failures, etc.)."""
        if self._prior is not None:
            logger.debug(f"ScaleManager: prior invalidated (was {self._prior:.3f})")
        self._prior = None
        self._depth_hint = None

    def reset(self) -> None:
        """Full reset for a new tracking session."""
        self._prior = None
        self._depth_hint = None
