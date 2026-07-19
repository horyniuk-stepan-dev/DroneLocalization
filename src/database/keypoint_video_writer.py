"""Debug keypoint-overlay rendering for DB builds (pure, no torch/self state).

Extracted verbatim from ``DatabaseBuilder._draw_keypoints_frame``
(IMPROVEMENT_PLAN п.1.3, splitting ``database_builder`` into modules). The
function draws detected keypoints, the YOLO dynamic-zone overlay, an info panel
and a legend onto a copy of the BGR frame — it depends only on its arguments,
so it is headless-testable and reusable by the optional keypoint-preview video.
"""

from __future__ import annotations

import cv2
import numpy as np


def draw_keypoints_frame(
    frame_bgr: np.ndarray,
    keypoints: np.ndarray,
    static_mask: np.ndarray,
    frame_id: int,
    total_frames: int,
) -> np.ndarray:
    """Return a copy of ``frame_bgr`` annotated with keypoints/mask/info panel.

    The input frame is not mutated. Logic preserved 1:1 with DatabaseBuilder.
    """
    vis = frame_bgr.copy()

    if static_mask is not None:
        dynamic_zone = static_mask == 0
        if dynamic_zone.any():
            overlay = vis.copy()
            overlay[dynamic_zone] = (0, 0, 200)
            cv2.addWeighted(overlay, 0.35, vis, 0.65, 0, vis)

    for x, y in keypoints:
        cx, cy = int(round(x)), int(round(y))
        cv2.circle(vis, (cx, cy), radius=3, color=(0, 255, 0), thickness=-1)
        cv2.circle(vis, (cx, cy), radius=4, color=(0, 180, 0), thickness=1)

    info_lines = [
        f"Frame: {frame_id:05d} / {total_frames:05d}",
        f"Keypoints: {len(keypoints)}",
        f"Dynamic mask: {'YES' if static_mask is not None else 'NO'}",
    ]
    panel_h = len(info_lines) * 28 + 14
    cv2.rectangle(vis, (0, 0), (340, panel_h), (0, 0, 0), -1)
    cv2.rectangle(vis, (0, 0), (340, panel_h), (80, 80, 80), 1)

    for idx, line in enumerate(info_lines):
        cv2.putText(
            vis,
            line,
            (8, 22 + idx * 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )

    legend_y = vis.shape[0] - 10
    cv2.circle(vis, (12, legend_y - 4), 5, (0, 255, 0), -1)
    cv2.putText(
        vis,
        "XFeat keypoint",
        (22, legend_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 255, 0),
        1,
        cv2.LINE_AA,
    )
    cv2.rectangle(vis, (200, legend_y - 10), (218, legend_y + 2), (0, 0, 200), -1)
    cv2.putText(
        vis,
        "YOLO dynamic zone",
        (224, legend_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 0, 200),
        1,
        cv2.LINE_AA,
    )

    return vis
