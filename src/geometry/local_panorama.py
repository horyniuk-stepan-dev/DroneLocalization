"""Render reference video frames in the coordinate system of a local map."""

import hashlib
import json
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class LocalPanoramaLayout:
    """An immutable-in-use snapshot; background rendering never reads a live DB."""

    affines: dict[int, np.ndarray]
    frame_width: int
    frame_height: int
    frame_step: int
    map_to_canvas: np.ndarray
    canvas_size: tuple[int, int]
    corners_yx: list[list[float]]
    map_signature: str

    @classmethod
    def from_database(cls, database, max_size=2048):
        if getattr(getattr(database, "converter", None), "mode", "") != "LOCAL":
            raise ValueError("A local panorama requires a LOCAL map")
        w = int(database.metadata.get("frame_width", 0))
        h = int(database.metadata.get("frame_height", 0))
        step = int(database.metadata.get("frame_step", 0))
        if min(w, h, step, max_size) <= 0:
            raise ValueError("Database is missing frame dimensions or frame_step; rebuild it")
        if database.frame_valid is None or database.frame_affine is None:
            raise ValueError("Build the local map before creating its panorama")
        affines = {
            int(i): np.array(database.frame_affine[i], dtype=np.float64, copy=True)
            for i in np.flatnonzero(database.frame_valid)
        }
        if not affines:
            raise ValueError("The local map has no connected frames")
        digest = hashlib.sha256()
        digest.update(np.asarray([w, h, step], dtype="<i8").tobytes())
        for fid, matrix in affines.items():
            digest.update(np.asarray([fid], dtype="<i8").tobytes())
            digest.update(np.asarray(matrix, dtype="<f8").tobytes())
        # Include the recorded video identity when available.
        digest.update(
            str(database.metadata.get("source_sha256") or getattr(database, "db_path", "")).encode()
        )
        corners = np.array([[0, 0, 1], [w, 0, 1], [w, h, 1], [0, h, 1]])
        footprints = np.concatenate([corners @ a.T for a in affines.values()])
        if not np.isfinite(footprints).all():
            raise ValueError("Non-finite local map geometry")
        lo, hi = footprints.min(axis=0), footprints.max(axis=0)
        span = hi - lo
        if span.min() <= 1e-9:
            raise ValueError("Degenerate local map geometry")
        scale = max_size / float(span.max())
        width, height = np.maximum(1, np.ceil(span * scale).astype(int))
        transform = np.array([[scale, 0, -lo[0] * scale], [0, -scale, hi[1] * scale], [0, 0, 1]])
        right, bottom = lo[0] + width / scale, hi[1] - height / scale
        yx = [[hi[1], lo[0]], [hi[1], right], [bottom, right], [bottom, lo[0]]]
        return cls(
            affines,
            w,
            h,
            step,
            transform,
            (int(width), int(height)),
            np.asarray(yx).tolist(),
            digest.hexdigest(),
        )

    def metadata(self):
        return {
            "version": 1,
            "coordinate_kind": "local_planar",
            "coordinate_units": "arbitrary",
            "corners_yx": self.corners_yx,
            "map_signature": self.map_signature,
        }


def render_local_panorama(video_path, layout, *, running=lambda: True, progress=lambda *a: None):
    """Feather-blend connected reference frames with bounded canvas memory.

    Returns BGRA, retaining valid black image pixels and transparent unmapped areas.
    Cancellation returns None and leaves saving to the caller.
    """
    cap = cv2.VideoCapture(str(video_path))
    try:
        if not cap.isOpened():
            raise ValueError(f"Cannot open reference video: {video_path}")
        cw, ch = layout.canvas_size
        colors = np.zeros((ch, cw, 3), dtype=np.float32)
        weights = np.zeros((ch, cw), dtype=np.float32)
        yy, xx = np.mgrid[: layout.frame_height, : layout.frame_width]
        feather = np.minimum.reduce(
            [xx + 1, layout.frame_width - xx, yy + 1, layout.frame_height - yy]
        ).astype(np.float32)
        feather /= feather.max()
        for count, (fid, affine) in enumerate(layout.affines.items(), 1):
            if not running():
                return None
            # frame_index_map lists occupied DB slots, not original video indices.
            video_frame = fid * layout.frame_step
            if not cap.set(cv2.CAP_PROP_POS_FRAMES, video_frame):
                raise ValueError(f"Cannot seek reference video frame {video_frame}")
            ok, frame = cap.read()
            if not ok:
                raise ValueError(
                    f"Cannot read reference video frame {video_frame}; check the source video"
                )
            frame = cv2.resize(frame, (layout.frame_width, layout.frame_height))
            matrix = (layout.map_to_canvas @ np.vstack([affine, [0, 0, 1]]))[:2]
            weighted = frame.astype(np.float32) * feather[..., None]
            colors += cv2.warpAffine(weighted, matrix, (cw, ch), flags=cv2.INTER_LINEAR)
            weights += cv2.warpAffine(feather, matrix, (cw, ch), flags=cv2.INTER_LINEAR)
            progress(
                int(count / len(layout.affines) * 95),
                f"Панорама локальної карти: {count}/{len(layout.affines)} кадрів",
            )
        if not running():
            return None
        valid = weights > 1e-6
        colors /= np.maximum(weights[..., None], 1e-6)
        image = np.empty((ch, cw, 4), dtype=np.uint8)
        image[..., :3] = np.clip(colors, 0, 255).astype(np.uint8)
        image[..., 3] = valid.astype(np.uint8) * 255
        return image
    finally:
        cap.release()


def load_local_panorama_metadata(path, layout):
    """Reject a saved overlay if the selected local map has since changed."""
    from pathlib import Path

    metadata = json.loads(Path(str(path) + ".json").read_text(encoding="utf-8"))
    if (
        metadata.get("coordinate_kind") != "local_planar"
        or metadata.get("map_signature") != layout.map_signature
    ):
        raise ValueError(
            "Панорама належить іншій або попередній версії локальної карти. Побудуйте її заново."
        )
    corners = np.asarray(metadata.get("corners_yx"), dtype=float)
    if corners.shape != (4, 2) or not np.isfinite(corners).all():
        raise ValueError("Invalid panorama coordinates")
    return corners.tolist()
