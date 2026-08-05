"""Spatial tile index for frame geo-filtering.

Built on frame_gps data from HDF5 database.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


class SpatialIndex:
    """Tiled spatial index for fast frame geo-filtering."""

    TILE_DEG: float = 0.005  # ≈ 500m

    def __init__(self, frame_gps: np.ndarray, tile_deg: float | None = None) -> None:
        if tile_deg is not None:
            self.tile_deg = tile_deg
        else:
            self.tile_deg = self.TILE_DEG

        self._tiles: dict[tuple[int, int], list[int]] = defaultdict(list)
        self._frame_gps = frame_gps
        self._num_indexed = 0

        self._build(frame_gps)

    def _build(self, frame_gps: np.ndarray) -> None:
        """Builds index from GPS coordinate array."""
        for frame_id in range(len(frame_gps)):
            lat, lon = frame_gps[frame_id]
            if np.isnan(lat) or np.isnan(lon):
                continue
            tile_key = self._to_tile(lat, lon)
            self._tiles[tile_key].append(frame_id)
            self._num_indexed += 1

        logger.info(
            f"SpatialIndex built: {self._num_indexed} frames in "
            f"{len(self._tiles)} tiles (tile_deg={self.tile_deg}°)"
        )

    def _to_tile(self, lat: float, lon: float) -> tuple[int, int]:
        """Converts GPS coordinate into tile key."""
        return int(lat / self.tile_deg), int(lon / self.tile_deg)

    def get_frame_ids_near(
        self,
        lat: float,
        lon: float,
        radius_tiles: int = 2,
    ) -> list[int]:
        """Returns frame_ids in tile square surrounding target point."""
        center_t_lat, center_t_lon = self._to_tile(lat, lon)
        result: list[int] = []

        for dt_lat in range(-radius_tiles, radius_tiles + 1):
            for dt_lon in range(-radius_tiles, radius_tiles + 1):
                key = (center_t_lat + dt_lat, center_t_lon + dt_lon)
                if key in self._tiles:
                    result.extend(self._tiles[key])

        return result

    def get_frame_gps(self, frame_id: int) -> tuple[float, float] | None:
        """Returns (lat, lon) for frame_id or None if invalid/missing."""
        if frame_id < 0 or frame_id >= len(self._frame_gps):
            return None
        lat, lon = self._frame_gps[frame_id]
        if np.isnan(lat) or np.isnan(lon):
            return None
        return float(lat), float(lon)

    @property
    def num_indexed(self) -> int:
        """Number of indexed frames."""
        return self._num_indexed

    @property
    def num_tiles(self) -> int:
        """Number of tiles."""
        return len(self._tiles)

    @property
    def is_available(self) -> bool:
        """Returns True if index contains at least one frame."""
        return self._num_indexed > 0
