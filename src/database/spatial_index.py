"""
spatial_index.py — Просторовий тайловий індекс для геофільтрації кадрів.

Будується поверх даних frame_gps з HDF5. Не вимагає зовнішніх
геосторонніх бібліотек (H3, S2 тощо).
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np

from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


class SpatialIndex:
    """
    Тайловий просторовий індекс для швидкої геофільтрації кадрів.
    
    Територія розбивається на рівні прямокутні тайли за формулою:
        tile_key = (int(lat / tile_deg), int(lon / tile_deg))
    де tile_deg ≈ 0.005° ≈ 500 метрів.
    
    Для кожного тайлу зберігається список frame_id кадрів,
    GPS-координати яких потрапляють у цей тайл.
    """

    TILE_DEG: float = 0.005  # ≈ 500m

    def __init__(self, frame_gps: np.ndarray, tile_deg: float | None = None) -> None:
        """
        Args:
            frame_gps: Масив (N, 2) з [lat, lon] для кожного кадру.
                       NaN значення ігноруються.
            tile_deg:  Розмір тайлу у градусах. За замовчуванням 0.005° ≈ 500m.
        """
        if tile_deg is not None:
            self.tile_deg = tile_deg
        else:
            self.tile_deg = self.TILE_DEG

        self._tiles: dict[tuple[int, int], list[int]] = defaultdict(list)
        self._frame_gps = frame_gps
        self._num_indexed = 0

        self._build(frame_gps)

    def _build(self, frame_gps: np.ndarray) -> None:
        """Будує індекс з масиву GPS-координат."""
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
        """Конвертує GPS у ключ тайлу."""
        return int(lat / self.tile_deg), int(lon / self.tile_deg)

    def get_frame_ids_near(
        self,
        lat: float,
        lon: float,
        radius_tiles: int = 2,
    ) -> list[int]:
        """
        Повертає об'єднаний список frame_id з квадрата тайлів навколо точки.
        
        При radius_tiles=2 та tile_deg=0.005° це дає покриття
        ≈ 2.5 км у кожному напрямку.
        
        Args:
            lat:           Широта центру пошуку.
            lon:           Довгота центру пошуку.
            radius_tiles:  Радіус пошуку у тайлах.
            
        Returns:
            Список frame_id кадрів у радіусі.
        """
        center_t_lat, center_t_lon = self._to_tile(lat, lon)
        result: list[int] = []

        for dt_lat in range(-radius_tiles, radius_tiles + 1):
            for dt_lon in range(-radius_tiles, radius_tiles + 1):
                key = (center_t_lat + dt_lat, center_t_lon + dt_lon)
                if key in self._tiles:
                    result.extend(self._tiles[key])

        return result

    def get_frame_gps(self, frame_id: int) -> tuple[float, float] | None:
        """Повертає (lat, lon) для frame_id або None."""
        if frame_id < 0 or frame_id >= len(self._frame_gps):
            return None
        lat, lon = self._frame_gps[frame_id]
        if np.isnan(lat) or np.isnan(lon):
            return None
        return float(lat), float(lon)

    @property
    def num_indexed(self) -> int:
        """Кількість проіндексованих кадрів."""
        return self._num_indexed

    @property
    def num_tiles(self) -> int:
        """Кількість тайлів."""
        return len(self._tiles)

    @property
    def is_available(self) -> bool:
        """True якщо індекс містить хоча б один кадр."""
        return self._num_indexed > 0
