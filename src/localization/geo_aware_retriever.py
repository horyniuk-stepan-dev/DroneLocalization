"""
geo_aware_retriever.py — Геозалежний ретривер з фоновою перебудовою FAISS.

Замінює FastRetrieval для джерел із frame_gps. При наявності
просторового контексту будує FAISS IndexFlatIP тільки з підмножини
кадрів в активному радіусі.
"""
from __future__ import annotations

import threading
from typing import TYPE_CHECKING

import faiss
import numpy as np

from src.utils.logging_utils import get_logger

if TYPE_CHECKING:
    from src.database.spatial_index import SpatialIndex

logger = get_logger(__name__)


class GeoAwareRetriever:
    """
    Геозалежний FAISS-ретривер з фоновою перебудовою індексу.
    
    При виклику update_position():
    - Визначає новий набір frame_id через SpatialIndex
    - Якщо набір змінився — перебудовує FAISS у daemon-потоці
    - Під час перебудови поточний індекс залишається активним
    
    Для джерел без frame_gps деградує до GlobalRetriever (повний масив).
    """

    def __init__(
        self,
        global_descriptors: np.ndarray,
        spatial_index: SpatialIndex | None = None,
    ) -> None:
        """
        Args:
            global_descriptors: Повний масив дескрипторів (N, D).
            spatial_index:      SpatialIndex або None для fallback.
        """
        self._full_descriptors = global_descriptors
        self._spatial_index = spatial_index
        self._dim = global_descriptors.shape[1]

        # Активний стан
        self._active_frame_ids: list[int] = list(range(len(global_descriptors)))
        self._lock = threading.Lock()

        # Починаємо з повного індексу
        self._index = self._build_faiss_index(
            global_descriptors, self._active_frame_ids
        )

        if spatial_index is not None and spatial_index.is_available:
            logger.info(
                f"GeoAwareRetriever initialized with SpatialIndex "
                f"({spatial_index.num_indexed} frames, {spatial_index.num_tiles} tiles). "
                f"Full index active until first position update."
            )
        else:
            logger.info(
                f"GeoAwareRetriever: no SpatialIndex available. "
                f"Operating in global mode ({len(global_descriptors)} vectors)."
            )

    def _build_faiss_index(
        self,
        descriptors: np.ndarray,
        frame_ids: list[int],
    ) -> faiss.IndexIDMap:
        """Будує FAISS IndexFlatIP з нормалізованих дескрипторів."""
        base_index = faiss.IndexFlatIP(self._dim)
        index = faiss.IndexIDMap(base_index)

        if len(frame_ids) == 0:
            return index

        normed = descriptors / (
            np.linalg.norm(descriptors, axis=1, keepdims=True) + 1e-8
        )
        ids = np.array(frame_ids, dtype=np.int64)
        index.add_with_ids(normed.astype(np.float32), ids)
        return index

    # ── Оновлення позиції ────────────────────────────────────────────────────

    def update_position(self, lat: float, lon: float, radius_tiles: int = 2) -> bool:
        """
        Оновлює активну підмножину кадрів за GPS-позицією.
        
        Якщо набір frame_id змінився — запускає перебудову FAISS у фоновому потоці.
        Під час перебудови поточний індекс залишається активним.
        
        Returns:
            True якщо набір змінився (перебудова ініційована).
        """
        if self._spatial_index is None or not self._spatial_index.is_available:
            return False

        new_frame_ids = sorted(
            self._spatial_index.get_frame_ids_near(lat, lon, radius_tiles)
        )

        if new_frame_ids == self._active_frame_ids:
            return False

        logger.info(
            f"GeoAwareRetriever: position update ({lat:.5f}, {lon:.5f}). "
            f"Active frames: {len(self._active_frame_ids)} → {len(new_frame_ids)}"
        )

        # Перебудова у фоновому потоці
        thread = threading.Thread(
            target=self._rebuild_in_background,
            args=(new_frame_ids,),
            daemon=True,
        )
        thread.start()
        return True

    def _rebuild_in_background(self, new_frame_ids: list[int]) -> None:
        """Перебудовує FAISS-індекс у daemon-потоці."""
        try:
            # Збираємо дескриптори для підмножини
            descriptors = self._full_descriptors[new_frame_ids]
            new_index = self._build_faiss_index(descriptors, new_frame_ids)

            # Атомарна заміна
            with self._lock:
                self._index = new_index
                self._active_frame_ids = new_frame_ids

            logger.info(
                f"GeoAwareRetriever: index rebuilt with {len(new_frame_ids)} frames"
            )
        except Exception as e:
            logger.error(
                f"GeoAwareRetriever: background rebuild failed: {e}",
                exc_info=True,
            )

    # ── Пошук ────────────────────────────────────────────────────────────────

    def find_similar_frames(
        self,
        query_desc: np.ndarray,
        top_k: int = 5,
    ) -> list[tuple[int, float]]:
        """
        Пошук схожих кадрів у активному індексі.
        
        Returns:
            Список (frame_id, cosine_score). frame_id — реальний ID у вихідній базі.
        """
        q = query_desc / (np.linalg.norm(query_desc) + 1e-8)
        q = q.astype(np.float32)
        if q.ndim == 1:
            q = q[None]

        with self._lock:
            if self._index.ntotal == 0:
                return []
            scores, ids = self._index.search(q, min(top_k, self._index.ntotal))

        results = [
            (int(idx), float(score))
            for idx, score in zip(ids[0], scores[0])
            if idx != -1
        ]
        return results

    # ── Утиліти ──────────────────────────────────────────────────────────────

    @property
    def num_active_frames(self) -> int:
        with self._lock:
            return len(self._active_frame_ids)

    @property
    def is_geo_aware(self) -> bool:
        return self._spatial_index is not None and self._spatial_index.is_available
