"""
multi_database_manager.py — Менеджер множинних баз даних.

Координує завантаження DatabaseLoader для кожного джерела,
просторову фільтрацію активних джерел та вибір найкращого збігу.
"""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from src.database.database_loader import DatabaseLoader
from src.localization.geo_aware_retriever import GeoAwareRetriever
from src.localization.matcher import FastRetrieval, LanceDBRetrieval
from src.utils.logging_utils import get_logger

if TYPE_CHECKING:
    from src.core.project_video_source import ProjectVideoSource

logger = get_logger(__name__)


class MultiDatabaseManager:
    """
    Центральний координаційний клас.
    Замінює прямий доступ до одного DatabaseLoader.

    Відповідає за:
    - Завантаження баз для enabled джерел
    - Створення retrievers (FAISS або LanceDB)
    - Просторову фільтрацію активних джерел
    - Вибір найкращого збігу через get_best_match
    """

    def __init__(
        self,
        sources: list[ProjectVideoSource],
        project_dir: Path,
        config: dict[str, Any] | None = None,
    ) -> None:
        self._sources: dict[str, ProjectVideoSource] = {}
        self._databases: dict[str, DatabaseLoader] = {}
        self._retrievers: dict[str, FastRetrieval | LanceDBRetrieval | GeoAwareRetriever] = {}
        self._active_source_ids: set[str] = set()
        self._project_dir = project_dir
        self._config = config or {}

        self._load_sources(sources)

    # ── Ініціалізація ────────────────────────────────────────────────────────

    def _load_sources(self, sources: list[ProjectVideoSource]) -> None:
        """Завантажує DatabaseLoader та створює retriever для кожного enabled джерела."""
        for src in sources:
            if not src.enabled:
                logger.debug(f"Skipping disabled source '{src.source_id}'")
                continue

            db_path = self._project_dir / src.database_file
            if not db_path.exists():
                logger.warning(
                    f"Database not found for source '{src.source_id}': {db_path}. "
                    f"Skipping this source."
                )
                continue

            try:
                loader = DatabaseLoader(str(db_path))
                self._databases[src.source_id] = loader
                self._sources[src.source_id] = src

                # Створюємо retriever (пріоритет LanceDB → GeoAware → FAISS)
                if loader.lance_table is not None:
                    retriever = LanceDBRetrieval(loader.lance_table)
                    logger.info(
                        f"Source '{src.source_id}': LanceDB retriever "
                        f"({loader.lance_table.count_rows()} vectors)"
                    )
                elif loader.global_descriptors is not None:
                    if loader.spatial_index is not None and loader.spatial_index.is_available:
                        # GeoAwareRetriever: геофільтрація через SpatialIndex
                        retriever = GeoAwareRetriever(
                            loader.global_descriptors,
                            spatial_index=loader.spatial_index,
                        )
                        logger.info(
                            f"Source '{src.source_id}': GeoAwareRetriever "
                            f"({len(loader.global_descriptors)} vectors, "
                            f"{loader.spatial_index.num_indexed} geo-indexed)"
                        )
                    else:
                        retriever = FastRetrieval(loader.global_descriptors)
                        logger.info(
                            f"Source '{src.source_id}': FAISS retriever "
                            f"({len(loader.global_descriptors)} vectors)"
                        )
                else:
                    logger.error(
                        f"Source '{src.source_id}': no descriptors available. "
                        f"Database may be corrupted."
                    )
                    continue

                self._retrievers[src.source_id] = retriever
                self._active_source_ids.add(src.source_id)

            except Exception as e:
                logger.error(
                    f"Failed to load database for source '{src.source_id}': {e}",
                    exc_info=True,
                )

        logger.info(
            f"MultiDatabaseManager initialized: {len(self._databases)} databases loaded, "
            f"{len(self._active_source_ids)} active"
        )

    def toggle_source(self, src: ProjectVideoSource) -> None:
        """Вмикає або вимикає джерело. Завантажує або вивантажує БД з пам'яті."""
        if src.enabled:
            if src.source_id not in self._databases:
                self._load_sources([src])
        else:
            if src.source_id in self._databases:
                try:
                    self._databases[src.source_id].close()
                except Exception as e:
                    logger.warning(f"Error closing database '{src.source_id}': {e}")
                del self._databases[src.source_id]
            if src.source_id in self._retrievers:
                del self._retrievers[src.source_id]
            if src.source_id in self._sources:
                del self._sources[src.source_id]
            if src.source_id in self._active_source_ids:
                self._active_source_ids.remove(src.source_id)
            logger.info(f"Source '{src.source_id}' disabled and unloaded from memory.")

    # ── Retrieval ────────────────────────────────────────────────────────────

    def get_best_match(
        self,
        global_desc: np.ndarray,
        top_k: int = 8,
    ) -> tuple[str | None, list[tuple[int, float]]]:
        """
        Виконує vectorний пошук у кожній активній базі.

        Returns:
            (source_id, candidates): source_id з найвищим top-1 score,
            candidates — список (frame_id, score). None якщо нічого не знайдено.
        """
        if not self._active_source_ids:
            logger.warning("No active sources for retrieval")
            return None, []

        best_source_id: str | None = None
        best_candidates: list[tuple[int, float]] = []
        best_top_score: float = -1.0

        # Сортуємо за priority (0 = найвищий) для детерміністичного tiebreak
        sorted_ids = sorted(
            self._active_source_ids,
            key=lambda sid: self._sources[sid].priority,
        )

        for source_id in sorted_ids:
            retriever = self._retrievers.get(source_id)
            if retriever is None:
                continue

            try:
                candidates = retriever.find_similar_frames(global_desc, top_k)
                if not candidates:
                    continue

                top_score = candidates[0][1]  # (frame_id, score)
                if top_score > best_top_score:
                    best_top_score = top_score
                    best_source_id = source_id
                    best_candidates = candidates

            except Exception as e:
                logger.error(
                    f"Retrieval failed for source '{source_id}': {e}",
                    exc_info=True,
                )

        if best_source_id is not None:
            logger.debug(
                f"Best match: source='{best_source_id}', "
                f"top_score={best_top_score:.4f}, "
                f"candidates={len(best_candidates)}"
            )

        return best_source_id, best_candidates

    # ── Доступ до об'єктів ───────────────────────────────────────────────────

    def get_database(self, source_id: str) -> DatabaseLoader | None:
        """Повертає DatabaseLoader для вказаного source_id."""
        return self._databases.get(source_id)

    def get_source_config(self, source_id: str) -> ProjectVideoSource | None:
        """Повертає ProjectVideoSource для вказаного source_id."""
        return self._sources.get(source_id)

    # ── Просторова фільтрація ────────────────────────────────────────────────

    def set_active_area(self, area_id: str) -> None:
        """Активує всі джерела вказаної зони."""
        new_active = {
            sid
            for sid, src in self._sources.items()
            if src.area_id == area_id and sid in self._databases
        }
        if new_active != self._active_source_ids:
            self._active_source_ids = new_active
            logger.info(
                f"Active area set to '{area_id}': "
                f"{sorted(self._active_source_ids)}"
            )

    def set_active_by_gps(
        self,
        lat: float,
        lon: float,
        radius_m: float = 2500.0,
    ) -> bool:
        """
        Активує джерела, geo_bounds яких містять точку (lat, lon).
        Джерела без geo_bounds завжди залишаються активними.

        Returns:
            True якщо набір активних джерел змінився.
        """
        new_active: set[str] = set()
        for sid, src in self._sources.items():
            if sid not in self._databases:
                continue
            if src.contains_point(lat, lon):
                new_active.add(sid)

        changed = new_active != self._active_source_ids
        if changed:
            old = sorted(self._active_source_ids)
            self._active_source_ids = new_active
            logger.info(
                f"Active sources changed by GPS ({lat:.5f}, {lon:.5f}): "
                f"{old} → {sorted(self._active_source_ids)}"
            )
        return changed

    def update_retriever_positions(self, lat: float, lon: float) -> None:
        """Оновлює позицію у всіх GeoAwareRetriever-ах для перебудови FAISS-підмножини."""
        for sid in self._active_source_ids:
            retriever = self._retrievers.get(sid)
            if isinstance(retriever, GeoAwareRetriever) and retriever.is_geo_aware:
                retriever.update_position(lat, lon)

    def set_all_active(self) -> None:
        """Активує всі завантажені джерела."""
        self._active_source_ids = set(self._databases.keys())

    # ── Утиліти ──────────────────────────────────────────────────────────────

    @property
    def active_source_ids(self) -> set[str]:
        return set(self._active_source_ids)

    @property
    def all_source_ids(self) -> list[str]:
        return list(self._databases.keys())

    @property
    def num_databases(self) -> int:
        return len(self._databases)

    def close_all(self) -> None:
        """Закриває всі DatabaseLoader."""
        for sid, db in self._databases.items():
            try:
                db.close()
            except Exception as e:
                logger.warning(f"Error closing database '{sid}': {e}")
        self._databases.clear()
        self._retrievers.clear()
        self._active_source_ids.clear()
        logger.info("All databases closed")

    def __contains__(self, source_id: str) -> bool:
        return source_id in self._databases

    def __len__(self) -> int:
        return len(self._databases)
