"""Multi-database manager.

Coordinates DatabaseLoader instances for multiple video sources,
spatial filtering of active sources, and multi-source vector retrieval.
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
    """Central coordination class managing multiple databases for video sources."""

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

    # ── Initialization ───────────────────────────────────────────────────────

    def _load_sources(self, sources: list[ProjectVideoSource]) -> None:
        """Loads DatabaseLoader and creates retriever for each enabled source."""
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

                # Create retriever (priority: LanceDB -> GeoAware -> FAISS)
                if loader.lance_table is not None:
                    retriever = LanceDBRetrieval(loader.lance_table)
                    logger.info(
                        f"Source '{src.source_id}': LanceDB retriever "
                        f"({loader.lance_table.count_rows()} vectors)"
                    )
                elif loader.global_descriptors is not None:
                    if loader.spatial_index is not None and loader.spatial_index.is_available:
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

        self._check_interchangeability()

    def _check_interchangeability(self) -> None:
        """Warn if loaded databases were built with incompatible schema settings."""
        import json as _json

        fps: dict[str, str] = {}
        comps: dict[str, dict] = {}
        for sid, loader in self._databases.items():
            fp = loader.metadata.get("schema_fingerprint")
            if fp is None:
                logger.warning(
                    f"Source '{sid}': no schema_fingerprint (older builder) — "
                    f"cannot verify interchangeability with the other databases."
                )
                continue
            fps[sid] = str(fp)
            raw = loader.metadata.get("schema_components")
            if raw:
                try:
                    comps[sid] = _json.loads(raw)
                except Exception as e:
                    logger.warning(
                        f"Source '{sid}': schema_components is not valid JSON ({e}) — "
                        f"per-field comparison unavailable, only the fingerprint is checked."
                    )

        distinct = set(fps.values())
        if len(distinct) <= 1:
            if fps:
                logger.info(
                    f"Interchangeability OK: {len(fps)} databases share schema "
                    f"{next(iter(distinct))}."
                )
            return

        from src.database.schema_fingerprint import compare

        ref_sid = next(iter(fps))
        ref = comps.get(ref_sid, {})
        logger.error(
            f"DATABASE MISMATCH: {len(distinct)} different schema fingerprints "
            f"among {len(fps)} databases — they are NOT interchangeable and "
            f"combined localization may be wrong."
        )
        for sid, fp in fps.items():
            if fp != fps[ref_sid] and comps.get(sid) and ref:
                logger.error(
                    f"  '{sid}' ({fp}) differs from '{ref_sid}' ({fps[ref_sid]}): "
                    f"{'; '.join(compare(ref, comps[sid]))}"
                )

    def unload_source(self, source_id: str) -> None:
        """Unloads a source from memory."""
        if source_id in self._databases:
            try:
                self._databases[source_id].close()
            except Exception as e:
                logger.warning(f"Error closing database '{source_id}': {e}")
            del self._databases[source_id]
        self._retrievers.pop(source_id, None)
        self._sources.pop(source_id, None)
        self._active_source_ids.discard(source_id)
        logger.info(f"Source '{source_id}' unloaded (e.g. pending rebuild)")

    def reload_source(self, src: ProjectVideoSource) -> bool:
        """Reloads source after database rebuild."""
        self.unload_source(src.source_id)
        self._load_sources([src])
        ok = src.source_id in self._databases
        if ok:
            logger.success(f"Source '{src.source_id}' reloaded after rebuild")
        else:
            logger.error(f"Failed to reload source '{src.source_id}' after rebuild")
        return ok

    def toggle_source(self, src: ProjectVideoSource) -> None:
        """Enables or disables a video source."""
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
        """Performs vector search across active databases, returning best match."""
        if not self._active_source_ids:
            logger.warning("No active sources for retrieval")
            return None, []

        best_source_id: str | None = None
        best_candidates: list[tuple[int, float]] = []
        best_top_score: float = -1.0

        # Sort by priority (0 = highest)
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

                top_score = candidates[0][1]
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

    # ── Object Access ────────────────────────────────────────────────────────

    def get_database(self, source_id: str) -> DatabaseLoader | None:
        """Returns DatabaseLoader for given source_id."""
        return self._databases.get(source_id)

    def get_source_config(self, source_id: str) -> ProjectVideoSource | None:
        """Returns ProjectVideoSource for given source_id."""
        return self._sources.get(source_id)

    # ── Spatial Filtering ────────────────────────────────────────────────────

    def set_active_area(self, area_id: str) -> None:
        """Activates all sources in specified area."""
        new_active = {
            sid
            for sid, src in self._sources.items()
            if src.area_id == area_id and sid in self._databases
        }
        if new_active != self._active_source_ids:
            self._active_source_ids = new_active
            logger.info(f"Active area set to '{area_id}': {sorted(self._active_source_ids)}")

    def set_active_by_gps(
        self,
        lat: float,
        lon: float,
        radius_m: float = 2500.0,
    ) -> bool:
        """Activates sources whose geo_bounds contain point (lat, lon)."""
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
        """Updates position in all GeoAwareRetrievers."""
        for sid in self._active_source_ids:
            retriever = self._retrievers.get(sid)
            if isinstance(retriever, GeoAwareRetriever) and retriever.is_geo_aware:
                retriever.update_position(lat, lon)

    def set_all_active(self) -> None:
        """Activates all loaded sources."""
        self._active_source_ids = set(self._databases.keys())

    # ── Utilities ────────────────────────────────────────────────────────────

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
        """Closes all DatabaseLoader instances."""
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
