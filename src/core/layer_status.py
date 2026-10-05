"""Qt-free bookkeeping for project layers (video sources).

A *layer* is one ``ProjectVideoSource``: its own video, ``database.h5`` and
``calibration.json``.  The GUI keeps one live ``DatabaseLoader`` and one live
``MultiAnchorCalibration`` per layer.  Status is computed from those live
objects (the same ones calibration and propagation mutate), and only falls
back to the filesystem for what is not loaded — so it changes the moment an
anchor is saved or propagation finishes, not on the next project reopen.

``find_layer_conflicts`` detects the ways two layers can end up sharing or
swapping calibration data: colliding file paths, a calibration file stamped
with another layer's id, anchors that do not fit the layer's database and
byte-identical anchor sets copied between layers.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np

# Key written into calibration.json by the GUI to record which layer owns it.
CALIBRATION_OWNER_KEY = "source_id"


class LayerState(str, Enum):
    DISABLED = "disabled"
    NO_DB = "no_db"
    DB_NOT_LOADED = "db_not_loaded"  # file exists, but the manager skipped/failed it
    NO_CALIBRATION = "no_calibration"
    NOT_PROPAGATED = "not_propagated"  # anchors exist, propagation never ran
    STALE = "stale"  # anchors changed after the last propagation
    READY = "ready"


_LABELS: dict[LayerState, str] = {
    LayerState.DISABLED: "🔇 Вимкнено",
    LayerState.NO_DB: "❌ Без БД",
    LayerState.DB_NOT_LOADED: "⛔ БД не завантажена",
    LayerState.NO_CALIBRATION: "⚠ Без калібр.",
    LayerState.NOT_PROPAGATED: "🟡 Потрібна пропагація",
    LayerState.STALE: "🟡 Пропагація застаріла",
    LayerState.READY: "✅ Готово",
}

_HINTS: dict[LayerState, str] = {
    LayerState.DISABLED: "Шар вимкнено — він не бере участі в локалізації.",
    LayerState.NO_DB: "Бази даних ще немає — побудуйте її (ПКМ → Побудувати БД).",
    LayerState.DB_NOT_LOADED: (
        "Файл бази є, але його не завантажено (несумісна схема дескрипторів "
        "або помилка читання) — див. лог."
    ),
    LayerState.NO_CALIBRATION: "Немає якорів і пропагації — відкрийте калібрування шару.",
    LayerState.NOT_PROPAGATED: "Якорі додано, але пропагацію GPS ще не запущено.",
    LayerState.STALE: "Якорі змінено після останньої пропагації — запустіть її знову.",
    LayerState.READY: "База має GPS-пропагацію, що відповідає поточним якорям.",
}


@dataclass(frozen=True)
class LayerStatus:
    source_id: str
    area_id: str
    enabled: bool
    state: LayerState
    has_db_file: bool
    db_loaded: bool
    num_anchors: int | None  # None = calibration not loaded (unknown)
    num_valid: int | None  # frames with GPS after propagation
    num_frames: int | None  # DB slots
    coordinate_mode: str = "GEOGRAPHIC"

    @property
    def label(self) -> str:
        if self.state == LayerState.READY and self.coordinate_mode == "LOCAL":
            return "✅ Локальна X/Y"
        return _LABELS[self.state]

    @property
    def hint(self) -> str:
        if self.state == LayerState.READY and self.coordinate_mode == "LOCAL":
            return "Відносна карта з міжкадрових зв’язків. Умовні X/Y, без GPS і метричного масштабу."
        return _HINTS[self.state]

    @property
    def anchors_text(self) -> str:
        return "—" if self.num_anchors is None else str(self.num_anchors)

    @property
    def gps_text(self) -> str:
        if self.num_valid is None or not self.num_frames:
            return "—"
        return f"{self.num_valid}/{self.num_frames}"


def _field(source: Any, name: str, default: Any = None) -> Any:
    if isinstance(source, Mapping):
        return source.get(name, default)
    return getattr(source, name, default)


def _anchor_frame_id(anchor: Any) -> int:
    return int(_field(anchor, "frame_id"))


def _anchor_matrix(anchor: Any) -> np.ndarray:
    return np.asarray(_field(anchor, "affine_matrix"), dtype=np.float64)


def anchor_sets_equal(a: Iterable[Any], b: Iterable[Any], rtol: float = 1e-9) -> bool:
    """True if both anchor collections have the same frame ids and matrices."""
    left = {_anchor_frame_id(x): _anchor_matrix(x) for x in a}
    right = {_anchor_frame_id(x): _anchor_matrix(x) for x in b}
    if left.keys() != right.keys():
        return False
    for fid, matrix in left.items():
        other = right[fid]
        if matrix.shape != other.shape or not np.allclose(matrix, other, rtol=rtol, atol=1e-9):
            return False
    return True


def propagation_matches_anchors(anchors_json: Any, anchors: Iterable[Any]) -> bool | None:
    """Compare anchors stored with the last propagation against current anchors.

    Returns None when the database carries no anchor record (older propagation
    or produced elsewhere) — staleness is then unknown, not assumed.
    """
    if anchors_json is None:
        return None
    if isinstance(anchors_json, bytes):
        anchors_json = anchors_json.decode("utf-8")
    try:
        stored = json.loads(anchors_json)
    except (TypeError, ValueError):
        return None
    if not isinstance(stored, list):
        return None
    try:
        return anchor_sets_equal(stored, anchors)
    except (KeyError, TypeError, ValueError):
        return None


def compute_layer_status(
    source: Any,
    project_dir: str | Path | None,
    *,
    database: Any = None,
    calibration: Any = None,
) -> LayerStatus:
    """Status of one layer from its live objects (``database``/``calibration``).

    ``database`` is the loaded ``DatabaseLoader`` of this layer or None when it
    is not loaded; ``calibration`` is its ``MultiAnchorCalibration`` or None
    when unknown.  The filesystem is consulted only for the DB file presence.
    """
    sid = str(_field(source, "source_id", "?"))
    area = str(_field(source, "area_id", "") or "")
    enabled = bool(_field(source, "enabled", True))
    db_file = _field(source, "database_file", "") or ""

    has_db_file = False
    if project_dir and db_file:
        has_db_file = (Path(project_dir) / db_file).is_file()

    num_anchors = len(calibration.anchors) if calibration is not None else None
    num_frames = num_valid = None
    propagated = False
    if database is not None:
        has_db_file = True
        try:
            num_frames = int(database.get_num_frames())
        except Exception:
            num_frames = None
        propagated = bool(getattr(database, "is_propagated", False))
        valid = getattr(database, "frame_valid", None)
        if propagated and valid is not None:
            num_valid = int(np.asarray(valid).sum())

    if not enabled:
        state = LayerState.DISABLED
    elif database is None:
        state = LayerState.DB_NOT_LOADED if has_db_file else LayerState.NO_DB
    elif propagated:
        state = LayerState.READY
        if num_anchors:
            same = propagation_matches_anchors(
                getattr(database, "propagation_anchors_json", None), calibration.anchors
            )
            if same is False:
                state = LayerState.STALE
    elif num_anchors:
        state = LayerState.NOT_PROPAGATED
    else:
        state = LayerState.NO_CALIBRATION

    return LayerStatus(
        source_id=sid,
        area_id=area,
        enabled=enabled,
        state=state,
        has_db_file=has_db_file,
        db_loaded=database is not None,
        num_anchors=num_anchors,
        num_valid=num_valid,
        num_frames=num_frames,
        coordinate_mode=getattr(getattr(database, "converter", None), "mode", "GEOGRAPHIC"),
    )


@dataclass(frozen=True)
class LayerConflict:
    kind: str  # "path" | "owner" | "range" | "duplicate"
    source_ids: tuple[str, ...]
    message: str


def _norm(project_dir: Path, rel: str) -> str:
    # casefold: the app runs on Windows, where "Layer" and "layer" are one folder.
    return str((project_dir / rel).resolve()).casefold()


def find_path_conflicts(sources: Iterable[Any], project_dir: str | Path) -> list[LayerConflict]:
    """Two layers must never share a DB, a calibration file or a DB folder.

    The DB folder matters too: ``vectors.lance`` lives next to ``database.h5``,
    so two databases in one folder silently share (and overwrite) one index.
    """
    root = Path(project_dir)
    owners: dict[tuple[str, str], list[str]] = {}
    for src in sources:
        sid = str(_field(src, "source_id", "?"))
        db = _field(src, "database_file", "") or ""
        cal = _field(src, "calibration_file", "") or ""
        keys = []
        if db:
            keys.append(("database_file", _norm(root, db)))
            keys.append(("database folder", _norm(root, str(Path(db).parent))))
        if cal:
            keys.append(("calibration_file", _norm(root, cal)))
        for key in keys:
            owners.setdefault(key, []).append(sid)

    shared = {
        (what, path): tuple(sorted(set(sids)))
        for (what, path), sids in owners.items()
        if len(set(sids)) > 1
    }
    same_db = {ids for (what, _), ids in shared.items() if what == "database_file"}
    conflicts = []
    for (what, path), ids in shared.items():
        if what == "database folder" and ids in same_db:
            continue  # already reported as a shared database_file
        conflicts.append(
            LayerConflict(
                kind="path",
                source_ids=ids,
                message=f"Шари {', '.join(ids)} мають спільний {what}: {path}",
            )
        )
    return conflicts


def find_layer_conflicts(
    sources: Iterable[Any],
    project_dir: str | Path,
    calibrations: Mapping[str, Any],
    num_frames: Mapping[str, int] | None = None,
) -> list[LayerConflict]:
    """All detectable ways calibration data got mixed between layers."""
    sources = list(sources)
    conflicts = find_path_conflicts(sources, project_dir)
    num_frames = num_frames or {}

    for sid, cal in calibrations.items():
        if cal is None:
            continue
        owner = (getattr(cal, "extra_metadata", None) or {}).get(CALIBRATION_OWNER_KEY)
        if owner and str(owner) != sid:
            conflicts.append(
                LayerConflict(
                    kind="owner",
                    source_ids=(sid,),
                    message=(
                        f"Калібрування шару «{sid}» записане для шару «{owner}» "
                        f"(ймовірно, файл скопійовано між шарами)."
                    ),
                )
            )
        n = num_frames.get(sid)
        if n:
            bad = sorted(
                _anchor_frame_id(a) for a in cal.anchors if not 0 <= _anchor_frame_id(a) < int(n)
            )
            if bad:
                conflicts.append(
                    LayerConflict(
                        kind="range",
                        source_ids=(sid,),
                        message=(
                            f"Якорі шару «{sid}» на кадрах {bad} виходять за межі його БД "
                            f"({n} слотів) — ймовірно, це якорі іншого шару."
                        ),
                    )
                )

    ids = sorted(sid for sid, cal in calibrations.items() if cal is not None and cal.anchors)
    for i, left in enumerate(ids):
        for right in ids[i + 1 :]:
            if anchor_sets_equal(calibrations[left].anchors, calibrations[right].anchors):
                conflicts.append(
                    LayerConflict(
                        kind="duplicate",
                        source_ids=(left, right),
                        message=(
                            f"Шари «{left}» і «{right}» мають однакові якорі — "
                            f"ймовірно, калібрування скопійоване між шарами."
                        ),
                    )
                )
    return conflicts
