"""Calibration contract between a recorded video and the database built from it.

A calibration anchor's affine belongs to one exact image, i.e. one DB slot.
FlightSimulator writes ``calibration.json`` next to its video and puts its anchors
on the slots that its own run of the database image selector kept. A later
database build can keep slightly different slots — borderline overlap decisions
flip between GPU runs, or ``database.*`` changed since the recording — and
propagation then refuses every anchor whose slot has no features. Reading that
calibration before the build lets the builder keep exactly those slots.
"""

from __future__ import annotations

import json
from pathlib import Path

from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

SIMULATOR_CALIBRATION_NAME = "calibration.json"


def simulator_anchor_frame_ids(
    video_path: str | Path, frame_step: int, num_slots: int | None = None
) -> set[int]:
    """Anchor slots of the FlightSimulator calibration recorded with this video.

    Returns an empty set unless ``calibration.json`` beside the video was written
    by FlightSimulator for this very video file and with the same ``frame_step``
    (otherwise its slot numbers mean other images). Slots outside
    ``[0, num_slots)`` are dropped.
    """
    video = Path(video_path)
    path = video.parent / SIMULATOR_CALIBRATION_NAME
    if not path.is_file():
        return set()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        logger.warning(f"Calibration beside the video is unreadable ({path}): {exc}")
        return set()
    if not isinstance(data, dict):
        return set()
    generator = data.get("generator") or {}
    if not isinstance(generator, dict) or generator.get("tool") != "FlightSimulator":
        return set()

    recorded_video = str(generator.get("video_file") or "").replace("\\", "/").rsplit("/", 1)[-1]
    if recorded_video and recorded_video.lower() != video.name.lower():
        logger.warning(
            f"{path} belongs to the video {recorded_video!r}, not {video.name!r} — "
            "its anchor slots are not kept"
        )
        return set()

    selection = data.get("keyframe_selection") or {}
    recorded_step = selection.get("frame_step") if isinstance(selection, dict) else None
    try:
        recorded_step = int(recorded_step)
    except (TypeError, ValueError):
        logger.warning(
            f"{path} does not record its frame_step (calibration older than v2.4) — "
            "its anchor slots are not kept"
        )
        return set()
    if recorded_step != int(frame_step):
        logger.warning(
            f"{path} was recorded with frame_step={recorded_step}, the database uses "
            f"frame_step={frame_step} — its anchor slots name other images and are not kept"
        )
        return set()

    slots: set[int] = set()
    for anchor in data.get("anchors") or []:
        try:
            slot = int(anchor["frame_id"])
        except (KeyError, TypeError, ValueError):
            continue
        if slot >= 0 and (num_slots is None or slot < int(num_slots)):
            slots.add(slot)
    return slots
