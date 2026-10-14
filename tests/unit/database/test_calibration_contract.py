"""Simulator anchor slots that a database build keeps as keyframes."""

import json

from config.app import AppConfig
from src.database.calibration_contract import simulator_anchor_frame_ids


def _write_calibration(
    folder, *, tool="FlightSimulator", video="flight.mp4", step=30, slots=(0, 7, 12)
):
    data = {
        "version": "2.4",
        "keyframe_selection": {"criterion": "overlap", "frame_step": step},
        "generator": {"tool": tool, "video_file": f"output\\layer\\{video}"},
        "anchors": [{"frame_id": slot, "affine_matrix": [[1, 0, 0], [0, 1, 0]]} for slot in slots],
    }
    if step is None:
        del data["keyframe_selection"]["frame_step"]
    (folder / "calibration.json").write_text(json.dumps(data), encoding="utf-8")
    return folder / video


def test_anchor_slots_of_this_simulator_video_are_returned(tmp_path):
    video = _write_calibration(tmp_path)

    assert simulator_anchor_frame_ids(video, 30) == {0, 7, 12}


def test_video_name_is_compared_without_case(tmp_path):
    _write_calibration(tmp_path, video="Flight.MP4")

    assert simulator_anchor_frame_ids(tmp_path / "flight.mp4", 30) == {0, 7, 12}


def test_slots_outside_the_video_are_dropped(tmp_path):
    video = _write_calibration(tmp_path, slots=(-1, 3, 40, 41))

    assert simulator_anchor_frame_ids(video, 30, num_slots=41) == {3, 40}


def test_no_calibration_beside_the_video(tmp_path):
    assert simulator_anchor_frame_ids(tmp_path / "flight.mp4", 30) == set()


def test_calibration_of_another_tool_is_ignored(tmp_path):
    video = _write_calibration(tmp_path, tool="DroneLocalization")

    assert simulator_anchor_frame_ids(video, 30) == set()


def test_calibration_of_another_video_is_ignored(tmp_path):
    _write_calibration(tmp_path, video="other.mp4")

    assert simulator_anchor_frame_ids(tmp_path / "flight.mp4", 30) == set()


def test_other_frame_step_names_other_images(tmp_path):
    video = _write_calibration(tmp_path, step=15)

    assert simulator_anchor_frame_ids(video, 30) == set()


def test_calibration_without_frame_step_is_ignored(tmp_path):
    video = _write_calibration(tmp_path, step=None)

    assert simulator_anchor_frame_ids(video, 30) == set()


def test_unreadable_calibration_is_ignored(tmp_path):
    (tmp_path / "calibration.json").write_text("{not json", encoding="utf-8")

    assert simulator_anchor_frame_ids(tmp_path / "flight.mp4", 30) == set()


def test_flag_is_off_by_default():
    assert AppConfig().database.require_simulator_anchor_frames is False
