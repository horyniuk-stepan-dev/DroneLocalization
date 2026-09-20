from types import SimpleNamespace

import numpy as np
import pytest

from src.localization.layer_search import LayerHandoff, LayerObservation, ScaleBelief
from src.localization.localizer import Localizer


def observation(source, quality=30, gps=(50.0, 30.0)):
    return LayerObservation(source, gps, quality, (), np.eye(3), (100, 100))


def test_bootstrap_needs_fresh_observations():
    controller = LayerHandoff(confirmations=2)
    candidate = observation("a")
    assert controller.choose([candidate], 1) is None
    assert controller.choose([candidate], 1) is None
    assert controller.active is None
    assert controller.choose([candidate], 2) is candidate
    assert controller.active is None  # selection alone must not commit source state
    controller.commit(candidate, 2)
    assert controller.active == "a"


def test_handoff_confirms_challenger_while_old_layer_still_tracks():
    controller = LayerHandoff(confirmations=2)
    old, new = observation("a"), observation("b", 50)
    controller.commit(old, 0)
    assert controller.choose([old, new], 1) is old
    controller.commit(old, 1)
    assert controller.state == "HANDOFF_PENDING"
    assert controller.choose([old, new], 2) is new
    controller.commit(new, 2)
    assert controller.active == "b"


def test_reacquisition_does_not_require_old_layer():
    controller = LayerHandoff(confirmations=2)
    controller.commit(observation("a"), 0)
    candidate = observation("b")
    assert controller.choose([candidate], 4) is None
    assert controller.choose([candidate], 5) is candidate


def test_near_equal_distant_solutions_are_ambiguous():
    controller = LayerHandoff(confirmations=1)
    a = observation("a")
    b = observation("b", gps=(50.01, 30.01))
    assert controller.choose([a, b], 1) is None
    assert controller.reason == "ambiguous_geometry"


def test_switch_does_not_hide_map_disagreement():
    controller = LayerHandoff(confirmations=1)
    a, b = observation("a"), observation("b", 100, gps=(50.01, 30.01))
    controller.commit(a, 0)
    assert controller.choose([a, b], 1) is a
    assert controller.reason == "layer_map_disagreement"


def test_hysteresis_keeps_active_layer_for_small_quality_difference():
    controller = LayerHandoff(confirmations=1)
    a, b = observation("a"), observation("b", 31)
    controller.commit(a, 0)
    assert controller.choose([a, b], 1) is a


def test_scale_uncertainty_grows_with_observation_age():
    belief = ScaleBelief(np.log(1.4), 0.05, 1, 90)
    fresh, stale = belief.candidates(1, 0.1), belief.candidates(5, 0.1)
    assert fresh[0] == pytest.approx(1.4)
    assert stale[1] < fresh[1] < fresh[0] < fresh[2] < stale[2]


class Extractor:
    def extract_global_descriptor(self, frame):
        return np.array([1.0, 0.0], dtype=np.float32)

    def extract_local_features(self, frame, static_mask=None):
        h, w = frame.shape[:2]
        xy = np.array(
            [
                (x, y)
                for x in np.linspace(w * 0.1, w * 0.9, 5)
                for y in np.linspace(h * 0.1, h * 0.9, 5)
            ],
            dtype=np.float32,
        )
        return {
            "keypoints": xy,
            "descriptors": np.ones((25, 8), dtype=np.float32),
            "image_size": np.array([h, w]),
        }


class Database:
    metadata = {"frame_width": 100, "frame_height": 100}
    frame_rmse = None
    frame_disagreement = None
    median_depth_scale = None

    def __init__(self, valid):
        self.valid = valid

    def get_local_features(self, candidate):
        return {"valid": self.valid}

    def get_frame_affine(self, candidate):
        return np.array([[1, 0, 0], [0, 1, 0]], dtype=float)

    def get_frame_size(self, candidate):
        return 100, 100


class Matcher:
    def match(self, query, reference):
        xy = query["keypoints"]
        return (xy, xy.copy()) if reference["valid"] else (xy[:0], xy[:0])


def localizer():
    databases = {"a": Database(False), "b": Database(True)}
    manager = SimpleNamespace(
        get_database=databases.get,
        get_matches_by_source=lambda *_args, **_kwargs: {"a": [(0, 0.99)], "b": [(0, 0.7)]},
    )
    converter = SimpleNamespace(metric_to_gps=lambda x, y: (50 + y * 1e-5, 30 + x * 1e-5))
    calibration = SimpleNamespace(converter=converter)
    return Localizer(
        databases["a"],
        Extractor(),
        Matcher(),
        calibration,
        config={
            "localization": {
                "auto_rotation": False,
                "layer_search": {
                    "enabled": True,
                    "confirmations": 2,
                    "budget_ms": 10000,
                },
            }
        },
        db_manager=manager,
        calib_manager=SimpleNamespace(get=lambda _: calibration),
    )


def test_lower_similarity_source_wins_after_geometry_with_shared_frame_ids():
    loc = localizer()
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    assert not loc.localize_frame(frame, timestamp=1)["success"]
    assert loc._active_source_id is None
    result = loc.localize_frame(frame, timestamp=2)
    assert result["success"]
    assert result["source_id"] == "b"
    assert result["matched_frame"] == 0
    assert loc.last_state["source_id"] == "b"
    assert set(loc._layer_search.beliefs) == {"b"}
    assert not loc.localize_frame(frame, timestamp=2)["success"]


def test_rejected_downstream_result_does_not_switch_source(monkeypatch):
    loc = localizer()
    original = loc.database
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    loc.localize_frame(frame, timestamp=1)
    monkeypatch.setattr(loc, "_localize_frame_impl", lambda *a, **kw: {"success": False})
    result = loc.localize_frame(frame, timestamp=2)
    assert not result["success"]
    assert loc.database is original
    assert loc._active_source_id is None
    assert not loc._layer_search.beliefs


def test_geometry_failure_reaches_other_scales_in_same_frame(monkeypatch):
    loc = localizer()
    loc._layer_search.handoff.confirmations = 1
    original = loc.feature_extractor.extract_local_features

    def extract(frame, static_mask=None):
        features = original(frame, static_mask)
        features["valid_scale"] = frame.shape[0] == 50
        return features

    def match(query, reference):
        xy = query["keypoints"]
        return (xy, xy * 2) if query["valid_scale"] and reference["valid"] else (xy[:0], xy[:0])

    monkeypatch.setattr(loc.feature_extractor, "extract_local_features", extract)
    monkeypatch.setattr(loc.matcher, "match", match)
    loc._scale_manager._prior = 1.4
    result = loc.localize_frame(np.zeros((100, 100, 3), dtype=np.uint8), timestamp=1)
    assert result["success"]
    assert result["source_id"] == "b"


def test_retrieval_fallback_is_not_a_confirmed_fix():
    loc = localizer()
    result = loc._result_builder.fallback(0, 0.99, loc.database, loc.calibration)
    assert result["success"] is False
    assert result["status"] == "candidate_only"


def test_zero_timestamp_is_a_valid_first_observation():
    loc = localizer()
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    assert loc.localize_frame(frame, timestamp=0)["error"] == "awaiting_confirmation"
    assert loc.localize_frame(frame, timestamp=0)["status"] == "stale"
    assert loc.localize_frame(frame, timestamp=1)["success"]


def test_downstream_exception_restores_source_filters_and_flow_state(monkeypatch):
    loc = localizer()
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    loc.localize_frame(frame, timestamp=0)
    database, trajectory = loc.database, loc.trajectory_filter
    state = {"source_id": "previous"}
    loc._last_state = state

    def fail(*args, **kwargs):
        loc._last_state = {"source_id": "b"}
        raise ValueError("conversion failed")

    monkeypatch.setattr(loc, "_localize_frame_impl", fail)
    with pytest.raises(ValueError, match="conversion failed"):
        loc.localize_frame(frame, timestamp=1)
    assert loc.database is database
    assert loc.trajectory_filter is trajectory
    assert loc.last_state is state
    assert loc._layer_search.handoff.active is None


def test_extrapolated_map_slot_cannot_confirm_localization():
    loc = localizer()
    loc.db_manager.get_database("b").frame_origin = np.array([4], dtype=np.uint8)
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    for timestamp in (0, 1):
        assert not loc.localize_frame(frame, timestamp=timestamp)["success"]
    assert not loc._layer_search.beliefs


def test_projective_tilt_can_confirm_without_an_altitude_measurement(monkeypatch):
    import cv2

    loc = localizer()
    loc._layer_search.handoff.confirmations = 1
    H = np.array([[1, .15, 3], [.1, .9, 4], [.001, .0005, 1]], dtype=float)

    def match(query, reference):
        xy = query["keypoints"]
        if not reference["valid"]:
            return xy[:0], xy[:0]
        return xy, cv2.perspectiveTransform(xy.reshape(-1, 1, 2), H).reshape(-1, 2)

    monkeypatch.setattr(loc.matcher, "match", match)
    result = loc.localize_frame(np.zeros((100, 100, 3), dtype=np.uint8), timestamp=0)
    assert result["status"] == "confirmed"
    assert result["coordinate_kind"] == "ground_observation"
    assert loc._layer_search.beliefs["b"].sigma > .05


def test_off_center_patch_does_not_support_image_center(monkeypatch):
    loc = localizer()
    loc._layer_search.handoff.confirmations = 1
    loc.config["localization"]["scale_pyramid"] = [1.0]
    loc._scale_manager._pyramid = (1.0,)
    original = loc.feature_extractor.extract_local_features

    def extract(frame, static_mask=None):
        features = original(frame, static_mask)
        features["keypoints"] *= .3
        return features

    monkeypatch.setattr(loc.feature_extractor, "extract_local_features", extract)
    result = loc.localize_frame(np.zeros((100, 100, 3), dtype=np.uint8), timestamp=0)
    assert not result["success"]
