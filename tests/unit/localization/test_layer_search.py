from types import SimpleNamespace

import numpy as np
import pytest

from src.database.multi_database_manager import MultiDatabaseManager
from src.localization.layer_search import LayerHandoff, LayerObservation, LayerSearch, ScaleBelief
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


def test_many_sources_keep_active_layer_and_rotate_recovery_probe():
    configs = {
        f"s{i:02d}": SimpleNamespace(scale_layer=None, geo_bounds=None)
        for i in range(15)
    }
    manager = SimpleNamespace(
        all_source_ids=list(configs),
        get_source_config=configs.get,
    )
    search = LayerSearch({"localization": {"layer_search": {"max_sources_per_frame": 3}}})
    search.handoff.commit(observation("s00"), 0)
    probed = [search._select_sources(manager) for _ in range(8)]
    assert all(len(selected) == 3 and selected[0] == "s00" for selected in probed)
    assert set().union(*(set(selected) for selected in probed)) == set(configs)


def test_source_routing_prefers_neighbor_and_nearby_without_excluding_others():
    layer = SimpleNamespace(layer_id="high", neighbor_layer_ids=("low",))
    configs = {
        "high": SimpleNamespace(scale_layer=layer, geo_bounds=None),
        "low": SimpleNamespace(
            scale_layer=SimpleNamespace(layer_id="low"), geo_bounds=None
        ),
        "near": SimpleNamespace(
            scale_layer=None, geo_bounds=(49.0, 29.0, 51.0, 31.0),
            contains_point=lambda lat, lon: 49 <= lat <= 51 and 29 <= lon <= 31,
        ),
        "far": SimpleNamespace(scale_layer=None, geo_bounds=None),
    }
    manager = SimpleNamespace(all_source_ids=list(configs), get_source_config=configs.get)
    search = LayerSearch({"localization": {"layer_search": {"max_sources_per_frame": 3}}})
    search.handoff.commit(observation("high"), 0)
    seen = [search._select_sources(manager) for _ in range(2)]
    assert all(selected[:2] == ["high", "low"] for selected in seen)
    assert {selected[2] for selected in seen} == {"near", "far"}


def test_layer_probe_can_recover_source_outside_stale_active_filter():
    class Retriever:
        def __init__(self, frame):
            self.frame = frame

        def find_similar_frames(self, _query, top_k):
            return [(self.frame, 0.8)][:top_k]

    manager = MultiDatabaseManager.__new__(MultiDatabaseManager)
    manager._sources = {
        sid: SimpleNamespace(priority=0) for sid in ("active", "hidden")
    }
    manager._databases = {
        sid: SimpleNamespace(metadata={}) for sid in ("active", "hidden")
    }
    manager._retrievers = {
        "active": Retriever(1), "hidden": Retriever(2)
    }
    manager._active_source_ids = {"active"}
    manager._config = {}
    result = manager.get_matches_by_source(
        np.array([1.0, 0.0], dtype=np.float32), 1, source_ids=["hidden"]
    )
    assert result == {"hidden": [(2, 0.8)]}


def test_noop_scale_belief_does_not_repeat_identical_geometry(monkeypatch):
    loc = localizer()
    loc._layer_search.beliefs["b"] = ScaleBelief(0.0, 0.05, 0.0, 0)
    original = loc._geometric_verifier.verify
    attempted = []

    def counted(features, candidates, database):
        attempted.append((id(database), candidates[0][0]))
        return original(features, candidates, database)

    monkeypatch.setattr(loc._geometric_verifier, "verify", counted)
    observations = loc._layer_search.search(
        loc, np.zeros((100, 100, 3), dtype=np.uint8), None, 1.0
    )
    assert observations
    assert len(attempted) == 2  # one candidate in each source, not 3x per source
    assert loc._layer_search.last_diagnostics["verifications"] == 2


def test_distinct_scale_descriptors_are_batched(monkeypatch):
    loc = localizer()
    loc._layer_search.beliefs["b"] = ScaleBelief(np.log(1.4), 0.1, 0.0, 0)
    batches = []

    def extract_multi(frames):
        batches.append(len(frames))
        return np.tile(np.array([[1.0, 0.0]], dtype=np.float32), (len(frames), 1))

    monkeypatch.setattr(
        loc.feature_extractor, "extract_global_descriptors_multi", extract_multi,
        raising=False,
    )
    loc._layer_search.search(loc, np.zeros((100, 100, 3), dtype=np.uint8), None, 1.0)
    assert batches == [3]  # one belief ratio overlaps the no-op 1.0 scale




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


# 50 m/s due east at 50 N, in degrees of longitude per second.
EAST_50MPS = 50.0 / 71_687.0


def moving_track(controller, source="a"):
    controller.commit(observation(source, gps=(50.0, 30.0)), 0)
    controller.commit(observation(source, gps=(50.0, 30.0 + EAST_50MPS)), 1)


def test_motion_gate_accepts_single_track_consistent_fix_after_lost():
    controller = LayerHandoff(confirmations=2, max_speed_mps=350, motion_gate=True)
    moving_track(controller)
    candidate = observation("b", gps=(50.0, 30.0 + 5 * EAST_50MPS))
    assert controller.choose([candidate], 5) is candidate
    assert controller.reason == "motion_gated"
    assert controller.state == "TRACKING"


def test_motion_gate_is_off_by_default():
    controller = LayerHandoff(confirmations=2, max_speed_mps=350)
    moving_track(controller)
    candidate = observation("b", gps=(50.0, 30.0 + 5 * EAST_50MPS))
    assert controller.choose([candidate], 5) is None
    assert controller.reason == "awaiting_confirmation"


def test_motion_gate_rejects_fix_off_the_predicted_track():
    controller = LayerHandoff(confirmations=2, max_speed_mps=350, motion_gate=True)
    moving_track(controller)
    # Last confirmed position, 200 m behind the prediction; radius is 30 + 30 * 4 m.
    stale = observation("b", gps=(50.0, 30.0 + EAST_50MPS))
    assert controller.choose([stale], 5) is None
    assert controller.reason == "awaiting_confirmation"


def test_motion_gate_needs_a_velocity_estimate():
    controller = LayerHandoff(confirmations=2, max_speed_mps=350, motion_gate=True)
    controller.commit(observation("a"), 0)
    assert controller.choose([observation("b")], 4) is None


def test_motion_gate_expires_after_max_gap():
    controller = LayerHandoff(
        confirmations=2, max_speed_mps=350, motion_gate=True, motion_gate_max_gap_s=10
    )
    moving_track(controller)
    candidate = observation("b", gps=(50.0, 30.0 + 21 * EAST_50MPS))
    assert controller.choose([candidate], 21) is None


def test_motion_gate_bridges_frame_where_active_layer_did_not_verify():
    controller = LayerHandoff(confirmations=2, max_speed_mps=350, motion_gate=True)
    moving_track(controller)
    candidate = observation("b", gps=(50.0, 30.0 + 2 * EAST_50MPS))
    assert controller.choose([candidate], 2) is candidate


def test_motion_gate_keeps_discretionary_switch_behind_confirmations():
    controller = LayerHandoff(confirmations=2, max_speed_mps=350, motion_gate=True)
    moving_track(controller)
    here = (50.0, 30.0 + 2 * EAST_50MPS)
    old, new = observation("a", gps=here), observation("b", 50, gps=here)
    assert controller.choose([old, new], 2) is old
    assert controller.state == "HANDOFF_PENDING"


def test_motion_gate_does_not_shortcut_bootstrap():
    controller = LayerHandoff(confirmations=2, max_speed_mps=350, motion_gate=True)
    assert controller.choose([observation("a")], 1) is None


def test_layer_search_passes_motion_gate_settings():
    search = LayerSearch(
        {
            "localization": {
                "layer_search": {
                    "motion_gate": True,
                    "motion_gate_growth_mps": 12.0,
                    "motion_gate_max_gap_s": 4.0,
                }
            }
        }
    )
    assert search.handoff.motion_gate is True
    assert search.handoff.motion_gate_growth_mps == 12.0
    assert search.handoff.motion_gate_max_gap_s == 4.0


def override_cfg(monkeypatch, search, **overrides):
    original = search.cfg
    monkeypatch.setattr(
        search, "cfg", lambda key, default: overrides.get(key, original(key, default))
    )


class OnlyFrameOne(Database):
    def get_local_features(self, candidate):
        return {"valid": int(candidate) == 1}


def verification_order(monkeypatch, diverse):
    loc = localizer()
    loc.db_manager.get_database = {"b": OnlyFrameOne(True)}.get
    loc.db_manager.get_matches_by_source = lambda *_a, **_k: {"b": [(0, 0.9), (1, 0.5)]}
    loc._layer_search.beliefs["b"] = ScaleBelief(np.log(1.4), 0.1, 0.0, 0)
    override_cfg(monkeypatch, loc._layer_search, diverse_candidates=diverse, max_verifications=2)
    original = loc._geometric_verifier.verify
    attempted = []

    def counted(features, candidates, database):
        attempted.append(int(candidates[0][0]))
        return original(features, candidates, database)

    monkeypatch.setattr(loc._geometric_verifier, "verify", counted)
    loc._layer_search.search(loc, np.zeros((100, 100, 3), dtype=np.uint8), None, 1.0)
    return attempted


def test_diverse_candidates_try_each_frame_before_rescaling_one(monkeypatch):
    assert verification_order(monkeypatch, diverse=False) == [0, 0]
    assert verification_order(monkeypatch, diverse=True) == [0, 1]


def early_stop_localizer(monkeypatch, enabled, track=(50.0005, 30.0005)):
    # Default track: the image centre through the identity map in localizer().
    loc = localizer()
    loc._layer_search.handoff.commit(observation("b", gps=track), 0.0)
    loc._layer_search.handoff.commit(observation("b", gps=track), 1.0)
    override_cfg(
        monkeypatch,
        loc._layer_search,
        early_stop=enabled,
        early_stop_min_quality=0.0,
        early_stop_max_scale_ratio=1.5,
    )
    return loc


def test_early_stop_ends_search_on_track_consistent_active_fix(monkeypatch):
    loc = early_stop_localizer(monkeypatch, enabled=True)
    observations = loc._layer_search.search(loc, np.zeros((100, 100, 3), dtype=np.uint8), None, 2.0)
    assert [o.source_id for o in observations] == ["b"]
    assert observations[0].scale_ratio == pytest.approx(1.0)
    assert loc._layer_search.last_diagnostics["verifications"] == 1
    assert loc._layer_search.last_diagnostics["early_stop"] is True


def test_early_stop_off_verifies_every_source(monkeypatch):
    loc = early_stop_localizer(monkeypatch, enabled=False)
    loc._layer_search.search(loc, np.zeros((100, 100, 3), dtype=np.uint8), None, 2.0)
    assert loc._layer_search.last_diagnostics["verifications"] == 2
    assert loc._layer_search.last_diagnostics["early_stop"] is False


def test_early_stop_ignores_fix_off_the_track(monkeypatch):
    # Hovering 200 m north of the fix; the gate radius one second later is 60 m.
    loc = early_stop_localizer(monkeypatch, enabled=True, track=(50.0005 + 0.0018, 30.0005))
    assert loc._layer_search.handoff.predict(2.0) is not None
    loc._layer_search.search(loc, np.zeros((100, 100, 3), dtype=np.uint8), None, 2.0)
    assert loc._layer_search.last_diagnostics["early_stop"] is False
    assert loc._layer_search.last_diagnostics["verifications"] == 2
