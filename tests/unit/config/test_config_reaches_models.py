"""Values from user_config.json must reach the objects they configure (audit 2026-10)."""

from __future__ import annotations

import numpy as np
import pytest

from config.app import AppConfig

# ── LightGlue / ALIKED constructor options ──────────────────────────────────


def test_lightglue_conf_reads_every_key():
    from src.models.model_manager import lightglue_conf

    block = {
        "depth_confidence": -1,
        "width_confidence": -1,
        "filter_threshold": 0.05,
        "flash": False,
        "mixed_precision": True,
    }
    assert lightglue_conf(block) == {
        "depth_confidence": -1.0,
        "width_confidence": -1.0,
        "filter_threshold": 0.05,
        "flash": False,
        "mp": True,
    }


def test_lightglue_defaults_equal_library_defaults():
    """Schema defaults = what ran before the keys were wired (no silent change)."""
    lightglue = pytest.importorskip("lightglue")
    lib = lightglue.LightGlue.default_conf
    lg = AppConfig().models.lightglue
    assert lg.depth_confidence == lib["depth_confidence"]
    assert lg.width_confidence == lib["width_confidence"]
    assert lg.filter_threshold == lib["filter_threshold"]
    assert lg.flash == lib["flash"]
    assert lg.mixed_precision == lib["mp"]


def test_aliked_defaults_equal_library_defaults():
    lightglue = pytest.importorskip("lightglue")
    lib = lightglue.ALIKED.default_conf
    al = AppConfig().models.aliked
    assert al.model_name == lib["model_name"]
    assert al.detection_threshold == lib["detection_threshold"]
    assert al.nms_radius == lib["nms_radius"]


def test_aliked_conf_from_config():
    from src.models.model_manager import aliked_conf_from

    cfg = AppConfig().model_dump()
    cfg["models"]["aliked"].update(detection_threshold=-1, nms_radius=3, max_keypoints=1024)
    assert aliked_conf_from(cfg) == {
        "model_name": "aliked-n16",
        "max_num_keypoints": 1024,
        "detection_threshold": -1.0,
        "nms_radius": 3,
    }


def test_load_lightglue_passes_config(monkeypatch):
    import src.models.model_manager as mm

    seen = {}

    class FakeLightGlue:
        def __init__(self, features, **conf):
            seen["features"] = features
            seen["conf"] = conf

        def eval(self):
            return self

        def to(self, device):
            return self

    monkeypatch.setattr(mm, "LightGlue", FakeLightGlue)
    cfg = AppConfig().model_dump()
    cfg["models"]["lightglue"].update(depth_confidence=-1, width_confidence=-1)
    manager = mm.ModelManager.__new__(mm.ModelManager)
    manager.config = cfg
    manager.device = "cpu"
    manager.models = {}
    import threading

    manager._model_lock = threading.RLock()
    monkeypatch.setattr(manager, "_ensure_vram_available", lambda *_: None, raising=False)
    monkeypatch.setattr(manager, "_register_model_usage", lambda *_: None, raising=False)
    manager.load_lightglue(features="aliked")
    assert seen["features"] == "aliked"
    assert seen["conf"]["depth_confidence"] == -1.0
    assert seen["conf"]["width_confidence"] == -1.0


# ── PoseLib inlier mask ─────────────────────────────────────────────────────


def _homography_with_outliers(n_in=160, n_out=40, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.uniform(0, 1000, (n_in + n_out, 2))
    H = np.array([[1.1, 0.05, 10.0], [0.02, 0.95, -5.0], [1e-5, 2e-5, 1.0]])
    xh = np.c_[x, np.ones(len(x))] @ H.T
    y = xh[:, :2] / xh[:, 2:]
    y[:n_out] += rng.uniform(50, 100, (n_out, 2))
    return x, y, n_out


@pytest.mark.parametrize("use_mad", [False, True])
def test_poselib_mask_marks_only_inliers(use_mad):
    pytest.importorskip("poselib")
    from src.geometry.transformations import GeometryTransforms

    x, y, n_out = _homography_with_outliers()
    H, mask = GeometryTransforms.estimate_homography(
        x, y, 3.0, backend="poselib", use_mad_ransac=use_mad
    )
    assert H is not None
    m = mask.ravel().astype(bool)
    # Before the fix any inlier set the WHOLE mask (numpy scalar-bool indexing).
    assert not m[:n_out].any()
    assert m[n_out:].sum() >= 0.95 * (len(m) - n_out)


# ── live_stream → VideoSourceConfig ─────────────────────────────────────────


def test_video_source_config_from_app_config():
    from src.video.video_source import VideoSourceConfig

    cfg = AppConfig().model_dump()
    cfg["live_stream"].update(reconnect_attempts=9, reconnect_delay_sec=0.5, buffer_size=3)
    vc = VideoSourceConfig.from_app_config("rtsp://x/y", cfg)
    assert (vc.reconnect_attempts, vc.reconnect_delay_sec, vc.buffer_size) == (9, 0.5, 3)
    assert vc.source == "rtsp://x/y"


# ── object_tracking.tracked_classes ─────────────────────────────────────────


def test_tracker_keeps_only_tracked_classes():
    pytest.importorskip("supervision")
    from src.tracking.object_tracker import ObjectTracker

    tracker = ObjectTracker({"track_activation_threshold": 0.25, "tracked_classes": [2]})
    dets = [
        {"class_id": 0, "confidence": 0.9, "bbox": [100.0, 100.0, 200.0, 300.0]},
        {"class_id": 2, "confidence": 0.9, "bbox": [500.0, 500.0, 600.0, 600.0]},
    ]
    tracked = tracker.update(dets, (1080, 1920))
    assert [t.class_id for t in tracked] == [2]


def test_tracker_empty_class_list_tracks_everything():
    pytest.importorskip("supervision")
    from src.tracking.object_tracker import ObjectTracker

    tracker = ObjectTracker({"track_activation_threshold": 0.25, "tracked_classes": []})
    dets = [
        {"class_id": 0, "confidence": 0.9, "bbox": [100.0, 100.0, 200.0, 300.0]},
        {"class_id": 2, "confidence": 0.9, "bbox": [500.0, 500.0, 600.0, 600.0]},
    ]
    assert len(tracker.update(dets, (1080, 1920))) == 2


# ── keypoint order before database truncation ───────────────────────────────


def test_keypoints_ordered_by_score_numpy_and_torch():
    from src.models.wrappers.feature_extractor import FeatureExtractor

    kp = np.array([[0, 0], [0, 10], [0, 20], [0, 30]], dtype=np.float32)  # raster order
    desc = np.arange(4, dtype=np.float32)[:, None]
    scores = np.array([0.1, 0.9, 0.3, 0.7], dtype=np.float32)
    k2, d2 = FeatureExtractor._by_score(kp, desc, scores)
    assert d2.ravel().tolist() == [1, 3, 2, 0]
    assert k2[0].tolist() == [0, 10]

    torch = pytest.importorskip("torch")
    kt, dt = FeatureExtractor._by_score(
        torch.from_numpy(kp), torch.from_numpy(desc), torch.from_numpy(scores)
    )
    assert dt.ravel().tolist() == [1, 3, 2, 0]


def test_by_score_without_scores_is_identity():
    from src.models.wrappers.feature_extractor import FeatureExtractor

    kp = np.zeros((3, 2), np.float32)
    desc = np.arange(3, dtype=np.float32)[:, None]
    k2, d2 = FeatureExtractor._by_score(kp, desc, None)
    assert d2 is desc and k2 is kp
