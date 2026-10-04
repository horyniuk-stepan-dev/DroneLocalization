"""Localizer with localization.rotation_mode = "continuous" (geometric fakes, CPU only).

The local "model" is ORB + brute-force Hamming matching and the global
descriptor is a grey thumbnail: both cheap, deterministic, and the thumbnail is
rotation-sensitive like DINO, so the angle scan has something to choose from.
What is checked is the geometry the mode adds: the rotation folded into H, the
reported centre / FOV / optical-flow point in ORIGINAL frame coordinates, and the
angle prior handed to the next keyframe.
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "fixtures"))
from localizer_fakes import FakeCalibration  # noqa: E402

W, H = 640, 360


def _scene(seed: int = 3) -> np.ndarray:
    rng = np.random.default_rng(seed)
    img = np.full((H, W, 3), 90, np.uint8)
    for _ in range(260):
        c = tuple(int(v) for v in rng.integers(0, 255, 3))
        x, y = int(rng.integers(0, W)), int(rng.integers(0, H))
        r = int(rng.integers(4, 22))
        if rng.random() < 0.5:
            cv2.circle(img, (x, y), r, c, -1)
        else:
            cv2.rectangle(img, (x, y), (x + r, y + int(r * 1.5)), c, -1)
    return cv2.GaussianBlur(img, (3, 3), 0)


def _thumb(image, valid_mask=None):
    g = cv2.cvtColor(np.ascontiguousarray(image), cv2.COLOR_RGB2GRAY).astype(np.float32)
    t = cv2.resize(g, (32, 18), interpolation=cv2.INTER_AREA).ravel()
    v = np.ones_like(t, bool)
    if valid_mask is not None:
        v = cv2.resize((valid_mask > 128).astype(np.float32), (32, 18)).ravel() > 0.9
    t = t - t[v].mean()
    t[~v] = 0
    return (t / (np.linalg.norm(t) + 1e-9)).astype(np.float32)


class _Extractor:
    def __init__(self):
        self.orb = cv2.ORB_create(nfeatures=1500)

    def extract_global_descriptor(self, image, valid_mask=None):
        return _thumb(image, valid_mask)

    def extract_global_descriptors_multi(self, images, valid_masks=None):
        masks = valid_masks or [None] * len(images)
        return np.stack([_thumb(i, m) for i, m in zip(images, masks)])

    def extract_local_features(self, image, static_mask=None):
        g = cv2.cvtColor(np.ascontiguousarray(image), cv2.COLOR_RGB2GRAY)
        m = None if static_mask is None else (static_mask > 128).astype(np.uint8) * 255
        kps, desc = self.orb.detectAndCompute(g, m)
        pts = np.array([k.pt for k in kps], np.float32).reshape(-1, 2)
        if desc is None:
            desc = np.zeros((0, 32), np.uint8)
        return {"keypoints": pts, "descriptors": desc, "image_size": np.array(g.shape[:2])}


class _Matcher:
    def __init__(self):
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    def match(self, q, r):
        if len(q["descriptors"]) < 2 or len(r["descriptors"]) < 2:
            return np.empty((0, 2)), np.empty((0, 2))
        ms = self.bf.match(q["descriptors"], r["descriptors"])
        return (
            np.array([q["keypoints"][m.queryIdx] for m in ms], np.float32),
            np.array([r["keypoints"][m.trainIdx] for m in ms], np.float32),
        )


class _DB:
    def __init__(self, ref, extractor):
        self.ref = ref
        self.features = extractor.extract_local_features(ref)
        self.global_descriptors = _thumb(ref)[None]
        self.lance_table = None
        self.frame_rmse = None
        self.frame_disagreement = None

    def get_num_frames(self):
        return 1

    def get_frame_size(self, i):
        return self.ref.shape[:2]

    def get_frame_affine(self, i):  # reference pixels == metres
        return np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])

    def get_local_features(self, i):
        return self.features


def _localizer(mode: str, ref):
    pytest.importorskip("torch")  # Localizer imports torch / faiss at module level
    from src.localization.localizer import Localizer

    ext = _Extractor()
    cfg = {
        "localization": {
            "rotation_mode": mode,
            "retrieval_top_k": 1,
            "rotation_rescan_min_score": 0.0,
            "temporal_candidate_prior": False,
            "max_geometric_rmse_px": 4.0,
        },
        "homography": {"backend": "opencv", "use_mad_ransac": True},
    }
    return Localizer(
        database=_DB(ref, ext),
        feature_extractor=ext,
        matcher=_Matcher(),
        calibration=FakeCalibration(),
        config=cfg,
        ref_frame_width=W,
        ref_frame_height=H,
    )


def _query(ref, true_deg, shift=(25.0, -15.0)):
    R = np.vstack([cv2.getRotationMatrix2D((W / 2, H / 2), true_deg, 1.0), [0, 0, 1]])
    T = np.array([[1, 0, shift[0]], [0, 1, shift[1]], [0, 0, 1.0]])
    Wq = T @ R  # reference pixel -> query pixel
    return cv2.warpAffine(ref, Wq[:2], (W, H)), Wq


def _ref_point(Wq, x, y):
    p = np.linalg.inv(Wq) @ [x, y, 1.0]
    return p[:2] / p[2]


def _gps_to_metric(lat, lon):  # inverse of FakeCalibration's converter
    return np.array([(lon - 30.0) / 1e-5, (lat - 50.0) / 1e-5])


@pytest.mark.parametrize("true_deg", [0, 30, 115, 250])
def test_continuous_centre_fov_and_prior(true_deg):
    ref = _scene()
    query, Wq = _query(ref, true_deg)
    loc = _localizer("continuous", ref)
    res = loc.localize_frame(query)
    assert res["success"], res.get("error")
    np.testing.assert_allclose(res["raw_metric"], _ref_point(Wq, W / 2, H / 2), atol=2.0)
    # rotation that undoes the query's: 360 - true
    assert abs(((res["rotation_deg"] + true_deg + 180) % 360) - 180) < 3.0
    assert abs(((loc._last_best_angle + true_deg + 180) % 360) - 180) < 1.0
    # H handed to OF / objects maps ORIGINAL frame pixels; angle is 0 downstream
    assert loc.last_state["global_angle"] == 0
    corner = np.array([[W - 1.0, H - 1.0]])
    from src.geometry.transformations import GeometryTransforms as G

    np.testing.assert_allclose(
        G.apply_homography(corner, loc.last_state["H"])[0],
        _ref_point(Wq, W - 1.0, H - 1.0),
        atol=3.0,
    )


def test_continuous_optical_flow_point_in_original_coords():
    ref = _scene()
    query, Wq = _query(ref, 115)
    loc = _localizer("continuous", ref)
    assert loc.localize_frame(query)["success"]
    dx, dy = 20.0, -12.0  # image content moved by (dx, dy) since the keyframe
    out = loc.localize_optical_flow(dx, dy, dt=1.0, rot_width=W, rot_height=H)
    assert out["success"]
    np.testing.assert_allclose(loc._last_of_raw, _ref_point(Wq, W / 2 - dx, H / 2 - dy), atol=2.0)


def test_quarter_mode_unchanged_for_cardinal_angle():
    """Exact 90° query: both modes must give the same centre."""
    ref = _scene()
    query, Wq = _query(ref, 90, shift=(0.0, 0.0))
    q = _localizer("quarter", ref).localize_frame(query)
    c = _localizer("continuous", ref).localize_frame(query)
    assert q["success"] and c["success"]
    np.testing.assert_allclose(q["raw_metric"], c["raw_metric"], atol=2.0)


def test_layer_search_path_continuous():
    """Layer search composes the same view rotation into its observation."""
    pytest.importorskip("torch")
    from types import SimpleNamespace

    from src.localization.localizer import Localizer

    ref = _scene()
    query, Wq = _query(ref, 115)
    ext = _Extractor()
    db = _DB(ref, ext)
    calibration = FakeCalibration()
    manager = SimpleNamespace(
        get_database=lambda _sid: db,
        get_matches_by_source=lambda desc, *_a, **_k: {
            "a": [(0, float(desc @ db.global_descriptors[0]))]
        },
    )
    loc = Localizer(
        db,
        ext,
        _Matcher(),
        calibration,
        config={
            "localization": {
                "rotation_mode": "continuous",
                "max_geometric_rmse_px": 4.0,
                "layer_search": {"enabled": True, "confirmations": 1, "budget_ms": 60000},
            },
            "homography": {"backend": "opencv", "use_mad_ransac": True},
        },
        db_manager=manager,
        calib_manager=SimpleNamespace(get=lambda _sid: calibration),
        ref_frame_width=W,
        ref_frame_height=H,
    )
    res = loc.localize_frame(query, timestamp=1.0)
    if not res.get("success"):
        res = loc.localize_frame(query, timestamp=2.0)
    assert res["success"], res.get("error")
    np.testing.assert_allclose(res["raw_metric"], _ref_point(Wq, W / 2, H / 2), atol=2.0)
