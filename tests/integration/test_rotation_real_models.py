"""Continuous vs quarter rotation with the real ALIKED + LightGlue (slow).

A frame of tests/fixtures/flight_clip.mp4 is the single reference; the query is
the same frame rotated about its centre and shifted, so the ground truth of the
query centre is known exactly. Global retrieval is a grey thumbnail (rotation
sensitive, like DINO) so the test needs no DINOv3 weights.

Skipped when lightglue or its ALIKED weights are not available offline
(models/.cache/torch/hub/checkpoints on the dev machines).

Measured 2026-10-04 (CPU, 1280x720, shift 60/-40 px), inliers / centre error:
  70 deg: quarter 113 / 5.4 px, continuous 458 / 0.25 px
  115 deg: quarter fails,       continuous 782 / 0.09 px
  290 deg: quarter 87 / 3.3 px, continuous 583 / 0.36 px
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

pytestmark = pytest.mark.slow

ROOT = Path(__file__).resolve().parents[2]
CLIP = ROOT / "tests" / "fixtures" / "flight_clip.mp4"
sys.path.insert(0, str(ROOT / "tests" / "fixtures"))
sys.path.insert(0, str(ROOT / "tests" / "unit" / "localization"))


def _models():
    torch = pytest.importorskip("torch")
    lightglue = pytest.importorskip("lightglue")
    import config  # noqa: F401  (points TORCH_HOME at models/.cache)

    hub = Path(torch.hub.get_dir()) / "checkpoints"
    if not (hub / "aliked-n16.pth").exists() or not any(hub.glob("aliked_lightglue*.pth")):
        pytest.skip(f"ALIKED/LightGlue weights not cached in {hub}")
    ext = lightglue.ALIKED(max_num_keypoints=2048).eval()
    lg = lightglue.LightGlue(features="aliked").eval()
    return torch, ext, lg


def _frame():
    if not CLIP.exists():
        pytest.skip("flight_clip.mp4 fixture missing")
    cap = cv2.VideoCapture(str(CLIP))
    cap.set(cv2.CAP_PROP_POS_FRAMES, 60)
    ok, f = cap.read()
    if not ok:
        pytest.skip("cannot decode flight_clip.mp4")
    return cv2.resize(f[:, :, ::-1], (640, 360), interpolation=cv2.INTER_AREA).copy()


@pytest.fixture(scope="module")
def rig():
    torch, ext, lg = _models()
    from test_rotation_continuous import _DB, _thumb

    def aliked(img, mask=None):
        t = torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1)[None].float() / 255.0
        with torch.no_grad():
            o = ext({"image": t})
        k, d = o["keypoints"][0].numpy(), o["descriptors"][0].numpy()
        if mask is not None and len(k):
            ix = np.clip(np.round(k[:, 0]).astype(int), 0, mask.shape[1] - 1)
            iy = np.clip(np.round(k[:, 1]).astype(int), 0, mask.shape[0] - 1)
            keep = mask[iy, ix] > 128
            k, d = k[keep], d[keep]
        return {"keypoints": k, "descriptors": d, "image_size": np.array(img.shape[:2])}

    class Extractor:
        def extract_global_descriptor(self, image, valid_mask=None):
            return _thumb(image, valid_mask)

        def extract_global_descriptors_multi(self, images, valid_masks=None):
            masks = valid_masks or [None] * len(images)
            return np.stack([_thumb(i, m) for i, m in zip(images, masks)])

        def extract_local_features(self, image, static_mask=None):
            return aliked(image, static_mask)

    class Matcher:
        def match(self, q, r):
            def pack(f):
                return {
                    "keypoints": torch.from_numpy(f["keypoints"]).float()[None],
                    "descriptors": torch.from_numpy(f["descriptors"]).float()[None],
                    "image_size": torch.tensor(
                        [[int(f["image_size"][1]), int(f["image_size"][0])]]
                    ),
                }

            with torch.no_grad():
                m = lg({"image0": pack(q), "image1": pack(r)})["matches"][0].numpy()
            return q["keypoints"][m[:, 0]], r["keypoints"][m[:, 1]]

    return Extractor, Matcher, _DB


def _run(rig, mode, ref, query):
    from localizer_fakes import FakeCalibration

    from src.localization.localizer import Localizer

    Extractor, Matcher, DB = rig
    ext = Extractor()
    h, w = ref.shape[:2]
    loc = Localizer(
        database=DB(ref, ext),
        feature_extractor=ext,
        matcher=Matcher(),
        calibration=FakeCalibration(),
        config={
            "localization": {
                "rotation_mode": mode,
                "retrieval_top_k": 1,
                "rotation_rescan_min_score": 0.0,
                "temporal_candidate_prior": False,
            },
            "homography": {"backend": "opencv", "use_mad_ransac": True},
        },
        ref_frame_width=w,
        ref_frame_height=h,
    )
    return loc.localize_frame(query)


@pytest.mark.parametrize("true_deg", [70, 115])
def test_continuous_beats_quarter_on_real_models(rig, true_deg):
    ref = _frame()
    h, w = ref.shape[:2]
    R = np.vstack([cv2.getRotationMatrix2D((w / 2, h / 2), true_deg, 1.0), [0, 0, 1]])
    Wq = np.array([[1, 0, 30], [0, 1, -20], [0, 0, 1.0]]) @ R
    query = cv2.warpAffine(ref, Wq[:2], (w, h))
    expected = (np.linalg.inv(Wq) @ [w / 2, h / 2, 1.0])[:2]

    cont = _run(rig, "continuous", ref, query)
    assert cont["success"], cont.get("error")
    assert np.hypot(*(np.asarray(cont["raw_metric"]) - expected)) < 2.0

    quarter = _run(rig, "quarter", ref, query)
    if quarter.get("success"):
        assert cont["inliers"] > 1.5 * quarter["inliers"]
