"""GEO accuracy profile with soft anchor points, end to end (graph -> refine -> save -> reload)."""

from unittest.mock import Mock

import numpy as np

from src.calibration.multi_anchor_calibration import MultiAnchorCalibration
from src.geometry.coordinates import CoordinateConverter
from src.geometry.frame_transform import get_frame_transform, project_points
from src.workers.propagation_pipeline import PropagationPipeline
from tests.unit.geometry.test_relative_map import make_database

QA = np.array([[100, 50], [40, 20], [160, 20], [160, 80], [40, 80]], float)


def tilted_truth(n=4):
    base = np.array([[0.7, 0.0, 3e6], [0.0, -0.7, 6e6], [0.0, 0.0, 1.0]])
    tilt = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [4e-4, -2e-4, 1.0]])
    return {
        i: base @ np.array([[1.0, 0, 25.0 * i], [0, 1.0, 0], [0, 0, 1.0]]) @ tilt for i in range(n)
    }


def run(tmp_path, name, monkeypatch, propagation):
    truth = tilted_truth()
    db = make_database(tmp_path / f"{name}.h5", n=4)
    calibration = MultiAnchorCalibration(CoordinateConverter("WEB_MERCATOR"))
    for a in (0, 3):
        ref = project_points(truth[a], QA)
        affine = np.linalg.lstsq(np.column_stack([QA, np.ones(5)]), ref, rcond=None)[0].T
        calibration.add_anchor(a, affine, {"points_2d": QA.tolist(), "points_metric": ref.tolist()})
    pixels = np.array([[x, y] for x in np.linspace(80, 190, 12) for y in np.linspace(10, 90, 9)])
    world = project_points(truth[0], pixels)
    features = {i: {"keypoints": project_points(np.linalg.inv(truth[i]), world)} for i in range(4)}
    matcher = Mock()
    matcher.match.side_effect = lambda a, b: (a["keypoints"], b["keypoints"])
    pipeline = PropagationPipeline(
        db,
        calibration,
        matcher,
        config={
            "propagation": {"high_accuracy": True, "subpixel_refinement": False, **propagation}
        },
        error_callback=Mock(),
        completed_callback=Mock(),
    )
    monkeypatch.setattr(pipeline, "_prefetch_features", lambda n: features)
    monkeypatch.setattr(pipeline, "_detect_loop_closures", lambda *a: 0)
    try:
        pipeline._propagate()
        pipeline._error_cb.assert_not_called()
        pipeline._completed_cb.assert_called_once()
        grid = np.array([[x, y] for x in (0, 100, 200) for y in (0, 50, 100)], float)
        errors = {
            i: float(
                np.linalg.norm(
                    project_points(get_frame_transform(db, i), grid)
                    - project_points(truth[i], grid),
                    axis=1,
                ).max()
            )
            for i in range(4)
        }
        return errors, pipeline._relative_alignment_report, db.frame_homography.copy()
    finally:
        db.close()


def test_soft_anchor_points_refine_anchor_frames_end_to_end(tmp_path, monkeypatch):
    frozen, frozen_report, _ = run(tmp_path, "frozen", monkeypatch, {})
    soft, report, homographies = run(
        tmp_path,
        "soft",
        monkeypatch,
        {
            "accuracy_anchor_points": True,
            "projective_solver": "lm_sparse",
            "projective_window_frames": 0,
        },
    )
    assert frozen_report.get("soft_anchor_frames", 0) == 0
    assert report["soft_anchor_frames"] == 2
    assert report["accuracy_options"]["projective_solver"] == "lm_sparse"
    # Frozen anchors keep their biased affine; soft anchors follow points + images.
    assert max(soft.values()) < 0.5 * max(frozen.values())
    assert abs(homographies[0][2, :2]).max() > 0  # anchor frame itself was refined
