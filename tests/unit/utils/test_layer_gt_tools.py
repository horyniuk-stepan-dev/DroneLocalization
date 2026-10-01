"""GT-oracle layer copies and map-error report (synthetic HDF5, no torch)."""

import json

import h5py
import numpy as np
import pytest

from scripts.layer_gt_tools import (
    LayerGTError,
    SimulatorRun,
    build_oracle_layer,
    map_error_report,
)
from src.calibration.multi_anchor_calibration import MultiAnchorCalibration
from src.core.layer_status import CALIBRATION_OWNER_KEY, propagation_matches_anchors
from src.geometry.coordinates import CoordinateConverter

W, H, N = 1280, 720, 6
KEYFRAMES = [0, 2, 3, 5]
PROJ = {"mode": "WEB_MERCATOR", "reference_gps": None}


def _affine(i):
    return np.array([[0.5, 0.0, 2905000.0 + 150.0 * i], [0.0, -0.5, 6175000.0]])


def _centre_gps(affine):
    x, y = affine @ np.array([W / 2, H / 2, 1.0])
    return CoordinateConverter.from_metadata(PROJ).metric_to_gps(float(x), float(y))


def _make_layer(tmp_path, sha="a" * 64, map_offset_m=40.0, anchor_shift=0.0):
    layer = tmp_path / "layer"
    layer.mkdir()
    db = layer / "database.h5"
    with h5py.File(db, "w") as f:
        md = f.create_group("metadata")
        md.attrs.update(
            {
                "num_frames": N,
                "frame_step": 30,
                "frame_width": W,
                "frame_height": H,
                "source_sha256": sha,
            }
        )
        lf = f.create_group("local_features")
        kp = np.zeros(N, np.int16)
        kp[KEYFRAMES] = 500
        lf.create_dataset("kp_counts", data=kp)
        grp = f.create_group("calibration")
        grp.attrs["projection_json"] = json.dumps(PROJ)
        affines = np.stack([_affine(i) for i in range(N)])
        affines[:, 0, 2] += map_offset_m  # a propagated map that is off by 40 m
        origin = np.full(N, 2, np.uint8)
        origin[0] = 1
        affines[0] = _affine(0)
        affines[0, 0, 2] += anchor_shift
        grp.create_dataset("frame_affine", data=affines)
        grp.create_dataset("frame_origin", data=origin)
        grp.create_dataset("frame_georef_status", data=np.ones(N, np.uint8))
        gps = np.array([_centre_gps(a) for a in affines])
        f.create_dataset("frame_gps", data=gps)
    (layer / "vectors.lance").mkdir()
    (layer / "vectors.lance" / "data.bin").write_bytes(b"idx")

    run_dir = tmp_path / "sim"
    run_dir.mkdir()
    slots = [
        {
            "slot": i,
            "affine": _affine(i).tolist(),
            "ground_center_valid": True,
            "ground_center_gps": list(_centre_gps(_affine(i))),
        }
        for i in range(N)
    ]
    gt = {"frame_step": 30, "frame_size": [W, H], "projection": PROJ, "slots": slots}
    (run_dir / "ground_truth.json").write_text(json.dumps(gt), encoding="utf-8")
    (run_dir / "manifest.json").write_text(
        json.dumps({"files": {"video": {"sha256": "a" * 64}}}), encoding="utf-8"
    )
    return db, SimulatorRun(run_dir)


def test_report_measures_map_error_against_gt(tmp_path):
    db, run = _make_layer(tmp_path)
    report = map_error_report(db, run)
    assert report["supported_keyframes"] == len(KEYFRAMES)
    assert report["supported_anchor"]["max_m"] < 0.5
    # Optimized keyframes carry the synthetic 40 projected-metre offset
    # (~26 ground metres at this latitude).
    assert 20 < report["supported_optimized"]["median_m"] < 30


def test_oracle_pins_keyframes_and_leaves_source_untouched(tmp_path):
    db, run = _make_layer(tmp_path)
    before = db.read_bytes()
    out = tmp_path / "oracle" / "low"
    result = build_oracle_layer(db, run, out, source_id="low")

    assert db.read_bytes() == before
    assert result["pinned_keyframes"] == len(KEYFRAMES)
    assert (out / "vectors.lance" / "data.bin").exists()
    assert not out.with_name("low.partial").exists()
    with h5py.File(out / "database.h5", "r") as f:
        grp = f["calibration"]
        assert grp.attrs["optimizer"] == "gt_oracle"
        assert grp["frame_valid"][:].nonzero()[0].tolist() == KEYFRAMES
        assert set(grp["frame_georef_status"][:][KEYFRAMES]) == {1}
        assert set(grp["frame_georef_status"][:][[1, 4]]) == {3}
        np.testing.assert_allclose(grp["frame_affine"][3], _affine(3))
        anchors_json = grp.attrs["anchors_json"]

    cal = MultiAnchorCalibration()
    cal.load(str(out / "calibration.json"))
    assert [a.frame_id for a in cal.anchors] == KEYFRAMES
    assert cal.extra_metadata[CALIBRATION_OWNER_KEY] == "low"
    # GUI status reads "ready", not "stale": HDF5 anchors == calibration anchors.
    assert propagation_matches_anchors(anchors_json, cal.anchors) is True
    assert map_error_report(out / "database.h5", run)["supported"]["max_m"] < 0.5


def test_refuses_ground_truth_of_another_recording(tmp_path):
    db, run = _make_layer(tmp_path, sha="b" * 64)
    with pytest.raises(LayerGTError, match="another recording"):
        build_oracle_layer(db, run, tmp_path / "o", source_id="x")
    assert not (tmp_path / "o").exists()


def test_refuses_when_anchor_slots_do_not_match_gt(tmp_path):
    db, run = _make_layer(tmp_path, anchor_shift=500.0)
    with pytest.raises(LayerGTError, match="does not describe these slots"):
        build_oracle_layer(db, run, tmp_path / "o", source_id="x")


def test_refuses_frame_step_mismatch_and_existing_output(tmp_path):
    db, run = _make_layer(tmp_path)
    gt = json.loads(run.ground_truth.read_text(encoding="utf-8"))
    gt["frame_step"] = 15
    run.ground_truth.write_text(json.dumps(gt), encoding="utf-8")
    with pytest.raises(LayerGTError, match="frame_step"):
        build_oracle_layer(db, run, tmp_path / "o", source_id="x")
    (tmp_path / "exists").mkdir()
    with pytest.raises(LayerGTError, match="already exists"):
        build_oracle_layer(db, run, tmp_path / "exists", source_id="x")
