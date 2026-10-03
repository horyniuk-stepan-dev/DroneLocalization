"""Retrieval-only A/B copies: scripts/swap_global_descriptors.py (no torch needed)."""

import json

import cv2
import h5py
import numpy as np
import pytest

import scripts.swap_global_descriptors as sgd
from src.database.schema_fingerprint import compute_fingerprint


def test_swapped_components_replace_only_global_fields():
    stored = {
        "global_backend": "dinov3",
        "descriptor_dim": 1024,
        "vlad_enabled": False,
        "local_extractor": "aliked",
        "frame_step": 30,
    }
    runtime = {
        "global_backend": "dinov3",
        "vlad_enabled": True,
        "vlad_vocab": "abc",
        "frame_step": 3,
    }
    result = sgd.swapped_components(stored, runtime, 256)
    assert result["vlad_enabled"] is True and result["vlad_vocab"] == "abc"
    assert result["descriptor_dim"] == 256
    assert result["frame_step"] == 30 and result["local_extractor"] == "aliked"
    assert stored["descriptor_dim"] == 1024  # input untouched


def _project(tmp_path):
    project = tmp_path / "proj"
    for sid in ("main", "2", "3_50mps"):
        layer = project / "sources" / sid
        layer.mkdir(parents=True)
        (layer / "database.h5").write_bytes(b"db")
        (layer / sgd.KEYPOINT_VIDEO).write_bytes(b"big")
    manifest = {
        "project_name": "proj",
        "video_sources": [
            {"source_id": sid, "database_file": f"sources/{sid}/database.h5"}
            for sid in ("main", "2")
        ],
    }
    (project / "project.json").write_text(json.dumps(manifest), encoding="utf-8")
    return project


def test_copy_project_copies_listed_layers_without_keypoint_videos(tmp_path):
    project = _project(tmp_path)
    out = tmp_path / "copy"
    manifest = sgd.copy_project(project, out)
    assert manifest["project_name"] == "copy"
    assert (out / "sources/main/database.h5").read_bytes() == b"db"
    assert not (out / "sources/main" / sgd.KEYPOINT_VIDEO).exists()
    assert not (out / "sources/3_50mps").exists()  # not listed in project.json
    assert all((out / name).is_dir() for name in sgd.STANDARD_DIRS)
    with pytest.raises(ValueError, match="already holds a project"):
        sgd.copy_project(project, out)
    reports_only = tmp_path / "reports_only"
    (reports_only / "reports").mkdir(parents=True)  # a replay run before the copy existed
    sgd.copy_project(project, reports_only)
    assert (reports_only / "sources/2/database.h5").exists()
    sgd.copy_project(project, tmp_path / "with_kp", include_keypoint_videos=True)
    assert (tmp_path / "with_kp/sources/2" / sgd.KEYPOINT_VIDEO).exists()


def test_keyframes_are_decoded_like_the_builder(tmp_path):
    path = tmp_path / "clip.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10, (16, 16))
    if not writer.isOpened():
        pytest.skip("no MJPG writer in this OpenCV build")
    for i in range(30):
        bgr = np.zeros((16, 16, 3), np.uint8)
        bgr[..., 0] = 8 * i  # blue carries the video frame index
        writer.write(bgr)
    writer.release()
    frames = list(sgd.keyframes_from_video(path, 3, [9, 0, 4]))
    assert [slot for slot, _ in frames] == [0, 4, 9]
    blue = [int(round(rgb[..., 2].mean() / 8)) for _, rgb in frames]  # RGB: blue last
    assert blue == [0, 12, 27]
    with pytest.raises(ValueError, match="ended before"):
        list(sgd.keyframes_from_video(path, 3, [10]))


def test_lance_table_is_rewritten_with_new_vectors(tmp_path):
    pytest.importorskip("lancedb")
    lance_dir = tmp_path / "vectors.lance"
    rng = np.random.default_rng(0)
    sgd.rewrite_lance(lance_dir, [(5, rng.normal(size=1024)), (9, rng.normal(size=1024))], 256)
    assert sgd.read_frame_ids(lance_dir) == [5, 9]
    rows = [(i, rng.normal(size=16).astype(np.float32)) for i in (1, 2, 3)]
    sgd.rewrite_lance(lance_dir, rows, 256)
    import lancedb

    table = lancedb.connect(str(lance_dir)).open_table("global_vectors").to_arrow()
    assert table.column("frame_id").to_pylist() == [1, 2, 3]
    np.testing.assert_allclose(table.column("vector").to_pylist()[2], rows[2][1], rtol=1e-6)


def test_metadata_gets_new_dim_components_and_fingerprint(tmp_path):
    h5_path = tmp_path / "database.h5"
    with h5py.File(h5_path, "w") as db:
        meta = db.create_group("metadata").attrs
        meta["descriptor_dim"] = 1024
        meta["schema_fingerprint"] = "old"
    components = {"global_backend": "dinov3", "descriptor_dim": 256, "vlad_enabled": True}
    fingerprint = sgd.update_metadata(h5_path, components)
    assert fingerprint == compute_fingerprint(components)
    with h5py.File(h5_path, "r") as db:
        meta = db["metadata"].attrs
        assert int(meta["descriptor_dim"]) == 256
        assert meta["schema_fingerprint"] == fingerprint
        assert json.loads(meta["schema_components"]) == components
