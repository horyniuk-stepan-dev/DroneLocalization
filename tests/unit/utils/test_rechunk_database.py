"""scripts/rechunk_database.py: one-frame chunks, identical content, safe swap."""

import h5py
import numpy as np

from scripts import rechunk_database as R

SLOTS, KPS = 10, 32


def _db(path, written=(0, 1, 2, 5)):
    rng = np.random.default_rng(0)
    with h5py.File(path, "w", libver="latest") as f:
        f.attrs["georef_generation"] = 3
        lf = f.create_group("local_features")
        lf.attrs["frame_width"] = 640
        specs = {
            "keypoints": (2, "float32"),
            "descriptors": (128, "float16"),
            "coords_2d": (2, "float32"),
        }
        for name, (dim, dtype) in specs.items():
            d = lf.create_dataset(
                name,
                shape=(SLOTS, KPS, dim),
                maxshape=(None, KPS, dim),
                dtype=dtype,
                chunks=(4, KPS, dim),
                compression="lzf",
                fillvalue=0.0,
            )
            for i in written:  # other slots stay pre-allocated and unwritten
                d[i] = rng.normal(size=(KPS, dim)).astype(dtype)
        counts = np.zeros(SLOTS, np.int16)
        counts[list(written)] = KPS
        lf.create_dataset("kp_counts", data=counts, chunks=(SLOTS,), compression="lzf")
        g = f.create_group("global_descriptors")
        g.create_dataset("frame_poses", data=rng.normal(size=(SLOTS, 3, 3)), chunks=(4, 3, 3))
        m = f.create_group("metadata")
        m.attrs["frame_step"] = 30
        m.create_dataset("frame_index_map", data=np.arange(4, dtype=np.int32))
    return path


def test_plan_picks_per_frame_feature_arrays_only(tmp_path):
    path = _db(tmp_path / "database.h5")
    with h5py.File(path, "r") as f:
        names = R.plan(f, min_frame_bytes=200)
    # frame_poses (72 B per frame) and kp_counts (1-D) stay as they are
    assert sorted(names) == [
        "local_features/coords_2d",
        "local_features/descriptors",
        "local_features/keypoints",
    ]


def test_rechunk_keeps_content_and_swaps_with_backup(tmp_path):
    path = _db(tmp_path / "database.h5")
    with h5py.File(path, "r") as f:
        before = {k: f["local_features"][k][:] for k in ("keypoints", "descriptors", "coords_2d")}
    msg = R.process(path, min_frame_bytes=200, dry_run=False, bench=False)
    assert msg.startswith("rechunked 3 datasets"), msg
    backup = tmp_path / "database.h5.chunk4"
    assert backup.is_file() and not (tmp_path / "database.h5.rechunk.tmp").exists()
    assert R.verify(backup, path) == []
    with h5py.File(path, "r") as f:
        lf = f["local_features"]
        assert lf["descriptors"].chunks == (1, KPS, 128)
        assert lf["descriptors"].maxshape == (None, KPS, 128)
        assert lf["descriptors"].compression == "lzf"
        for k, v in before.items():
            np.testing.assert_array_equal(lf[k][:], v)
        assert f.attrs["georef_generation"] == 3 and lf.attrs["frame_width"] == 640
        assert f["global_descriptors/frame_poses"].chunks == (4, 3, 3)  # copied as is
        assert f["metadata"].attrs["frame_step"] == 30
    # second run: nothing to do; the backup is not touched
    assert R.process(path, 200, False, False).startswith("already one frame per chunk")


def test_dry_run_and_existing_backup_leave_the_file_alone(tmp_path):
    path = _db(tmp_path / "database.h5")
    stat = path.stat()
    assert R.process(path, 200, dry_run=True, bench=False).startswith("would rechunk (from 4")
    (tmp_path / "database.h5.chunk4").write_bytes(b"keep")
    assert R.process(path, 200, False, False).startswith("backup database.h5.chunk4 already exists")
    assert path.stat().st_mtime_ns == stat.st_mtime_ns
    assert (tmp_path / "database.h5.chunk4").read_bytes() == b"keep"


def test_verify_reports_a_changed_value(tmp_path):
    a = _db(tmp_path / "a.h5")
    b = _db(tmp_path / "b.h5")
    with h5py.File(b, "r+") as f:
        f["local_features/descriptors"][1, 0, 0] += 1
    assert R.verify(a, b) == ["local_features/descriptors: values differ"]


def test_encrypted_database_is_refused(tmp_path):
    from src.security.at_rest import MAGIC

    path = tmp_path / "database.h5"
    path.write_bytes(MAGIC + b"\x01" + b"\x00" * 64)
    assert R.process(path, 200, False, False).startswith("encrypted")


def test_main_with_project_folder(tmp_path, capsys):
    (tmp_path / "sources" / "main").mkdir(parents=True)
    _db(tmp_path / "sources" / "main" / "database.h5")
    assert R.main(["--project", str(tmp_path), "--min-frame-bytes", "200"]) == 0
    assert "rechunked 3 datasets" in capsys.readouterr().out
