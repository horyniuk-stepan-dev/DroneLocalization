from pathlib import Path

import pytest

import src.workers.database_worker as worker_module
from src.workers.database_worker import DatabaseGenerationWorker


def _write_database_artifacts(root: Path, db_name: str, marker: str) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / db_name).write_text(f"db-{marker}", encoding="utf-8")
    vectors = root / "vectors.lance"
    vectors.mkdir()
    (vectors / "marker.txt").write_text(f"vectors-{marker}", encoding="utf-8")


def test_promote_staged_database_replaces_db_and_vector_index(tmp_path):
    final = tmp_path / "database.h5"
    _write_database_artifacts(tmp_path, final.name, "old")
    staging = tmp_path / ".database-rebuild-test"
    _write_database_artifacts(staging, final.name, "new")

    DatabaseGenerationWorker._promote_staged_database(staging, final)

    assert final.read_text(encoding="utf-8") == "db-new"
    assert (tmp_path / "vectors.lance" / "marker.txt").read_text(encoding="utf-8") == (
        "vectors-new"
    )


def test_incomplete_staging_never_replaces_existing_database(tmp_path):
    final = tmp_path / "database.h5"
    _write_database_artifacts(tmp_path, final.name, "old")
    staging = tmp_path / ".database-rebuild-test"
    staging.mkdir()
    (staging / final.name).write_text("db-incomplete", encoding="utf-8")

    with pytest.raises(RuntimeError, match="required artifacts"):
        DatabaseGenerationWorker._promote_staged_database(staging, final)

    assert final.read_text(encoding="utf-8") == "db-old"
    assert (tmp_path / "vectors.lance" / "marker.txt").read_text(encoding="utf-8") == (
        "vectors-old"
    )


def test_worker_forwards_exact_calibration_anchor_slots(monkeypatch, tmp_path):
    observed = {}

    class FakeBuilder:
        def __init__(self, output_path, config):
            self.output_path = Path(output_path)

        def build_from_video(self, **kwargs):
            observed["required_frame_ids"] = kwargs["required_frame_ids"]
            _write_database_artifacts(
                self.output_path.parent, self.output_path.name, "generated"
            )

    monkeypatch.setattr(worker_module, "DatabaseBuilder", FakeBuilder)
    final = tmp_path / "database.h5"
    worker = DatabaseGenerationWorker(
        video_path="reference.mp4",
        output_path=str(final),
        model_manager=object(),
        required_frame_ids={41, 83, 125},
    )

    worker.run()

    assert observed["required_frame_ids"] == {41, 83, 125}
    assert final.read_text(encoding="utf-8") == "db-generated"
