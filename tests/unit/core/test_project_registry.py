import json
from pathlib import Path

import pytest

from src.core.project_registry import ProjectRegistry


@pytest.fixture
def registry(tmp_path, monkeypatch):
    home = tmp_path / "home"
    monkeypatch.setattr(Path, "home", lambda: home)
    return ProjectRegistry()


def make_project(folder: Path, name: str = "", layers: tuple[str, ...] = (), **fields) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    manifest = {"project_name": name or folder.name, "video_path": "v.mp4", **fields}
    (folder / "project.json").write_text(json.dumps(manifest), encoding="utf-8")
    for layer in layers:
        (folder / "sources" / layer).mkdir(parents=True)
        (folder / "sources" / layer / "database.h5").write_bytes(b"x")
        (folder / "sources" / layer / "calibration.json").write_text("{}", encoding="utf-8")
    return folder


def test_find_projects_root_itself_is_a_project(tmp_path):
    root = make_project(tmp_path / "proj", layers=("main",))
    # A nested project.json under the project must not be reported separately.
    make_project(root / "sources" / "main" / "inner")
    assert ProjectRegistry.find_projects(root) == [root.resolve()]


def test_find_projects_in_parent_folder(tmp_path):
    a = make_project(tmp_path / "a")
    c = make_project(tmp_path / "b" / "c")
    make_project(tmp_path / "d" / "e" / "f")  # 3 levels below root: out of reach
    make_project(tmp_path / ".hidden")
    (tmp_path / "empty").mkdir()
    (tmp_path / "file.txt").write_text("not a folder", encoding="utf-8")

    assert ProjectRegistry.find_projects(tmp_path) == sorted(
        [a.resolve(), c.resolve()], key=lambda p: str(p).casefold()
    )
    assert len(ProjectRegistry.find_projects(tmp_path, max_depth=3)) == 3
    assert ProjectRegistry.find_projects(tmp_path / "missing") == []


def test_import_adds_projects_with_manifest_fields(registry, tmp_path):
    proj = make_project(
        tmp_path / "testtopboch_hh",
        name="Bochkivtsi HH",
        layers=("main", "2"),
        created_at="2026-09-28T22:41:50",
    )
    result = registry.import_projects([proj])

    assert result.added == [str(proj.resolve())]
    assert result.already_known == [] and result.skipped == []
    (entry,) = registry.get_all()
    assert entry["name"] == "Bochkivtsi HH"
    assert entry["video_path"] == "v.mp4"
    assert entry["created_at"] == "2026-09-28T22:41:50"
    assert entry["has_database"] and entry["has_calibration"]
    # Persisted and visible as a recent project in a fresh registry instance.
    assert [p["path"] for p in ProjectRegistry().get_recent()] == [str(proj.resolve())]


def test_reimport_keeps_dates_and_refreshes_status(registry, tmp_path):
    proj = make_project(tmp_path / "p")
    registry.import_projects([proj])
    before = dict(registry.get_all()[0])
    assert not before["has_database"]

    (proj / "sources" / "main").mkdir(parents=True)
    (proj / "sources" / "main" / "database.h5").write_bytes(b"x")
    result = registry.import_projects([proj])

    assert result.added == [] and result.already_known == [str(proj.resolve())]
    (after,) = registry.get_all()
    assert after["has_database"]
    assert after["last_opened"] == before["last_opened"]
    assert after["created_at"] == before["created_at"]


def test_import_skips_non_projects_and_tolerates_encrypted_manifest(registry, tmp_path):
    plain = tmp_path / "not_a_project"
    plain.mkdir()
    encrypted = tmp_path / "enc_copy"
    encrypted.mkdir()
    (encrypted / "project.json").write_bytes(b"\x00\x9fENCRYPTED\xff")

    result = registry.import_projects([plain, encrypted])

    assert result.skipped == [str(plain.resolve())]
    assert result.added == [str(encrypted.resolve())]
    assert registry.get_all()[0]["name"] == "enc_copy"
