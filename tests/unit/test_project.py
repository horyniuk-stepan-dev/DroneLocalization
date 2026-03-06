import pytest
import json
from pathlib import Path
from src.core.project import ProjectManager, ProjectSettings

def test_project_settings_from_dict():
    data = {
        "project_name": "TestMission", 
        "created_at": "2023-01-01T00:00:00", 
        "video_path": "/fake/video.mp4"
    }
    settings = ProjectSettings.from_dict(data)
    assert settings.project_name == "TestMission"
    assert settings.video_path == "/fake/video.mp4"
    assert settings.altitude_m == 100.0  # default

def test_project_manager_create(tmp_path):
    manager = ProjectManager()
    
    # Not loaded initially
    assert not manager.is_loaded
    assert manager.project_name == "No Project"
    assert manager.database_path is None
    assert manager.calibration_path is None
    
    workspace = str(tmp_path)
    mission_data = {
        "mission_name": "My Mission",
        "video_path": "path.mp4",
        "altitude_m": 120.5
    }
    
    success = manager.create_project(workspace, mission_data)
    
    assert success is True
    assert manager.is_loaded
    assert manager.project_name == "My Mission"
    # Should replace space with underscore
    assert "My_Mission" in str(manager.project_dir)
    assert manager.settings.altitude_m == 120.5
    
    assert manager.database_path is not None and "database.h5" in manager.database_path
    assert manager.calibration_path is not None and "calibration.json" in manager.calibration_path
    
    # Check folder structure creation
    assert (manager.project_dir / "panoramas").exists()
    assert (manager.project_dir / "test_photos").exists()
    assert (manager.project_dir / "test_videos").exists()
    assert (manager.project_dir / "project.json").exists()

def test_project_manager_load(tmp_path):
    # Setup - let's reuse create_project to generate the folder structure
    manager = ProjectManager()
    mission_data = {"mission_name": "LoadTest", "video_path": "test.mp4"}
    manager.create_project(str(tmp_path), mission_data)
    
    proj_dir = manager.project_dir
    
    # New manager instance to test loading
    loader = ProjectManager()
    assert loader.load_project(str(proj_dir)) is True
    assert loader.is_loaded
    assert loader.project_name == "LoadTest"
    assert loader.settings.video_path == "test.mp4"

def test_project_manager_load_fail(tmp_path):
    loader = ProjectManager()
    # Path doesn't exist
    assert loader.load_project(str(tmp_path / "NonExistent")) is False
    assert not loader.is_loaded
    
    # Folder exists but no project.json
    empty_dir = tmp_path / "Empty"
    empty_dir.mkdir()
    assert loader.load_project(str(empty_dir)) is False
