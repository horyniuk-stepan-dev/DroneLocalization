import os

import pytest

from config.config import APP_CONFIG
from src.database.database_builder import DatabaseBuilder
from src.models.model_manager import ModelManager


@pytest.fixture
def model_manager():
    # Initialize basic model manager for testing
    cfg = APP_CONFIG.copy()
    cfg["models"] = {
        "device": "cpu",  # Force CPU to avoid CUDA OOM in parallel tests if run on small GPUs
        "performance": {"fp16_enabled": False},
    }
    mm = ModelManager(cfg)
    return mm


def test_db_build_short(benchmark, tmp_path, dummy_video_path, model_manager):
    """
    Benchmarks standard short DB build (50 frames).
    """
    output_db = str(tmp_path / "test_short.h5")
    builder = DatabaseBuilder(output_path=output_db, config=APP_CONFIG)

    # We benchmark the entire build process
    benchmark.pedantic(
        builder.build_from_video,
        kwargs={
            "video_path": dummy_video_path,
            "model_manager": model_manager,
            "save_keypoint_video": False,
        },
        rounds=1,  # Since DB building is heavy, run only 1 round per test
        iterations=1,
    )

    assert os.path.exists(output_db)


@pytest.mark.slow
def test_db_build_long(benchmark, tmp_path, long_dummy_video_path, model_manager):
    """
    Benchmarks long DB build (300 frames). Checks for performance degradation.
    """
    output_db = str(tmp_path / "test_long.h5")
    builder = DatabaseBuilder(output_path=output_db, config=APP_CONFIG)

    benchmark.pedantic(
        builder.build_from_video,
        kwargs={
            "video_path": long_dummy_video_path,
            "model_manager": model_manager,
            "save_keypoint_video": False,
        },
        rounds=1,
        iterations=1,
    )

    assert os.path.exists(output_db)
