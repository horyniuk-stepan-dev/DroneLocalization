import json
from config import APP_CONFIG
from config.access import load_user_config
from config.app import AppConfig


def test_config_structure():
    """Verify that the APP_CONFIG dictionary has all required main sections."""
    required_sections = [
        "global_descriptor",
        "models",
        "localization",
        "tracking",
        "preprocessing",
        "gui",
        "projection",
        "database",
        "homography",
    ]
    for section in required_sections:
        assert section in APP_CONFIG, f"Missing section: {section}"


def test_config_values():
    """Verify some critical default values in the config."""
    assert APP_CONFIG["global_descriptor"]["dinov2"]["descriptor_dim"] == 1024
    assert APP_CONFIG["localization"]["min_matches"] >= 4
    assert APP_CONFIG["tracking"]["kalman_process_noise"] > 0
    assert APP_CONFIG["preprocessing"]["clahe_tile_grid"] == [8, 8]
    # Нові поля Горизонту 1
    assert APP_CONFIG["preprocessing"]["masking_strategy"] in ("yolo", "none")
    # Інваріант замість літерала: тест на конкретний дефолт дрейфує разом
    # із конфігом і фейлиться без реальної причини (антипатерн)
    assert APP_CONFIG["homography"]["backend"] in ("poselib", "opencv")
    assert APP_CONFIG["models"]["yolo"]["model_path"] == "models/yolo11n-seg.pt"
    assert APP_CONFIG["models"]["engines_cache"]["engine_cache_dir"] == "models/engines/"


def test_config_types():
    """Verify that config values have the correct data types."""
    assert isinstance(APP_CONFIG["global_descriptor"]["dinov2"]["descriptor_dim"], int)
    assert isinstance(APP_CONFIG["localization"]["ransac_threshold"], float)
    assert isinstance(APP_CONFIG["tracking"]["process_fps"], int | float)
    assert isinstance(APP_CONFIG["preprocessing"]["histogram_matching"], bool)
    assert isinstance(APP_CONFIG["preprocessing"]["clahe_tile_grid"], list)
    assert isinstance(APP_CONFIG["homography"]["backend"], str)
    assert isinstance(APP_CONFIG["homography"]["ransac_threshold"], float)


def test_auto_create_default_config(tmp_path, monkeypatch):
    """Verify that if no user_config.json exists, load_user_config creates a new file with defaults."""
    fake_config_file = tmp_path / "user_config.json"
    monkeypatch.setattr("config.access.CONFIG_FILE_PATH", str(fake_config_file))
    monkeypatch.setattr("config.access.user_config_candidates", lambda: [fake_config_file])

    assert not fake_config_file.exists()
    cfg = load_user_config()
    assert fake_config_file.exists()

    with open(fake_config_file, encoding="utf-8") as f:
        data = json.load(f)

    assert "global_descriptor" in data
    assert cfg.global_descriptor == AppConfig().global_descriptor

