import pytest
from config.config import APP_CONFIG

def test_config_structure():
    """Verify that the APP_CONFIG dictionary has all required main sections."""
    required_sections = [
        'dinov2', 'lightglue', 'localization', 
        'tracking', 'preprocessing', 'gui'
    ]
    for section in required_sections:
        assert section in APP_CONFIG, f"Missing section: {section}"

def test_config_values():
    """Verify some critical default values in the config."""
    assert APP_CONFIG['dinov2']['descriptor_dim'] == 1024
    assert APP_CONFIG['localization']['min_matches'] >= 4
    assert APP_CONFIG['tracking']['kalman_process_noise'] > 0
    assert APP_CONFIG['preprocessing']['clahe_tile_grid'] == [8, 8]

def test_config_types():
    """Verify that config values have the correct data types."""
    assert isinstance(APP_CONFIG['dinov2']['descriptor_dim'], int)
    assert isinstance(APP_CONFIG['localization']['ransac_threshold'], float)
    assert isinstance(APP_CONFIG['tracking']['process_fps'], (int, float))
    assert isinstance(APP_CONFIG['preprocessing']['histogram_matching'], bool)
    assert isinstance(APP_CONFIG['preprocessing']['clahe_tile_grid'], list)
