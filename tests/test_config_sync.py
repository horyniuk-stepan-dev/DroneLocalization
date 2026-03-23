"""Тест на синхронність конфігурації та відсутність дублікатів.

Захищає від ситуацій, коли конфіг-ключі дублюються
або обов'язкові секції відсутні.
"""
import pytest
from config.config import APP_CONFIG, get_cfg


class TestGetCfgHelper:
    """Тести для хелпера get_cfg()"""

    def test_simple_path(self):
        assert get_cfg(APP_CONFIG, 'dinov2.descriptor_dim') == 1024

    def test_nested_path(self):
        assert get_cfg(APP_CONFIG, 'localization.confidence.confidence_max_inliers') == 80

    def test_missing_key_returns_default(self):
        assert get_cfg(APP_CONFIG, 'nonexistent.key', 42) == 42

    def test_partial_path_returns_default(self):
        assert get_cfg(APP_CONFIG, 'localization.nonexistent', 'fallback') == 'fallback'

    def test_empty_config_returns_default(self):
        assert get_cfg({}, 'any.path', 99) == 99


class TestNoDuplicateConfigKeys:
    """confidence_max_inliers має бути ТІЛЬКИ в localization.confidence"""

    def test_no_duplicate_confidence_max_inliers(self):
        assert 'confidence_max_inliers' in APP_CONFIG['localization']['confidence']
        assert 'confidence_max_inliers' not in APP_CONFIG.get('projection', {})


class TestConfigSectionsExist:
    """Перевірка наявності обов'язкових секцій"""

    @pytest.mark.parametrize("section", [
        'dinov2', 'database', 'localization', 'tracking',
        'preprocessing', 'gui', 'models', 'projection',
    ])
    def test_section_exists(self, section):
        assert section in APP_CONFIG, f"Відсутня секція: {section}"


class TestConfigTypes:
    """Перевірка типів критичних параметрів"""

    def test_tracking_types(self):
        trk = APP_CONFIG['tracking']
        assert isinstance(trk['outlier_threshold_std'], (int, float))
        assert isinstance(trk['max_speed_mps'], (int, float))
        assert isinstance(trk['outlier_window'], int)

    def test_localization_types(self):
        loc = APP_CONFIG['localization']
        assert isinstance(loc['min_matches'], int)
        assert isinstance(loc['min_inliers_accept'], int)
        assert isinstance(loc['ransac_threshold'], float)
