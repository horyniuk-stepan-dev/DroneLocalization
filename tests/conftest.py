"""Pytest configuration"""

import pytest
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture
def test_data_dir():
    """Test data directory fixture"""
    return Path(__file__).parent / "fixtures"


@pytest.fixture
def sample_image(test_data_dir):
    """Sample image fixture"""
    # TODO: Load sample image
    pass


@pytest.fixture
def sample_features():
    """Sample features fixture"""
    # TODO: Return sample feature dict
    pass
