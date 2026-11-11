"""Pytest configuration"""

import os
import sys
from pathlib import Path

import pytest

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Tests must not append to the user's logs/localization_failures.csv
# (src/localization/failure_log.py). Set it explicitly to keep a CSV.
os.environ.setdefault("DRONELOC_FAILURE_CSV", "off")


@pytest.fixture
def test_data_dir():
    """Test data directory fixture"""
    return Path(__file__).parent / "fixtures"
