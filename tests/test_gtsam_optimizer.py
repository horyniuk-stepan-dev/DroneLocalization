import importlib.util

import pytest

# Skip if gtsam is not installed
GTSAM_AVAILABLE = importlib.util.find_spec("gtsam") is not None

pytestmark = pytest.mark.skipif(not GTSAM_AVAILABLE, reason="GTSAM is not installed")

# TODO
