"""Root pytest conftest — loaded before test collection and any test module.

Windows/torch DLL fix: some torch builds fail to initialise ``c10.dll``
([WinError 1114]) if another native runtime (numpy/MKL, OpenMP, PoseLib, cv2) is
loaded into the process first. Importing torch here, before anything else, lets
it set up its DLL search dirs in a clean process. Guarded so it is a no-op where
torch is not installed (e.g. pure-Python CI / the dev sandbox).
"""

import os
import tempfile

# Same as main.py: one OpenBLAS thread (a 120x120 numpy solve took 34 ms instead of
# 0.05 ms with the default thread count on Windows). Set before torch, which loads numpy.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

try:
    import torch  # noqa: F401
except Exception:
    pass

# Telemetry dumps a report at interpreter exit; keep test runs out of the
# application's logs/telemetry_report.json (src/utils/telemetry.py).
os.environ.setdefault(
    "DRONELOC_TELEMETRY_DIR", os.path.join(tempfile.gettempdir(), "droneloc_test_telemetry")
)
