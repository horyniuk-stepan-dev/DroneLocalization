"""Root pytest conftest — loaded before test collection and any test module.

Windows/torch DLL fix: some torch builds fail to initialise ``c10.dll``
([WinError 1114]) if another native runtime (numpy/MKL, OpenMP, PoseLib, cv2) is
loaded into the process first. Importing torch here, before anything else, lets
it set up its DLL search dirs in a clean process. Guarded so it is a no-op where
torch is not installed (e.g. pure-Python CI / the dev sandbox).
"""

try:
    import torch  # noqa: F401
except Exception:
    pass
