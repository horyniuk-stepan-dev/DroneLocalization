"""PyInstaller runtime hook: fix torch DLL loading in frozen apps.

torch.__init__ uses sysconfig/sys.exec_prefix to locate its DLL directory,
but these paths are wrong inside a PyInstaller bundle. This hook adds
torch/lib to the DLL search path before torch gets imported.
"""
import os
import sys

if getattr(sys, "frozen", False):
    # _MEIPASS points to _internal/ in --onedir mode
    base = sys._MEIPASS
    torch_lib = os.path.join(base, "torch", "lib")

    if os.path.isdir(torch_lib):
        os.add_dll_directory(torch_lib)
        # Also prepend to PATH as fallback for LoadLibraryW
        os.environ["PATH"] = torch_lib + ";" + os.environ.get("PATH", "")

    # TensorRT libs
    trt_lib = os.path.join(base, "tensorrt_libs")
    if os.path.isdir(trt_lib):
        os.add_dll_directory(trt_lib)
        os.environ["PATH"] = trt_lib + ";" + os.environ.get("PATH", "")

    # Also set TORCH_HOME / HF_HOME for bundled caches
    app_dir = os.path.dirname(sys.executable)
    cache_dir = os.path.join(app_dir, "_internal", ".cache")
    if not os.path.isdir(cache_dir):
        cache_dir = os.path.join(app_dir, ".cache")

    if os.path.isdir(cache_dir):
        torch_home = os.path.join(cache_dir, "torch")
        if os.path.isdir(torch_home):
            os.environ.setdefault("TORCH_HOME", torch_home)

        hf_hub = os.path.join(cache_dir, "huggingface", "hub")
        if os.path.isdir(hf_hub):
            os.environ.setdefault("HF_HOME", os.path.join(cache_dir, "huggingface"))
            os.environ.setdefault("HUGGINGFACE_HUB_CACHE", hf_hub)
