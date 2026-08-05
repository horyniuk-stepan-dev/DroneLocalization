import os
import sys

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# --- PyInstaller: fix DLL loading & redirect caches for frozen builds ---
if getattr(sys, "frozen", False):
    from pathlib import Path

    _meipass = getattr(sys, "_MEIPASS", str(Path(sys.executable).parent))
    _app_dir = str(Path(sys.executable).parent)

    # 1) Fix tensorrt DLL loading if needed
    _trt_lib = os.path.join(_meipass, "tensorrt_libs")
    if os.path.isdir(_trt_lib):
        os.add_dll_directory(_trt_lib)
        os.environ["PATH"] = _trt_lib + ";" + os.environ.get("PATH", "")

    # 2) Redirect TORCH_HOME / HF_HOME
    _cache_dir = os.path.join(_meipass, ".cache")
    if os.path.isdir(_cache_dir):
        os.environ.setdefault("TORCH_HOME", os.path.join(_cache_dir, "torch"))
        _hf_hub = os.path.join(_cache_dir, "huggingface", "hub")
        if os.path.isdir(_hf_hub):
            os.environ.setdefault("HF_HOME", os.path.join(_cache_dir, "huggingface"))
            os.environ.setdefault("HUGGINGFACE_HUB_CACHE", _hf_hub)

# WORKAROUND FOR PYINSTALLER + PYTORCH + WINDOWS WinError 1114:
# Import torch BEFORE anything else to prevent DLL conflicts
import warnings
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))

os.environ["YOLO_VERBOSE"] = "False"
os.environ["OPENCV_LOG_LEVEL"] = "ERROR"
os.environ["TRT_LOGGER_SEVERITY"] = "3"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
warnings.filterwarnings("ignore", category=DeprecationWarning, module="torch")
warnings.filterwarnings("ignore", category=UserWarning, module="torchvision")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="pkg_resources")
warnings.filterwarnings("ignore", category=UserWarning, message="xFormers is not available")

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import argparse
import traceback

from PyQt6.QtCore import Qt, QThread
from PyQt6.QtWidgets import QApplication

from config import APP_SETTINGS, CONFIG_LOAD_STATUS, CONFIG_LOADED_FROM, user_data_dir
from src.core.headless_runner import HeadlessRunner
from src.gui.main_window import MainWindow
from src.utils.logging_utils import enable_crash_handler, get_logger, setup_logging


class StartupWorker(QThread):
    def __init__(self, model_manager):
        super().__init__()
        self.model_manager = model_manager

    def run(self):
        try:
            self.model_manager.prewarm()
        except Exception as e:
            logger = get_logger(__name__)
            logger.warning(f"Startup prewarm failed: {e}. Models will load on first use.")


def _build_exception_hook(log):
    """Return sys.excepthook that logs unhandled exceptions before exit."""

    def hook(exctype, value, tb):
        # Ctrl+C is a deliberate cancellation, not a crash — it must not print a
        # traceback or be logged as one. It reaches here because KeyboardInterrupt
        # is a BaseException and slips past every `except Exception`.
        if issubclass(exctype, KeyboardInterrupt):
            log.info("Interrupted by user (Ctrl+C) — exiting")
            sys.exit(130)

        log.critical(
            "Unhandled exception caught — application will exit",
            exc_info=(exctype, value, tb),
        )

        traceback.print_exception(exctype, value, tb)
        sys.exit(1)

    return hook


def _run_supervised(args, logger) -> int:
    """HARDENING P0-3: run the headless pipeline in a child process and restart
    it on crash with exponential backoff.

    Because the pipeline runs as a separate process, even a native segfault
    (CUDA/cv2/torch) is survivable — the supervisor sees a non-zero exit and
    relaunches. Turns "silent death" into "logged auto-recovery". Active only
    with ``--supervise --headless``; default behavior is unchanged.
    """
    import subprocess
    import time as _time

    if getattr(sys, "frozen", False):
        base_cmd = [sys.executable]
    else:
        base_cmd = [sys.executable, os.path.abspath(sys.argv[0])]

    child_cmd = base_cmd + [
        "--headless",
        "--project",
        args.project,
        "--source",
        args.source,
    ]
    # Порти передаємо дитині ЛИШЕ якщо їх явно задали в CLI: інакше дитина має
    # взяти їх із user_config.json так само, як це зробив би одиночний запуск.
    if args.ws_port is not None:
        child_cmd += ["--ws-port", str(args.ws_port)]
    if args.rest_port is not None:
        child_cmd += ["--rest-port", str(args.rest_port)]

    # HARDENING P1-6: an encrypted project needs the passphrase on EVERY child
    # start. Ask once here, in the supervisor, and pipe it to each child on its
    # stdin — unattended restart is the whole point of this mode, so re-prompting
    # a human per restart would defeat it. A pipe (unlike an env var) is invisible
    # in process listings, not inherited by grandchildren, and consumed once.
    child_input = None
    try:
        from src.security.at_rest import get_passphrase, prompt_and_verify_passphrase
        from src.security.project_scan import encrypted_artifacts_at

        encrypted = encrypted_artifacts_at(args.project)
        if encrypted:
            if not prompt_and_verify_passphrase(str(encrypted[0])):
                logger.error(
                    "Project is encrypted and no valid passphrase was given — "
                    "supervisor will not start."
                )
                return 1
            child_input = get_passphrase().encode("utf-8") + b"\n"
            logger.info("Map passphrase accepted — it will be piped to each child")
    except Exception as e:
        logger.error(f"Could not resolve the map passphrase: {e}")
        return 1

    backoff = 1.0
    backoff_max = 60.0
    restarts = 0
    max_restarts = args.max_restarts  # 0 = unlimited

    logger.info(
        f"Supervisor starting | max_restarts={max_restarts or 'unlimited'} | "
        f"child={' '.join(child_cmd)}"
    )

    while True:
        start = _time.monotonic()
        try:
            # Passing input= replaces the child's stdin with a pipe, which is what
            # its get_passphrase reads. Plaintext projects pass None and keep
            # inheriting the terminal exactly as before.
            rc = subprocess.run(child_cmd, input=child_input).returncode
        except KeyboardInterrupt:
            logger.info("Supervisor received interrupt — shutting down")
            return 0
        runtime = _time.monotonic() - start

        if rc == 0:
            logger.info("Supervised pipeline exited cleanly (rc=0) — supervisor stopping")
            return 0

        restarts += 1
        logger.error(
            f"Supervised pipeline crashed (rc={rc}) after {runtime:.1f}s — restart #{restarts}"
        )

        if max_restarts and restarts >= max_restarts:
            logger.critical(f"Supervisor reached max_restarts={max_restarts}; giving up.")
            return 1

        # A child that ran long enough is 'healthy'; reset the backoff.
        if runtime >= backoff_max:
            backoff = 1.0

        logger.warning(f"Restarting in {backoff:.0f}s (backoff)")
        try:
            _time.sleep(backoff)
        except KeyboardInterrupt:
            logger.info("Supervisor interrupted during backoff — shutting down")
            return 0
        backoff = min(backoff * 2, backoff_max)


def main() -> None:
    try:
        log_level = APP_SETTINGS.models.performance.log_level
    except Exception:
        log_level = "INFO"  # Safe default
    setup_logging(log_level=log_level, log_file=str(user_data_dir() / "logs" / "app.log"))
    logger = get_logger(__name__)

    # HARDENING P0-2: capture native segfaults/aborts before the excepthook.
    enable_crash_handler(user_data_dir() / "logs")

    sys.excepthook = _build_exception_hook(logger)

    logger.info("=" * 70)
    logger.info("DRONE TOPOMETRIC LOCALIZATION SYSTEM STARTING")
    logger.info("=" * 70)

    logger.info(f"Python: {sys.version}")
    logger.info(f"PyTorch: {torch.__version__}")

    # Звідки взято налаштування. Якщо файл не знайдено — це WARNING, а не тиша:
    # старт на вбудованих дефолтах вимикає smoother, edge-гейти й torch_compile,
    # і раніше про це не було жодного сигналу в логах.
    if CONFIG_LOADED_FROM:
        logger.info(CONFIG_LOAD_STATUS)
    else:
        logger.warning(CONFIG_LOAD_STATUS)

    # ── HARDENING P2-12: weight-integrity preflight (fail-closed go/no-go) ────
    try:
        _wi_mode = APP_SETTINGS.models.performance.weight_integrity_mode
    except Exception:
        _wi_mode = "off"
    if _wi_mode and _wi_mode.lower() != "off":
        from src.utils.weight_integrity import WeightIntegrityError, run_preflight

        _models_root = Path(__file__).parent / "models"
        try:
            run_preflight(
                _models_root,
                _models_root / "weights_manifest.json",
                mode=_wi_mode,
                logger=logger,
            )
        except WeightIntegrityError as e:
            logger.critical(f"Weight integrity preflight failed — aborting startup.\n{e}")
            sys.exit(1)

    # ── Hardware auto-detection & compute auto-tuning ────────────────────────
    from src.utils.hardware_profile import HardwareProfile

    hw_profile = HardwareProfile.detect()
    hw_profile.log_summary()

    # Apply PyTorch backend optimizations (TF32, cudnn.benchmark, thread counts).
    # HARDENING P1-8: deterministic mode bounds worst-case latency when enabled.
    try:
        _deterministic = APP_SETTINGS.models.performance.deterministic
    except Exception:
        _deterministic = False
    hw_profile.apply_torch_backends(deterministic=_deterministic)

    # Auto-tune config values if enabled
    if APP_SETTINGS.models.performance.auto_tune:
        import config as _cfg_module

        overrides = hw_profile.auto_tune(_cfg_module.APP_CONFIG)
        if overrides:
            hw_profile.apply_overrides(_cfg_module.APP_CONFIG, overrides)
            hw_profile.log_overrides(overrides)
            # Перезаливаємо APP_SETTINGS У МІСЦІ. Переприв'язка (`= AppConfig(...)`)
            # оновлювала лише config.* і main.*, а headless_runner/main_window/
            # config_dialog лишалися на старому об'єкті — auto-tune до них не
            # доходив, а --ws-port/--rest-port писалися в об'єкт, якого ніхто
            # не читає.
            _cfg_module.reload_settings_in_place(_cfg_module.APP_CONFIG)
        else:
            logger.info("Auto-tune: all settings already optimal or user-customized")
    else:
        logger.info("Auto-tune disabled (models.performance.auto_tune = false)")

    parser = argparse.ArgumentParser(description="Drone Topometric Localization")
    parser.add_argument("--headless", action="store_true", help="Run without GUI")
    parser.add_argument(
        "--project", type=str, help="Path to project directory (required for headless)"
    )
    parser.add_argument(
        "--source", type=str, help="Video source URL or path (required for headless)"
    )
    # ВАЖЛИВО: default=None, а не число. Раніше дефолти argparse (8765/8080)
    # БЕЗУМОВНО перезаписували network_api.ws_port / rest_port із user_config.json
    # навіть коли прапорець не передавали — headless завжди слухав 8080, хоча
    # конфіг казав 8081.
    parser.add_argument(
        "--ws-port",
        type=int,
        default=None,
        help="WebSocket port (default: network_api.ws_port from config)",
    )
    parser.add_argument(
        "--rest-port",
        type=int,
        default=None,
        help="REST API port (default: network_api.rest_port from config)",
    )
    parser.add_argument(
        "--supervise",
        action="store_true",
        help="Headless: run the pipeline in a child process and auto-restart it on crash",
    )
    parser.add_argument(
        "--max-restarts",
        type=int,
        default=0,
        help="Supervisor: stop after N restarts (0 = unlimited)",
    )

    args = parser.parse_args()

    # HARDENING P0-3: supervisor mode wraps the headless pipeline in an
    # auto-restarting parent process. Flag-gated; off = current behavior.
    if args.supervise:
        if not args.headless:
            logger.error("--supervise requires --headless")
            sys.exit(1)
        if not args.project or not args.source:
            logger.error("--project and --source are required with --supervise")
            sys.exit(1)
        sys.exit(_run_supervised(args, logger))

    # HARDENING P1-6 SP3: a crashed run leaves decrypted global descriptors in a
    # temp directory. Wipe only those whose owner process is gone. Applies to both
    # modes — a headless run decrypts the same LanceDB index.
    try:
        from src.database.database_loader import sweep_stale_lance_tempdirs

        sweep_stale_lance_tempdirs()
    except Exception as e:
        logger.warning(f"Stale decrypted-index sweep failed: {e}")

    try:
        if args.headless:
            logger.info("Running in headless mode")
            if not args.project or not args.source:
                logger.error("--project and --source are required in headless mode")
                sys.exit(1)

            # Перекриваємо конфіг лише тим, що явно задане в CLI.
            if args.ws_port is not None:
                APP_SETTINGS.network_api.ws_port = args.ws_port
            if args.rest_port is not None:
                APP_SETTINGS.network_api.rest_port = args.rest_port
            logger.info(
                f"Network API ports: ws={APP_SETTINGS.network_api.ws_port}, "
                f"rest={APP_SETTINGS.network_api.rest_port}"
            )

            runner = HeadlessRunner(args.project, args.source)
            runner.run()
            exit_code = 0
        else:
            QApplication.setHighDpiScaleFactorRoundingPolicy(
                Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
            )
            app = QApplication(sys.argv)
            app.setApplicationName("Drone Localization")
            app.setOrganizationName("UAV Systems")
            logger.info("Qt application initialized")

            window = MainWindow()
            window.show()

            # Запускаємо prewarm у фоновому потоці
            if hasattr(window, "model_manager") and window.model_manager:
                app._startup_worker = StartupWorker(window.model_manager)
                app._startup_worker.start()

            logger.success("Application startup complete")

            exit_code = app.exec()

    except KeyboardInterrupt:
        # Most likely Ctrl+C at the map passphrase prompt. Cancelling a decrypt is
        # a normal outcome, so exit quietly with the conventional 130.
        logger.info("Interrupted by user (Ctrl+C) — exiting")
        sys.exit(130)
    except Exception as e:
        logger.critical(f"Fatal error during startup: {e}", exc_info=True)
        sys.exit(1)

    logger.info(f"Application exiting | code={exit_code}")
    logger.info("=" * 70)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
