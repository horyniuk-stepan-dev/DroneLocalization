"""Command-line arguments for main.py."""

import argparse


def build_parser() -> argparse.ArgumentParser:
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

    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)
