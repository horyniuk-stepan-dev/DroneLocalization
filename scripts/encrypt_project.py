"""HARDENING P1-6: encrypt a project's geo-revealing artifacts at rest.

Encrypts the small, whole-file artifacts that leak the operational area in
plaintext — ``calibration.json`` (geo anchors) and ``database_graph.geojson``
(per-frame GPS) — in place, so the app auto-detects and decrypts them on load
given the passphrase (``DRONELOC_PASSPHRASE`` env or an interactive prompt).

    python scripts/encrypt_project.py --project <dir>

The passphrase is prompted twice (confirmation) and never stored. Each file is
encrypted only after a round-trip decrypt is verified in RAM, and written
atomically, so an interrupted run cannot corrupt an artifact. Keep the passphrase
safe: it cannot be recovered, and without it the map is unreadable — that is the
point.

The big map artifacts (``database.h5``, ``vectors.lance/``) are handled by later
sub-projects and are NOT touched here.
"""

from __future__ import annotations

import argparse
import getpass
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _resolve_targets(project_dir: str) -> list[Path]:
    """The geo-revealing whole-file artifacts of a project, in load order."""
    from src.core.project import ProjectManager

    pm = ProjectManager()
    if not pm.load_project(project_dir):
        raise SystemExit(f"Could not load project: {project_dir}")

    targets: list[Path] = []
    if pm.calibration_path and Path(pm.calibration_path).exists():
        targets.append(Path(pm.calibration_path))
    if pm.database_path:
        geojson = Path(str(pm.database_path).replace(".h5", "_graph.geojson"))
        if geojson.exists():
            targets.append(geojson)
    return targets


def _prompt_new_passphrase() -> str:
    pw = getpass.getpass("New map passphrase: ")
    if not pw:
        raise SystemExit("Empty passphrase — aborting.")
    if getpass.getpass("Confirm passphrase: ") != pw:
        raise SystemExit("Passphrases do not match — aborting.")
    return pw


def _write_atomic(path: Path, data: bytes) -> None:
    tmp = path.with_name(path.name + ".enc-tmp")
    with open(tmp, "wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)  # atomic on the same filesystem


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--project", required=True, help="Project directory")
    args = parser.parse_args()

    from src.security.at_rest import decrypt_bytes, encrypt_bytes, is_encrypted

    targets = _resolve_targets(args.project)
    if not targets:
        print("No geo artifacts found (calibration.json / database_graph.geojson).")
        return 1

    pending = [p for p in targets if not is_encrypted(p.read_bytes())]
    for p in targets:
        mark = "already encrypted" if p not in pending else "will encrypt"
        print(f"  [{mark}] {p}")
    if not pending:
        print("Nothing to do — all target artifacts are already encrypted.")
        return 0

    pw = _prompt_new_passphrase()

    for path in pending:
        data = path.read_bytes()
        container = encrypt_bytes(data, pw)
        # Verify the round-trip in RAM BEFORE overwriting — never destroy a
        # plaintext we cannot decrypt back.
        if decrypt_bytes(container, pw) != data:
            raise SystemExit(f"Round-trip verify FAILED for {path} — aborting, file untouched.")
        _write_atomic(path, container)
        print(f"  encrypted in place: {path.name} ({len(data)} -> {len(container)} bytes)")

    print(
        "\nDone. Keep the passphrase safe — it cannot be recovered.\n"
        "Run the app with DRONELOC_PASSPHRASE set (or enter it when prompted)."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
