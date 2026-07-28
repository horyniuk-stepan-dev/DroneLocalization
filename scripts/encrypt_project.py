"""HARDENING P1-6: build an encrypted COPY of a project (non-destructive).

Produces a separate, fully-encrypted copy of a project directory, leaving the
plaintext master untouched. Deployment model: the ground station keeps the
master; the drone carries only the encrypted copy, so airframe capture yields
ciphertext.

    python scripts/encrypt_project.py --project <src> --output <dst>

Every sensitive artifact is encrypted (passphrase-derived AES-256-GCM); every
other file is copied verbatim. The passphrase is prompted twice and never
stored — keep it safe, it cannot be recovered.

Sensitive artifacts, encrypted per-file into the copy, matched by name at any
depth (projects store them per source, e.g. ``sources/main/database.h5``):
  * calibration.json          (geo anchors)
  * database_graph.geojson    (per-frame GPS)
  * database.h5               (keypoints/descriptors)  — loads decrypted (SP2)
  * database_keypoints.mp4    (used by calibration)
  * vectors.lance/**          (retrieval index, every file within)

The app auto-detects and decrypts calibration.json and database.h5 on load given
the passphrase. Load support for vectors.lance/ and database_keypoints.mp4 is
SP3 (pending) — they are encrypted in the copy already so the build is complete.
"""

from __future__ import annotations

import argparse
import getpass
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Whole-file artifacts that leak the mission; encrypted into the copy wherever
# they appear in the tree (root, or per-source under sources/<id>/).
SENSITIVE_FILES = {
    "calibration.json",
    "database_graph.geojson",
    "database.h5",
    "database_keypoints.mp4",
}
# Directories whose every file is sensitive (the retrieval index).
SENSITIVE_DIRS = {"vectors.lance"}
# Names are not fixed per source: a source may declare its own database file, and
# the keypoint video is named after it (<db_stem>_keypoints.mp4, see
# DatabaseBuilder). Match by suffix so a renamed map cannot dodge encryption —
# every .h5 in a project IS map data.
SENSITIVE_SUFFIXES = ("_keypoints.mp4", ".h5")


def _is_sensitive(rel_dir: Path, name: str) -> bool:
    """Match sensitive artifacts by basename at ANY depth.

    Projects keep their artifacts per source (``sources/main/database.h5``,
    ``sources/area2/calibration.json`` — see ProjectSettings defaults), so a
    root-only match would leave the geo-anchors of every real project in
    plaintext inside a copy advertised as encrypted."""
    if any(part in SENSITIVE_DIRS for part in rel_dir.parts):
        return True
    return name in SENSITIVE_FILES or name.endswith(SENSITIVE_SUFFIXES)


def build_encrypted_copy(src_dir: str, dst_dir: str, passphrase: str) -> dict:
    """Copy ``src_dir`` to ``dst_dir``, encrypting every sensitive artifact.

    Returns a summary ``{"encrypted": [...], "copied": n}``. The source is never
    modified. ``dst_dir`` must not already exist (refuse to overwrite)."""
    from src.security.at_rest import encrypt_file

    src, dst = Path(src_dir), Path(dst_dir)
    if not src.is_dir():
        raise SystemExit(f"Source project not found: {src}")
    if dst.exists():
        raise SystemExit(f"Output already exists (refusing to overwrite): {dst}")

    encrypted: list[str] = []
    copied = 0
    for path in sorted(src.rglob("*")):
        rel = path.relative_to(src)
        target = dst / rel
        if path.is_dir():
            target.mkdir(parents=True, exist_ok=True)
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        if _is_sensitive(rel.parent, path.name):
            encrypt_file(str(path), str(target), passphrase)
            encrypted.append(str(rel))
        else:
            shutil.copy2(path, target)
            copied += 1
    return {"encrypted": encrypted, "copied": copied}


def _prompt_new_passphrase() -> str:
    pw = getpass.getpass("New map passphrase: ")
    if not pw:
        raise SystemExit("Empty passphrase — aborting.")
    if getpass.getpass("Confirm passphrase: ") != pw:
        raise SystemExit("Passphrases do not match — aborting.")
    return pw


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--project", required=True, help="Source project directory (master)")
    parser.add_argument("--output", required=True, help="Destination for the encrypted copy")
    args = parser.parse_args()

    pw = _prompt_new_passphrase()
    summary = build_encrypted_copy(args.project, args.output, pw)

    for rel in summary["encrypted"]:
        print(f"  encrypted: {rel}")
    print(f"  copied verbatim: {summary['copied']} file(s)")
    if not summary["encrypted"]:
        print("WARNING: no sensitive artifacts were found to encrypt.")
    print(
        "\nDone. The plaintext master is untouched. Keep the passphrase safe — it "
        "cannot be recovered.\nRun the app on the copy with DRONELOC_PASSPHRASE set "
        "(or enter it when prompted)."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
