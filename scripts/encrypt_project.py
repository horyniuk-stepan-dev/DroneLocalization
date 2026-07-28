"""HARDENING P1-6: build an encrypted COPY of a project (non-destructive).

Produces a separate, fully-encrypted copy of a project directory, leaving the
plaintext master untouched. Deployment model: the ground station keeps the
master; the drone carries only the encrypted copy, so airframe capture yields
ciphertext.

    python scripts/encrypt_project.py --project <src> --output <dst>

EVERY file is encrypted (passphrase-derived AES-256-GCM) — there is no allowlist
of "sensitive" files to get wrong. Two earlier allowlist bugs (root-only matching,
then a renamed database) each shipped geo-anchors in plaintext inside a copy
advertised as encrypted; encrypting everything removes that class of bug.

The passphrase is prompted twice and never stored — keep it safe, it cannot be
recovered.

Because ``project.json`` is encrypted too, it doubles as the marker: a project
whose manifest carries the container header is an encrypted deployment copy. The
app detects that before loading, prompts for the passphrase, and refuses every
write into such a project — the copy is immutable by design. Rebuilds,
calibration saves and propagation belong on the plaintext master.
"""

from __future__ import annotations

import argparse
import getpass
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def build_encrypted_copy(src_dir: str, dst_dir: str, passphrase: str) -> dict:
    """Copy ``src_dir`` to ``dst_dir``, encrypting every file without exception.

    Returns a summary ``{"encrypted": [...], "total": n}``. The source is never
    modified. ``dst_dir`` must not already exist (refuse to overwrite)."""
    from src.security.at_rest import encrypt_file

    src, dst = Path(src_dir), Path(dst_dir)
    if not src.is_dir():
        raise SystemExit(f"Source project not found: {src}")
    if dst.exists():
        raise SystemExit(f"Output already exists (refusing to overwrite): {dst}")

    encrypted: list[str] = []
    for path in sorted(src.rglob("*")):
        rel = path.relative_to(src)
        target = dst / rel
        if path.is_dir():
            target.mkdir(parents=True, exist_ok=True)
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        encrypt_file(str(path), str(target), passphrase)
        encrypted.append(str(rel))
    return {"encrypted": encrypted, "total": len(encrypted)}


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
    print(f"\n  {summary['total']} file(s) encrypted, none left in plaintext.")
    if not summary["encrypted"]:
        print("WARNING: the source project is empty — nothing was encrypted.")
    print(
        "\nDone. The plaintext master is untouched. Keep the passphrase safe — it "
        "cannot be recovered.\nThe copy is read-only: the app refuses to write into "
        "it. Rebuild and recalibrate on the master."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
