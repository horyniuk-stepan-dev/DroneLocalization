"""HARDENING P2-12: weight-integrity pinning (offline / air-gap assurance).

The models are pre-staged offline; at startup we can verify every weight file
against a pinned SHA-256 manifest and **fail closed** if any file is missing or
altered. This detects a swapped/corrupted weight (supply-chain or on-disk
tamper) before it is ever loaded — a single go/no-go preflight rather than a
hook threaded through every model-load site.

Modes (``models.performance.weight_integrity_mode``):
- ``off``     — skip entirely (default; current behavior).
- ``warn``    — log any mismatch/missing but continue.
- ``enforce`` — raise on the first problem set; caller aborts startup.

Generate the manifest with ``scripts/generate_weights_manifest.py`` after
staging weights on the target, then commit/ship ``models/weights_manifest.json``.
"""

import hashlib
import json
from pathlib import Path

WEIGHT_EXTENSIONS = (".pth", ".pt", ".onnx", ".engine")
# The redirected torch/HF download cache is not part of the pinned weight set.
_EXCLUDE_DIR_PARTS = (".cache",)


class WeightIntegrityError(RuntimeError):
    """Raised in 'enforce' mode when weights are missing or altered."""


def compute_sha256(path: str | Path, chunk_size: int = 1 << 20) -> str:
    """Streaming SHA-256 of a file (chunked so large .engine files don't OOM)."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def iter_weight_files(root: str | Path):
    """Yield weight files under ``root`` (relative Path), excluding the cache."""
    root = Path(root)
    for p in sorted(root.rglob("*")):
        if not p.is_file() or p.suffix.lower() not in WEIGHT_EXTENSIONS:
            continue
        rel = p.relative_to(root)
        if any(part in _EXCLUDE_DIR_PARTS for part in rel.parts):
            continue
        yield rel


def generate_manifest(root: str | Path) -> dict:
    """Build a manifest dict {relative_posix_path: sha256} for all weight files."""
    root = Path(root)
    return {rel.as_posix(): compute_sha256(root / rel) for rel in iter_weight_files(root)}


def verify_manifest(root: str | Path, manifest: dict) -> list[str]:
    """Return a list of human-readable problems (empty = all pinned files match).

    Checks every file listed in the manifest: reports it if missing or if its
    hash differs. Files on disk not in the manifest are ignored (the manifest is
    the authority on what must match).
    """
    root = Path(root)
    problems: list[str] = []
    for rel, expected in manifest.items():
        fpath = root / rel
        if not fpath.is_file():
            problems.append(f"MISSING: {rel}")
            continue
        actual = compute_sha256(fpath)
        if actual != expected:
            problems.append(f"MISMATCH: {rel} (expected {expected[:12]}…, got {actual[:12]}…)")
    return problems


def run_preflight(
    root: str | Path,
    manifest_path: str | Path,
    mode: str = "off",
    logger=None,
) -> bool:
    """Run the startup integrity gate. Returns True if OK / skipped.

    In ``enforce`` mode, raises ``WeightIntegrityError`` on any problem. In
    ``warn`` mode, logs and returns False. In ``off`` mode, returns True.
    A missing manifest is a soft skip in ``off``/``warn`` but a hard failure in
    ``enforce`` (you asked to enforce but gave nothing to enforce against).
    """
    mode = (mode or "off").lower()
    if mode == "off":
        return True

    manifest_path = Path(manifest_path)
    if not manifest_path.is_file():
        msg = f"weight_integrity_mode={mode} but manifest not found: {manifest_path}"
        if mode == "enforce":
            raise WeightIntegrityError(msg)
        if logger:
            logger.warning(msg)
        return False

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    problems = verify_manifest(root, manifest)

    if not problems:
        if logger:
            logger.info(f"Weight integrity OK — {len(manifest)} files verified.")
        return True

    detail = f"Weight integrity check FAILED ({len(problems)} problem(s)):\n  " + "\n  ".join(
        problems
    )
    if mode == "enforce":
        raise WeightIntegrityError(detail)
    if logger:
        logger.warning(detail)
    return False
