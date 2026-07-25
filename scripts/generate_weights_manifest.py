"""Generate models/weights_manifest.json — the pinned SHA-256 set for the
weight-integrity preflight (HARDENING P2-12).

Run this on the target AFTER staging all weights offline:

    python scripts/generate_weights_manifest.py                # -> models/weights_manifest.json
    python scripts/generate_weights_manifest.py --root models --out models/weights_manifest.json

Then enable enforcement in user_config.json:

    "models": { "performance": { "weight_integrity_mode": "enforce" } }

Re-run whenever weights legitimately change (e.g. after rebuilding a TensorRT
.engine on new hardware) so the manifest reflects the intended set.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.weight_integrity import generate_manifest  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="models", help="Weights root directory")
    parser.add_argument(
        "--out", default="models/weights_manifest.json", help="Output manifest path"
    )
    args = parser.parse_args()

    root = Path(args.root)
    if not root.is_dir():
        print(f"ERROR: root directory not found: {root}", file=sys.stderr)
        sys.exit(1)

    manifest = generate_manifest(root)
    out = Path(args.out)
    out.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Wrote {len(manifest)} weight hashes to {out}")


if __name__ == "__main__":
    main()
