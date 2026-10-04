"""Estimate flight_data.camera_yaw_offset_deg from a continuous-rotation replay.

In localization.rotation_mode=continuous every confirmed fix reports the angle
it applied to the frame (rotation_deg), measured by vision. Telemetry predicts
the same angle as ``reference_heading - heading - offset``. The circular mean of
the difference is the camera mount offset; a large spread means the heading
convention is wrong (the script also tries the mirrored heading).

    python scripts/fit_yaw_offset.py --replay-csv R.csv --telemetry T.csv [--preset flightsim]
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.flight_data import CsvFlightData, heading_to_yaw_hint  # noqa: E402


def circular_stats(angles_deg: list[float]) -> tuple[float, float]:
    """(mean in [0, 360), circular std in degrees)."""
    s = sum(math.sin(math.radians(a)) for a in angles_deg)
    c = sum(math.cos(math.radians(a)) for a in angles_deg)
    n = max(1, len(angles_deg))
    r = min(1.0, math.hypot(s, c) / n)
    mean = math.degrees(math.atan2(s, c)) % 360.0
    std = math.degrees(math.sqrt(-2.0 * math.log(max(r, 1e-12))))
    return mean, std


def fit(rows: list[tuple[float, float]], telemetry, reference_heading_deg: float = 0.0) -> dict:
    """rows: (timestamp, measured rotation_deg). Returns offsets for both heading senses."""
    out = {}
    for name, sign in (("as_logged", 1.0), ("mirrored", -1.0)):
        diffs = []
        for t, measured in rows:
            sample = telemetry.sample_at(t)
            if sample is None or sample.heading_deg is None:
                continue
            hint0 = heading_to_yaw_hint(sign * sample.heading_deg, reference_heading_deg, 0.0)
            diffs.append((hint0 - measured) % 360.0)
        mean, std = circular_stats(diffs) if diffs else (float("nan"), float("nan"))
        out[name] = {"offset_deg": mean, "spread_deg": std, "n": len(diffs)}
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--replay-csv", required=True, type=Path)
    p.add_argument("--telemetry", required=True, type=Path)
    p.add_argument("--preset", default="flightsim", choices=["flightsim", "generic"])
    p.add_argument("--reference-heading", type=float, default=0.0)
    args = p.parse_args(argv)
    rows = []
    with open(args.replay_csv, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if str(r.get("confirmed")).lower() == "true" and r.get("rotation_deg") not in (
                "",
                None,
            ):
                rows.append((float(r["timestamp"]), float(r["rotation_deg"])))
    if not rows:
        print(
            "no confirmed rows with rotation_deg (was the replay run with rotation_mode=continuous?)"
        )
        return 1
    res = fit(rows, CsvFlightData(args.telemetry, preset=args.preset), args.reference_heading)
    for name, r in res.items():
        print(
            f"{name:10s} offset {r['offset_deg']:7.2f} deg  spread {r['spread_deg']:6.2f} deg  n={r['n']}"
        )
    best = min(res, key=lambda k: res[k]["spread_deg"])
    print(
        f"-> use the '{best}' heading sense; flight_data.camera_yaw_offset_deg = "
        f"{((res[best]['offset_deg'] + 180) % 360) - 180:.1f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
