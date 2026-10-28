"""Does the Depth-Anything-V2 depth hint carry altitude information at all?

LOCALIZER_IMPROVEMENT_PLAN 0.3. ``DepthEstimator.get_relative_scale`` returns
1/median(disparity), which grows with distance, and ScaleManager uses the ratio
query/DB as r = query_alt / db_alt. Because DA-V2 output is affine-invariant
(scale and shift unknown per image), that ratio may be noise on nadir frames.

Input: a CSV with columns ``image,altitude_m`` (AGL or any common datum), e.g.
frames exported from DroneSimUE at several altitudes over the same area.
The first row is the reference ("DB") frame.

Output: per-frame measured ratio vs true altitude ratio, plus Spearman rank
correlation and median relative error. A useful hint needs rho close to +1.

Usage (Windows, project venv, GPU):
    python scripts/check_depth_scale_vs_altitude.py frames.csv --device cuda
"""

import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.depth.depth_estimator import DepthEstimator  # noqa: E402


def _rank(a: np.ndarray) -> np.ndarray:
    order = np.argsort(a)
    r = np.empty(len(a), dtype=np.float64)
    r[order] = np.arange(len(a), dtype=np.float64)
    return r


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    rx, ry = _rank(x), _rank(y)
    rx -= rx.mean()
    ry -= ry.mean()
    den = float(np.sqrt((rx**2).sum() * (ry**2).sum()))
    return float((rx * ry).sum() / den) if den > 0 else float("nan")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("csv", type=Path)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    rows = list(csv.DictReader(args.csv.open(newline="", encoding="utf-8")))
    if len(rows) < 3:
        print("need at least 3 frames (reference + 2)")
        return 2

    est = DepthEstimator.build("depth_anything_v2", device=args.device)
    scales, alts = [], []
    for row in rows:
        path = Path(row["image"])
        if not path.is_absolute():
            path = args.csv.parent / path
        bgr = cv2.imread(str(path))
        if bgr is None:
            print(f"cannot read {path}")
            return 2
        scales.append(est.get_relative_scale(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)))
        alts.append(float(row["altitude_m"]))

    s = np.asarray(scales)
    a = np.asarray(alts)
    meas = s / s[0]
    true = a / a[0]
    print(f"{'image':40s} {'alt_m':>8s} {'true_r':>8s} {'meas_r':>8s}")
    for row, al, tr, mr in zip(rows, a, true, meas):
        print(f"{Path(row['image']).name[:40]:40s} {al:8.1f} {tr:8.3f} {mr:8.3f}")
    rel = np.abs(meas[1:] - true[1:]) / true[1:]
    print(f"spearman(meas_r, true_r) = {spearman(meas, true):+.3f}  (useful hint: close to +1)")
    print(f"median |meas_r - true_r| / true_r = {np.median(rel):.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
