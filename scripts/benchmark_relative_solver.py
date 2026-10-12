"""Synthetic scaling benchmark: four temporal neighbors, 512 points per link."""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import psutil

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.geometry.frame_transform import project_points
from src.geometry.relative_alignment import PointLink
from src.geometry.relative_projective import refine_projective_map


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames", type=int, nargs="+", default=[25, 50, 100, 200])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    results = []
    for count in args.frames:
        rng = np.random.default_rng(102)
        truth = {
            i: np.array([[1.0, 0.002 * np.sin(i), 30 * i], [0.0, 1.0, 2 * i], [0.0, 0.0, 1.0]])
            for i in range(count)
        }
        initial = {i: H.copy() for i, H in truth.items()}
        for i in range(1, count):
            initial[i][:2, 2] += rng.normal(0, 1, 2)
        links = []
        for i in range(count):
            for j in range(i + 1, min(count, i + 5)):
                p = rng.uniform([150, 80], [1100, 640], (512, 2))
                q = project_points(np.linalg.inv(truth[j]) @ truth[i], p) + rng.normal(
                    0, 0.15, p.shape
                )
                links.append(PointLink(i, j, p, q))
        start = time.perf_counter()
        _, qa, _ = refine_projective_map(initial, links, 1280, 720, {0}, max_nfev=50)
        row = {
            "frames": count,
            "links": len(links),
            "seconds": time.perf_counter() - start,
            "peak_working_set_mb": psutil.Process().memory_info().peak_wset / 2**20,
            "qa": qa,
        }
        results.append(row)
        print(json.dumps(row), flush=True)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(results, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
