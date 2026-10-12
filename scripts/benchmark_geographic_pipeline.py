"""Full GEO pipeline comparison on workspace copies of an existing project.

Runs retrieval, gates, pruning, terrain, relative refinement, atomic save/reload
and GeoJSON export. Only the explicitly selected new output directory is written.
"""

import argparse
import copy
import hashlib
import json
import shutil
import sys
import time
from pathlib import Path

import torch  # noqa: F401 -- initialize Windows native runtime first

# isort: split
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import APP_CONFIG
from src.calibration.multi_anchor_calibration import MultiAnchorCalibration
from src.database.database_loader import DatabaseLoader
from src.geometry.frame_transform import as_homography, get_frame_transform, project_points
from src.geometry.pair_refinement import symmetric_errors
from src.localization.matcher import FeatureMatcher
from src.models.model_manager import ModelManager
from src.workers.propagation_pipeline import PropagationPipeline


def digest(path):
    sha = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            sha.update(block)
    return sha.hexdigest()


def distribution(values):
    return (
        {
            "count": len(values),
            "median": float(np.median(values)),
            "p95": float(np.percentile(values, 95)),
            "max": float(np.max(values)),
        }
        if len(values)
        else None
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--source", default="main")
    parser.add_argument("--ground-truth", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    root = args.output_dir.resolve()
    if root.exists() or root == args.project.resolve() or args.project.resolve() in root.parents:
        parser.error("Use a new output directory outside the input project")
    source = args.project / "sources" / args.source
    original_hash = digest(source / "database.h5")
    root.mkdir(parents=True)
    project = json.loads((args.project / "project.json").read_text(encoding="utf-8"))
    truth = {
        int(row["slot"]): row
        for row in json.loads(args.ground_truth.read_text(encoding="utf-8"))["slots"]
    }
    conf = copy.deepcopy(APP_CONFIG)
    conf["models"]["performance"]["torch_compile"] = False
    matcher = FeatureMatcher(ModelManager(config=conf), conf)
    output = {
        "project": str(args.project),
        "source": args.source,
        "scope": "full_GEO_pipeline",
        "input_sha256": original_hash,
        "results": {},
    }
    maps, validation = {}, []
    for mode in ("legacy", "accuracy"):
        directory = root / mode
        directory.mkdir()
        (directory / "project.json").write_text(json.dumps(project), encoding="utf-8")
        shutil.copy2(source / "database.h5", directory / "database.h5")
        if (source / "vectors.lance").exists():
            shutil.copytree(source / "vectors.lance", directory / "vectors.lance")
        calibration = MultiAnchorCalibration()
        calibration.load(str(source / "calibration.json"))
        cfg = copy.deepcopy(conf)
        cfg.setdefault("propagation", {})["high_accuracy"] = mode == "accuracy"
        # Each mode may cache public DEM tiles only in its own workspace copy.
        cfg["graph_optimization"]["terrain_dem_path"] = str(root / "terrain")
        errors, completed = [], []
        db = DatabaseLoader(str(directory / "database.h5"))
        try:
            w, h = db.metadata["frame_width"], db.metadata["frame_height"]
            p = PropagationPipeline(
                db,
                calibration,
                matcher,
                cfg,
                error_callback=errors.append,
                completed_callback=lambda completed=completed: completed.append(True),
            )
            start = time.perf_counter()
            p._propagate()
            transforms = {
                int(i): get_frame_transform(db, int(i)).copy()
                for i in np.flatnonzero(db.frame_valid)
            }
            maps[mode] = transforms
            anchors = {a.frame_id for a in calibration.anchors}
            if mode == "accuracy":
                validation = [
                    ((link.first, link.second), (link.first_points, link.second_points))
                    for link in p._validated_alignment_links
                ]
            center = [[w / 2, h / 2]]
            occupied = set(map(int, np.flatnonzero(db.db_file["local_features/kp_counts"][:])))
            errors_ground = [
                np.linalg.norm(project_points(T, center)[0] - truth[i]["center_mercator"])
                for i, T in transforms.items()
                if i in truth and i in occupied and i not in anchors
            ]
            qa = {
                "seconds": time.perf_counter() - start,
                "completed": bool(completed),
                "errors": errors,
                "slots": db.get_num_frames(),
                "occupied": len(occupied),
                "anchors": len(anchors),
                "saved_valid_frames": len(transforms),
                "relative_alignment": p._relative_alignment_report,
                "nonanchor_center_error_mercator_m": distribution(errors_ground),
            }
            if (directory / "database_graph.geojson").exists():
                graph = json.loads(
                    (directory / "database_graph.geojson").read_text(encoding="utf-8")
                )
                edges = [f for f in graph["features"] if f["geometry"]["type"] == "LineString"]
                qa["exported_edges"] = len(edges)
                qa["edges_with_residual"] = sum(
                    e["properties"].get("residual") is not None for e in edges
                )
            output["results"][mode] = qa
            (root / "report.json").write_text(json.dumps(output, indent=2), encoding="utf-8")
            print(json.dumps({mode: qa}), flush=True)
        finally:
            db.close()
    for mode, transforms in maps.items():
        residuals = []
        for (first, second), (p, q) in validation:
            if all(first in M and second in M for M in maps.values()):
                order = np.lexsort((p[:, 1], p[:, 0]))[::5]
                H = np.linalg.inv(as_homography(transforms[second])) @ as_homography(
                    transforms[first]
                )
                residuals.extend(symmetric_errors(H, p[order], q[order]))
        output["results"][mode]["common_transfer_px"] = distribution(residuals)
    output["original_database_unchanged"] = digest(source / "database.h5") == original_hash
    (root / "report.json").write_text(json.dumps(output, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
