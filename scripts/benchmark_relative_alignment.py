"""Read-only comparison of relative maps using existing database observations.

Only the JSON report is written. Database calibration and source videos are never
modified. Ground-truth center errors use a single first-frame affine gauge; no
other ground-truth frame is used to fit the map.
"""

import argparse
import copy
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import torch  # noqa: F401 -- Windows native runtime must initialize first

# isort: split
import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import APP_CONFIG
from src.geometry.frame_transform import as_homography, project_points
from src.geometry.pair_refinement import symmetric_errors
from src.geometry.pose_graph.optimizer import PoseGraphOptimizer
from src.geometry.relative_alignment import refine_relative_affines
from src.geometry.relative_projective import initialize_from_pairs, refine_projective_map
from src.localization.matcher import FeatureMatcher
from src.models.model_manager import ModelManager
from src.utils.logging_utils import get_logger
from src.workers.propagation_pipeline import PropagationPipeline


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, required=True)
    parser.add_argument("--ground-truth", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=24)
    parser.add_argument("--start-slot", type=int, default=0)
    parser.add_argument("--point-limit", type=int, default=512)
    parser.add_argument("--no-subpixel", action="store_true")
    args = parser.parse_args()
    if args.frames < 2:
        parser.error("--frames must be at least 2")
    if args.start_slot < 0 or args.point_limit < 0:
        parser.error("--start-slot and --point-limit must be nonnegative")
    if args.output.suffix.lower() != ".json" or args.output.resolve() == args.db.resolve():
        parser.error("--output must be a separate JSON report")
    if args.ground_truth and args.output.resolve() == args.ground_truth.resolve():
        parser.error("--output cannot overwrite ground truth")
    logger = get_logger(__name__)
    with h5py.File(args.db, "r") as db:
        metadata = dict(db["metadata"].attrs)
        counts = db["local_features/kp_counts"][:]
        ids = [int(i) for i in np.flatnonzero(counts) if i >= args.start_slot][: args.frames]
        features = {
            i: {
                "keypoints": db["local_features/keypoints"][i, : counts[i]].astype(float),
                "descriptors": db["local_features/descriptors"][i, : counts[i]].astype(np.float32),
                "image_size": np.array([metadata["frame_height"], metadata["frame_width"]]),
            }
            for i in ids
        }
    if len(ids) < 2:
        parser.error("Selected range has fewer than two keyframes")
    config = copy.deepcopy(APP_CONFIG)
    config["models"]["performance"]["torch_compile"] = False
    matcher = FeatureMatcher(ModelManager(config=config), config)
    w, h = int(metadata["frame_width"]), int(metadata["frame_height"])
    gauge = np.array([[1.0, 0.0, -w / 2], [0.0, -1.0, h / 2]])
    gt = None
    if args.ground_truth:
        gt_data = json.loads(args.ground_truth.read_text(encoding="utf-8"))
        if (
            list(gt_data.get("frame_size", [])) != [w, h]
            or gt_data.get("frame_step") != metadata["frame_step"]
        ):
            raise ValueError("Ground-truth dimensions/frame_step do not match the database")
        gt = {int(row["slot"]): row for row in gt_data["slots"]}
    output = {
        "database": str(args.db),
        "frame_ids": ids,
        "frame_size": [w, h],
        "point_limit": args.point_limit,
        "results": {},
    }
    solved, validation_links = {}, []
    for mode in ("legacy_local", "accuracy"):
        started = time.perf_counter()
        conf = copy.deepcopy(config)
        conf["propagation"]["high_accuracy"] = mode == "accuracy"
        conf["propagation"]["point_pair_limit"] = args.point_limit
        pipeline = PropagationPipeline(
            SimpleNamespace(metadata=metadata),
            SimpleNamespace(converter=SimpleNamespace(mode="LOCAL")),
            matcher,
            conf,
        )
        pipeline._relative_feature_ids = {id(f): i for i, f in features.items()}
        if mode == "accuracy" and not args.no_subpixel:
            from src.geometry.pair_refinement import SourceFrames

            pipeline._source_frames = SourceFrames(metadata)
        graph = PoseGraphOptimizer(w, h, isotropy_weight=pipeline.isotropy_weight)
        for fid in ids:
            graph.add_node(fid)
        graph.fix_node(ids[0], gauge)
        try:
            pipeline._build_temporal_edges(graph, features, max(ids) + 1)
            links = pipeline._accuracy_links(graph)
            if mode == "accuracy":
                initial = initialize_from_pairs(
                    pipeline._pair_homographies,
                    [(e.from_id, e.to_id, e.weight) for e in graph.edges],
                    {ids[0]: gauge},
                    w,
                    h,
                )
                transforms, report, _ = refine_projective_map(initial, links, w, h, {ids[0]})
            else:
                graph.initialize_from_bfs()
                initial = graph.optimize(
                    max_iterations=50,
                    use_analytic_jac=True,
                    two_stage_prune=pipeline.two_stage_prune,
                )
                transforms, report, _ = refine_relative_affines(initial, links, w, h, ids[0])
            metrics = {
                "frames": len(transforms),
                "edges": len(graph.edges),
                "seconds": time.perf_counter() - started,
                "alignment": report,
                "subpixel_points": sum(
                    q.get("subpixel_points", 0) for q in pipeline._pair_diagnostics.values()
                ),
            }
            solved[mode] = transforms
            if mode == "accuracy":
                validation_links = links
                metrics["pair_diagnostics"] = {
                    f"{a}:{b}": q for (a, b), q in pipeline._pair_diagnostics.items()
                }
            if gt:
                first = gt[ids[0]]
                seed = as_homography(first["affine"])
                center = np.array([[w / 2, h / 2]])
                # Fit only the first ground center, retaining the seed image's linear scale.
                seed[:2, 2] += (
                    np.asarray(first["center_mercator"]) - project_points(seed, center)[0]
                )
                to_metric = seed @ np.linalg.inv(as_homography(gauge))
                scale = np.sqrt(abs(np.linalg.det(seed[:2, :2])))
                errors = [
                    float(
                        np.linalg.norm(
                            project_points(to_metric @ as_homography(T), center)[0]
                            - gt[fid]["center_mercator"]
                        )
                        / scale
                    )
                    for fid, T in transforms.items()
                    if fid in gt and fid != ids[0]
                ]
                metrics["ground_center_error_seed_pixels"] = {
                    "median": float(np.median(errors)),
                    "p95": float(np.percentile(errors, 95)),
                    "max": max(errors),
                    "count": len(errors),
                    "scope": "first-frame affine scale and center only; planar model over real simulator terrain",
                }
            output["results"][mode] = metrics
            logger.info(f"{mode}: {metrics}")
        except ValueError as exc:
            output["results"][mode] = {
                "error": str(exc),
                "edges": len(graph.edges),
                "seconds": time.perf_counter() - started,
            }
            logger.error(f"{mode} could not produce a valid map: {exc}")
        finally:
            if pipeline._source_frames is not None:
                pipeline._source_frames.close()
    # Same observations and same units for both methods. Hold-out is from the
    # global adjustment only: the pairwise estimators already saw these points.
    common_ids = set.intersection(*(set(T) for T in solved.values())) if solved else set()
    output["common_frame_ids"] = sorted(common_ids)
    for mode, transforms in solved.items():
        errors = []
        for link in validation_links:
            if link.first not in common_ids or link.second not in common_ids:
                continue
            order = np.lexsort((link.first_points[:, 1], link.first_points[:, 0]))[::5]
            H = np.linalg.inv(as_homography(transforms[link.second])) @ as_homography(
                transforms[link.first]
            )
            errors.extend(symmetric_errors(H, link.first_points[order], link.second_points[order]))
        if errors:
            output["results"][mode]["common_transfer_error_px"] = {
                "median": float(np.median(errors)),
                "p95": float(np.percentile(errors, 95)),
                "count": len(errors),
                "scope": "same accuracy pair observations; withheld from global adjustment only",
            }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
