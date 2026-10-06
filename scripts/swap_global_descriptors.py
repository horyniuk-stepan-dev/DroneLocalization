"""Swap the global descriptors of a project's layers in a copy (retrieval-only A/B).

Keyframes, local features and calibration stay byte-identical; only each layer's
``vectors.lance`` and the schema metadata change, so a replay against the copy differs
from the original only in global retrieval (e.g. VLAD vs CLS). A full rebuild would
also re-run propagation, whose loop closures use the same descriptors, and so change
the maps themselves.

    python scripts/swap_global_descriptors.py \\
        --project "D:/My Projects/TEST/testtopboch_max" \\
        --out "D:/My Projects/TEST/testtopboch_max_vlad" \\
        --set models.vlad.enabled=true \\
        --set models.vlad.vocab_path=models/vlad_vocab_v1_c32_p256.npz

Query the copy with the same overrides (replay ``--set ...``): with VLAD off in
user_config.json its layers are refused, by design. Each layer's source video is
re-decoded exactly like the database builder does (cv2, every ``frame_step``-th
frame, BGR -> RGB) and must match the size recorded in the database. Keypoint
visualisation videos are not copied unless ``--copy-keypoint-videos`` is given.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections.abc import Iterable, Iterator
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Settings that define what a global descriptor is; everything else in a database's
# schema (local features, keypoints, frame step, ...) is kept from the original.
GLOBAL_DESCRIPTOR_FIELDS = (
    "global_backend",
    "descriptor_dim",
    "vlad_enabled",
    "vlad_pca_dim",
    "dino_cpu_resize",
    "vlad_vocab",
    "vlad_layer",
    "vlad_low_norm_fraction",
    "dino_input_size",
)
KEYPOINT_VIDEO = "database_keypoints.mp4"
STANDARD_DIRS = ("panoramas", "reports", "test_photos", "test_videos")


def swapped_components(stored: dict, runtime: dict, descriptor_dim: int) -> dict:
    """The stored schema with the global-descriptor fields taken from the runtime."""
    result = dict(stored)
    for key in GLOBAL_DESCRIPTOR_FIELDS:
        result[key] = runtime.get(key)
    result["descriptor_dim"] = int(descriptor_dim)
    return result


def check_output(out: Path) -> None:
    """Refuse an output folder that already holds a project or layer data.

    A folder with only empty helper folders or replay reports (e.g. created by a
    replay script run too early) is accepted.
    """
    if (out / "project.json").exists() or (out / "sources").exists():
        raise ValueError(
            f"{out} already holds a project (project.json or sources/); "
            "delete it or choose another --out"
        )


def copy_project(project: Path, out: Path, include_keypoint_videos: bool = False) -> dict:
    """Copy project.json and every listed layer folder; returns the copied manifest."""
    check_output(out)
    manifest = json.loads((project / "project.json").read_text(encoding="utf-8"))
    sources = manifest.get("video_sources") or []
    if not sources:
        raise ValueError(f"{project / 'project.json'} lists no video_sources")
    out.mkdir(parents=True, exist_ok=True)
    ignore = None if include_keypoint_videos else shutil.ignore_patterns(KEYPOINT_VIDEO)
    for source in sources:
        layer_dir = Path(source["database_file"]).parent
        shutil.copytree(project / layer_dir, out / layer_dir, ignore=ignore)
    for name in STANDARD_DIRS:
        (out / name).mkdir(exist_ok=True)
    manifest["project_name"] = out.name
    (out / "project.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return manifest


def keyframes_from_video(
    video: Path, frame_step: int, slots: Iterable[int]
) -> Iterator[tuple[int, object]]:
    """(slot, RGB frame) for the wanted slots, decoded like the database builder."""
    import cv2

    wanted = set(int(s) for s in slots)
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise ValueError(f"cannot open video {video}")
    try:
        index = 0
        while wanted:
            if not cap.grab():
                break
            if index % frame_step == 0 and index // frame_step in wanted:
                ok, frame = cap.retrieve()
                if not ok:
                    raise ValueError(f"cannot decode frame {index} of {video}")
                wanted.discard(index // frame_step)
                yield index // frame_step, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            index += 1
    finally:
        cap.release()
    if wanted:
        raise ValueError(f"{video} ended before slots {sorted(wanted)[:5]}... were reached")


def read_frame_ids(lance_dir: Path) -> list[int]:
    import lancedb

    table = lancedb.connect(str(lance_dir)).open_table("global_vectors")
    return [int(v) for v in table.to_arrow().column("frame_id").to_pylist()]


def rewrite_lance(lance_dir: Path, rows: list[tuple[int, object]], index_min_frames: int) -> None:
    """Replace the table like DBWriter does: frame_id + fixed-size float32 vector."""
    import lancedb
    import numpy as np
    import pyarrow as pa

    if not rows:
        raise ValueError("no descriptors to write")
    dim = int(np.asarray(rows[0][1]).size)
    if lance_dir.exists():
        shutil.rmtree(lance_dir)
    schema = pa.schema(
        [pa.field("frame_id", pa.int32()), pa.field("vector", pa.list_(pa.float32(), dim))]
    )
    table = lancedb.connect(str(lance_dir)).create_table(
        "global_vectors", schema=schema, mode="create"
    )
    table.add(
        [
            {"frame_id": int(fid), "vector": np.asarray(vec, dtype=np.float32).tolist()}
            for fid, vec in rows
        ]
    )
    if len(rows) >= index_min_frames:
        table.create_index(
            metric="cosine", num_partitions=min(256, len(rows) // 8), num_sub_vectors=32
        )


def update_metadata(h5_path: Path, components: dict) -> str:
    """Write descriptor_dim, schema_components and the new fingerprint; returns it."""
    import h5py

    from src.database.schema_fingerprint import compute_fingerprint

    fingerprint = compute_fingerprint(components)
    with h5py.File(h5_path, "r+") as db:
        meta = db["metadata"].attrs
        meta["descriptor_dim"] = int(components["descriptor_dim"])
        meta["schema_components"] = json.dumps(components, sort_keys=True)
        meta["schema_fingerprint"] = fingerprint
    return fingerprint


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--project", required=True, type=Path, help="existing project folder")
    parser.add_argument(
        "--out", required=True, type=Path, help="new project folder (must not exist)"
    )
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        dest="config_set",
        metavar="SECTION.KEY=VALUE",
        help="config override for the new descriptors, e.g. models.vlad.enabled=true",
    )
    parser.add_argument("--batch", type=int, default=8, help="frames per descriptor batch")
    parser.add_argument("--copy-keypoint-videos", action="store_true")
    args = parser.parse_args(argv)
    from scripts.replay_multilayer_ground_truth import config_overrides

    try:
        args.overrides = config_overrides(args.config_set)
    except ValueError as exc:
        parser.error(str(exc))
    return args


def main(argv=None) -> int:
    args = parse_args(argv)
    project, out = args.project.resolve(), args.out.resolve()
    try:  # before the models load
        check_output(out)
        if not (project / "project.json").is_file():
            raise ValueError(f"no project.json in {project}")
    except ValueError as exc:
        print(f"ERROR: {exc}")
        return 1

    import h5py

    from config import APP_CONFIG
    from scripts.propagation_sweep import apply_overrides
    from src.database.schema_fingerprint import build_components
    from src.models.model_manager import ModelManager
    from src.models.wrappers.feature_extractor import FeatureExtractor

    config = apply_overrides(APP_CONFIG, args.overrides)
    manager = ModelManager(config=config)
    extractor = FeatureExtractor(
        manager.load_local_extractor(), manager.load_dinov2(), manager.device, config=config
    )
    if config["models"]["vlad"]["enabled"] and extractor.vlad_aggregator is None:
        print("ERROR: models.vlad.enabled but the vocabulary did not load (see the log above)")
        return 1
    dim = int(extractor.global_descriptor_dim)
    runtime = build_components(config, descriptor_dim=dim, local_descriptor_dim=0)
    index_min = int(config["database"].get("lancedb_index_min_frames", 256))

    try:
        manifest = copy_project(project, out, args.copy_keypoint_videos)
    except ValueError as exc:
        print(f"ERROR: {exc}")
        return 1
    print(f"copied {project} -> {out} (descriptor dim {dim})")
    report = {
        "created": datetime.now().astimezone().isoformat(timespec="seconds"),
        "source_project": str(project),
        "overrides": args.overrides,
        "layers": {},
    }
    for source in manifest["video_sources"]:
        sid = source["source_id"]
        db_path = out / source["database_file"]
        with h5py.File(db_path, "r") as db:
            meta = dict(db["metadata"].attrs)
        stored = json.loads(meta["schema_components"])
        frame_step = int(meta.get("frame_step", stored.get("frame_step", 30)))
        video = Path(str(meta.get("source_path") or source["video_path"]))
        expected_size = int(meta.get("source_size_bytes", -1))
        if not video.is_file() or (expected_size >= 0 and video.stat().st_size != expected_size):
            print(f"ERROR: layer {sid}: source video missing or changed: {video}")
            return 1
        lance_dir = db_path.parent / "vectors.lance"
        frame_ids = read_frame_ids(lance_dir)
        rows, batch_ids, batch_frames = [], [], []
        for slot, rgb in keyframes_from_video(video, frame_step, frame_ids):
            batch_ids.append(slot)
            batch_frames.append(rgb)
            if len(batch_frames) == args.batch:
                rows += zip(batch_ids, extractor.extract_global_descriptors_multi(batch_frames))
                batch_ids, batch_frames = [], []
                print(f"  layer {sid}: {len(rows)}/{len(frame_ids)}", end="\r")
        if batch_frames:
            rows += zip(batch_ids, extractor.extract_global_descriptors_multi(batch_frames))
        rows.sort(key=lambda row: row[0])
        rewrite_lance(lance_dir, rows, index_min)
        components = swapped_components(stored, runtime, dim)
        fingerprint = update_metadata(db_path, components)
        layer = source.get("scale_layer")
        if isinstance(layer, dict) and layer.get("descriptor_schema_fingerprint"):
            layer["descriptor_schema_fingerprint"] = fingerprint
        report["layers"][sid] = {
            "keyframes": len(rows),
            "old_fingerprint": str(meta.get("schema_fingerprint")),
            "new_fingerprint": fingerprint,
        }
        print(f"  layer {sid}: {len(rows)} descriptors, fingerprint {fingerprint}")
    (out / "project.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (out / "swap_global_descriptors.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"done: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
