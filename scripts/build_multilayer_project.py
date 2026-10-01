"""Build a multi-layer project from FlightSimulator reference recordings, headless.

For every layer: database from the recorded video (same builder and settings as
the GUI, no forced slots), a check that every calibration anchor landed on a
database keyframe, the layer's calibration.json stamped with its layer id,
graph propagation, and — when the recording has ground_truth.json — the layer's
map error against that truth. Finally project.json is written, so the result
opens in the GUI like any other project.

    python scripts/build_multilayer_project.py --project "D:/My Projects/TEST/testtopboch_hh" \\
        --layer main 1000 "D:/My Projects/FlightSimulator/output/bochkivtsi_hh_1000m" \\
        --layer 2    2000 "D:/My Projects/FlightSimulator/output/bochkivtsi_hh_2000m" \\
        --layer 3    500  "D:/My Projects/FlightSimulator/output/bochkivtsi_hh_500m"

``--resume`` continues after an interruption: finished layers are not rebuilt.
Layers are processed one after another (one GPU model set is shared).
``--resnap-anchors`` repairs anchors whose slot the database did not keep as a
keyframe (see ``resnap_anchors``); the recording itself is never modified.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.layer_gt_tools import SimulatorRun, map_error_report  # noqa: E402


class BuildError(RuntimeError):
    pass


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--project", required=True, type=Path, help="new project folder")
    parser.add_argument("--name", help="project name (default: folder name)")
    parser.add_argument(
        "--layer",
        action="append",
        nargs=3,
        required=True,
        metavar=("ID", "ALTITUDE_M", "SIMULATOR_RUN_DIR"),
        help="reference layer; repeat. The first one becomes the project's main video.",
    )
    parser.add_argument("--area", default="area_main", help="area_id shared by all layers")
    parser.add_argument("--pin-exact-anchors", action="store_true")
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="SECTION.KEY=VALUE",
        help="override an existing config key for this build only (JSON value, e.g. "
        "graph_optimization.isotropy_weight=10.0); repeatable, user_config.json is untouched",
    )
    parser.add_argument(
        "--resnap-anchors",
        action="store_true",
        help="move simulator anchors that missed a database keyframe onto the nearest one "
        f"(within {RESNAP_MAX_SHIFT_SLOTS} slots) using ground_truth.json",
    )
    parser.add_argument("--resume", action="store_true", help="skip finished layers")
    args = parser.parse_args(argv)
    args.overrides = {}
    for item in args.set:
        name, sep, text = item.partition("=")
        if not sep or "." not in name:
            parser.error(f"--set expects SECTION.KEY=VALUE, got {item!r}")
        try:
            args.overrides[name.strip()] = json.loads(text)
        except json.JSONDecodeError:
            args.overrides[name.strip()] = text
    ids = [layer[0] for layer in args.layer]
    if len(set(i.casefold() for i in ids)) != len(ids):
        parser.error(f"layer IDs must be unique (case-insensitive): {ids}")
    for sid, altitude, _ in args.layer:
        if not sid.replace("_", "").replace("-", "").isalnum():
            parser.error(f"layer ID '{sid}': use letters, digits, _ and - only")
        try:
            if float(altitude) <= 0:
                raise ValueError
        except ValueError:
            parser.error(f"layer '{sid}': altitude must be a positive number")
    return args


def check_recording(run: SimulatorRun) -> tuple[Path, Path]:
    """The simulator run must be complete and contain video + calibration."""
    video = run.folder / "video.mp4"
    calibration = run.folder / "calibration.json"
    missing = [str(p) for p in (video, calibration, run.manifest) if not p.is_file()]
    if missing:
        raise BuildError(f"recording incomplete, missing: {missing}")
    status = json.loads(run.manifest.read_text(encoding="utf-8")).get("status")
    if status != "complete":
        raise BuildError(f"recording status is '{status}', not 'complete': {run.manifest}")
    return video, calibration


def keyframe_slots(db_path: Path) -> set[int]:
    import h5py

    with h5py.File(db_path, "r") as db:
        counts = db["local_features"]["kp_counts"][:]
    return {int(i) for i in (counts > 0).nonzero()[0]}


def check_anchor_contract(calibration_path: Path, db_path: Path) -> int:
    """Every anchor must sit on a keyframe the database actually stored."""
    anchors = json.loads(calibration_path.read_text(encoding="utf-8")).get("anchors") or []
    slots = {int(a["frame_id"]) for a in anchors}
    if not slots:
        raise BuildError(f"no anchors in {calibration_path}")
    missing = sorted(slots - keyframe_slots(db_path))
    if missing:
        raise BuildError(
            f"{len(missing)} anchor slot(s) are not database keyframes: {missing[:20]}. "
            "The recording and the database used different selector settings "
            "(check database.* in user_config.json against video.keyframes.json), or a "
            "borderline keyframe decision differed between the two runs; for a simulator "
            "recording --resnap-anchors moves such anchors onto the nearest keyframe."
        )
    return len(slots)


RESNAP_MAX_SHIFT_SLOTS = 3


def resnap_anchors(
    calibration_path: Path,
    db_path: Path,
    run: SimulatorRun,
    out_path: Path,
    max_shift: int = RESNAP_MAX_SHIFT_SLOTS,
) -> list[tuple[int, int | None]]:
    """Move simulator anchors that missed a database keyframe onto the nearest one.

    The simulator predicts keyframes with the database's own selector, but a
    borderline overlap decision can still differ between the two runs (different
    GPU numerics or package versions), shifting a few keyframes by a slot. A
    simulator anchor is ground truth for its slot, so the anchor is re-created
    from the ground truth of the nearest database keyframe within ``max_shift``
    slots that is not already an anchor; anchors without one are dropped. Writes
    the repaired calibration to ``out_path`` and returns (old slot, new slot or
    None) for every moved or dropped anchor.
    """
    import numpy as np

    from src.geometry.coordinates import CoordinateConverter

    data = json.loads(calibration_path.read_text(encoding="utf-8"))
    keyframes = keyframe_slots(db_path)
    truth = {int(s["slot"]): s for s in run.load_gt().get("slots", [])}
    projection = data.get("projection")
    converter = CoordinateConverter.from_metadata(projection) if projection else None
    taken = {int(a["frame_id"]) for a in data["anchors"] if int(a["frame_id"]) in keyframes}
    anchors, moves = [], []
    for anchor in data["anchors"]:
        slot = int(anchor["frame_id"])
        if slot in keyframes:
            anchors.append(anchor)
            continue
        candidates = sorted(
            (abs(k - slot), k)
            for k in range(slot - max_shift, slot + max_shift + 1)
            if k in keyframes and k not in taken and (truth.get(k) or {}).get("affine") is not None
        )
        if not candidates:
            moves.append((slot, None))
            continue
        new = candidates[0][1]
        taken.add(new)
        affine = np.asarray(truth[new]["affine"], dtype=float).reshape(2, 3)
        moved = json.loads(json.dumps(anchor))
        moved["frame_id"] = new
        moved["affine_matrix"] = affine.tolist()
        qa = moved.setdefault("qa_data", {})
        if truth[new].get("rmse_m") is not None:
            qa["rmse_m"] = float(truth[new]["rmse_m"])
        points = np.asarray(qa.get("points_2d") or [], dtype=float).reshape(-1, 2)
        if len(points):
            metric = points @ affine[:, :2].T + affine[:, 2]
            qa["points_metric"] = metric.tolist()
            if converter is not None:
                qa["points_gps"] = [
                    list(converter.metric_to_gps(float(x), float(y))) for x, y in metric
                ]
        note = f"re-snapped from slot {slot} to DB keyframe {new} (simulator ground truth)"
        qa["notes"] = f"{qa['notes']} | {note}" if qa.get("notes") else note
        anchors.append(moved)
        moves.append((slot, new))
    data["anchors"] = sorted(anchors, key=lambda a: int(a["frame_id"]))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    return moves


def is_propagated(db_path: Path) -> bool:
    import h5py

    with h5py.File(db_path, "r") as db:
        return "calibration" in db and "frame_affine" in db["calibration"]


def build_database(video: Path, db_path: Path, config: dict, manager) -> None:
    from src.database.database_builder import DatabaseBuilder

    db_path.parent.mkdir(parents=True, exist_ok=True)
    builder = DatabaseBuilder(output_path=str(db_path), config=config)
    last = [-1]

    def progress(percent: int) -> None:
        if percent // 10 != last[0]:
            last[0] = percent // 10
            print(f"    [{percent:3d}%] building {db_path.parent.name}", flush=True)

    builder.build_from_video(
        video_path=str(video),
        model_manager=manager,
        progress_callback=progress,
        save_keypoint_video=True,
    )


def propagate(db_path: Path, calibration_path: Path, config: dict, manager) -> None:
    from src.calibration.multi_anchor_calibration import MultiAnchorCalibration
    from src.database.database_loader import DatabaseLoader
    from src.localization.matcher import FeatureMatcher
    from src.workers.propagation_pipeline import PropagationPipeline

    database = DatabaseLoader(str(db_path))
    calibration = MultiAnchorCalibration()
    calibration.load(str(calibration_path))
    errors: list[str] = []
    completed = [False]
    pipeline = PropagationPipeline(
        database,
        calibration,
        FeatureMatcher(model_manager=manager, config=config),
        config=config,
        progress_callback=lambda p, m: print(f"    [{p:3d}%] {m}", flush=True)
        if p % 10 == 0
        else None,
        error_callback=errors.append,
        completed_callback=lambda: completed.__setitem__(0, True),
    )
    try:
        pipeline._run_propagation()
    finally:
        database.close()
    if errors or not completed[0]:
        raise BuildError(f"propagation failed: {errors or 'no completion signal'}")


def write_project(project: Path, name: str, layers: list[dict], area: str) -> Path:
    from src.core.project import ProjectSettings
    from src.core.project_video_source import ProjectVideoSource
    from src.utils.atomic_io import atomic_write_text

    first = layers[0]
    sources = [
        ProjectVideoSource(
            source_id=layer["id"],
            area_id=area,
            video_path=layer["video"],
            database_file=f"sources/{layer['id']}/database.h5",
            calibration_file=f"sources/{layer['id']}/calibration.json",
            description=f"{layer['altitude']:g} м",
            enabled=True,
            priority=index,
            camera_params={"altitude_m": layer["altitude"]},
        ).to_dict()
        for index, layer in enumerate(layers)
    ]
    settings = ProjectSettings(
        project_name=name,
        created_at=datetime.now().isoformat(),
        video_path=first["video"],
        database_filename=f"sources/{first['id']}/database.h5",
        calibration_filename=f"sources/{first['id']}/calibration.json",
        video_sources=sources,
        altitude_m=first["altitude"],
    )
    for sub in ("panoramas", "test_photos", "test_videos"):
        (project / sub).mkdir(exist_ok=True)
    path = project / "project.json"
    atomic_write_text(str(path), json.dumps(asdict(settings), indent=4, ensure_ascii=False))
    return path


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    project = args.project.resolve()
    if (project / "project.json").exists() and not args.resume:
        print(f"ERROR: {project / 'project.json'} exists (use --resume to continue it)")
        return 1

    from config import APP_CONFIG
    from scripts.propagation_sweep import apply_overrides
    from src.calibration.multi_anchor_calibration import MultiAnchorCalibration
    from src.calibration.multi_calibration_manager import save_layer_calibration
    from src.models.model_manager import ModelManager

    try:
        config = apply_overrides(APP_CONFIG, args.overrides)
    except ValueError as exc:
        print(f"ERROR: {exc}")
        return 1
    for name, value in args.overrides.items():
        print(f"override {name} = {value!r}")
    project.mkdir(parents=True, exist_ok=True)
    if args.pin_exact_anchors:
        config["graph_optimization"]["pin_exact_anchors"] = True
    manager = ModelManager(config=config)

    layers, summary = [], {}
    for sid, altitude, sim_dir in args.layer:
        run = SimulatorRun(Path(sim_dir).resolve())
        print(f"== layer {sid} ({altitude} m) from {run.folder}", flush=True)
        video, sim_calibration = check_recording(run)
        layer_dir = project / "sources" / sid
        db_path = layer_dir / "database.h5"
        cal_path = layer_dir / "calibration.json"

        if db_path.exists() and not args.resume:
            raise BuildError(f"{db_path} exists (use --resume)")
        if not db_path.exists():
            build_database(video, db_path, config, manager)
        try:
            n_anchors = check_anchor_contract(sim_calibration, db_path)
        except BuildError:
            if not (args.resnap_anchors and run.ground_truth.is_file()):
                raise
            repaired = layer_dir / "calibration_from_simulator_resnapped.json"
            moves = resnap_anchors(sim_calibration, db_path, run, repaired)
            dropped = [old for old, new in moves if new is None]
            print(
                f"   re-snapped {len(moves) - len(dropped)} anchor(s) onto DB keyframes "
                f"{[m for m in moves if m[1] is not None]}; dropped {dropped}",
                flush=True,
            )
            sim_calibration = repaired
            n_anchors = check_anchor_contract(sim_calibration, db_path)

        if not (args.resume and cal_path.exists()):
            calibration = MultiAnchorCalibration()
            calibration.load(str(sim_calibration))
            save_layer_calibration(calibration, cal_path, sid)
        if not (args.resume and is_propagated(db_path)):
            propagate(db_path, cal_path, config, manager)

        report = map_error_report(db_path, run) if run.ground_truth.is_file() else None
        summary[sid] = {"anchors": n_anchors, "map_error_vs_gt": report}
        print(json.dumps({sid: summary[sid]}, indent=2, ensure_ascii=False), flush=True)
        layers.append({"id": sid, "altitude": float(altitude), "video": str(video)})

    path = write_project(project, args.name or project.name, layers, args.area)
    print(f"Project written: {path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except BuildError as exc:
        print(f"ERROR: {exc}")
        raise SystemExit(1) from exc
