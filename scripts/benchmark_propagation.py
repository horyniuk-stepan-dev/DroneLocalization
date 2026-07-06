"""
End-to-end бенчмарк графової пропагації на даних із відомою істиною (Етап 0.1).

Ідея: FlightSimulator знає точний GPS/affine кожного пікселя кожного кадру →
ідеальний генератор еталонів. Бенчмарк бере відео + ground_truth.json, будує БД,
ставить якорі з GT ПРОГРАМНО (без кліків), запускає пропагацію та рахує метрики
проти GT. Гейт: median не гірше базлайну +10%, max +20%.

Схема ground_truth.json:
{
  "scene": "straight_run",
  "frame_width": 1920, "frame_height": 1080,
  "projection": "UTM",
  "anchors": [0, 150, 300],                 # frame_id, що беруться як GT-якорі
  "frames": [
    {"frame_id": 0,   "affine": [[..],[..]], "gps": [lat, lon]},
    ...
  ]
}

Три еталонні сцени (окремі теки): пряма траса; маршрут із поверненням (реальні
loop closures); hover-сегмент (порожні keyframe-гепи).

Запуск:
  python scripts/benchmark_propagation.py --dataset datasets/straight_run
  python scripts/benchmark_propagation.py --dataset datasets/loop --update-baseline

Пер-PR-правило: ганяємо на 3 сценах, метрики в опис зміни; регресія > гейта =
не мержимо (див. .agents/POSE_GRAPH_IMPROVEMENT_PLAN.md).

Примітка щодо середовища: цей скрипт потребує GPU/моделі/відео та повний пайплайн
(faiss, h5py, PyQt6). Чисті функції метрик/гейта (compute_metrics, check_gate)
винесені окремо й покриті tests/test_benchmark_metrics.py.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

# Пороги гейта (плану): median +10%, max +20%.
MEDIAN_TOL = 0.10
MAX_TOL = 0.20

BASELINE_PATH = Path("benchmarks/propagation_baseline.json")


# ── Чисті функції (тестовані без GPU) ────────────────────────────────────────

def _center(affine: np.ndarray, cx: float, cy: float) -> np.ndarray:
    p = np.array([cx, cy], dtype=np.float64)
    return affine[:2, :2] @ p + affine[:2, 2]


def _iso_scale(affine: np.ndarray) -> float:
    """Ізотропний масштаб-проксі = √|det| = геом. середнє sx,sy."""
    return float(np.sqrt(abs(np.linalg.det(affine[:2, :2]))))


def compute_metrics(
    pred: dict[int, np.ndarray],
    gt: dict[int, np.ndarray],
    frame_w: int,
    frame_h: int,
) -> dict:
    """Метрики пропагації проти GT.

    pred/gt: frame_id → 2x3 афінна матриця (px→metric). Метри = метричний простір
    проєкції (UTM). Похибка = зсув центру кадру vs GT.
    """
    cx, cy = frame_w / 2.0, frame_h / 2.0
    errs: list[float] = []
    scale_dev: list[float] = []
    det_ok = 0
    n = 0
    for fid, g in gt.items():
        p = pred.get(fid)
        if p is None:
            continue
        n += 1
        errs.append(float(np.linalg.norm(_center(p, cx, cy) - _center(g, cx, cy))))
        if np.sign(np.linalg.det(p[:2, :2])) == np.sign(np.linalg.det(g[:2, :2])):
            det_ok += 1
        s_p, s_g = _iso_scale(p), _iso_scale(g)
        if s_g > 0:
            scale_dev.append(abs(s_p - s_g) / s_g)

    if n == 0:
        raise ValueError("Жоден передбачений кадр не збігся з GT — перевірте frame_id.")

    errs_a = np.array(errs)
    return {
        "n_frames": int(n),
        "coverage": float(n / max(len(gt), 1)),
        "median_err_m": float(np.median(errs_a)),
        "p95_err_m": float(np.percentile(errs_a, 95)),
        "max_err_m": float(np.max(errs_a)),
        "det_sign_ok": float(det_ok / n),
        "scale_drift": float(np.median(scale_dev)) if scale_dev else 0.0,
    }


def check_gate(
    metrics: dict,
    baseline: dict | None,
    median_tol: float = MEDIAN_TOL,
    max_tol: float = MAX_TOL,
) -> tuple[bool, str]:
    """Порівнює метрики з базлайном. Повертає (passed, human-readable details)."""
    if baseline is None:
        return True, "базлайн відсутній (перший прогін) — записуємо як еталон"

    med_limit = baseline["median_err_m"] * (1 + median_tol)
    max_limit = baseline["max_err_m"] * (1 + max_tol)
    med_ok = metrics["median_err_m"] <= med_limit + 1e-9
    max_ok = metrics["max_err_m"] <= max_limit + 1e-9

    lines = [
        f"median: {metrics['median_err_m']:.3f} м  (ліміт {med_limit:.3f} = "
        f"базлайн {baseline['median_err_m']:.3f} +{median_tol:.0%})  "
        f"{'OK' if med_ok else 'РЕГРЕСІЯ'}",
        f"max:    {metrics['max_err_m']:.3f} м  (ліміт {max_limit:.3f} = "
        f"базлайн {baseline['max_err_m']:.3f} +{max_tol:.0%})  "
        f"{'OK' if max_ok else 'РЕГРЕСІЯ'}",
    ]
    return (med_ok and max_ok), "\n".join(lines)


def load_ground_truth(path: Path) -> dict:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    gt_affines = {
        int(fr["frame_id"]): np.array(fr["affine"], dtype=np.float64)
        for fr in data["frames"]
    }
    return {
        "scene": data.get("scene", Path(path).parent.name),
        "frame_width": int(data.get("frame_width", 1920)),
        "frame_height": int(data.get("frame_height", 1080)),
        "anchors": [int(a) for a in data.get("anchors", [])],
        "gt_affines": gt_affines,
        "raw": data,
    }


def load_baseline(scene: str, path: Path = BASELINE_PATH) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8")).get(scene)


def save_baseline(scene: str, metrics: dict, path: Path = BASELINE_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    allb = {}
    if path.exists():
        allb = json.loads(path.read_text(encoding="utf-8"))
    allb[scene] = metrics
    path.write_text(json.dumps(allb, indent=2, ensure_ascii=False), encoding="utf-8")


# ── Оркестрація повного пайплайну (потребує GPU/моделей/відео) ────────────────

def read_pred_affines_from_hdf5(db_path: str) -> dict[int, np.ndarray]:
    """Читає результати пропагації (frame_affine, frame_valid) з HDF5."""
    import h5py

    with h5py.File(db_path, "r") as f:
        grp = f["calibration"]
        fa = grp["frame_affine"][:]
        fv = grp["frame_valid"][:].astype(bool)
    return {i: fa[i] for i in range(len(fa)) if fv[i]}


def run_pipeline(dataset_dir: Path, gt: dict) -> dict[int, np.ndarray]:
    """Будує БД з відео, ставить GT-якорі програмно, запускає пропагацію.

    Використовує реальні класи проєкту. Викидає зрозумілу помилку, якщо чогось
    бракує (модель/відео) — бенчмарк не має «тихо» деградувати.
    """
    from config.config import APP_CONFIG
    from src.calibration.multi_anchor_calibration import MultiAnchorCalibration
    from src.database.database_builder import DatabaseBuilder  # noqa: F401  (реальний білдер)
    from src.database.database_loader import DatabaseLoader
    from src.geometry.coordinates import CoordinateConverter
    from src.localization.matcher import Matcher  # noqa: F401
    from src.workers.calibration_propagation_worker import CalibrationPropagationWorker

    video = next((p for p in dataset_dir.glob("*.mp4")), None)
    if video is None:
        raise FileNotFoundError(f"У {dataset_dir} немає *.mp4 для збудови БД")

    db_path = dataset_dir / "benchmark.h5"
    # 1) Збудова БД (реальний білдер проєкту). Точний виклик залежить від API
    #    DatabaseBuilder; тут очікується, що БД уже збудована або білдер її створить.
    if not db_path.exists():
        raise FileNotFoundError(
            f"БД {db_path} не знайдена. Збудуйте її білдером проєкту для {video.name} "
            f"(окремий крок: потребує GPU/моделей)."
        )

    database = DatabaseLoader(str(db_path))
    database.load()

    # 2) GT-якорі ПРОГРАМНО (без кліків)
    converter = CoordinateConverter(gt["raw"].get("projection", "UTM"))
    calibration = MultiAnchorCalibration(converter=converter)
    for aid in gt["anchors"]:
        calibration.add_anchor(aid, gt["gt_affines"][aid])

    # 3) Пропагація (синхронно; сигнали → прінти)
    matcher = Matcher(APP_CONFIG)
    worker = CalibrationPropagationWorker(database, calibration, matcher, config=APP_CONFIG)
    worker.progress.connect(lambda pct, msg: print(f"  [{pct:3d}%] {msg}"))
    worker.error.connect(lambda msg: print(f"  ERROR: {msg}", file=sys.stderr))
    worker._propagate()

    return read_pred_affines_from_hdf5(str(db_path))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Бенчмарк графової пропагації проти GT")
    ap.add_argument("--dataset", required=True, type=Path,
                    help="тека з відео + ground_truth.json")
    ap.add_argument("--update-baseline", action="store_true",
                    help="перезаписати базлайн цією сценою")
    ap.add_argument("--pred-hdf5", type=Path, default=None,
                    help="(опційно) готовий HDF5 з пропагацією — оцінка без збудови/прогону")
    args = ap.parse_args(argv)

    gt_path = args.dataset / "ground_truth.json"
    gt = load_ground_truth(gt_path)
    scene = gt["scene"]
    print(f"Сцена: {scene}  (кадрів у GT: {len(gt['gt_affines'])}, якорів: {len(gt['anchors'])})")

    if args.pred_hdf5:
        pred = read_pred_affines_from_hdf5(str(args.pred_hdf5))
    else:
        pred = run_pipeline(args.dataset, gt)

    metrics = compute_metrics(pred, gt["gt_affines"], gt["frame_width"], gt["frame_height"])
    print("\nМетрики:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    baseline = load_baseline(scene)
    passed, details = check_gate(metrics, baseline)
    print("\nГейт:\n" + details)

    if args.update_baseline or baseline is None:
        save_baseline(scene, metrics)
        print(f"\nБазлайн для сцени '{scene}' записано у {BASELINE_PATH}")

    if not passed:
        print("\nРЕГРЕСІЯ > гейта — НЕ мержимо.", file=sys.stderr)
        return 1
    print("\nГейт пройдено.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
