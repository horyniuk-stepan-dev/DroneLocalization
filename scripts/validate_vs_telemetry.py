"""
GT-валідатор пропагації проти істини симулятора (Етап 0.1).

Симулятор знає точну GT-позу кожного кадру. Цей інструмент бере результат
пропагації з `database.h5` (calibration/frame_affine + frame_valid) і числово
порівнює його з GT з двох джерел:

  1. `ground_truth.json` — точний ПО-СЛОТНИЙ експорт CalibrationLogger (Етап 0.2):
     GT-афінна кожного слота (px→Mercator) + heading + позначки якорів. Головне
     джерело: дає похибку центру, кута ТА масштабу без часової інтерполяції.
  2. `telemetry.csv` — щільна GT-позиція дрона (локальні метри). Вторинна звірка
     позиції: локальні → Mercator через `map_center`, лінійна інтерполяція на слот.

Головна цінність — РОЗРІЗИ: медіана/p95/max похибки окремо на
    прямих / дугах розворотів / ±k слотів від якоря / серединах прольотів,
бо саме вони показують, ЩО дає похибку.

Чисті функції метрик/розрізів (без h5py/pyproj) винесені й покриті
tests/test_validate_vs_telemetry.py. Це read-only інструмент (жодного гейта).

Запуск:
  python scripts/validate_vs_telemetry.py --db path/to/database.h5 \
      --ground-truth path/to/ground_truth.json
  python scripts/validate_vs_telemetry.py --db database.h5 \
      --telemetry telemetry.csv --map-center 2903967.4,6171063.1 --fps 30
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

# ── Чисті функції (тестуються без h5py/pyproj) ───────────────────────────────


def frame_center(affine: np.ndarray, cx: float, cy: float) -> np.ndarray:
    """Метричний центр кадру: affine застосована до пікселя (cx, cy)."""
    p = np.array([cx, cy], dtype=np.float64)
    return affine[:2, :2] @ p + affine[:2, 2]


def iso_scale(affine: np.ndarray) -> float:
    """Ізотропний масштаб-проксі = √|det| (геом. середнє sx, sy)."""
    return float(np.sqrt(abs(np.linalg.det(affine[:2, :2]))))


def affine_angle_deg(affine: np.ndarray) -> float:
    """Кут повороту матриці (градуси), atan2(M[1,0], M[0,0])."""
    return float(np.degrees(np.arctan2(affine[1, 0], affine[0, 0])))


def _wrap_deg(d: np.ndarray) -> np.ndarray:
    """Приводить різницю кутів до (−180, 180]."""
    return (np.asarray(d, dtype=np.float64) + 180.0) % 360.0 - 180.0


def summarize(errs: np.ndarray) -> dict:
    """{n, median, p95, max, mean} для масиву похибок (порожній → нулі)."""
    e = np.asarray(errs, dtype=np.float64)
    e = e[~np.isnan(e)]
    if e.size == 0:
        return {"n": 0, "median": 0.0, "p95": 0.0, "max": 0.0, "mean": 0.0}
    return {
        "n": int(e.size),
        "median": float(np.median(e)),
        "p95": float(np.percentile(e, 95)),
        "max": float(np.max(e)),
        "mean": float(np.mean(e)),
    }


def interp_telemetry(
    times: np.ndarray, xs: np.ndarray, ys: np.ndarray, slot_times: np.ndarray
) -> np.ndarray:
    """Лінійна інтерполяція локальної GT-позиції на моменти слотів.

    Повертає (N, 2). Поза діапазоном телеметрії — NaN (щоб не екстраполювати
    хибно). times мусить бути відсортований.
    """
    st = np.asarray(slot_times, dtype=np.float64)
    out = np.full((st.size, 2), np.nan, dtype=np.float64)
    in_range = (st >= times[0]) & (st <= times[-1])
    out[in_range, 0] = np.interp(st[in_range], times, xs)
    out[in_range, 1] = np.interp(st[in_range], times, ys)
    return out


def classify_slots(
    slots: np.ndarray,
    headings_deg: np.ndarray | None,
    anchor_slots: set[int],
    turn_rate_deg_per_slot: float = 3.0,
    near_anchor_k: int = 3,
) -> dict[str, np.ndarray]:
    """Булеві маски розрізів по слотах (розрізи МОЖУТЬ перетинатися).

    - straights   : |Δheading|/слот  <  turn_rate (або heading невідомий)
    - turn_arcs   : |Δheading|/слот  >= turn_rate
    - near_anchor : відстань до найближчого якоря <= near_anchor_k
    - mid_leg     : відстань до найближчого якоря  >  near_anchor_k
    """
    slots = np.asarray(slots)
    n = slots.size

    # Розворот: швидкість зміни heading між сусідніми слотами (за slot-різницею).
    turning = np.zeros(n, dtype=bool)
    if headings_deg is not None and n >= 2:
        h = np.unwrap(np.radians(np.asarray(headings_deg, dtype=np.float64)))
        h_deg = np.degrees(h)
        gaps = np.maximum(np.diff(slots.astype(np.float64)), 1.0)
        rate = np.abs(np.diff(h_deg)) / gaps  # °/слот між сусідами
        turn_pt = rate >= float(turn_rate_deg_per_slot)
        # Слот вважається дугою, якщо суміжний інтервал (ліворуч чи праворуч) крутиться.
        turning[:-1] |= turn_pt
        turning[1:] |= turn_pt

    if anchor_slots:
        a = np.array(sorted(anchor_slots), dtype=np.float64)
        dist = np.abs(slots.astype(np.float64)[:, None] - a[None, :]).min(axis=1)
    else:
        dist = np.full(n, np.inf)
    near = dist <= near_anchor_k

    return {
        "straights": ~turning,
        "turn_arcs": turning,
        "near_anchor": near,
        "mid_leg": ~near,
    }


def compute_report(
    pred: dict[int, np.ndarray],
    gt_affine: dict[int, np.ndarray],
    frame_w: int,
    frame_h: int,
    headings_deg: dict[int, float] | None = None,
    anchor_slots: set[int] | None = None,
    turn_rate_deg_per_slot: float = 3.0,
    near_anchor_k: int = 3,
    telemetry_centers: dict[int, np.ndarray] | None = None,
) -> dict:
    """Повний звіт: похибка центру/кута/масштабу загалом і в розрізах.

    pred / gt_affine: slot → 2x3 афінна (px→Mercator).
    telemetry_centers: slot → 2-вектор GT-центру в Mercator (незалежна звірка).
    """
    cx, cy = frame_w / 2.0, frame_h / 2.0
    anchor_slots = anchor_slots or set()

    common = sorted(s for s in gt_affine if s in pred)
    if not common:
        raise ValueError("Жоден слот пропагації не збігся з GT — перевірте frame_step/слоти.")

    slots = np.array(common, dtype=np.int64)
    err = np.empty(slots.size, dtype=np.float64)
    ang = np.empty(slots.size, dtype=np.float64)
    sca = np.empty(slots.size, dtype=np.float64)
    for k, s in enumerate(common):
        p, g = pred[s], gt_affine[s]
        err[k] = float(np.linalg.norm(frame_center(p, cx, cy) - frame_center(g, cx, cy)))
        ang[k] = abs(float(_wrap_deg(affine_angle_deg(p) - affine_angle_deg(g))))
        sg = iso_scale(g)
        sca[k] = abs(iso_scale(p) - sg) / sg if sg > 0 else np.nan

    head_arr = (
        np.array([headings_deg.get(int(s), np.nan) for s in slots], dtype=np.float64)
        if headings_deg
        else None
    )
    if head_arr is not None and np.all(np.isnan(head_arr)):
        head_arr = None
    masks = classify_slots(
        slots, head_arr, anchor_slots, turn_rate_deg_per_slot, near_anchor_k
    )

    def cut(mask: np.ndarray) -> dict:
        d = summarize(err[mask])
        d["angle_deg"] = summarize(ang[mask])["median"]
        d["scale_rel"] = summarize(sca[mask])["median"]
        return d

    report = {
        "n_slots_compared": int(slots.size),
        "n_gt_slots": int(len(gt_affine)),
        "coverage": float(slots.size / max(len(gt_affine), 1)),
        "overall": {
            **summarize(err),
            "angle_deg_median": summarize(ang)["median"],
            "angle_deg_max": summarize(ang)["max"],
            "scale_rel_median": summarize(sca)["median"],
        },
        "cuts": {name: cut(m) for name, m in masks.items()},
    }

    # Незалежна телеметрична звірка позиції (за наявності).
    if telemetry_centers:
        tel_err = []
        for k, s in enumerate(common):
            c = telemetry_centers.get(int(s))
            if c is None or np.any(np.isnan(c)):
                continue
            p_center = frame_center(pred[s], cx, cy)
            tel_err.append(float(np.linalg.norm(p_center - np.asarray(c, dtype=np.float64))))
        if tel_err:
            report["telemetry_cross_check"] = summarize(np.array(tel_err))

    return report


def format_report(rep: dict, scene: str = "") -> str:
    """Короткий текстовий звіт для лога/CI."""
    lines = [f"GT-валідація пропагації{(' — ' + scene) if scene else ''}"]
    o = rep["overall"]
    lines.append(
        f"  слотів звірено: {rep['n_slots_compared']}/{rep['n_gt_slots']} "
        f"(покриття {rep['coverage']:.0%})"
    )
    lines.append(
        f"  ЗАГАЛОМ: median={o['median']:.2f} м, p95={o['p95']:.2f} м, "
        f"max={o['max']:.2f} м | кут med={o['angle_deg_median']:.2f}° "
        f"(max {o['angle_deg_max']:.2f}°), масштаб med={o['scale_rel_median']:.3%}"
    )
    lines.append("  Розрізи (median / p95 / max, м; кут°; масштаб%):")
    for name in ("straights", "turn_arcs", "near_anchor", "mid_leg"):
        c = rep["cuts"].get(name, {})
        if not c or c.get("n", 0) == 0:
            lines.append(f"    {name:<12}: —")
            continue
        lines.append(
            f"    {name:<12}: {c['median']:6.2f} / {c['p95']:6.2f} / {c['max']:6.2f}"
            f"  (n={c['n']}, кут {c['angle_deg']:.2f}°, масшт {c['scale_rel']:.3%})"
        )
    if "telemetry_cross_check" in rep:
        t = rep["telemetry_cross_check"]
        lines.append(
            f"  Телеметрична звірка позиції: median={t['median']:.2f} м, "
            f"p95={t['p95']:.2f} м, max={t['max']:.2f} м (n={t['n']})"
        )
    return "\n".join(lines)


# ── I/O (h5py / json / csv) ──────────────────────────────────────────────────


def read_pred_from_hdf5(db_path: str) -> tuple[dict[int, np.ndarray], int, int]:
    """Читає (frame_affine, frame_valid) + розмір кадру з HDF5."""
    import h5py

    with h5py.File(db_path, "r") as f:
        grp = f["calibration"]
        fa = grp["frame_affine"][:]
        fv = grp["frame_valid"][:].astype(bool)
        meta = f["metadata"].attrs if "metadata" in f else {}
        fw = int(meta.get("frame_width", meta.get("width", 1920)))
        fh = int(meta.get("frame_height", meta.get("height", 1080)))
    pred = {i: fa[i] for i in range(len(fa)) if fv[i]}
    return pred, fw, fh


def load_ground_truth(path: str) -> dict:
    """Читає ground_truth.json (Етап 0.2) у зручні словники."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    gt_affine: dict[int, np.ndarray] = {}
    headings: dict[int, float] = {}
    anchor_slots: set[int] = set()
    for row in data["slots"]:
        s = int(row["slot"])
        gt_affine[s] = np.array(row["affine"], dtype=np.float64)
        if row.get("heading_deg") is not None:
            headings[s] = float(row["heading_deg"])
        if row.get("is_anchor"):
            anchor_slots.add(s)
    fs = data.get("frame_size") or [1920, 1080]
    return {
        "gt_affine": gt_affine,
        "headings": headings,
        "anchor_slots": anchor_slots,
        "frame_size": (int(fs[0]), int(fs[1])),
        "frame_step": int(data.get("frame_step", 30)),
        "fps": float(data.get("fps", 30.0)),
        "map_center": data.get("map_center"),
    }


def load_telemetry(path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Читає telemetry.csv → (sim_time, pos_x, pos_y), відсортовані за часом."""
    t, x, y = [], [], []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            t.append(float(row["sim_time"]))
            x.append(float(row["pos_x"]))
            y.append(float(row["pos_y"]))
    t = np.array(t)
    order = np.argsort(t)
    return t[order], np.array(x)[order], np.array(y)[order]


def telemetry_slot_centers(
    tel_path: str,
    slots: list[int],
    map_center: tuple[float, float],
    frame_step: int,
    fps: float,
) -> dict[int, np.ndarray]:
    """GT-центр кадру в Mercator на кожен слот з телеметрії.

    slot → video_frame = slot*frame_step → t = video_frame/fps → interp позиції.
    Mercator = local + map_center (див. OrthophotoMap.local_to_metric).
    """
    times, xs, ys = load_telemetry(tel_path)
    slot_arr = np.array(sorted(set(int(s) for s in slots)), dtype=np.int64)
    slot_times = slot_arr.astype(np.float64) * frame_step / fps
    local = interp_telemetry(times, xs, ys, slot_times)
    cx0, cy0 = float(map_center[0]), float(map_center[1])
    out: dict[int, np.ndarray] = {}
    for k, s in enumerate(slot_arr):
        out[int(s)] = np.array([local[k, 0] + cx0, local[k, 1] + cy0], dtype=np.float64)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="GT-валідатор пропагації проти симулятора")
    ap.add_argument("--db", required=True, help="database.h5 після пропагації")
    ap.add_argument("--ground-truth", default=None, help="ground_truth.json (Етап 0.2, точний)")
    ap.add_argument("--telemetry", default=None, help="telemetry.csv (щільна позиція)")
    ap.add_argument("--map-center", default=None,
                    help="cx,cy Mercator-центру мапи для телеметрії (якщо нема GT-json)")
    ap.add_argument("--frame-step", type=int, default=None, help="DB frame_step (для телеметрії)")
    ap.add_argument("--fps", type=float, default=None, help="target_fps відео (для телеметрії)")
    ap.add_argument("--turn-rate-deg", type=float, default=3.0, help="поріг дуги (°/слот)")
    ap.add_argument("--near-anchor-k", type=int, default=3, help="радіус розрізу near_anchor")
    ap.add_argument("--out", default=None, help="куди писати JSON-звіт (дефолт: поряд із db)")
    args = ap.parse_args(argv)

    pred, fw, fh = read_pred_from_hdf5(args.db)
    if not pred:
        print("У database.h5 немає валідних кадрів пропагації.", file=sys.stderr)
        return 2

    if not args.ground_truth and not args.telemetry:
        print("Потрібне хоча б одне джерело GT: --ground-truth або --telemetry.",
              file=sys.stderr)
        return 2

    gt_affine: dict[int, np.ndarray] = {}
    headings: dict[int, float] | None = None
    anchor_slots: set[int] = set()
    frame_step = args.frame_step or 30
    fps = args.fps or 30.0
    map_center = None
    frame_w, frame_h = fw, fh

    if args.ground_truth:
        g = load_ground_truth(args.ground_truth)
        gt_affine = g["gt_affine"]
        headings = g["headings"] or None
        anchor_slots = g["anchor_slots"]
        frame_w, frame_h = g["frame_size"]
        frame_step = args.frame_step or g["frame_step"]
        fps = args.fps or g["fps"]
        map_center = g["map_center"]

    if args.map_center:
        map_center = [float(v) for v in args.map_center.split(",")]

    telemetry_centers = None
    if args.telemetry and map_center is not None:
        target_slots = list(gt_affine.keys()) if gt_affine else list(pred.keys())
        telemetry_centers = telemetry_slot_centers(
            args.telemetry, target_slots, map_center, frame_step, fps
        )
    elif args.telemetry and map_center is None:
        print("  (телеметрію задано без --map-center/ground_truth → пропущено звірку позиції)",
              file=sys.stderr)

    # Якщо GT-json немає — будуємо «GT-афінну» з телеметрії неможливо (нема кута/масштабу).
    # Тоді звіряємося лише позиційно: використовуємо предикт як gt-плейсхолдер для розрізів
    # немає сенсу, тож вимагаємо GT-json для повного звіту.
    if not gt_affine:
        if telemetry_centers is None:
            print("Без --ground-truth і без map-center немає з чим порівнювати.", file=sys.stderr)
            return 2
        # Позиційний-лише режим: рахуємо похибку центрів пред↔телеметрія.
        errs = []
        cxp, cyp = frame_w / 2.0, frame_h / 2.0
        for s, aff in pred.items():
            c = telemetry_centers.get(int(s))
            if c is None or np.any(np.isnan(c)):
                continue
            errs.append(float(np.linalg.norm(frame_center(aff, cxp, cyp) - c)))
        rep = {"telemetry_only": summarize(np.array(errs))}
        print(json.dumps(rep, ensure_ascii=False, indent=2))
    else:
        rep = compute_report(
            pred, gt_affine, frame_w, frame_h,
            headings_deg=headings, anchor_slots=anchor_slots,
            turn_rate_deg_per_slot=args.turn_rate_deg, near_anchor_k=args.near_anchor_k,
            telemetry_centers=telemetry_centers,
        )
        print(format_report(rep, scene=Path(args.db).stem))

    out_path = Path(args.out) if args.out else Path(args.db).with_suffix(".gt_validation.json")
    out_path.write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nЗвіт збережено: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
