"""Acceptance-тест scale-інваріантності (IMPLEMENTATION_PLAN, Фаза 1, п. 4).

Синтетичні GSD-ратіо r ∈ {0.5, 0.7, 1.4, 2.0} над кадрами еталонного відео →
``localize_frame`` → медіанна помилка ≤ 2× базової (r = 1.0); окремо — що
FOV-полігон масштабується ~r (ловить старий баг, де полігон не залежав від r).

GPU-частина запускається ЛИШЕ на Windows-машині з моделями і готовим проєктом:

    set DRONELOC_ACCEPT_DIR=D:\\шлях\\до\\проєкту     (database.h5 + calibration.json
                                                     [+ ground_truth.json від симулятора])
    set DRONELOC_ACCEPT_VIDEO=D:\\шлях\\до\\flight.mp4
    python -m pytest tests/test_scale_invariance.py -q -s

Опційно: DRONELOC_ACCEPT_FRAMES (=12), DRONELOC_ACCEPT_TOL_M (=15, гейт без GT),
DRONELOC_ACCEPT_MIN_SUCCESS (=0.5).

Без цих змінних (чи без torch) GPU-тести skip'аються; чисті synth-хелпери
тестуються будь-де. База має бути збудована з ТОГО Ж відео (той самий рейс).
"""

from __future__ import annotations

import json
import math
import os
import statistics

import cv2
import numpy as np
import pytest

SCALES = (0.5, 0.7, 1.4, 2.0)
WARMUP = 3  # кадри на прогрів rotation/scale-prior — у метрику не йдуть


# ── чисті synth-хелпери (тестуються без GPU) ─────────────────────────────────


def synth_gsd(frame: np.ndarray, r: float) -> np.ndarray:
    """Імітує політ на висоті r × db_altitude над тим самим центром.

    r < 1 (нижче): центр-кроп частки r + upscale до вихідних розмірів —
        менша площа, дрібніший GSD.
    r > 1 (вище): downscale у 1/r + чорні поля до вихідних розмірів —
        DB-покриття займає центральну 1/r кадру, периферія невідома.
    Розміри кадру завжди зберігаються (ResolutionNormalizer їх не чіпає).
    """
    h, w = frame.shape[:2]
    if abs(r - 1.0) < 1e-9:
        return frame.copy()
    if r < 1.0:
        cw, ch = max(32, int(w * r)), max(32, int(h * r))
        x1, y1 = (w - cw) // 2, (h - ch) // 2
        crop = frame[y1 : y1 + ch, x1 : x1 + cw]
        return cv2.resize(crop, (w, h), interpolation=cv2.INTER_CUBIC)
    nw, nh = max(32, int(round(w / r))), max(32, int(round(h / r)))
    small = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_AREA)
    out = np.zeros_like(frame)
    x1, y1 = (w - nw) // 2, (h - nh) // 2
    out[y1 : y1 + nh, x1 : x1 + nw] = small
    return out


def geo_dist_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Гаверсин, метри."""
    rad = math.radians
    dlat, dlon = rad(lat2 - lat1), rad(lon2 - lon1)
    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(rad(lat1)) * math.cos(rad(lat2)) * math.sin(dlon / 2) ** 2
    )
    return 6371000.0 * 2 * math.asin(min(1.0, math.sqrt(a)))


def poly_diameter_m(poly: list) -> float | None:
    """Максимальна попарна геовідстань між кутами полігона."""
    if not poly or len(poly) < 3:
        return None
    best = 0.0
    for i in range(len(poly)):
        for j in range(i + 1, len(poly)):
            best = max(
                best, geo_dist_m(poly[i][0], poly[i][1], poly[j][0], poly[j][1])
            )
    return best


def test_synth_gsd_low_altitude_is_center_crop():
    rng = np.random.default_rng(3)
    frame = rng.integers(0, 255, (360, 640, 3), dtype=np.uint8)
    out = synth_gsd(frame, 0.5)
    assert out.shape == frame.shape
    cw, ch = int(640 * 0.5), int(360 * 0.5)
    x1, y1 = (640 - cw) // 2, (360 - ch) // 2
    ref = cv2.resize(
        frame[y1 : y1 + ch, x1 : x1 + cw], (640, 360), interpolation=cv2.INTER_CUBIC
    )
    np.testing.assert_array_equal(out, ref)


def test_synth_gsd_high_altitude_pads_borders():
    rng = np.random.default_rng(4)
    frame = rng.integers(1, 255, (360, 640, 3), dtype=np.uint8)  # без нулів
    out = synth_gsd(frame, 2.0)
    assert out.shape == frame.shape
    nw, nh = 320, 180
    x1, y1 = (640 - nw) // 2, (360 - nh) // 2
    assert out[: y1 - 1].max() == 0 and out[y1 + nh + 1 :].max() == 0
    assert out[y1 : y1 + nh, x1 : x1 + nw].mean() > 10  # центр — реальний контент


def test_synth_gsd_identity():
    frame = np.full((64, 64, 3), 7, dtype=np.uint8)
    np.testing.assert_array_equal(synth_gsd(frame, 1.0), frame)


# ── GPU-acceptance (Windows) ─────────────────────────────────────────────────


def _env():
    d = os.environ.get("DRONELOC_ACCEPT_DIR")
    v = os.environ.get("DRONELOC_ACCEPT_VIDEO")
    if not d or not v:
        pytest.skip(
            "set DRONELOC_ACCEPT_DIR + DRONELOC_ACCEPT_VIDEO (Windows acceptance run)"
        )
    if not os.path.isfile(os.path.join(d, "database.h5")):
        pytest.skip(f"no database.h5 in {d}")
    if not os.path.isfile(v):
        pytest.skip(f"video not found: {v}")
    pytest.importorskip("torch")
    return d, v


@pytest.fixture(scope="session")
def stack():
    """Моделі + БД + калібрування (один раз на сесію) — патерн HeadlessRunner."""
    proj_dir, video = _env()

    from config import APP_CONFIG
    from src.calibration.multi_anchor_calibration import MultiAnchorCalibration
    from src.database.database_loader import DatabaseLoader
    from src.localization.matcher import FeatureMatcher
    from src.models.model_manager import ModelManager
    from src.models.wrappers.feature_extractor import FeatureExtractor

    mm = ModelManager(config=APP_CONFIG)
    fe = FeatureExtractor(
        mm.load_local_extractor(), mm.load_dinov2(), mm.device, config=APP_CONFIG
    )
    matcher = FeatureMatcher(model_manager=mm, config=APP_CONFIG)

    db = DatabaseLoader(os.path.join(proj_dir, "database.h5"))
    calib = MultiAnchorCalibration()
    calib_path = os.path.join(proj_dir, "calibration.json")
    if os.path.isfile(calib_path):
        calib.load(calib_path)
    if not calib.converter.is_initialized:
        ref_gps = calib.converter.reference_gps
        if calib.is_calibrated and ref_gps:
            calib.converter.gps_to_metric(ref_gps[0], ref_gps[1])
        else:
            pytest.skip("calibration missing / converter uninitialized")

    gt = {}
    gt_path = os.path.join(proj_dir, "ground_truth.json")
    if os.path.isfile(gt_path):
        with open(gt_path, encoding="utf-8") as f:
            data = json.load(f)
        for slot in data.get("slots", []):
            mx, my = slot["center_mercator"]
            gt[int(slot["video_frame"])] = calib.converter.metric_to_gps(mx, my)

    return {
        "config": {**APP_CONFIG, "_model_manager": mm},
        "db": db,
        "fe": fe,
        "matcher": matcher,
        "calib": calib,
        "video": video,
        "gt": gt,
    }


def _grab_frames(video: str, wanted: list[int]) -> dict[int, np.ndarray]:
    cap = cv2.VideoCapture(video)
    frames = {}
    try:
        for idx in wanted:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame = cap.read()
            if ok and frame is not None:
                frames[idx] = frame
    finally:
        cap.release()
    return frames


@pytest.fixture(scope="session")
def frames(stack):
    n = int(os.environ.get("DRONELOC_ACCEPT_FRAMES", "12"))
    cap = cv2.VideoCapture(stack["video"])
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    if total <= 0:
        pytest.skip("video has no frames")

    if stack["gt"]:  # семплюємо тільки кадри, для яких є GT-слоти
        gt_frames = sorted(f for f in stack["gt"] if 0 <= f < total)
        if len(gt_frames) < WARMUP + 4:
            pytest.skip(f"too few GT slots in video range ({len(gt_frames)})")
        step = max(1, len(gt_frames) // n)
        wanted = gt_frames[::step][:n]
    else:
        lo, hi = int(total * 0.05), int(total * 0.95)
        wanted = list(np.linspace(lo, hi, n, dtype=int))

    got = _grab_frames(stack["video"], wanted)
    if len(got) < WARMUP + 4:
        pytest.skip(f"decoded only {len(got)} frames")
    return got


def _run(stack, frames_map, r: float) -> list[dict]:
    """Свіжий Localizer → послідовна локалізація кадрів із synth-GSD r."""
    from src.localization.localizer import Localizer

    db = stack["db"]
    loc = Localizer(
        db,
        stack["fe"],
        stack["matcher"],
        stack["calib"],
        config=stack["config"],
        ref_frame_width=int(db.metadata.get("frame_width", 0)),
        ref_frame_height=int(db.metadata.get("frame_height", 0)),
    )
    out = []
    for idx in sorted(frames_map):
        res = loc.localize_frame(synth_gsd(frames_map[idx], r), dt=1.0)
        ok = bool(res.get("success")) and not res.get("fallback_mode")
        out.append(
            {
                "frame": idx,
                "ok": ok,
                "lat": res.get("lat"),
                "lon": res.get("lon"),
                "poly": res.get("fov_polygon"),
            }
        )
    return out


@pytest.fixture(scope="session")
def runs(stack, frames):
    return {r: _run(stack, frames, r) for r in (1.0, *SCALES)}


def _errors(stack, results: list[dict], baseline: list[dict]) -> list[float]:
    """Помилки (м) по кадрах: проти GT, якщо є; інакше — проти baseline-позицій."""
    gt = stack["gt"]
    base_pos = {b["frame"]: (b["lat"], b["lon"]) for b in baseline if b["ok"]}
    errs = []
    for res in results[WARMUP:]:
        if not res["ok"]:
            continue
        if gt:
            ref = gt.get(res["frame"])
        else:
            ref = base_pos.get(res["frame"])
        if ref is not None:
            errs.append(geo_dist_m(res["lat"], res["lon"], ref[0], ref[1]))
    return errs


def test_baseline_localizes(runs):
    base = runs[1.0][WARMUP:]
    rate = sum(r["ok"] for r in base) / max(1, len(base))
    assert rate >= 0.7, (
        f"baseline (r=1.0) success rate {rate:.2f} < 0.7 — розберись із базовим "
        f"прогоном перед перевіркою scale-інваріантності"
    )


@pytest.mark.parametrize("r", SCALES)
def test_median_error_within_2x_baseline(stack, runs, r):
    min_rate = float(os.environ.get("DRONELOC_ACCEPT_MIN_SUCCESS", "0.5"))
    results = runs[r][WARMUP:]
    rate = sum(x["ok"] for x in results) / max(1, len(results))
    assert rate >= min_rate, f"r={r}: success rate {rate:.2f} < {min_rate}"

    errs = _errors(stack, runs[r], runs[1.0])
    assert len(errs) >= 3, f"r={r}: замало успішних кадрів для медіани ({len(errs)})"
    med = statistics.median(errs)

    if stack["gt"]:
        base_errs = _errors(stack, runs[1.0], runs[1.0])
        base_med = statistics.median(base_errs) if base_errs else 0.0
        limit = max(2.0 * base_med, 3.0)  # 3 м — підлога проти іграшково малої бази
        assert med <= limit, (
            f"r={r}: median err {med:.1f} м > {limit:.1f} м (2×baseline {base_med:.1f})"
        )
    else:
        tol = float(os.environ.get("DRONELOC_ACCEPT_TOL_M", "15"))
        assert med <= tol, f"r={r}: median |pos_r − pos_base| {med:.1f} м > {tol} м"


@pytest.mark.parametrize("r,lo,hi", [(0.5, 0.3, 0.75), (2.0, 1.4, 3.0)])
def test_fov_polygon_scales_with_r(runs, r, lo, hi):
    """Лінійний розмір полігона ~ r×baseline. Старий баг давав ratio ≈ 1 — обидві
    смуги його виключають."""
    base_d = {
        b["frame"]: poly_diameter_m(b["poly"])
        for b in runs[1.0][WARMUP:]
        if b["ok"] and poly_diameter_m(b["poly"])
    }
    ratios = []
    for res in runs[r][WARMUP:]:
        d = poly_diameter_m(res["poly"]) if res["ok"] else None
        if d and base_d.get(res["frame"]):
            ratios.append(d / base_d[res["frame"]])
    assert len(ratios) >= 3, f"r={r}: замало пар полігонів ({len(ratios)})"
    med = statistics.median(ratios)
    assert lo <= med <= hi, (
        f"r={r}: медіана відношення діаметрів FOV {med:.2f} поза [{lo}, {hi}] — "
        f"полігон не масштабується разом із висотою"
    )
