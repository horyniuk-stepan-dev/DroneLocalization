"""
Бенчмарк заліза + стадій пайплайна для порівняння двох машин (див.
docs/PERF_BENCHMARK_PLAN.md).

Три яруси:
  0. Паспорт середовища (CPU/GPU/версії) — завжди.
  1. Мікробенчі заліза (GPU matmul, bandwidth, CLAHE, OF, MAGSAC, диск) —
     потрібні лише numpy+cv2(+torch для GPU-частини).
  2. Стадії пайплайна на реальних моделях (DINOv3, ALIKED, LightGlue,
     retrieval-проксі) на детермінованому синтетичному вході.
  3. (опційно, --video) той самий ланцюг по реальних кадрах.

Запуск:
  python scripts/benchmark_hardware.py                       # яруси 0-2
  python scripts/benchmark_hardware.py --skip-models         # лише 0-1
  python scripts/benchmark_hardware.py --video flight.mp4 --frames 60
  python scripts/benchmark_hardware.py --compare old.json new.json

Результат: benchmarks/hw_<host>_<дата>.json + таблиця в консоль.
Порівнювати треба однаковий комміт коду, однакові ваги та user_config.json.
Без PyQt6/h5py/lancedb: працює до повного перенесення проєктів.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import socket
import statistics
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

# Проєктний корінь у sys.path (запуск як `python scripts/benchmark_hardware.py`)
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

SCHEMA_VERSION = 1
SEED = 42

# Бюджети зі структури пайплайна: keyframe раз на ~1 c (frame_step=30 @ 30fps),
# трекінг — кожен кадр.
KEYFRAME_BUDGET_MS = 1000.0
TRACKING_BUDGET_MS = 33.0


# ── Тайминг ──────────────────────────────────────────────────────────────────


def bench(fn, warmup: int = 3, iters: int = 15, sync=None) -> dict:
    """Ганяє fn: warmup відкидається (перший виклик пишеться як cold start),
    далі iters вимірів. Повертає median/p95/min/max/mean у мс."""
    first_ms = None
    for i in range(warmup):
        t0 = time.perf_counter()
        fn()
        if sync:
            sync()
        if i == 0:
            first_ms = (time.perf_counter() - t0) * 1000.0
    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        if sync:
            sync()
        times.append((time.perf_counter() - t0) * 1000.0)
    times.sort()
    return {
        "median_ms": round(statistics.median(times), 3),
        "mean_ms": round(sum(times) / len(times), 3),
        "p95_ms": round(times[min(len(times) - 1, int(len(times) * 0.95))], 3),
        "min_ms": round(times[0], 3),
        "max_ms": round(times[-1], 3),
        "iters": iters,
        "first_call_ms": round(first_ms, 3) if first_ms is not None else None,
    }


# ── Ярус 0: паспорт середовища ───────────────────────────────────────────────


def collect_env() -> dict:
    env = {
        "hostname": socket.gethostname(),
        "os": f"{platform.system()} {platform.release()}",
        "python": platform.python_version(),
        "cpu": platform.processor() or platform.machine(),
        "cpu_cores": os.cpu_count(),
        "numpy": np.__version__,
        "opencv": cv2.__version__,
        "ram_gb": None,
        "torch": None,
        "cuda": None,
        "cudnn": None,
        "gpu": None,
        "gpu_vram_gb": None,
        "gpu_capability": None,
        "nvidia_driver": None,
    }
    try:
        import psutil

        env["ram_gb"] = round(psutil.virtual_memory().total / 1024**3, 1)
    except ImportError:
        pass
    try:
        import torch

        env["torch"] = torch.__version__
        if torch.cuda.is_available():
            env["cuda"] = torch.version.cuda
            env["cudnn"] = torch.backends.cudnn.version()
            props = torch.cuda.get_device_properties(0)
            env["gpu"] = props.name
            env["gpu_vram_gb"] = round(props.total_memory / 1024**3, 2)
            env["gpu_capability"] = f"{props.major}.{props.minor}"
    except ImportError:
        pass
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if out.returncode == 0:
            env["nvidia_driver"] = out.stdout.strip().splitlines()[0]
    except (OSError, subprocess.TimeoutExpired):
        pass
    return env


# ── Синтетичні входи (детерміновані, однакові на обох машинах) ───────────────


def make_synthetic_frame(w: int = 1920, h: int = 1080, seed: int = SEED) -> np.ndarray:
    """Аеро-подібна текстура: великі плями (поля) + дрібна фактура + дороги.
    Той самий seed → побайтово той самий кадр на обох машинах."""
    rng = np.random.default_rng(seed)
    small = rng.integers(0, 255, (h // 40, w // 40, 3), dtype=np.uint8)
    img = cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC)
    noise = rng.integers(0, 40, (h, w, 3), dtype=np.uint8)
    img = cv2.add(img, noise)
    for i in range(8):  # «дороги» — прямі лінії, дають кутові фічі
        p1 = (int(rng.integers(0, w)), int(rng.integers(0, h)))
        p2 = (int(rng.integers(0, w)), int(rng.integers(0, h)))
        cv2.line(img, p1, p2, (90 + 10 * i, 90, 90), rng.integers(3, 12))
    return img


def warp_frame(img: np.ndarray, seed: int = SEED) -> np.ndarray:
    """Невеликий зсув+поворот+масштаб — «сусідній кадр» для матчингу."""
    rng = np.random.default_rng(seed + 1)
    h, w = img.shape[:2]
    m = cv2.getRotationMatrix2D(
        (w / 2, h / 2), rng.uniform(3, 8), rng.uniform(0.95, 1.05)
    )
    m[:, 2] += rng.uniform(-40, 40, 2)
    return cv2.warpAffine(img, m, (w, h))


# ── Ярус 1: мікробенчі ───────────────────────────────────────────────────────


def run_micro(results: dict) -> None:
    rng = np.random.default_rng(SEED)

    a = rng.standard_normal((2048, 2048)).astype(np.float32)
    b = rng.standard_normal((2048, 2048)).astype(np.float32)
    results["cpu_matmul_2048_f32"] = bench(lambda: a @ b, warmup=2, iters=10)

    frame = make_synthetic_frame()

    def clahe_proxy():
        lab = cv2.cvtColor(frame, cv2.COLOR_RGB2LAB)
        l_ch, a_ch, b_ch = cv2.split(lab)
        cl = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(l_ch)
        cv2.cvtColor(cv2.merge((cl, a_ch, b_ch)), cv2.COLOR_LAB2RGB)

    results["cpu_clahe_1080p"] = bench(clahe_proxy, warmup=2, iters=15)

    # Optical flow проксі: як у tracking_worker (forward + backward перевірка)
    prev_g = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    curr_g = cv2.cvtColor(warp_frame(frame), cv2.COLOR_RGB2GRAY)
    pts = (
        np.stack(
            [
                rng.uniform(50, frame.shape[1] - 50, 2000),
                rng.uniform(50, frame.shape[0] - 50, 2000),
            ],
            axis=1,
        )
        .astype(np.float32)
        .reshape(-1, 1, 2)
    )

    def of_proxy():
        nxt, st, _ = cv2.calcOpticalFlowPyrLK(prev_g, curr_g, pts, None)
        cv2.calcOpticalFlowPyrLK(curr_g, prev_g, nxt, None)

    results["of_pyrlk_fb_2000pts"] = bench(of_proxy, warmup=2, iters=15)

    # MAGSAC-проксі: 2000 пар, 30% викидів (як transformations.py, USAC_MAGSAC)
    src = rng.uniform(0, 1900, (2000, 2)).astype(np.float64)
    h_true = np.array([[1.02, 0.01, 25.0], [-0.01, 0.99, -13.0], [1e-6, 1e-6, 1.0]])
    dst_h = np.hstack([src, np.ones((2000, 1))]) @ h_true.T
    dst = dst_h[:, :2] / dst_h[:, 2:3] + rng.normal(0, 0.5, (2000, 2))
    out_idx = rng.choice(2000, 600, replace=False)
    dst[out_idx] = rng.uniform(0, 1900, (600, 2))

    def magsac_proxy():
        cv2.findHomography(
            src.reshape(-1, 1, 2),
            dst.reshape(-1, 1, 2),
            method=cv2.USAC_MAGSAC,
            ransacReprojThreshold=5.0,
            maxIters=2000,
            confidence=0.999,
        )

    results["homography_magsac_2000pts"] = bench(magsac_proxy, warmup=2, iters=15)

    # Диск: послідовне читання найбільшого файлу з models/ (прогрів моделей).
    # УВАГА: повторні читання йдуть з кешу ОС — чесне лише перше після ребута;
    # тому пишемо і перший, і кешований час.
    models_dir = _ROOT / "models"
    if models_dir.exists():
        files = [
            p for p in models_dir.rglob("*") if p.is_file() and p.stat().st_size > 2**20
        ]
        if files:
            big = max(files, key=lambda p: p.stat().st_size)
            read_mb = min(big.stat().st_size, 512 * 2**20) / 2**20

            def disk_read():
                with open(big, "rb") as f:
                    while f.read(8 * 2**20):
                        if f.tell() >= read_mb * 2**20:
                            break

            r = bench(disk_read, warmup=1, iters=3)
            r["file"] = big.name
            r["read_mb"] = round(read_mb, 1)
            r["first_read_mb_s"] = round(read_mb / (r["first_call_ms"] / 1000.0), 1)
            r["cached_mb_s"] = round(read_mb / (r["median_ms"] / 1000.0), 1)
            results["disk_read_models"] = r

    tmp = _ROOT / "benchmarks" / "_disk_write_probe.bin"
    tmp.parent.mkdir(exist_ok=True)
    blob = rng.integers(0, 255, 128 * 2**20, dtype=np.uint8).tobytes()

    def disk_write():
        with open(tmp, "wb") as f:
            f.write(blob)
            f.flush()
            os.fsync(f.fileno())

    r = bench(disk_write, warmup=1, iters=3)
    r["write_mb_s"] = round(128 / (r["median_ms"] / 1000.0), 1)
    results["disk_write_128mb"] = r
    try:
        tmp.unlink()
    except OSError:
        pass

    # GPU мікробенчі
    try:
        import torch
    except ImportError:
        print("[micro] torch not installed - GPU micro-benches skipped")
        return
    if not torch.cuda.is_available():
        print("[micro] CUDA not available - GPU micro-benches skipped")
        return
    torch.manual_seed(SEED)
    dev = torch.device("cuda")
    sync = torch.cuda.synchronize

    for dtype, name in [(torch.float32, "f32"), (torch.float16, "f16")]:
        x = torch.randn(4096, 4096, device=dev, dtype=dtype)
        y = torch.randn(4096, 4096, device=dev, dtype=dtype)
        r = bench(lambda: x @ y, warmup=5, iters=20, sync=sync)
        r["tflops"] = round(2 * 4096**3 / (r["median_ms"] / 1000.0) / 1e12, 2)
        results[f"gpu_matmul_4096_{name}"] = r

    host = torch.empty(64 * 2**20 // 4, dtype=torch.float32, pin_memory=True)  # 64 MB
    devt = torch.empty_like(host, device=dev)
    r = bench(
        lambda: devt.copy_(host, non_blocking=True), warmup=3, iters=15, sync=sync
    )
    r["gb_s"] = round(64 / 1024 / (r["median_ms"] / 1000.0), 2)
    results["gpu_h2d_64mb"] = r
    r = bench(
        lambda: host.copy_(devt, non_blocking=True), warmup=3, iters=15, sync=sync
    )
    r["gb_s"] = round(64 / 1024 / (r["median_ms"] / 1000.0), 2)
    results["gpu_d2h_64mb"] = r
    torch.cuda.empty_cache()


# ── Ярус 2: стадії пайплайна ─────────────────────────────────────────────────


def run_model_stages(results: dict, config_snapshot: dict) -> None:
    try:
        import torch
    except ImportError:
        print("[models] torch not installed - model stages skipped")
        return

    from config import APP_CONFIG, get_active_descriptor_cfg, get_cfg
    from src.localization.matcher import FastRetrieval, FeatureMatcher
    from src.models.model_manager import ModelManager
    from src.models.wrappers.feature_extractor import FeatureExtractor

    torch.manual_seed(SEED)
    sync = torch.cuda.synchronize if torch.cuda.is_available() else None

    desc_cfg = get_active_descriptor_cfg(APP_CONFIG)
    config_snapshot.update(
        {
            "local_extractor": get_cfg(APP_CONFIG, "models.local_extractor", "aliked"),
            "descriptor_backend": type(desc_cfg).__name__,
            "descriptor_input_size": getattr(desc_cfg, "input_size", None),
            "max_keypoints": get_cfg(APP_CONFIG, "models.aliked.max_keypoints", None),
            "max_local_edge": get_cfg(APP_CONFIG, "localization.max_local_edge", 1600),
            "retrieval_top_k": get_cfg(APP_CONFIG, "localization.retrieval_top_k", 12),
            "torch_compile": get_cfg(
                APP_CONFIG, "models.performance.torch_compile", False
            ),
        }
    )

    def vram_peak_reset():
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

    def vram_peak_mb():
        if torch.cuda.is_available():
            return round(torch.cuda.max_memory_allocated() / 2**20, 1)
        return None

    mm = ModelManager(config=APP_CONFIG)

    # Холодне завантаження (= внесок у час старту апки)
    t0 = time.perf_counter()
    local_model = mm.load_local_extractor()
    results["load_local_extractor"] = {
        "first_call_ms": round((time.perf_counter() - t0) * 1e3, 1)
    }
    t0 = time.perf_counter()
    global_model = mm.load_dinov2()
    results["load_dinov2"] = {
        "first_call_ms": round((time.perf_counter() - t0) * 1e3, 1)
    }
    t0 = time.perf_counter()
    matcher = FeatureMatcher(model_manager=mm, config=APP_CONFIG)
    results["load_lightglue"] = {
        "first_call_ms": round((time.perf_counter() - t0) * 1e3, 1)
    }

    fe = FeatureExtractor(local_model, global_model, mm.device, config=APP_CONFIG)

    frame = make_synthetic_frame()
    frame2 = warp_frame(frame)

    # Global descriptor (DINOv3 + препроцес). Перший виклик = torch.compile cold.
    vram_peak_reset()
    results["global_descriptor"] = bench(
        lambda: fe.extract_global_descriptor(frame), warmup=3, iters=15, sync=sync
    )
    results["global_descriptor"]["peak_vram_mb"] = vram_peak_mb()

    # Local features (ALIKED/RDD FP32)
    vram_peak_reset()
    results["local_features"] = bench(
        lambda: fe.extract_local_features(frame), warmup=3, iters=15, sync=sync
    )
    results["local_features"]["peak_vram_mb"] = vram_peak_mb()

    # Матчинг: query повний, ref обрізаний до 2048 (як max_keypoints_stored у БД)
    qf = fe.extract_local_features(frame)
    rf = fe.extract_local_features(frame2)
    keep = min(2048, len(rf["keypoints"]))
    rf_trim = {
        "keypoints": rf["keypoints"][:keep],
        "descriptors": rf["descriptors"][:keep],
        "coords_2d": rf["coords_2d"][:keep],
        "image_size": rf["image_size"],
    }
    results["match_stats"] = {
        "query_kpts": int(len(qf["keypoints"])),
        "ref_kpts": int(keep),
    }
    if matcher.lightglue is not None:
        vram_peak_reset()
        results["match_lightglue"] = bench(
            lambda: matcher.match(qf, rf_trim), warmup=3, iters=15, sync=sync
        )
        results["match_lightglue"]["peak_vram_mb"] = vram_peak_mb()
    else:
        print("[models] LightGlue unavailable - match_lightglue skipped")
    results["match_mnn"] = bench(
        lambda: matcher.match_mnn(qf, rf_trim), warmup=2, iters=10
    )

    # Retrieval-проксі: brute force по 8192 дескрипторах реальної розмірності
    gd = fe.extract_global_descriptor(frame)
    rng = np.random.default_rng(SEED)
    db = rng.standard_normal((8192, gd.shape[0])).astype(np.float32)
    retr = FastRetrieval(db)
    results["retrieval_numpy_8192"] = bench(
        lambda: retr.find_similar_frames(gd, top_k=12), warmup=2, iters=20
    )

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ── Ярус 3: реальне відео (опційно) ──────────────────────────────────────────


def run_video(results: dict, video_path: str, n_frames: int) -> None:
    try:
        from config import APP_CONFIG
        from src.localization.matcher import FeatureMatcher
        from src.models.model_manager import ModelManager
        from src.models.wrappers.feature_extractor import FeatureExtractor
        import torch
    except ImportError as e:
        print(f"[video] skipped: {e}")
        return

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[video] cannot open {video_path}")
        return
    sync = torch.cuda.synchronize if torch.cuda.is_available() else None

    mm = ModelManager(config=APP_CONFIG)
    fe = FeatureExtractor(
        mm.load_local_extractor(), mm.load_dinov2(), mm.device, config=APP_CONFIG
    )
    matcher = FeatureMatcher(model_manager=mm, config=APP_CONFIG)

    stages: dict[str, list] = {"global": [], "local": [], "match": []}
    prev_feats = None
    processed = 0
    while processed < n_frames:
        ok, bgr = cap.read()
        if not ok:
            break
        frame = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        t0 = time.perf_counter()
        fe.extract_global_descriptor(frame)
        if sync:
            sync()
        stages["global"].append((time.perf_counter() - t0) * 1000)

        t0 = time.perf_counter()
        feats = fe.extract_local_features(frame)
        if sync:
            sync()
        stages["local"].append((time.perf_counter() - t0) * 1000)

        if prev_feats is not None:
            t0 = time.perf_counter()
            matcher.match(feats, prev_feats)
            if sync:
                sync()
            stages["match"].append((time.perf_counter() - t0) * 1000)
        prev_feats = feats
        processed += 1
    cap.release()

    out = {"frames": processed, "video": os.path.basename(video_path)}
    for name, ts in stages.items():
        if len(ts) > 3:
            ts_w = sorted(ts[2:])  # відкинути warmup-кадри
            out[name] = {
                "median_ms": round(statistics.median(ts_w), 2),
                "p95_ms": round(ts_w[int(len(ts_w) * 0.95)], 2),
            }
    chain = sum(out[s]["median_ms"] for s in ("global", "local", "match") if s in out)
    if chain:
        out["keyframe_chain_median_ms"] = round(chain, 1)
        out["keyframe_chain_fps"] = round(1000.0 / chain, 2)
    results["video_pipeline"] = out


# ── Агрегати та вивід ────────────────────────────────────────────────────────


def compute_aggregates(results: dict, config_snapshot: dict) -> dict:
    agg = {}

    def med(name):
        r = results.get(name)
        return r.get("median_ms") if r else None

    g, loc, m = med("global_descriptor"), med("local_features"), med("match_lightglue")
    retr, hom = med("retrieval_numpy_8192"), med("homography_magsac_2000pts")
    if None not in (g, loc, m, retr, hom):
        top_k = config_snapshot.get("retrieval_top_k") or 12
        agg["steady_keyframe_ms"] = round(g + loc + m + retr + hom, 1)
        agg["worst_keyframe_ms"] = round(g + loc + top_k * m + retr + top_k * hom, 1)
        agg["keyframe_budget_ms"] = KEYFRAME_BUDGET_MS
    of = med("of_pyrlk_fb_2000pts")
    if of is not None:
        agg["tracking_frame_ms"] = of
        agg["tracking_budget_ms"] = TRACKING_BUDGET_MS
    return agg


def print_report(report: dict) -> None:
    env = report["env"]
    print("\n" + "=" * 72)
    print(f"Host: {env['hostname']}  |  {env['os']}  |  {env.get('cpu')}")
    print(
        f"GPU:  {env.get('gpu')}  VRAM {env.get('gpu_vram_gb')} GB  "
        f"cap {env.get('gpu_capability')}  driver {env.get('nvidia_driver')}"
    )
    print(
        f"torch {env.get('torch')}  CUDA {env.get('cuda')}  "
        f"cv2 {env.get('opencv')}  numpy {env.get('numpy')}"
    )
    print("=" * 72)
    print(f"{'stage':38s} {'median':>9s} {'p95':>9s} {'cold':>10s}")
    for name, r in report["results"].items():
        if not isinstance(r, dict):
            continue
        if "median_ms" not in r:
            if "first_call_ms" in r:  # холодні завантаження моделей
                print(
                    f"{name:38s} {'-':>9s} {'-':>9s} {round(r['first_call_ms']):>8d}ms"
                )
            continue
        cold = r.get("first_call_ms")
        extra = ""
        if "tflops" in r:
            extra = f"  {r['tflops']} TFLOPS"
        elif "gb_s" in r:
            extra = f"  {r['gb_s']} GB/s"
        elif "peak_vram_mb" in r and r["peak_vram_mb"]:
            extra = f"  peak {r['peak_vram_mb']} MB"
        print(
            f"{name:38s} {r['median_ms']:>7.1f}ms {r['p95_ms']:>7.1f}ms "
            f"{(str(round(cold)) + 'ms') if cold else '-':>10s}{extra}"
        )
    if report.get("aggregates"):
        print("-" * 72)
        for k, v in report["aggregates"].items():
            print(f"{k:38s} {v}")
    print("=" * 72)


# ── Порівняння двох звітів ───────────────────────────────────────────────────


def compare_reports(path_a: str, path_b: str) -> None:
    with open(path_a, encoding="utf-8") as f:
        a = json.load(f)
    with open(path_b, encoding="utf-8") as f:
        b = json.load(f)
    ea, eb = a["env"], b["env"]
    print(
        f"\nA = {ea['hostname']} ({ea.get('gpu')})   B = {eb['hostname']} ({eb.get('gpu')})"
    )
    for key in ("torch", "cuda", "opencv", "numpy", "nvidia_driver"):
        if ea.get(key) != eb.get(key):
            print(
                f"  WARNING: {key} differs: A={ea.get(key)}  B={eb.get(key)} "
                f"- ratio includes software stack, not just hardware"
            )

    print(f"\n{'stage':38s} {'A median':>10s} {'B median':>10s} {'B/A':>7s}")
    print("-" * 70)
    flagged = []
    for name, ra in a["results"].items():
        rb = b["results"].get(name)
        if not (isinstance(ra, dict) and isinstance(rb, dict)):
            continue
        ma, mb = ra.get("median_ms"), rb.get("median_ms")
        if ma is None or mb is None or ma == 0:
            continue
        ratio = mb / ma
        mark = ""
        if ratio > 1.3:
            mark = "  <-- slower"
            flagged.append((name, ratio))
        print(f"{name:38s} {ma:>8.1f}ms {mb:>8.1f}ms {ratio:>6.2f}x{mark}")

    print("-" * 70)
    for k in ("steady_keyframe_ms", "worst_keyframe_ms", "tracking_frame_ms"):
        va, vb = a.get("aggregates", {}).get(k), b.get("aggregates", {}).get(k)
        if va and vb:
            print(f"{k:38s} {va:>8.1f}   {vb:>8.1f}   {vb / va:.2f}x")
    budget = b.get("aggregates", {})
    if (
        budget.get("worst_keyframe_ms")
        and budget["worst_keyframe_ms"] > KEYFRAME_BUDGET_MS
    ):
        print(
            f"\nVERDICT: worst-case keyframe {budget['worst_keyframe_ms']:.0f}ms > "
            f"budget {KEYFRAME_BUDGET_MS:.0f}ms on B - "
            f"apply EFFICIENCY_OPTIONS.md knobs (top-k gating, MNN prefilter, 2048 kpts)"
        )
    if (
        budget.get("tracking_frame_ms")
        and budget["tracking_frame_ms"] > TRACKING_BUDGET_MS
    ):
        print(
            f"VERDICT: tracking {budget['tracking_frame_ms']:.1f}ms > {TRACKING_BUDGET_MS}ms "
            f"budget on B - real-time tracking at risk"
        )
    if flagged:
        worst = max(flagged, key=lambda x: x[1])
        print(
            f"\nMost degraded stage: {worst[0]} ({worst[1]:.2f}x). If it breaks the overall "
            f"trend, suspect software stack (CPU fallback, missing weights), not hardware."
        )


# ── main ─────────────────────────────────────────────────────────────────────


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Hardware/pipeline benchmark (A/B two machines)"
    )
    ap.add_argument(
        "--skip-models", action="store_true", help="only tiers 0-1 (no weights needed)"
    )
    ap.add_argument(
        "--video", type=str, default=None, help="optional real video for tier 3"
    )
    ap.add_argument(
        "--frames", type=int, default=60, help="frames to process from --video"
    )
    ap.add_argument("--output", type=str, default=None, help="output json path")
    ap.add_argument(
        "--compare", nargs=2, metavar=("OLD", "NEW"), help="compare two reports"
    )
    args = ap.parse_args()

    if args.compare:
        compare_reports(args.compare[0], args.compare[1])
        return

    np.random.seed(SEED)
    report = {
        "schema_version": SCHEMA_VERSION,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "env": collect_env(),
        "config_snapshot": {},
        "results": {},
        "aggregates": {},
    }

    print("[tier 1] hardware micro-benchmarks...")
    run_micro(report["results"])

    if not args.skip_models:
        print("[tier 2] pipeline model stages (first calls include compile/warmup)...")
        try:
            run_model_stages(report["results"], report["config_snapshot"])
        except Exception as e:  # noqa: BLE001 - бенч не має падати цілком
            print(f"[models] FAILED: {type(e).__name__}: {e}")
            report["results"]["model_stages_error"] = f"{type(e).__name__}: {e}"

    if args.video:
        print(f"[tier 3] real video pass: {args.video}")
        run_video(report["results"], args.video, args.frames)

    report["aggregates"] = compute_aggregates(
        report["results"], report["config_snapshot"]
    )

    out_dir = _ROOT / "benchmarks"
    out_dir.mkdir(exist_ok=True)
    host = report["env"]["hostname"].replace(" ", "_")
    out_path = (
        Path(args.output)
        if args.output
        else (out_dir / f"hw_{host}_{datetime.now():%Y%m%d-%H%M}.json")
    )
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print_report(report)
    print(f"\nSaved: {out_path}")
    print(
        "Compare: python scripts/benchmark_hardware.py --compare <old.json> <new.json>"
    )


if __name__ == "__main__":
    main()
