"""Time the DINOv3 global-descriptor pass at several input sizes on this GPU.

Answers the speed side of "maximum DINO resolution at maximum speed": the
forward pass (``forward_features``, what VLAD aggregates) at each size, batch 1
(a query) and batch 8 (database building), fp32 and, if requested, fp16
autocast. Accuracy is measured separately by replays against database copies
built at the same size (scratch/testtopboch_analysis/dino_resolution_ab.ps1).

    python scripts/benchmark_dino_resolution.py --sizes 224 336 448 --fp16

Run it on every target machine (RTX 5070 Ti and the GTX 1650 lower bound):
the numbers are per GPU.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def time_pass(model, size: int, batch: int, iters: int, warmup: int, fp16: bool, device: str):
    import torch

    x = torch.randn(batch, 3, size, size, device=device)
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    times = []
    for i in range(warmup + iters):
        if device == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.inference_mode(), torch.autocast(device, enabled=fp16 and device == "cuda"):
            model.forward_features(x)
        if device == "cuda":
            torch.cuda.synchronize()
        if i >= warmup:
            times.append((time.perf_counter() - start) * 1000.0)
    times.sort()
    peak = torch.cuda.max_memory_allocated() / 2**20 if device == "cuda" else None
    return {
        "size": size,
        "tokens": (size // 16) ** 2,
        "batch": batch,
        "fp16": fp16,
        "median_ms": round(times[len(times) // 2], 2),
        "p90_ms": round(times[int(len(times) * 0.9)], 2),
        "ms_per_image": round(times[len(times) // 2] / batch, 2),
        "peak_mb": round(peak, 1) if peak is not None else None,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--sizes", nargs="+", type=int, default=[224, 336, 448])
    parser.add_argument("--batches", nargs="+", type=int, default=[1, 8])
    parser.add_argument("--iters", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--fp16", action="store_true", help="also time fp16 autocast")
    parser.add_argument("--out", type=Path, help="write results as JSON")
    args = parser.parse_args(argv)

    import torch

    from config import APP_CONFIG, get_active_descriptor_cfg
    from src.models.wrappers.dinov3_wrapper import DINOv3Wrapper

    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = get_active_descriptor_cfg(APP_CONFIG)
    model = DINOv3Wrapper(
        cfg.hf_model_id, device=device, revision=getattr(cfg, "hf_revision", "") or None
    ).eval()
    gpu = torch.cuda.get_device_name(0) if device == "cuda" else "cpu"
    print(f"device: {gpu}")
    results = []
    for size in args.sizes:
        if size % 16:
            print(f"skip {size}: not a multiple of 16")
            continue
        for batch in args.batches:
            for fp16 in [False, True] if args.fp16 else [False]:
                try:
                    r = time_pass(model, size, batch, args.iters, args.warmup, fp16, device)
                except RuntimeError as exc:  # out of memory on small GPUs
                    r = {"size": size, "batch": batch, "fp16": fp16, "error": str(exc)[:200]}
                    if device == "cuda":
                        torch.cuda.empty_cache()
                results.append(r)
                print(json.dumps(r))
    if args.out:
        args.out.write_text(json.dumps({"device": gpu, "results": results}, indent=2), "utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
