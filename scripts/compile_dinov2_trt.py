#!/usr/bin/env python3
"""
Скрипт для компіляції DINOv2 ViT-L/14 → ONNX → TensorRT FP16 engine.

Використання:
    python scripts/compile_dinov2_trt.py --output models/engines/

Залежності:
    - torch, torchvision (для завантаження моделі)
    - onnx (для валідації)
    - tensorrt або trtexec CLI (для компіляції engine)
"""

import argparse
import subprocess
import sys
from pathlib import Path


def export_onnx(output_dir: Path, input_size: int = 336) -> Path:
    """Експортує DINOv2 у ONNX формат."""
    import torch

    print("[1/3] Завантаження DINOv2 vitl14...")
    model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitl14")
    model = model.eval().cpu()

    onnx_path = output_dir / "dinov2_vitl14.onnx"
    dummy_input = torch.randn(1, 3, input_size, input_size)

    print(f"[2/3] Експорт у ONNX: {onnx_path}")
    torch.onnx.export(
        model,
        dummy_input,
        str(onnx_path),
        opset_version=17,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes=None,  # фіксований вхід для TRT оптимізації
    )

    # Валідація ONNX
    try:
        import onnx

        onnx_model = onnx.load(str(onnx_path))
        onnx.checker.check_model(onnx_model)
        print("  ONNX валідація пройдена ✓")
    except ImportError:
        print("  [WARNING] onnx не встановлений, пропускаю валідацію")
    except Exception as e:
        print(f"  [WARNING] ONNX валідація: {e}")

    file_size_mb = onnx_path.stat().st_size / (1024 * 1024)
    print(f"  Розмір ONNX: {file_size_mb:.1f} MB")
    return onnx_path


def compile_trt(onnx_path: Path, output_dir: Path) -> Path:
    """Компілює ONNX → TensorRT FP16 engine через trtexec."""
    engine_path = output_dir / "dinov2_vitl14_fp16.engine"

    print(f"[3/3] Компіляція TensorRT FP16: {engine_path}")

    cmd = [
        "trtexec",
        f"--onnx={onnx_path}",
        f"--saveEngine={engine_path}",
        "--fp16",
        "--workspace=4096",
    ]

    print(f"  Команда: {' '.join(cmd)}")
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        if result.returncode != 0:
            print(f"  [ERROR] trtexec failed:\n{result.stderr}")
            sys.exit(1)
        print(f"  TensorRT engine saved: {engine_path}")
        file_size_mb = engine_path.stat().st_size / (1024 * 1024)
        print(f"  Розмір engine: {file_size_mb:.1f} MB")
    except FileNotFoundError:
        print(
            "  [ERROR] trtexec не знайдено. Встановіть TensorRT або додайте trtexec до PATH.\n"
            "  Альтернатива: використайте ONNX файл з onnxruntime-gpu."
        )
        sys.exit(1)
    except subprocess.TimeoutExpired:
        print("  [ERROR] Компіляція перевищила таймаут (600с)")
        sys.exit(1)

    return engine_path


def main():
    parser = argparse.ArgumentParser(description="Compile DINOv2 → TensorRT FP16 engine")
    parser.add_argument(
        "--output",
        type=str,
        default="models/engines/",
        help="Директорія для збереження engine (default: models/engines/)",
    )
    parser.add_argument(
        "--onnx-only",
        action="store_true",
        help="Тільки ONNX експорт (без TensorRT компіляції)",
    )
    parser.add_argument(
        "--input-size",
        type=int,
        default=336,
        help="Розмір вхідного зображення (default: 336)",
    )
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    onnx_path = export_onnx(output_dir, args.input_size)

    if not args.onnx_only:
        compile_trt(onnx_path, output_dir)

    print("\n✓ Готово!")


if __name__ == "__main__":
    main()
