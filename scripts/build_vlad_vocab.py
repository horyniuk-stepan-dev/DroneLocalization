"""RESEARCH 2.1: офлайн-побудова VLAD-словника (AnyLoc) для DroneLocalization.

Проходить референсне відео, збирає патч-токени DINOv3 з кожного K-го кадру,
будує k-means словник + PCA-whitening і зберігає .npz, який вмикається через:

    models.vlad.enabled = true
    models.vlad.vocab_path = "models/vlad_vocab.npz"

Після цього базу даних треба ПЕРЕБУДУВАТИ (розмірність дескриптора змінюється).

Запуск (Windows, у venv проєкту, потрібен GPU):
    python scripts/build_vlad_vocab.py --video path/to/reference.mp4 \
        --output models/vlad_vocab.npz [--every 30] [--layer N] [--max-frames 500]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2  # noqa: E402
import numpy as np  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--video", required=True, help="Референсне відео (те, з якого будується БД)")
    ap.add_argument("--output", default="models/vlad_vocab.npz")
    ap.add_argument("--every", type=int, default=30, help="Кожен N-й кадр (default 30)")
    ap.add_argument("--max-frames", type=int, default=500)
    ap.add_argument("--clusters", type=int, default=None, help="Перекрити models.vlad.n_clusters")
    ap.add_argument("--pca-dim", type=int, default=None, help="Перекрити models.vlad.pca_dim")
    ap.add_argument("--layer", type=int, default=None, help="Проміжний шар ViT (default: конфіг)")
    args = ap.parse_args()

    import torch
    import torchvision.transforms as T

    from config import APP_CONFIG, get_active_descriptor_cfg, get_cfg
    from src.models.wrappers.dinov3_wrapper import DINOv3Wrapper
    from src.models.wrappers.vlad_aggregator import VladAggregator

    device = "cuda" if torch.cuda.is_available() else "cpu"
    desc_cfg = get_active_descriptor_cfg(APP_CONFIG)
    n_clusters = args.clusters or get_cfg(APP_CONFIG, "models.vlad.n_clusters", 32)
    pca_dim = args.pca_dim or get_cfg(APP_CONFIG, "models.vlad.pca_dim", 512)
    layer = args.layer if args.layer is not None else get_cfg(APP_CONFIG, "models.vlad.layer", None)

    backend = get_cfg(APP_CONFIG, "models.global_descriptor.backend", "dinov3")
    if backend != "dinov3":
        print(f"ERROR: словник VLAD підтримано лише для DINOv3 (зараз backend={backend})")
        return 1

    model = DINOv3Wrapper(
        desc_cfg.hf_model_id,
        device=device,
        revision=getattr(desc_cfg, "hf_revision", "") or None,
    )
    transform = T.Compose(
        [
            T.Resize((desc_cfg.input_size, desc_cfg.input_size), antialias=True),
            T.Normalize(mean=desc_cfg.normalize_mean, std=desc_cfg.normalize_std),
        ]
    )

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"ERROR: не вдалося відкрити відео {args.video}")
        return 1

    tokens_per_image: list[np.ndarray] = []
    idx = 0
    while len(tokens_per_image) < args.max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % args.every == 0:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            t = torch.from_numpy(rgb).float().div_(255.0).permute(2, 0, 1)[None].to(device)
            with torch.no_grad():
                feats = (
                    model.forward_features(transform(t), layer=layer)
                    if layer is not None
                    else model.forward_features(transform(t))
                )
            tokens_per_image.append(feats["x_norm_patchtokens"][0].float().cpu().numpy())
            if len(tokens_per_image) % 50 == 0:
                print(f"  зібрано кадрів: {len(tokens_per_image)}")
        idx += 1
    cap.release()

    print(f"Зібрано {len(tokens_per_image)} кадрів × {tokens_per_image[0].shape} токенів")
    if len(tokens_per_image) < pca_dim + 1:
        print(
            f"УВАГА: кадрів ({len(tokens_per_image)}) < pca_dim+1 ({pca_dim + 1}) — "
            f"PCA буде обрізано до {len(tokens_per_image) - 1} вимірів. "
            f"Зменшіть --every або збільшіть --max-frames."
        )

    agg = VladAggregator(
        n_clusters=n_clusters,
        pca_dim=pca_dim,
        low_norm_fraction=get_cfg(APP_CONFIG, "models.vlad.low_norm_fraction", 0.0),
    )
    agg.fit(tokens_per_image)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    agg.save(args.output)
    print(f"Готово: {args.output} (out_dim={agg.out_dim})")
    print("Наступні кроки: увімкніть models.vlad.enabled + vocab_path і ПЕРЕБУДУЙТЕ базу даних.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
