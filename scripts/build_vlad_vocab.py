"""RESEARCH 2.1: офлайн-побудова VLAD-словника (AnyLoc) для DroneLocalization.

Проходить референсні відео, збирає патч-токени DINOv3, будує k-means словник +
PCA-whitening і зберігає .npz, який вмикається через:

    models.vlad.enabled = true
    models.vlad.vocab_path = "models/vlad_vocab.npz"

Після цього базу даних треба ПЕРЕБУДУВАТИ (розмірність дескриптора змінюється).

СЛОВНИК — НЕ ПРО-ПРОЄКТНИЙ. AnyLoc показує, що в аеродомені domain-specific
словник б'є map-specific (підігнаний під одну карту). Тому подавайте сюди
КІЛЬКА різних обльотів, а отриманий .npz фіксуйте як спільний ассет для всіх
проєктів: `vocab_path` — глобальний ключ конфігу, а fingerprint бази не містить
ідентичності словника, тож змішування баз із різними словниками не буде
виявлене й дасть тихе сміття.

Семплінг: бюджет `--max-frames` ділиться між відео порівну, і всередині кожного
кадри беруться рівномірно по ВСІЙ довжині (а не з початку).

Запуск (Windows, у venv проєкту, потрібен GPU):
    python scripts/build_vlad_vocab.py --video flight_a.mp4 flight_b.mp4 flight_c.mp4 \
        --output models/vlad_vocab_c32_p256_v2.npz --max-frames 3000 [--layer N]
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
    ap.add_argument(
        "--video",
        required=True,
        nargs="+",
        help="Одне або КІЛЬКА референсних відео (кілька — краще, див. докстрінг)",
    )
    ap.add_argument("--output", default="models/vlad_vocab.npz")
    ap.add_argument(
        "--max-frames",
        type=int,
        default=2000,
        help="СУМАРНИЙ бюджет кадрів на всі відео; ділиться між ними порівну (default 2000)",
    )
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
    # ВАЖЛИВО: препроцес мусить збігатися з FeatureExtractor._dino_input, інакше
    # словник/PCA підганяються під інший розподіл токенів, ніж той, що буде на
    # побудові БД і на запиті. models.performance.dino_cpu_resize перемикає
    # torchvision Resize(antialias) на cv2 INTER_AREA/INTER_CUBIC — інший фільтр,
    # тому він і сидить у SCHEMA_FIELDS.
    cpu_resize = bool(get_cfg(APP_CONFIG, "models.performance.dino_cpu_resize", False))
    s = int(desc_cfg.input_size)
    normalize = T.Normalize(mean=desc_cfg.normalize_mean, std=desc_cfg.normalize_std)
    resize_gpu = T.Resize((s, s), antialias=True)

    def prep(rgb: np.ndarray) -> torch.Tensor:
        """(H, W, 3) uint8 RGB -> (1, 3, S, S) нормалізований тензор на device."""
        if cpu_resize:
            h, w = rgb.shape[:2]
            interp = cv2.INTER_AREA if (h > s or w > s) else cv2.INTER_CUBIC
            rgb = cv2.resize(np.ascontiguousarray(rgb), (s, s), interpolation=interp)
            t = torch.from_numpy(rgb).permute(2, 0, 1)[None].to(device).float().div_(255.0)
            return normalize(t)
        t = torch.from_numpy(rgb).float().div_(255.0).permute(2, 0, 1)[None].to(device)
        return normalize(resize_gpu(t))

    print(
        f"Препроцес DINO: {'cv2 CPU-resize (INTER_AREA/CUBIC)' if cpu_resize else 'torchvision Resize(antialias)'} -> {s}x{s}"
    )

    videos = list(args.video)
    quota_base, rem = divmod(args.max_frames, len(videos))
    if quota_base < 1:
        print(f"ERROR: --max-frames {args.max_frames} менший за кількість відео ({len(videos)})")
        return 1

    tokens_per_image: list[np.ndarray] = []
    for vi, path in enumerate(videos):
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            print(f"ERROR: не вдалося відкрити відео {path}")
            return 1
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total <= 0:
            print(f"ERROR: не вдалося визначити довжину {path}")
            cap.release()
            return 1
        quota = quota_base + (1 if vi < rem else 0)
        # Рівномірно по ВСІЙ довжині. Стара схема (`idx % every` зі зупинкою на
        # max_frames) обривалася на перших max_frames*every кадрах — словник
        # бачив лише початок польоту.
        step = max(1, total // max(quota, 1))
        got = 0
        for k in range(quota):
            cap.set(cv2.CAP_PROP_POS_FRAMES, min(k * step, total - 1))
            ok, frame = cap.read()
            if not ok:
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            t = prep(rgb)
            with torch.no_grad():
                feats = (
                    model.forward_features(t, layer=layer)
                    if layer is not None
                    else model.forward_features(t)
                )
            tokens_per_image.append(feats["x_norm_patchtokens"][0].float().cpu().numpy())
            got += 1
            if got % 100 == 0:
                print(f"  [{vi + 1}/{len(videos)}] зібрано {got}/{quota}")
        cap.release()
        print(f"  {path}: {got} кадрів із {total} (крок {step})")

    print(f"Зібрано {len(tokens_per_image)} кадрів × {tokens_per_image[0].shape} токенів")
    if len(tokens_per_image) < pca_dim + 1:
        print(
            f"УВАГА: кадрів ({len(tokens_per_image)}) < pca_dim+1 ({pca_dim + 1}) — "
            f"PCA буде обрізано до {len(tokens_per_image) - 1} вимірів. "
            f"Збільшіть --max-frames або додайте ще відео."
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
