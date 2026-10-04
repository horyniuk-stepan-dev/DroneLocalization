"""RESEARCH 2.1: офлайн-побудова VLAD-словника (AnyLoc) для DroneLocalization.

Проходить референсні відео, збирає патч-токени DINOv3, будує k-means словник +
PCA-whitening і зберігає .npz, який вмикається через:

    models.vlad.enabled = true
    models.vlad.vocab_path = "models/vlad_vocab.npz"

Після цього базу даних треба ПЕРЕБУДУВАТИ (розмірність дескриптора змінюється).

СЛОВНИК — НЕ ПРО-ПРОЄКТНИЙ. AnyLoc показує, що в аеродомені domain-specific
словник б'є map-specific (підігнаний під одну карту). Тому подавайте сюди
КІЛЬКА різних обльотів, а отриманий .npz фіксуйте як спільний ассет для всіх
проєктів.

Vocabulary identity: the database schema fingerprint records the vocabulary's
content hash (plus models.vlad.layer and low_norm_fraction), and a source built
with another vocabulary is refused when the project is loaded. Replacing the
vocabulary therefore means rebuilding every database that should stay
queryable. The .npz records its provenance (sources, layer, input size, DINO
resize, model revision); FeatureExtractor warns when the config differs.

Sources: --video files and/or --images folders (searched recursively, e.g.
exported frames or map tiles). The --max-frames budget is split evenly between
sources and sampled evenly over the whole of each source.

Запуск (Windows, у venv проєкту, потрібен GPU):
    python scripts/build_vlad_vocab.py --video flight_a.mp4 flight_b.mp4 \
        --images D:/tiles/region_a D:/tiles/region_b \
        --output models/vlad_vocab_c32_p256_v2.npz --max-frames 3000 [--layer N]
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

IMAGE_SUFFIXES = frozenset({".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"})


def split_budget(total: int, parts: int) -> list[int]:
    """Split a frame budget evenly; the first parts take the remainder."""
    base, rem = divmod(int(total), int(parts))
    return [base + (1 if i < rem else 0) for i in range(parts)]


def even_indices(total: int, quota: int) -> list[int]:
    """Up to ``quota`` distinct indices spread evenly over the whole of range(total)."""
    if total <= 0 or quota <= 0:
        return []
    quota = min(quota, total)
    step = total / quota
    return [int(k * step) for k in range(quota)]


def list_images(folder: str | Path) -> list[Path]:
    """Image files under ``folder`` (recursive, sorted)."""
    root = Path(folder)
    if not root.is_dir():
        raise ValueError(f"image folder does not exist: {folder}")
    files = sorted(
        path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    if not files:
        raise ValueError(f"no images in {folder}")
    return files


def iter_video_frames(path: str | Path, quota: int):
    """RGB frames sampled evenly over the whole video."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f"cannot open video {path}")
    try:
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total <= 0:
            raise ValueError(f"cannot determine the length of {path}")
        for index in even_indices(total, quota):
            cap.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, frame = cap.read()
            if not ok:
                break
            yield cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    finally:
        cap.release()


def iter_image_files(folder: str | Path, quota: int):
    """RGB images sampled evenly over a folder's sorted file list."""
    files = list_images(folder)
    for index in even_indices(len(files), quota):
        image = cv2.imread(str(files[index]), cv2.IMREAD_COLOR)
        if image is None:
            print(f"  WARNING: unreadable image skipped: {files[index]}")
            continue
        yield cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--video",
        nargs="+",
        default=[],
        help="Одне або КІЛЬКА референсних відео (кілька — краще, див. докстрінг)",
    )
    ap.add_argument(
        "--images",
        nargs="+",
        default=[],
        help="folder(s) of images or map tiles, searched recursively; each folder is one source",
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
    if not args.video and not args.images:
        ap.error("give at least one --video or --images source")

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

    # global_descriptor is a top-level section; the old "models.global_descriptor"
    # path never existed, so this check always saw the default "dinov3".
    backend = get_cfg(APP_CONFIG, "global_descriptor.backend", "dinov3")
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

    sources = [("video", path) for path in args.video]
    sources += [("images", path) for path in args.images]
    quotas = split_budget(args.max_frames, len(sources))
    if min(quotas) < 1:
        print(
            f"ERROR: --max-frames {args.max_frames} is smaller than the number of "
            f"sources ({len(sources)})"
        )
        return 1

    tokens_per_image: list[np.ndarray] = []
    used_sources: list[dict] = []
    for si, ((kind, path), quota) in enumerate(zip(sources, quotas)):
        frames = (
            iter_video_frames(path, quota) if kind == "video" else iter_image_files(path, quota)
        )
        got = 0
        try:
            for rgb in frames:
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
                    print(f"  [{si + 1}/{len(sources)}] {got}/{quota}")
        except ValueError as exc:
            print(f"ERROR: {exc}")
            return 1
        used_sources.append({"kind": kind, "path": str(Path(path).resolve()), "frames": got})
        print(f"  {kind} {path}: {got} frames (quota {quota})")

    if len(tokens_per_image) < 2:
        print(f"ERROR: only {len(tokens_per_image)} frame(s) collected; VLAD needs at least 2")
        return 1
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
    agg.save(
        args.output,
        provenance={
            "created": datetime.now().astimezone().isoformat(timespec="seconds"),
            "sources": used_sources,
            "images": len(tokens_per_image),
            "layer": layer,
            "input_size": s,
            "dino_cpu_resize": cpu_resize,
            "hf_model_id": desc_cfg.hf_model_id,
            "hf_revision": getattr(desc_cfg, "hf_revision", "") or "",
            "n_clusters": n_clusters,
            "pca_dim": pca_dim,
        },
    )
    print(f"Готово: {args.output} (out_dim={agg.out_dim})")
    print("Наступні кроки: увімкніть models.vlad.enabled + vocab_path і ПЕРЕБУДУЙТЕ базу даних.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
