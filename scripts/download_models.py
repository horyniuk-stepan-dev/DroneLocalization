#!/usr/bin/env python3
"""
Скрипт для автоматичного завантаження ваг нейромереж
"""

import sys
import urllib.request
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))


class DownloadProgressBar(tqdm):
    """Кастомний прогрес-бар для urllib.request"""

    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_url(url: str, output_path: Path):
    """Завантаження файлу з відображенням прогресу"""
    with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=output_path.name) as t:
        urllib.request.urlretrieve(url, filename=output_path, reporthook=t.update_to)


def download_models():
    """Завантаження всіх необхідних моделей"""
    models_dir = Path("data/models")
    models_dir.mkdir(parents=True, exist_ok=True)

    models_to_download = {
        "yolov8x-seg.pt": "https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8x-seg.pt",
        "superpoint_v1.pth": "https://github.com/magicleap/SuperPointPretrainedNetwork/raw/master/superpoint_v1.pth",
        "netvlad_v1.pth": "https://storage.googleapis.com/netvlad_weights/netvlad_v1.pth",
        "lightglue_superpoint.pth": "https://github.com/cvg/LightGlue/releases/download/v0.1_arxiv/superpoint_lightglue.pth",
        "depth_anything_vitl14.pth": "https://huggingface.co/spaces/LiheYoung/Depth-Anything/resolve/main/checkpoints/depth_anything_vitl14.pth"
    }

    print("Ініціалізація завантаження нейромережевих моделей...")

    for filename, url in models_to_download.items():
        filepath = models_dir / filename

        if filepath.exists():
            print(f"Модель {filename} вже існує у {models_dir}. Пропускаємо.")
            continue

        print(f"\nЗавантаження {filename}...")
        try:
            download_url(url, filepath)
        except Exception as e:
            print(f"Критична помилка під час завантаження {filename}: {str(e)}")
            print("Будь ласка, перевірте з'єднання з інтернетом або завантажте файл вручну.")

    print("\nПроцес перевірки та завантаження моделей завершено!")


if __name__ == "__main__":
    download_models()