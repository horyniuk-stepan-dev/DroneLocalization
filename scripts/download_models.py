#!/usr/bin/env python3
"""
Скрипт для автоматичного завантаження ваг нейромереж
"""

import sys
import urllib.request
from pathlib import Path

try:
    from tqdm import tqdm
except ImportError:
    print("Встановлення tqdm для відображення прогресу...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "tqdm"])
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
    # Створюємо папку якщо її немає
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with DownloadProgressBar(unit="B", unit_scale=True, miniters=1, desc=output_path.name) as t:
        urllib.request.urlretrieve(url, filename=output_path, reporthook=t.update_to)


def download_models():
    """Завантаження всіх необхідних моделей у папку models/"""
    models_dir = Path("models")
    models_dir.mkdir(parents=True, exist_ok=True)

    # Список моделей з прямими посиланнями
    models_to_download = {
        "yolo11n-seg.pt": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n-seg.pt",
        "depth_anything_v2_vits.pth": "https://huggingface.co/depth-anything/Depth-Anything-V2-Small/resolve/main/depth_anything_v2_vits.pth",
        # RDD та LightGlue ваги зазвичай завантажуються автоматично або потребують Google Drive
    }

    print(f"Ініціалізація завантаження нейромережевих моделей у {models_dir.absolute()}...")

    for filename, url in models_to_download.items():
        filepath = models_dir / filename

        if filepath.exists() and filepath.stat().st_size > 1000:
            print(f"Модель {filename} вже існує. Пропускаємо.")
            continue

        print(f"\nЗавантаження {filename}...")
        try:
            download_url(url, filepath)
        except Exception as e:
            print(f"Критична помилка під час завантаження {filename}: {str(e)}")
            print("Будь ласка, перевірте з'єднання з інтернетом або завантажте файл вручну.")

    print("\nПроцес перевірки та завантаження моделей завершено!")
    print("Примітка: для моделей RDD та CESP може знадобитися ручне завантаження з Google Drive (див. документацію).")


if __name__ == "__main__":
    download_models()
