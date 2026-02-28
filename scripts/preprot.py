import cv2
import matplotlib.pyplot as plt
import numpy as np
import os

from src.utils.image_preprocessor import ImagePreprocessor


def run(source_path, reference_path):
    if not os.path.exists(source_path) or not os.path.exists(reference_path):
        print("Помилка: Один з файлів не знайдено.")
        return

    # Зчитуємо зображення
    src_bgr = cv2.imread(source_path)
    ref_bgr = cv2.imread(reference_path)

    # Переводимо в RGB
    src_rgb = cv2.cvtColor(src_bgr, cv2.COLOR_BGR2RGB)
    ref_rgb = cv2.cvtColor(ref_bgr, cv2.COLOR_BGR2RGB)

    # Конфіг для зведення гістограм
    config = {
        'preprocessing': {
            'histogram_matching': True,
            'reference_image_path': reference_path
        }
    }

    preprocessor = ImagePreprocessor(config)

    # Застосовуємо перенесення кольорів та освітлення
    matched_rgb = preprocessor.preprocess(src_rgb)

    # Візуалізація результату
    fig, axes = plt.subplots(1, 3, figsize=(20, 8))

    axes[0].imshow(src_rgb)
    axes[0].set_title("Оригінал (інший)", fontsize=14)
    axes[0].axis('off')

    axes[1].imshow(ref_rgb)
    axes[1].set_title("Еталон (сонячний)", fontsize=14)
    axes[1].axis('off')

    axes[2].imshow(matched_rgb)
    axes[2].set_title("Результат гістограм", fontsize=14)
    axes[2].axis('off')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Файл, який ми хочемо змінити (поточний кадр з камери)
    SOURCE_IMAGE = "C:/Users/horyn/OneDrive/Desktop/big/Screenshot1.png"

    # Файл, З ЯКОГО ми беремо ідеальні кольори (еталон з бази даних)
    REFERENCE_IMAGE = "C:/Users/horyn/OneDrive/Desktop/big/Screenshot.png"

    run(SOURCE_IMAGE, REFERENCE_IMAGE)