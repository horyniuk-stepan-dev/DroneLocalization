import cv2
import matplotlib.pyplot as plt
import numpy as np
import sys
import os

# Підключаємо ваш існуючий клас
from src.utils.image_preprocessor import ImagePreprocessor


def run(image_path, clip_limit=2.0):
    if not os.path.exists(image_path):
        print(f"Помилка: Файл {image_path} не знайдено.")
        return

    # Зчитуємо зображення через OpenCV (він читає у форматі BGR)
    img_bgr = cv2.imread(image_path)

    # Переводимо в RGB, оскільки ваш пайплайн працює з RGB
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # Ініціалізуємо ваш препроцесор з тестовим лімітом
    config = {'preprocessing': {'clahe_clip_limit': clip_limit, 'clahe_tile_size': (8, 8)}}
    preprocessor = ImagePreprocessor(config)

    # Застосовуємо покращення
    enhanced_rgb = preprocessor.preprocess(img_rgb)

    # Налаштовуємо відображення до і після через matplotlib
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    axes[0].imshow(img_rgb)
    axes[0].set_title("Оригінал (до обробки)", fontsize=14)
    axes[0].axis('off')

    axes[1].imshow(enhanced_rgb)
    axes[1].set_title(f"Після CLAHE (clip_limit={clip_limit})", fontsize=14)
    axes[1].axis('off')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Вкажіть тут шлях до будь-якого темного або контрастного кадру з вашого дрона
    TEST_IMAGE_PATH = "C:/Users/horyn/OneDrive/Desktop/big/Screenshot1.png"

    # Можете поекспериментувати з цим значенням (рекомендую від 1.5 до 4.0)
    TEST_CLIP_LIMIT = 4

    run(TEST_IMAGE_PATH, TEST_CLIP_LIMIT)