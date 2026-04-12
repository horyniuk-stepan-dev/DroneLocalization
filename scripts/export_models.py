#!/usr/bin/env python3
"""
Скрипт для примусового експорту моделей у TorchScript (.pth) або TensorRT (.engine).
Це дозволяє підготувати систему до роботи без залежності від вихідних бібліотек (наприклад, LightGlue).
"""

import sys
import os
from pathlib import Path
import torch

# Додаємо корінь проєкту до path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config.config import APP_SETTINGS
from src.models.model_manager import ModelManager
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

def export_all_models():
    """Експортує всі підтримувані моделі."""
    logger.info("Starting manual model export sequence...")
    
    # Ініціалізуємо менеджер моделей з дефолтним конфігом
    manager = ModelManager(config=APP_SETTINGS)
    
    # 1. LightGlue (ALIKED)
    logger.info("--- Exporting LightGlue (ALIKED) ---")
    try:
        # Примусово завантажуємо через бібліотеку (backend='git')
        # Для цього тимчасово змінимо конфіг
        original_backend = APP_SETTINGS.models.lightglue.backend
        APP_SETTINGS.models.lightglue.backend = "git"
        APP_SETTINGS.models.lightglue.auto_convert = True # Щоб спрацював _auto_export
        
        manager.load_lightglue(features="aliked")
        
        # Повертаємо бекенд
        APP_SETTINGS.models.lightglue.backend = original_backend
    except Exception as e:
        logger.error(f"Failed to export LightGlue ALIKED: {e}")

    # 2. LightGlue (SuperPoint)
    logger.info("--- Exporting LightGlue (SuperPoint) ---")
    try:
        original_backend = APP_SETTINGS.models.lightglue_superpoint.backend
        APP_SETTINGS.models.lightglue_superpoint.backend = "git"
        APP_SETTINGS.models.lightglue_superpoint.auto_convert = True
        
        manager.load_lightglue(features="superpoint")
        
        APP_SETTINGS.models.lightglue_superpoint.backend = original_backend
    except Exception as e:
        logger.error(f"Failed to export LightGlue SuperPoint: {e}")

    # 3. YOLO (уже має вбудований експорт в Ultralytics, але ми можемо його тригернути)
    logger.info("--- Exporting YOLO to TensorRT ---")
    try:
        if APP_SETTINGS.models.performance.use_tensorrt_for_yolo:
            manager.load_yolo()
    except Exception as e:
        logger.error(f"Failed to export YOLO: {e}")

    logger.success("Model export sequence complete!")

if __name__ == "__main__":
    if torch.cuda.is_available():
        export_all_models()
    else:
        logger.error("CUDA is required for model export (to ensure FP16 and TensorRT compatibility).")
