"""Тести CPU-resize перед входом DINO (аудит §2.1, повна форма).

Кадр цілком їхав на GPU, щоб там зменшитись до (S, S) — для DINOv3 це 224.
Прапорець ``models.performance.dino_cpu_resize`` переносить зменшення на CPU і
на uint8, тобто по PCIe їде ~S²·3 байт замість H·W·3.

cv2.INTER_AREA і torchvision ``Resize(antialias=True)`` — РІЗНІ фільтри, отже
значення дескрипторів зміщуються. Тому головне, що тут перевіряється, — це не
швидкість, а те, що прапорець ЗМІНЮЄ schema fingerprint: база, збудована з
іншим значенням, має детектуватись як несумісна, а не тихо псувати матчі.

FeatureExtractor імпортує torch, тож самий клас у пісочниці не піднімається —
логіка resize тестується на чистій функції-двійнику з тією самою реалізацією.
"""

import cv2
import numpy as np
import pytest

from src.database import schema_fingerprint as sf

# ── Fingerprint: прапорець мусить розрізняти бази ────────────────────────────


def _components(**over):
    cfg = {
        "global_descriptor": {"backend": "dinov3"},
        "models": {
            "vlad": {"enabled": False, "pca_dim": 512},
            "local_extractor": "aliked",
            "performance": {"dino_cpu_resize": False},
        },
        "database": {
            "max_keypoints_stored": 2048,
            "keypoint_video_scale": 0.5,
            "frame_step": 30,
            "store_sift_features": False,
            "sift_max_keypoints": 2048,
        },
    }
    if "dino_cpu_resize" in over:
        cfg["models"]["performance"]["dino_cpu_resize"] = over["dino_cpu_resize"]
    return sf.build_components(cfg, descriptor_dim=1024, local_descriptor_dim=128)


class TestSchemaFingerprint:
    def test_flag_is_part_of_the_schema(self):
        assert "dino_cpu_resize" in sf.SCHEMA_FIELDS

    def test_flag_is_collected_from_config(self):
        assert _components(dino_cpu_resize=False)["dino_cpu_resize"] is False
        assert _components(dino_cpu_resize=True)["dino_cpu_resize"] is True

    def test_databases_with_different_flag_are_not_interchangeable(self):
        """Головна властивість: інший прапорець → інший хеш → база відхиляється."""
        a = _components(dino_cpu_resize=False)
        b = _components(dino_cpu_resize=True)
        assert sf.compute_fingerprint(a) != sf.compute_fingerprint(b)
        assert any("dino_cpu_resize" in d for d in sf.compare(a, b))

    def test_same_flag_gives_same_fingerprint(self):
        assert sf.compute_fingerprint(_components(dino_cpu_resize=True)) == (
            sf.compute_fingerprint(_components(dino_cpu_resize=True))
        )

    def test_default_is_off_so_existing_fingerprints_are_unaffected(self):
        """Дефолт має збігатися зі значенням, яке дає відсутність ключа."""
        no_key = sf.build_components(
            {"global_descriptor": {"backend": "dinov3"}},
            descriptor_dim=1024,
            local_descriptor_dim=128,
        )
        assert no_key["dino_cpu_resize"] is False


# ── Логіка самого resize (двійник реалізації FeatureExtractor) ───────────────


def _cpu_resize_dino(image: np.ndarray, dino_size: int) -> np.ndarray:
    """Копія FeatureExtractor._cpu_resize_dino без torch-залежності."""
    s = int(dino_size)
    h, w = image.shape[:2]
    interp = cv2.INTER_AREA if (h > s or w > s) else cv2.INTER_CUBIC
    return cv2.resize(np.ascontiguousarray(image), (s, s), interpolation=interp)


class TestCpuResize:
    @pytest.mark.parametrize(
        "shape", [(1080, 1920, 3), (2160, 3840, 3), (224, 224, 3), (100, 150, 3)]
    )
    def test_output_shape_is_square_target(self, shape):
        img = np.random.default_rng(0).integers(0, 256, shape, dtype=np.uint8)
        out = _cpu_resize_dino(img, 224)
        assert out.shape == (224, 224, 3)

    def test_output_stays_uint8(self):
        """Ключова властивість: на девайс має їхати uint8, а не float32."""
        img = np.random.default_rng(1).integers(0, 256, (1080, 1920, 3), dtype=np.uint8)
        assert _cpu_resize_dino(img, 224).dtype == np.uint8

    def test_aspect_ratio_is_deliberately_not_preserved(self):
        """Форма квадратна — точно як у T.Resize((S, S)), який цей шлях заміщає.

        Якби ми зберігали аспект, геометрія входу розійшлася б із GPU-варіантом
        і дескриптори БД/запиту стали б непорівнянними.
        """
        img = np.random.default_rng(2).integers(0, 256, (100, 900, 3), dtype=np.uint8)
        assert _cpu_resize_dino(img, 224).shape[:2] == (224, 224)

    def test_downscale_uses_area_upscale_uses_cubic(self):
        """Дзеркалить ResolutionNormalizer: AREA на зменшення, CUBIC на збільшення."""
        big = np.random.default_rng(3).integers(0, 256, (1080, 1920, 3), dtype=np.uint8)
        small = np.random.default_rng(4).integers(0, 256, (64, 64, 3), dtype=np.uint8)
        assert np.array_equal(
            _cpu_resize_dino(big, 224),
            cv2.resize(big, (224, 224), interpolation=cv2.INTER_AREA),
        )
        assert np.array_equal(
            _cpu_resize_dino(small, 224),
            cv2.resize(small, (224, 224), interpolation=cv2.INTER_CUBIC),
        )

    def test_is_deterministic(self):
        img = np.random.default_rng(5).integers(0, 256, (1080, 1920, 3), dtype=np.uint8)
        assert np.array_equal(_cpu_resize_dino(img, 224), _cpu_resize_dino(img, 224))

    def test_transfer_volume_actually_drops(self):
        """Кількісно: те, заради чого все це робиться."""
        img = np.random.default_rng(6).integers(0, 256, (1080, 1920, 3), dtype=np.uint8)
        before = img.nbytes  # uint8, повний кадр
        after = _cpu_resize_dino(img, 224).nbytes
        assert after < before / 30, f"очікували ≥30× менше, отримали {before / after:.1f}×"

    def test_differs_from_gpu_filter_which_is_why_the_flag_is_in_the_schema(self):
        """AREA ≠ bilinear-antialias — саме тому бази несумісні між режимами."""
        img = np.random.default_rng(7).integers(0, 256, (1080, 1920, 3), dtype=np.uint8)
        area = _cpu_resize_dino(img, 224).astype(np.float32)
        linear = cv2.resize(img, (224, 224), interpolation=cv2.INTER_LINEAR).astype(np.float32)
        assert not np.allclose(area, linear, atol=1e-3)
