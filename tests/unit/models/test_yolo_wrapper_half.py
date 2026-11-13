"""YOLOWrapper half precision follows models.performance.fp16_enabled."""

import pytest

torch = pytest.importorskip("torch")

from src.models.wrappers import yolo_wrapper  # noqa: E402
from src.models.wrappers.yolo_wrapper import YOLOWrapper  # noqa: E402


@pytest.fixture
def cuda_on(monkeypatch):
    monkeypatch.setattr(yolo_wrapper.torch.cuda, "is_available", lambda: True)


def test_explicit_half_flag(cuda_on):
    assert YOLOWrapper(object(), "cuda", half=True).use_half is True
    assert YOLOWrapper(object(), "cuda", half=False).use_half is False


def test_cpu_never_half(cuda_on):
    assert YOLOWrapper(object(), "cpu", half=True).use_half is False


@pytest.mark.parametrize("flag", [True, False])
def test_default_reads_config(cuda_on, monkeypatch, flag):
    import config

    cfg = {"models": {"performance": {"fp16_enabled": flag}}}
    monkeypatch.setattr(config, "APP_CONFIG", cfg)
    assert YOLOWrapper(object(), "cuda").use_half is flag
