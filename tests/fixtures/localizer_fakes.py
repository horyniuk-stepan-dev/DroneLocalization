"""Pure-numpy fakes for characterizing Localizer without GPU/torch models.

They implement the structural contracts in ``src.interfaces`` so a Localizer can
be driven deterministically. Import is torch-free (usable in any environment);
the Localizer that consumes them still imports torch/faiss, so the actual
characterization test runs on a full install. See docs/REFACTOR_LOCALIZER_PLAN.md.
"""

from __future__ import annotations

import hashlib
from typing import Any

import numpy as np
from numpy.typing import NDArray

_DIM = 32


def _seeded(image: NDArray[Any], n: int, cols: int) -> NDArray[np.float64]:
    """Deterministic ``(n, cols)`` array keyed by the image bytes.

    Uses a stable hash (blake2b): Python's builtin ``hash()`` is per-process
    randomised (PYTHONHASHSEED), which would make snapshots non-reproducible.
    """
    seed = int.from_bytes(hashlib.blake2b(image.tobytes(), digest_size=8).digest(), "little")
    rng = np.random.default_rng(seed % (2**32))
    return rng.standard_normal((n, cols)).astype(np.float64)


class FakeGlobalExtractor:
    """Deterministic unit global descriptors (GlobalDescriptorExtractor)."""

    def __init__(self, dim: int = _DIM) -> None:
        self.dim = dim
        self.calls = 0            # inspect to assert A3-prior avoids full scan

    def extract_global_descriptor(self, image: NDArray[Any]) -> NDArray[np.float64]:
        self.calls += 1
        v = _seeded(image, 1, self.dim)[0]
        return v / (np.linalg.norm(v) + 1e-12)

    def extract_global_descriptors_multi(
        self, images: list[NDArray[Any]]
    ) -> NDArray[np.float64]:
        self.calls += 1          # one batched forward pass
        return np.stack([self._one(im) for im in images])

    def _one(self, image: NDArray[Any]) -> NDArray[np.float64]:
        v = _seeded(image, 1, self.dim)[0]
        return v / (np.linalg.norm(v) + 1e-12)


class FakeLocalExtractor:
    """Grid keypoints + descriptors (LocalFeatureExtractor)."""

    def __init__(self, grid: int = 5) -> None:
        self.grid = grid

    def extract_local_features(
        self, image: NDArray[Any], static_mask: NDArray[Any] | None = None
    ) -> dict[str, Any]:
        h, w = image.shape[:2]
        xs = np.linspace(w * 0.2, w * 0.8, self.grid)
        ys = np.linspace(h * 0.2, h * 0.8, self.grid)
        kpts = np.array([[x, y] for y in ys for x in xs], dtype=np.float32)
        return {
            "keypoints": kpts,
            "descriptors": _seeded(image, len(kpts), _DIM).astype(np.float32),
            "scores": np.ones(len(kpts), dtype=np.float32),
        }


class FakeMatcher:
    """Identity correspondence -> valid homography with many inliers."""

    def match(
        self, query_features: dict[str, Any], ref_features: dict[str, Any]
    ) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
        q = np.asarray(query_features["keypoints"], dtype=np.float32)
        r = np.asarray(ref_features["keypoints"], dtype=np.float32)
        n = min(len(q), len(r))
        return q[:n], r[:n]


class _FakeConverter:
    def metric_to_gps(self, x: float, y: float) -> tuple[float, float]:
        return (50.0 + y * 1e-5, 30.0 + x * 1e-5)


class FakeCalibration:
    def __init__(self) -> None:
        self.converter = _FakeConverter()

    def set_gsd_calculator(self, gsd: Any) -> None:  # noqa: D401 - no-op
        pass


class FakeDatabase:
    """In-memory keyframe DB (FrameDatabase) + attrs Localizer reads directly."""

    def __init__(self, num_frames: int = 8, dim: int = _DIM,
                 frame_w: int = 640, frame_h: int = 480) -> None:
        self._n = num_frames
        self.frame_w, self.frame_h = frame_w, frame_h
        rng = np.random.default_rng(0)
        g = rng.standard_normal((num_frames, dim)).astype(np.float32)
        self.global_descriptors = g / (np.linalg.norm(g, axis=1, keepdims=True) + 1e-12)
        self.lance_table = None            # -> Localizer uses FastRetrieval
        self.frame_rmse = None             # -> confidence uses 0.0 branch
        self.frame_disagreement = None
        self._local = FakeLocalExtractor()

    def get_num_frames(self) -> int:
        return self._n

    def get_frame_size(self, frame_id: int) -> tuple[int, int]:
        return (self.frame_h, self.frame_w)

    def get_frame_affine(self, frame_id: int) -> NDArray[np.float64] | None:
        # simple translation per frame (valid 2x3 affine)
        return np.array([[1.0, 0.0, 10.0 * frame_id], [0.0, 1.0, 5.0 * frame_id]])

    def get_local_features(self, frame_id: int) -> dict[str, NDArray[np.float64]]:
        img = np.full((self.frame_h, self.frame_w, 3), frame_id % 255, dtype=np.uint8)
        return self._local.extract_local_features(img)


def synthetic_frame(seed: int, w: int = 640, h: int = 480) -> NDArray[np.uint8]:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 255, size=(h, w, 3), dtype=np.uint8)


def build_localizer(config: dict[str, Any] | None = None) -> Any:
    """Construct a Localizer wired with fakes. Imports Localizer lazily so this
    module stays torch-free; call this only where torch/faiss are installed."""
    from src.localization.localizer import Localizer  # noqa: PLC0415 (lazy by design)
    db = FakeDatabase()
    return Localizer(
        database=db,
        feature_extractor=_CombinedExtractor(),
        matcher=FakeMatcher(),
        calibration=FakeCalibration(),
        config=config or {},
        ref_frame_width=db.frame_w,
        ref_frame_height=db.frame_h,
    )


class _CombinedExtractor(FakeGlobalExtractor, FakeLocalExtractor):
    """Localizer's ``feature_extractor`` is used for BOTH global and local."""

    def __init__(self) -> None:
        FakeGlobalExtractor.__init__(self)
        FakeLocalExtractor.__init__(self)
