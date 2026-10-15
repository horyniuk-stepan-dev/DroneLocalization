"""PatchifyRetrieval: batched descriptors equal the sequential ones; score scales.

The batched path used to stack crops of different sizes (RuntimeError) and call
the backbone directly (CLS instead of VLAD, no dino_cpu_resize). It now goes
through FeatureExtractor.extract_global_descriptors_multi.
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")

from src.localization.patchify import PatchifyRetrieval  # noqa: E402

GRIDS = [[1, 1], [2, 2], [3, 3]]
TOKEN_DIM = 8


class TinyBackbone(torch.nn.Module):
    """DINO-shaped stand-in: 16 px patches -> tokens; CLS = mean token."""

    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        self.embed = torch.nn.Conv2d(3, TOKEN_DIM, kernel_size=16, stride=16)

    def forward_features(self, x, layer=None):
        tokens = self.embed(x).flatten(2).transpose(1, 2)
        return {"x_norm_clstoken": tokens.mean(1), "x_norm_patchtokens": tokens}

    def forward(self, x):
        return self.forward_features(x)["x_norm_clstoken"]


def _vlad(k=3):
    from src.models.wrappers.vlad_aggregator import VladAggregator

    rng = np.random.default_rng(1)
    agg = VladAggregator(n_clusters=k, pca_dim=4)
    agg.centers = rng.standard_normal((k, TOKEN_DIM)).astype(np.float32)
    return agg  # no PCA: out_dim = k * TOKEN_DIM


def _extractor(cpu_resize: bool, batch_max: int = 0, vlad: bool = False):
    from src.models.wrappers.feature_extractor import FeatureExtractor

    config = {
        "global_descriptor": {"backend": "dinov3", "dinov3": {"input_size": 64}},
        "models": {
            "performance": {
                "dino_cpu_resize": cpu_resize,
                "fp16_enabled": False,
                "global_batch_max": batch_max,
            }
        },
    }
    fe = FeatureExtractor(None, TinyBackbone().eval(), device="cpu", config=config)
    if vlad:
        fe.vlad_aggregator = _vlad()
    return fe


def _image(h=193, w=211):
    # Odd size: the 2x2 and 3x3 crops differ in size (remainder goes to the last).
    return np.random.default_rng(2).integers(0, 255, (h, w, 3), dtype=np.uint8)


@pytest.mark.parametrize(
    "cpu_resize,vlad,batch_max",
    [(True, True, 0), (False, True, 0), (True, False, 0), (True, True, 4), (False, False, 3)],
)
@pytest.mark.parametrize("batch_size", [2, 4, 64])
def test_batched_descriptors_equal_sequential(cpu_resize, vlad, batch_max, batch_size):
    fe = _extractor(cpu_resize, batch_max, vlad)
    dim = int(fe.global_descriptor_dim) if vlad else TOKEN_DIM
    image = _image()

    sequential = PatchifyRetrieval(fe, descriptor_dim=dim, grids=GRIDS, batch_size=1)
    batched = PatchifyRetrieval(fe, descriptor_dim=dim, grids=GRIDS, batch_size=batch_size)

    a = sequential.compute_patch_descriptors(image)
    b = batched.compute_patch_descriptors(image)
    assert a.shape == b.shape == (14, dim)
    np.testing.assert_allclose(b, a, rtol=1e-5, atol=1e-5)


class _NoMultiExtractor:
    def __init__(self, dim):
        self.dim = dim
        self.calls = 0

    def extract_global_descriptor(self, image):
        self.calls += 1
        return np.full(self.dim, image.shape[0] + image.shape[1], dtype=np.float32)


def test_extractor_without_multi_falls_back_to_single_calls():
    fe = _NoMultiExtractor(dim=5)
    batched = PatchifyRetrieval(fe, descriptor_dim=5, grids=GRIDS, batch_size=8)
    out = batched.compute_patch_descriptors(_image())
    assert out.shape == (14, 5)
    assert fe.calls == 14


class _WrongDimExtractor(_NoMultiExtractor):
    def extract_global_descriptors_multi(self, images, valid_masks=None):
        return np.zeros((len(images), self.dim * 4), dtype=np.float32)


def test_wrong_batched_dimension_is_reported():
    batched = PatchifyRetrieval(
        _WrongDimExtractor(dim=5), descriptor_dim=5, grids=GRIDS, batch_size=4
    )
    with pytest.raises(ValueError, match="expected"):
        batched.compute_patch_descriptors(_image())


# -- scoring (localization.patchify_unique_coverage) -------------------------


def _unit(i, dim=6):
    v = np.zeros(dim, dtype=np.float32)
    v[i] = 1.0
    return v


def _index(unique: bool, frames: list[np.ndarray]) -> PatchifyRetrieval:
    pytest.importorskip("faiss")
    retrieval = PatchifyRetrieval(object(), descriptor_dim=6, grids=GRIDS, unique_coverage=unique)
    retrieval.build_index(np.stack(frames), list(range(len(frames))))
    return retrieval


def test_legacy_score_of_identical_frame_exceeds_one():
    # Old behaviour, kept by default: every reference-patch hit counts.
    frames = [np.stack([_unit(0)] * 14), np.stack([_unit(1)] * 14)]
    query = np.stack([_unit(0)] * 14)
    best = dict(_index(False, frames).search(query, top_k=2))
    assert best[0] == pytest.approx(6.0)  # search_k = 6 hits of 1.0 per query patch


def test_unique_coverage_scores_identical_frame_one():
    frames = [np.stack([_unit(0)] * 14), np.stack([_unit(1)] * 14)]
    query = np.stack([_unit(0)] * 14)
    best = dict(_index(True, frames).search(query, top_k=2))
    assert best[0] == pytest.approx(1.0)


def test_unique_coverage_is_the_share_of_query_patches():
    frames = [np.stack([_unit(0)] * 14), np.stack([_unit(2)] * 14)]
    query = np.stack([_unit(0)] * 7 + [_unit(2)] * 7)
    scores = dict(_index(True, frames).search(query, top_k=2))
    assert scores[0] == pytest.approx(0.5)
    assert scores[1] == pytest.approx(0.5)
    assert max(scores.values()) <= 1.0 + 1e-6
