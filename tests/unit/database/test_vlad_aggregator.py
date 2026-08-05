"""RESEARCH 2.1: тести VladAggregator (чистий numpy/scipy — без GPU/torch)."""

import numpy as np
import pytest

from src.models.wrappers.vlad_aggregator import VladAggregator


def _make_tokens(rng, centers, n_tokens=64, noise=0.05):
    """Кадр = випадкові токени навколо заданих 'семантичних' центрів."""
    idx = rng.integers(0, len(centers), size=n_tokens)
    return (centers[idx] + rng.normal(0, noise, (n_tokens, centers.shape[1]))).astype(np.float32)


@pytest.fixture()
def fitted_agg():
    rng = np.random.default_rng(0)
    world = rng.normal(0, 1, (6, 16)).astype(np.float32)  # 6 "семантик" у 16D
    images = [_make_tokens(rng, world) for _ in range(40)]
    agg = VladAggregator(n_clusters=4, pca_dim=8, seed=1)
    agg.fit(images)
    return agg, world, rng


def test_out_dim_and_norm(fitted_agg):
    agg, world, rng = fitted_agg
    assert agg.out_dim == 8
    desc = agg.aggregate(_make_tokens(rng, world))
    assert desc.shape == (8,)
    assert desc.dtype == np.float32
    np.testing.assert_allclose(np.linalg.norm(desc), 1.0, atol=1e-5)


def test_similar_scenes_closer_than_different(fitted_agg):
    """Два кадри однієї 'сцени' ближчі, ніж кадри різних сцен."""
    agg, world, rng = fitted_agg
    scene_a = world[:3]  # сцена A — перші 3 семантики
    scene_b = world[3:]  # сцена B — інші 3
    a1 = agg.aggregate(_make_tokens(rng, scene_a))
    a2 = agg.aggregate(_make_tokens(rng, scene_a))
    b1 = agg.aggregate(_make_tokens(rng, scene_b))
    sim_aa = float(a1 @ a2)
    sim_ab = float(a1 @ b1)
    assert sim_aa > sim_ab + 0.2, f"same-scene {sim_aa:.3f} vs cross {sim_ab:.3f}"


def test_save_load_roundtrip(tmp_path, fitted_agg):
    agg, world, rng = fitted_agg
    tokens = _make_tokens(rng, world)
    ref = agg.aggregate(tokens)
    p = str(tmp_path / "vocab.npz")
    agg.save(p)
    loaded = VladAggregator.load(p)
    np.testing.assert_allclose(loaded.aggregate(tokens), ref, atol=1e-6)
    assert loaded.out_dim == agg.out_dim


def test_low_norm_filter_drops_weak_tokens():
    rng = np.random.default_rng(2)
    agg = VladAggregator(n_clusters=2, pca_dim=4, low_norm_fraction=0.5)
    strong = rng.normal(0, 1, (32, 8)).astype(np.float32) * 10
    weak = rng.normal(0, 1, (32, 8)).astype(np.float32) * 0.01
    mixed = np.concatenate([strong, weak])
    kept = agg._filter_low_norm(mixed)
    assert len(kept) == 32  # рівно половина
    assert np.linalg.norm(kept, axis=1).min() > 1.0  # слабкі викинуто


def test_pca_degrades_gracefully_with_few_images():
    """< pca_dim кадрів: PCA обрізається, а не падає."""
    rng = np.random.default_rng(3)
    world = rng.normal(0, 1, (4, 8)).astype(np.float32)
    images = [_make_tokens(rng, world, n_tokens=32) for _ in range(5)]
    agg = VladAggregator(n_clusters=2, pca_dim=64, seed=1)
    agg.fit(images)
    assert agg.out_dim == 4  # min(64, n_samples-1=4, K*D=16)
    d = agg.aggregate(_make_tokens(rng, world, n_tokens=32))
    assert d.shape == (4,)


def test_unfitted_raises():
    agg = VladAggregator()
    with pytest.raises(RuntimeError):
        agg.aggregate(np.zeros((10, 8), dtype=np.float32))
    with pytest.raises(RuntimeError):
        _ = agg.out_dim
