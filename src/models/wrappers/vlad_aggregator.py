"""RESEARCH 2.1 (AnyLoc, arXiv:2308.00688): ненавчена VLAD-агрегація патч-токенів.

Замінює CLS-токен глобального дескриптора на VLAD поверх патч-токенів
фундаментальної моделі: словник — k-means по токенах референсних кадрів,
дескриптор — конкатенація нормованих кластерних залишків + PCA-whitening.

Модуль свідомо без torch: fit/aggregate працюють на numpy (k-means — faiss,
якщо доступний, інакше scipy), тому юніт-тестується без GPU і вантажиться
у DatabaseBuilder / FeatureExtractor без додаткових залежностей.

Пайплайн:
    offline (scripts/build_vlad_vocab.py):
        tokens_per_image = DINOv3.forward_features(...)  # (N, D) кожен
        agg = VladAggregator(n_clusters=32, pca_dim=512)
        agg.fit(list_of_tokens)
        agg.save("vlad_vocab.npz")
    online (FeatureExtractor):
        agg = VladAggregator.load("vlad_vocab.npz")
        desc = agg.aggregate(tokens)  # (out_dim,), L2-нормований
"""

from __future__ import annotations

import numpy as np

from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


class VladAggregator:
    """VLAD з жорстким призначенням + intra-нормалізація + PCA-whitening."""

    def __init__(
        self,
        n_clusters: int = 32,
        pca_dim: int = 512,
        low_norm_fraction: float = 0.0,
        seed: int = 42,
    ) -> None:
        self.n_clusters = int(n_clusters)
        self.pca_dim = int(pca_dim)
        self.low_norm_fraction = float(low_norm_fraction)
        self.seed = int(seed)

        self.centers: np.ndarray | None = None      # (K, D)
        self.pca_mean: np.ndarray | None = None      # (K*D,)
        self.pca_components: np.ndarray | None = None  # (pca_dim, K*D)
        self.pca_eigvals: np.ndarray | None = None   # (pca_dim,)

    # ── Властивості ──────────────────────────────────────────────────────

    @property
    def is_fitted(self) -> bool:
        return self.centers is not None

    @property
    def out_dim(self) -> int:
        """Розмірність фінального дескриптора."""
        if self.centers is None:
            raise RuntimeError("VladAggregator is not fitted")
        if self.pca_components is not None:
            return int(self.pca_components.shape[0])
        return int(self.n_clusters * self.centers.shape[1])

    # ── Fit ──────────────────────────────────────────────────────────────

    def fit(
        self,
        tokens_per_image: list[np.ndarray],
        max_kmeans_tokens: int = 200_000,
    ) -> VladAggregator:
        """Будує словник (k-means) і PCA-whitening по референсних кадрах.

        Args:
            tokens_per_image: список (N_i, D) патч-токенів окремих кадрів.
            max_kmeans_tokens: стеля вибірки токенів для k-means (пам'ять).
        """
        if len(tokens_per_image) < 2:
            raise ValueError("fit() needs at least 2 images of tokens")

        rng = np.random.default_rng(self.seed)
        stacked = np.concatenate(
            [np.asarray(t, dtype=np.float32) for t in tokens_per_image], axis=0
        )
        if len(stacked) > max_kmeans_tokens:
            idx = rng.choice(len(stacked), size=max_kmeans_tokens, replace=False)
            stacked = stacked[idx]

        self.centers = self._kmeans(stacked)
        logger.info(
            f"VLAD vocabulary: k-means done | k={self.n_clusters}, "
            f"tokens={len(stacked)}, dim={stacked.shape[1]}"
        )

        # PCA-whitening по VLAD-векторах референсних кадрів (як в AnyLoc).
        vlads = np.stack([self._vlad(t) for t in tokens_per_image])  # (M, K*D)
        n_samples, full_dim = vlads.shape
        eff_dim = min(self.pca_dim, n_samples - 1, full_dim)
        if eff_dim < self.pca_dim:
            logger.warning(
                f"PCA dim reduced {self.pca_dim} → {eff_dim}: "
                f"only {n_samples} reference images (need ≥ pca_dim+1 for full rank)"
            )
        if eff_dim < 2:
            logger.warning("PCA disabled: not enough reference images")
            self.pca_mean = None
            self.pca_components = None
            self.pca_eigvals = None
            return self

        self.pca_mean = vlads.mean(axis=0)
        centered = vlads - self.pca_mean
        # SVD економного розміру: components — праві сингулярні вектори
        _, s, vt = np.linalg.svd(centered, full_matrices=False)
        self.pca_components = vt[:eff_dim].astype(np.float32)
        # Дисперсія компонент; epsilon від ділення на ~0 для хвостових компонент
        self.pca_eigvals = (s[:eff_dim] ** 2 / max(n_samples - 1, 1)).astype(np.float32)
        logger.info(f"VLAD PCA-whitening fitted: {full_dim} → {eff_dim}")
        return self

    def _kmeans(self, tokens: np.ndarray) -> np.ndarray:
        """k-means: faiss (швидко, GPU-able) з фолбеком на scipy."""
        try:
            import faiss

            km = faiss.Kmeans(
                d=int(tokens.shape[1]),
                k=self.n_clusters,
                niter=25,
                seed=self.seed,
                verbose=False,
            )
            km.train(np.ascontiguousarray(tokens, dtype=np.float32))
            return km.centroids.reshape(self.n_clusters, -1).astype(np.float32)
        except ImportError:
            logger.info("faiss not available — falling back to scipy kmeans2")
            from scipy.cluster.vq import kmeans2

            centers, _ = kmeans2(
                tokens.astype(np.float64),
                self.n_clusters,
                minit="++",
                seed=self.seed,
            )
            return centers.astype(np.float32)

    # ── Aggregate ────────────────────────────────────────────────────────

    def _filter_low_norm(self, tokens: np.ndarray) -> np.ndarray:
        """Dustbin-сурогат (SALAD): відкидає частку токенів з найнижчою нормою
        (небо, однорідні поверхні несуть мало просторової інформації)."""
        if self.low_norm_fraction <= 0.0 or len(tokens) < 8:
            return tokens
        norms = np.linalg.norm(tokens, axis=1)
        thresh = np.quantile(norms, self.low_norm_fraction)
        kept = tokens[norms > thresh]
        return kept if len(kept) >= 4 else tokens

    def _vlad(self, tokens: np.ndarray) -> np.ndarray:
        """VLAD-вектор без PCA: (K*D,) з intra- та глобальною L2-нормалізацією."""
        if self.centers is None:
            raise RuntimeError("VladAggregator is not fitted")
        t = self._filter_low_norm(np.asarray(tokens, dtype=np.float32))
        k, d = self.centers.shape

        # Жорстке призначення до найближчого центру: argmin ||t - c||²
        # через розклад (економія пам'яті проти повної матриці відстаней)
        dots = t @ self.centers.T                      # (N, K)
        c_sq = np.sum(self.centers**2, axis=1)         # (K,)
        assign = np.argmax(dots - 0.5 * c_sq, axis=1)  # (N,)

        vlad = np.zeros((k, d), dtype=np.float32)
        for ci in range(k):
            sel = t[assign == ci]
            if len(sel):
                vlad[ci] = (sel - self.centers[ci]).sum(axis=0)

        # Intra-нормалізація (по кластеру) — пригнічує burstiness
        norms = np.linalg.norm(vlad, axis=1, keepdims=True)
        np.divide(vlad, norms, out=vlad, where=norms > 1e-12)

        flat = vlad.reshape(-1)
        n = np.linalg.norm(flat)
        return flat / n if n > 1e-12 else flat

    def aggregate(self, tokens: np.ndarray) -> np.ndarray:
        """Патч-токени (N, D) → глобальний дескриптор (out_dim,), L2-норм."""
        v = self._vlad(tokens)
        if self.pca_components is not None:
            v = (v - self.pca_mean) @ self.pca_components.T
            v = v / np.sqrt(self.pca_eigvals + 1e-8)  # whitening
            n = np.linalg.norm(v)
            if n > 1e-12:
                v = v / n
        return v.astype(np.float32)

    def aggregate_batch(self, tokens_batch: np.ndarray | list[np.ndarray]) -> np.ndarray:
        """(B, N, D) або список (N_i, D) → (B, out_dim)."""
        return np.stack([self.aggregate(t) for t in tokens_batch])

    # ── Persistence ──────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        if self.centers is None:
            raise RuntimeError("Nothing to save: not fitted")
        np.savez_compressed(
            path,
            centers=self.centers,
            pca_mean=self.pca_mean if self.pca_mean is not None else np.empty(0),
            pca_components=(
                self.pca_components if self.pca_components is not None else np.empty((0, 0))
            ),
            pca_eigvals=self.pca_eigvals if self.pca_eigvals is not None else np.empty(0),
            meta=np.array(
                [self.n_clusters, self.pca_dim, self.seed], dtype=np.int64
            ),
            low_norm_fraction=np.float64(self.low_norm_fraction),
        )
        logger.info(f"VLAD vocabulary saved: {path} (out_dim={self.out_dim})")

    @classmethod
    def load(cls, path: str, low_norm_fraction: float | None = None) -> VladAggregator:
        data = np.load(path, allow_pickle=False)
        n_clusters, pca_dim, seed = (int(x) for x in data["meta"])
        lnf = (
            float(data["low_norm_fraction"])
            if low_norm_fraction is None
            else float(low_norm_fraction)
        )
        agg = cls(n_clusters=n_clusters, pca_dim=pca_dim, low_norm_fraction=lnf, seed=seed)
        agg.centers = data["centers"].astype(np.float32)
        if data["pca_components"].size:
            agg.pca_mean = data["pca_mean"].astype(np.float32)
            agg.pca_components = data["pca_components"].astype(np.float32)
            agg.pca_eigvals = data["pca_eigvals"].astype(np.float32)
        logger.info(
            f"VLAD vocabulary loaded: {path} | k={n_clusters}, out_dim={agg.out_dim}"
        )
        return agg
