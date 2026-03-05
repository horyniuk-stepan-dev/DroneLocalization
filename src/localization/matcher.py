import numpy as np
import torch

from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


class FastRetrieval:
    """Fast candidate search using DINOv2 global descriptors"""

    def __init__(self, global_descriptors: np.ndarray):
        logger.info(f"Initializing FastRetrieval with {len(global_descriptors)} descriptors")
        self.global_descriptors = self.normalize_vectors(global_descriptors)
        logger.success("FastRetrieval initialized and descriptors normalized")

    @staticmethod
    def normalize_vectors(vectors: np.ndarray) -> np.ndarray:
        logger.debug(f"Normalizing {len(vectors)} vectors")
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        normalized = vectors / (norms + 1e-8)
        logger.debug("Vector normalization complete")
        return normalized

    def find_similar_frames(self, query_desc: np.ndarray, top_k: int = 5) -> list:
        logger.debug(f"Finding top-{top_k} similar frames...")

        query_norm = query_desc / (np.linalg.norm(query_desc) + 1e-8)

        # Матричне множення для швидкого пошуку косинусної схожості
        similarities = np.dot(self.global_descriptors, query_norm.T).flatten()
        top_indices = np.argsort(similarities)[::-1][:top_k]

        results = [(int(idx), float(similarities[idx])) for idx in top_indices]
        return results


from typing import Any


class FeatureMatcher:
    """Matches local keypoints (XFeat or SuperPoint+LightGlue)"""

    def __init__(self, model_manager: Any | None = None, config: dict | None = None):
    def __init__(self, model_manager: Any | None = None, config: dict | None = None):
        self.config = config or {}
        self.model_manager = model_manager
        self.ratio_threshold = self.config.get("localization", {}).get("ratio_threshold", 0.95)
        self.ratio_threshold = self.config.get("localization", {}).get("ratio_threshold", 0.95)

        # Намагаємося завантажити LightGlue, якщо він потрібен
        self.lightglue = None
        if self.model_manager and "lightglue" in self.model_manager.models:
            self.lightglue = self.model_manager.models["lightglue"]
        if self.model_manager and "lightglue" in self.model_manager.models:
            self.lightglue = self.model_manager.models["lightglue"]
            logger.info("FeatureMatcher configured to use LightGlue (if descriptor dims match)")
        else:
            logger.info("FeatureMatcher configured to use fast Numpy L2 matching (ideal for XFeat)")

    def match(self, query_features: dict, ref_features: dict) -> tuple:
        """
        Dynamically routes to LightGlue (for 256-dim SuperPoint)
        or Fast L2 Matcher (for 64-dim XFeat).
        """
        desc_dim = (
            query_features["descriptors"].shape[1] if len(query_features["descriptors"]) > 0 else 0
        )
        desc_dim = (
            query_features["descriptors"].shape[1] if len(query_features["descriptors"]) > 0 else 0
        )

        # Якщо є LightGlue і розмірність дескриптора 256 (SuperPoint)
        if self.lightglue is not None and desc_dim == 256:
            return self._lightglue_match(query_features, ref_features)

        # Для XFeat (64) або якщо LightGlue не завантажено
        return self._fast_numpy_match(query_features, ref_features, self.ratio_threshold)

    def _fast_numpy_match(
        self, query_features: dict, ref_features: dict, ratio_threshold: float = 0.80
    ) -> tuple:
    def _fast_numpy_match(
        self, query_features: dict, ref_features: dict, ratio_threshold: float = 0.80
    ) -> tuple:
        """
        Highly optimized L2 matching using dot product and Mutual Nearest Neighbor (MNN).
        """
        desc_q = query_features["descriptors"]
        desc_r = ref_features["descriptors"]
        kpts_q = query_features["keypoints"]
        kpts_r = ref_features["keypoints"]
        desc_q = query_features["descriptors"]
        desc_r = ref_features["descriptors"]
        kpts_q = query_features["keypoints"]
        kpts_r = ref_features["keypoints"]

        if len(desc_q) < 2 or len(desc_r) < 2:
            return np.empty((0, 2)), np.empty((0, 2))

        # 1. Нормалізація дескрипторів
        desc_q_n = desc_q / (np.linalg.norm(desc_q, axis=1, keepdims=True) + 1e-8)
        desc_r_n = desc_r / (np.linalg.norm(desc_r, axis=1, keepdims=True) + 1e-8)

        # 2. Розрахунок косинусної схожості через швидке матричне множення
        sim = np.dot(desc_q_n, desc_r_n.T)

        # 3. Lowe's Ratio Test — argpartition O(n) замість argsort O(n log n)
        # Потрібні лише top-2 для ratio test
        top2_idx = np.argpartition(-sim, kth=1, axis=1)[:, :2]
        top2_sim = np.take_along_axis(sim, top2_idx, axis=1)
        # Сортуємо лише 2 елементи щоб best >= second_best
        order = np.argsort(-top2_sim, axis=1)
        top2_idx = np.take_along_axis(top2_idx, order, axis=1)
        top2_sim = np.take_along_axis(top2_sim, order, axis=1)

        best_sim = top2_sim[:, 0]
        second_best_sim = top2_sim[:, 1]
        best_matches_indices = top2_idx[:, 0]

        # Переводимо схожість у L2-відстань: D = sqrt(2 - 2*sim)
        best_dist = np.sqrt(np.clip(2.0 - 2.0 * best_sim, 0, None))
        second_best_dist = np.sqrt(np.clip(2.0 - 2.0 * second_best_sim, 0, None))

        valid_ratio = (best_dist / (second_best_dist + 1e-8)) < ratio_threshold

        # 4. Mutual Nearest Neighbor (MNN) check
        reverse_best_indices = np.argmax(sim, axis=0)
        is_mutual = reverse_best_indices[best_matches_indices] == np.arange(len(desc_q))

        valid_matches = valid_ratio & is_mutual

        mkpts_q = kpts_q[valid_matches]
        mkpts_r = kpts_r[best_matches_indices[valid_matches]]

        return mkpts_q, mkpts_r

    def _lightglue_match(self, query_features: dict, ref_features: dict) -> tuple:
        """Matches features using Neural LightGlue Matcher"""
        try:
            device = next(self.lightglue.parameters()).device

            # Підготовка тензорів для LightGlue
            data = {
                "image0": {
                    "keypoints": torch.from_numpy(query_features["keypoints"])
                    .float()[None]
                    .to(device),
                    "descriptors": torch.from_numpy(query_features["descriptors"])
                    .float()[None]
                    .to(device),
                "image0": {
                    "keypoints": torch.from_numpy(query_features["keypoints"])
                    .float()[None]
                    .to(device),
                    "descriptors": torch.from_numpy(query_features["descriptors"])
                    .float()[None]
                    .to(device),
                },
                "image1": {
                    "keypoints": torch.from_numpy(ref_features["keypoints"])
                    .float()[None]
                    .to(device),
                    "descriptors": torch.from_numpy(ref_features["descriptors"])
                    .float()[None]
                    .to(device),
                },
                "image1": {
                    "keypoints": torch.from_numpy(ref_features["keypoints"])
                    .float()[None]
                    .to(device),
                    "descriptors": torch.from_numpy(ref_features["descriptors"])
                    .float()[None]
                    .to(device),
                },
            }

            with torch.no_grad():
                res = self.lightglue(data)

            matches = res["matches"][0].cpu().numpy()
            matches = res["matches"][0].cpu().numpy()

            if len(matches) == 0:
                return np.empty((0, 2)), np.empty((0, 2))

            m_q = matches[:, 0]
            m_r = matches[:, 1]

            mkpts_q = query_features["keypoints"][m_q]
            mkpts_r = ref_features["keypoints"][m_r]
            mkpts_q = query_features["keypoints"][m_q]
            mkpts_r = ref_features["keypoints"][m_r]

            return mkpts_q, mkpts_r

        except Exception as e:
            logger.error(f"LightGlue match failed: {e}")
            return np.empty((0, 2)), np.empty((0, 2))

