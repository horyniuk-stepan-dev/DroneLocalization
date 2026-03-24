import faiss
import numpy as np
import torch

from config.config import get_cfg
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


class FastRetrieval:
    """Fast candidate search using DINOv2 global descriptors (optimized with FAISS)"""

    def __init__(self, global_descriptors: np.ndarray):
        logger.info(
            f"Initializing FastRetrieval with {len(global_descriptors)} descriptors using FAISS"
        )
        self.dim = global_descriptors.shape[1]

        # Inner Product index (для косинусної схожості нормалізованих векторів)
        self.index = faiss.IndexFlatIP(self.dim)

        # Нормалізуємо і додаємо в індекс
        normed = self.normalize_vectors(global_descriptors)
        self.index.add(normed.astype(np.float32))

        logger.success(f"FAISS index built with {self.index.ntotal} vectors")

    @staticmethod
    def normalize_vectors(vectors: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        return vectors / (norms + 1e-8)

    def find_similar_frames(self, query_desc: np.ndarray, top_k: int = 5) -> list:
        # Підготовка запиту
        q = query_desc / (np.linalg.norm(query_desc) + 1e-8)
        q = q.astype(np.float32)

        if q.ndim == 1:
            q = q[None]

        # Пошук у FAISS
        scores, ids = self.index.search(q, top_k)

        # Повертаємо список (id, score)
        results = [(int(idx), float(score)) for idx, score in zip(ids[0], scores[0]) if idx != -1]
        return results


class FeatureMatcher:
    """Matches local keypoints (XFeat or SuperPoint+LightGlue)"""

    def __init__(self, model_manager=None, config=None):
        self.config = config or {}
        self.model_manager = model_manager
        self.ratio_threshold = get_cfg(self.config, "localization.ratio_threshold", 0.95)

        # Завантажуємо LightGlue (ALIKED) через ModelManager
        self.lightglue = None
        if self.model_manager:
            try:
                self.lightglue = self.model_manager.load_lightglue_aliked()
                logger.info("FeatureMatcher configured to use LightGlue (ALIKED)")
            except Exception as e:
                logger.warning(f"Failed to load LightGlue ALIKED: {e}. Falling back to Numpy L2.")
        else:
            logger.info("FeatureMatcher configured to use fast Numpy L2 matching")

    def match(self, query_features: dict, ref_features: dict) -> tuple:
        """
        Dynamically routes to LightGlue (for 256-dim SuperPoint)
        or Fast L2 Matcher (for 64-dim XFeat / 128-dim ALIKED).
        """
        desc_dim = (
            query_features["descriptors"].shape[1] if len(query_features["descriptors"]) > 0 else 0
        )

        # Якщо є LightGlue і розмірність дескриптора 128 (ALIKED)
        if self.lightglue is not None and desc_dim == 128:
            return self._lightglue_match(query_features, ref_features)

        # Fallback (якщо немає LightGlue або інші ознаки)
        return self._fast_numpy_match(query_features, ref_features, self.ratio_threshold)

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

            if len(matches) == 0:
                return np.empty((0, 2)), np.empty((0, 2))

            m_q = matches[:, 0]
            m_r = matches[:, 1]

            mkpts_q = query_features["keypoints"][m_q]
            mkpts_r = ref_features["keypoints"][m_r]

            return mkpts_q, mkpts_r

        except Exception as e:
            logger.error(f"LightGlue match failed: {e}")
            return np.empty((0, 2)), np.empty((0, 2))
