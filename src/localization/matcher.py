"""
matcher.py — Feature matching module.

Key fixes:
- ratio_threshold default set to 0.75 per Lowe's ratio test for normalized descriptors,
  preventing false positive matches on homogeneous textures.
"""

import faiss
import numpy as np
import torch

from config import get_cfg
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


def extract_sift_features(
    image: np.ndarray, static_mask: np.ndarray | None = None, max_keypoints: int = 2048
) -> dict:
    """RESEARCH 2.2: SIFT features in a format compatible with LightGlue(features="sift").

    Used by both DatabaseBuilder (offline, database storage) and
    Localizer (online, emergency fallback) — an identical pipeline guarantees
    descriptor compatibility. Descriptors are rootSIFT (L1-norm + sqrt), like in
    the lightglue library extractor, on which the sift-matcher weights were trained.
    """
    import cv2

    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if image.ndim == 3 else image
    mask8 = None
    if static_mask is not None:
        mask8 = (static_mask > 128).astype(np.uint8) * 255

    sift = cv2.SIFT_create(nfeatures=int(max_keypoints))
    kps, descs = sift.detectAndCompute(gray, mask8)
    if descs is None or len(kps) == 0:
        return {
            "keypoints": np.empty((0, 2), dtype=np.float32),
            "descriptors": np.empty((0, 128), dtype=np.float32),
            "image_size": np.array(gray.shape[:2], dtype=np.int32),
        }

    # rootSIFT: L1-normalise + elementwise sqrt → L2-norm ≈1
    descs = descs.astype(np.float32)
    descs /= np.maximum(descs.sum(axis=1, keepdims=True), 1e-12)
    descs = np.sqrt(descs)

    pts = np.array([kp.pt for kp in kps], dtype=np.float32)
    return {
        "keypoints": pts,
        "descriptors": descs,
        "image_size": np.array(gray.shape[:2], dtype=np.int32),
    }


class FastRetrieval:
    """Fast candidate search using DINOv2 global descriptors (optimized with FAISS)"""

    def __init__(self, global_descriptors: np.ndarray):
        logger.info(
            f"Initializing FastRetrieval with {len(global_descriptors)} descriptors using FAISS"
        )
        self.dim = global_descriptors.shape[1]

        # Inner Product index for cosine similarity of normalised vectors
        base_index = faiss.IndexFlatIP(self.dim)
        self.index = faiss.IndexIDMap(base_index)

        # Normalise and add to index
        normed = self.normalize_vectors(global_descriptors)
        ids = np.arange(len(global_descriptors), dtype=np.int64)
        self.index.add_with_ids(normed.astype(np.float32), ids)

        logger.success(f"FAISS index built with {self.index.ntotal} vectors")

    @staticmethod
    def normalize_vectors(vectors: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        return vectors / (norms + 1e-8)

    def add_descriptor(self, query_desc: np.ndarray, frame_id: int):
        """Incrementally adds a new descriptor to FAISS index."""
        normed = self.normalize_vectors(query_desc)
        if normed.ndim == 1:
            normed = normed[None]
        self.index.add_with_ids(normed.astype(np.float32), np.array([frame_id], dtype=np.int64))
        logger.debug(f"Added descriptor for frame {frame_id} to FAISS. Total: {self.index.ntotal}")

    def find_similar_frames(self, query_desc: np.ndarray, top_k: int = 5) -> list:
        q = query_desc / (np.linalg.norm(query_desc) + 1e-8)
        q = q.astype(np.float32)

        if q.ndim == 1:
            q = q[None]

        scores, ids = self.index.search(q, top_k)
        results = [(int(idx), float(score)) for idx, score in zip(ids[0], scores[0]) if idx != -1]
        return results


class LanceDBRetrieval:
    """Fast candidate search using LanceDB for vector similarity."""

    def __init__(self, lance_table):
        logger.info("Initializing LanceDBRetrieval using LanceDB table natively")
        self.lance_table = lance_table

    def add_descriptor(self, query_desc: np.ndarray, frame_id: int):
        # LanceDB insertion is usually handled batch-wise in DatabaseLoader.
        pass

    def find_similar_frames(self, query_desc: np.ndarray, top_k: int = 5) -> list:
        if self.lance_table is None:
            return []

        q = query_desc / (np.linalg.norm(query_desc) + 1e-8)

        try:
            res = (
                self.lance_table.search(q.astype(np.float32).flatten())
                .metric("cosine")
                .limit(top_k)
                .select(["frame_id", "_distance"])
                .to_list()
            )
            # Returns [(frame_id, similarity)]
            return [(int(r["frame_id"]), float(max(0.0, 1.0 - r["_distance"]))) for r in res]
        except Exception as e:
            logger.error(f"LanceDB query failed: {e}")
            return []


class FeatureMatcher:
    """Matches local keypoints (XFeat or SuperPoint+LightGlue)"""

    def __init__(self, model_manager=None, config=None):
        self.config = config or {}
        self.model_manager = model_manager

        # Ratio threshold lowered from 0.95 to 0.75.
        # 0.95 allowed too many false matches on homogeneous textures
        # (fields, forests, rooftops), causing degenerate homographies in MAGSAC++/LMEDS
        # and coordinate micro-jumps between adjacent frames.
        # 0.75 is the standard Lowe's ratio test value for normalised L2 descriptors.
        self.ratio_threshold = get_cfg(self.config, "localization.ratio_threshold", 0.75)

        # Load LightGlue (ALIKED/RDD) via ModelManager
        self.lightglue = None
        if self.model_manager:
            local_extractor = get_cfg(self.config, "models.local_extractor", "aliked")
            lg_features = "rdd" if local_extractor == "rdd" else "aliked"
            try:
                self.lightglue = self.model_manager.load_lightglue(features=lg_features)
                logger.info(f"FeatureMatcher configured to use LightGlue ({lg_features})")
            except Exception as e:
                logger.warning(
                    f"Failed to load LightGlue ({lg_features}): {e}. "
                    f"Cause: model files may be missing or VRAM insufficient. "
                    f"Falling back to Numpy L2 matching.",
                    exc_info=True,
                )
        else:
            logger.info("FeatureMatcher configured to use fast Numpy L2 matching")

        logger.info(f"FeatureMatcher ratio_threshold = {self.ratio_threshold:.2f}")

        # For warn-once logging of incompatible descriptor dimensions
        self._dim_mismatch_warned: set = set()

        # LightGlue(sift) loaded lazily — only when the emergency fallback fires
        # for the first time (VRAM is not wasted otherwise)
        self._lightglue_sift = None
        self._lightglue_sift_failed = False

    def match(self, query_features: dict, ref_features: dict) -> tuple:
        """
        Dynamically routes to LightGlue (for 256-dim SuperPoint)
        or Fast L2 Matcher (for 64-dim XFeat / 128-dim ALIKED).
        """
        desc_dim = (
            query_features["descriptors"].shape[1] if len(query_features["descriptors"]) > 0 else 0
        )
        ref_dim = (
            ref_features["descriptors"].shape[1] if len(ref_features["descriptors"]) > 0 else 0
        )

        # Guard: mismatched descriptor dimensions (e.g. query=128-dim ALIKED,
        # ref=256-dim RDD/SuperPoint from an old database) cannot be matched at all —
        # neither LightGlue nor L2. That source's database was built with a
        # different extractor and must be regenerated.
        if desc_dim and ref_dim and desc_dim != ref_dim:
            key = (desc_dim, ref_dim)
            if key not in self._dim_mismatch_warned:
                self._dim_mismatch_warned.add(key)
                logger.error(
                    f"Descriptor dimension mismatch: query={desc_dim}, ref={ref_dim}. "
                    f"Reference database was built with a different local extractor. "
                    f"Rebuild that source's database with the current extractor. "
                    f"Skipping all matches for this dim pair (logged once)."
                )
            return np.empty((0, 2)), np.empty((0, 2))

        # Use LightGlue if available and descriptor dim is 128 (ALIKED) or 256 (RDD/SuperPoint)
        if self.lightglue is not None and desc_dim in (128, 256):
            return self._lightglue_match(query_features, ref_features)

        if self.lightglue is not None and desc_dim not in (128, 256):
            logger.debug(
                f"LightGlue available but descriptor dim={desc_dim} is unsupported. "
                f"Using Numpy L2 matching instead."
            )

        # Fallback (no LightGlue or unsupported descriptor)
        return self._fast_numpy_match(query_features, ref_features, self.ratio_threshold)

    def _fast_numpy_match(
        self, query_features: dict, ref_features: dict, ratio_threshold: float = 0.75
    ) -> tuple:
        """
        Highly optimized L2 matching using dot product and Mutual Nearest Neighbor (MNN).
        """
        desc_q = query_features["descriptors"]
        desc_r = ref_features["descriptors"]
        kpts_q = query_features["keypoints"]
        kpts_r = ref_features["keypoints"]

        if len(desc_q) < 2 or len(desc_r) < 2:
            logger.debug(
                f"Numpy L2 match aborted: insufficient descriptors | "
                f"query={len(desc_q)}, ref={len(desc_r)} (minimum=2)"
            )
            return np.empty((0, 2)), np.empty((0, 2))

        # 1. Normalise descriptors
        desc_q_n = desc_q / (np.linalg.norm(desc_q, axis=1, keepdims=True) + 1e-8)
        desc_r_n = desc_r / (np.linalg.norm(desc_r, axis=1, keepdims=True) + 1e-8)

        # 2. Cosine similarity via fast matrix multiplication
        sim = np.dot(desc_q_n, desc_r_n.T)

        # 3. Lowe's Ratio Test — argpartition O(n) instead of argsort O(n log n)
        top2_idx = np.argpartition(-sim, kth=1, axis=1)[:, :2]
        top2_sim = np.take_along_axis(sim, top2_idx, axis=1)
        order = np.argsort(-top2_sim, axis=1)
        top2_idx = np.take_along_axis(top2_idx, order, axis=1)
        top2_sim = np.take_along_axis(top2_sim, order, axis=1)

        best_sim = top2_sim[:, 0]
        second_best_sim = top2_sim[:, 1]
        best_matches_indices = top2_idx[:, 0]

        # Convert similarity to L2 distance: D = sqrt(2 − 2*sim)
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

    def match_mnn(self, query_features: dict, ref_features: dict) -> tuple:
        """Deterministic mutual-NN (L2) matching BYPASSING LightGlue.

        Фолбек для temporal-ребер пропагації: на повторюваній ріллі LightGlue
        місцями віддає 12–28 матчів там, де MNN по тих самих дескрипторах
        знаходить 100–800 пар (перевірено на lasttest). Викликається воркером,
        коли LightGlue дав < min_matches."""
        return self._fast_numpy_match(query_features, ref_features, self.ratio_threshold)

    def match_sift(self, query_features: dict, ref_features: dict) -> tuple:
        """RESEARCH 2.2: SIFT feature matching via LightGlue(features="sift").

        Окремий метод (не через match()): SIFT-дескриптори 128-вимірні, як
        ALIKED, тож маршрутизація за розмірністю відправила б їх у
        ALIKED-матчер з ловом сміттєвих збігів.
        """
        if self._lightglue_sift is None and not self._lightglue_sift_failed:
            if self.model_manager is None:
                self._lightglue_sift_failed = True
            else:
                try:
                    self._lightglue_sift = self.model_manager.load_lightglue(features="sift")
                    logger.info("LightGlue (sift) loaded for emergency fallback")
                except Exception as e:
                    self._lightglue_sift_failed = True
                    logger.warning(f"Failed to load LightGlue (sift): {e} — fallback disabled")
        if self._lightglue_sift is None:
            return np.empty((0, 2)), np.empty((0, 2))
        return self._lightglue_match(query_features, ref_features, model=self._lightglue_sift)

    def _lightglue_match(self, query_features: dict, ref_features: dict, model=None) -> tuple:
        """Matches features using Neural LightGlue Matcher"""
        try:
            if model is None:
                model = self.lightglue
            if len(query_features["keypoints"]) == 0 or len(ref_features["keypoints"]) == 0:
                logger.warning(
                    f"Empty keypoints provided to LightGlue | "
                    f"query_kpts={len(query_features['keypoints'])}, "
                    f"ref_kpts={len(ref_features['keypoints'])}. "
                    f"Cannot match without keypoints."
                )
                return np.empty((0, 2)), np.empty((0, 2))

            device = next(model.parameters()).device

            # image_size needed for correct [-1, 1] coordinate normalisation in LightGlue.
            # Without it, cross-resolution pairs (4K query vs 1080p ref) produce ~0 matches.
            image0_data = {
                "keypoints": torch.from_numpy(query_features["keypoints"]).float()[None].to(device),
                "descriptors": torch.from_numpy(query_features["descriptors"])
                .float()[None]
                .to(device),
            }
            image1_data = {
                "keypoints": torch.from_numpy(ref_features["keypoints"]).float()[None].to(device),
                "descriptors": torch.from_numpy(ref_features["descriptors"])
                .float()[None]
                .to(device),
            }

            q_size = query_features.get("image_size")
            r_size = ref_features.get("image_size")
            if q_size is not None:
                # image_size expected as (W, H) in LightGlue
                image0_data["image_size"] = torch.tensor(
                    [[int(q_size[1]), int(q_size[0])]], device=device
                )
            if r_size is not None:
                image1_data["image_size"] = torch.tensor(
                    [[int(r_size[1]), int(r_size[0])]], device=device
                )

            data = {"image0": image0_data, "image1": image1_data}

            with torch.no_grad():
                res = model(data)

            matches = res["matches"][0].cpu().numpy()

            if len(matches) == 0:
                return np.empty((0, 2)), np.empty((0, 2))

            m_q = matches[:, 0]
            m_r = matches[:, 1]

            mkpts_q = query_features["keypoints"][m_q]
            mkpts_r = ref_features["keypoints"][m_r]

            return mkpts_q, mkpts_r

        except Exception as e:
            logger.error(
                f"LightGlue match failed: {e} | "
                f"query_kpts={len(query_features.get('keypoints', []))}, "
                f"query_desc_shape={query_features.get('descriptors', np.empty(0)).shape}, "
                f"ref_kpts={len(ref_features.get('keypoints', []))}, "
                f"ref_desc_shape={ref_features.get('descriptors', np.empty(0)).shape}. "
                f"Possible causes: CUDA OOM, tensor shape mismatch, or model corruption.",
                exc_info=True,
            )
            return np.empty((0, 2)), np.empty((0, 2))
