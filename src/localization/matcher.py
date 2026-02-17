import numpy as np
import torch

from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


class FastRetrieval:
    """Fast candidate search using NetVLAD global descriptors"""

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
        similarities = np.dot(self.global_descriptors, query_norm)
        top_indices = np.argsort(similarities)[::-1][:top_k]

        results = []
        for idx in top_indices:
            results.append((int(idx), float(similarities[idx])))
            logger.debug(f"Candidate {idx}: similarity = {similarities[idx]:.4f}")

        logger.debug(f"Found {len(results)} candidates")
        return results


class FeatureMatcher:
    """Precise keypoint matching using LightGlue"""

    def __init__(self, lightglue_model, device='cuda'):
        self.lightglue = lightglue_model
        self.device = device
        logger.info(f"FeatureMatcher initialized on device: {device}")

    @torch.no_grad()
    def match(self, query_features: dict, ref_features: dict) -> tuple:
        logger.debug("Starting feature matching with LightGlue...")

        # --- ЗАХИСТ ВІД БАГУ З ФОРМОЮ МАТРИЦЬ ---
        # Якщо в базі HDF5 або з FeatureExtractor прийшла матриця (256, N) замість (N, 256)
        if query_features['descriptors'].ndim == 2 and query_features['descriptors'].shape[0] == 256 and \
                query_features['descriptors'].shape[1] != 256:
            query_features['descriptors'] = query_features['descriptors'].T

        if ref_features['descriptors'].ndim == 2 and ref_features['descriptors'].shape[0] == 256 and \
                ref_features['descriptors'].shape[1] != 256:
            ref_features['descriptors'] = ref_features['descriptors'].T
        # ----------------------------------------

        # Prepare query features
        kpts_q = torch.from_numpy(query_features['keypoints']).float().unsqueeze(0).to(self.device)
        desc_q = torch.from_numpy(query_features['descriptors']).float().to(self.device)

        # ВАЖЛИВО: LightGlue очікує форму (B, N, D), тому ми просто додаємо batch_size = 1
        if desc_q.ndim == 2:
            desc_q = desc_q.unsqueeze(0)  # (N, D) -> (1, N, D)

        # Prepare reference features
        kpts_r = torch.from_numpy(ref_features['keypoints']).float().unsqueeze(0).to(self.device)
        desc_r = torch.from_numpy(ref_features['descriptors']).float().to(self.device)

        if desc_r.ndim == 2:
            desc_r = desc_r.unsqueeze(0)  # (N, D) -> (1, N, D)

        logger.debug(f"Query: {kpts_q.shape[1]} keypoints, desc shape: {desc_q.shape}")
        logger.debug(f"Ref: {kpts_r.shape[1]} keypoints, desc shape: {desc_r.shape}")

        # Динамічний розрахунок розміру зображення
        def get_img_size(kpts):
            if kpts.shape[1] == 0:
                return torch.tensor([[1920, 1080]]).to(self.device)
            max_vals, _ = torch.max(kpts[0], dim=0)
            return torch.tensor([[max_vals[0].item() + 10, max_vals[1].item() + 10]]).to(self.device)

        data0 = {
            'keypoints': kpts_q,
            'descriptors': desc_q,
            'image_size': get_img_size(kpts_q)
        }

        data1 = {
            'keypoints': kpts_r,
            'descriptors': desc_r,
            'image_size': get_img_size(kpts_r)
        }

        try:
            # Perform matching
            result = self.lightglue({'image0': data0, 'image1': data1})

            # Extract matches
            matches0 = result['matches0'][0].cpu().numpy()  # (N_query,)

            # Get valid matches
            valid_mask = matches0 >= 0
            query_indices = np.where(valid_mask)[0]
            ref_indices = matches0[valid_mask]

            if len(query_indices) == 0:
                logger.warning("No matches found between query and reference")
                return np.array([]), np.array([])

            # Get matched keypoint coordinates
            mkpts_query = query_features['keypoints'][query_indices]
            mkpts_ref = ref_features['keypoints'][ref_indices.astype(int)]

            logger.debug(f"LightGlue found {len(mkpts_query)} matches")

            return mkpts_query, mkpts_ref

        except Exception as e:
            logger.error(f"Error during LightGlue matching: {e}", exc_info=True)
            logger.warning("Falling back to simple L2 distance matching")
            return self._fallback_match(query_features, ref_features)





    def _fallback_match(self, query_features: dict, ref_features: dict, ratio_threshold: float = 0.8) -> tuple:
        """Fallback matching using descriptor L2 distance and Lowe's ratio test"""
        logger.debug("Using fallback matcher with ratio test")

        desc_q = query_features['descriptors']
        desc_r = ref_features['descriptors']

        # Normalize descriptors
        desc_q_norm = desc_q / (np.linalg.norm(desc_q, axis=1, keepdims=True) + 1e-8)
        desc_r_norm = desc_r / (np.linalg.norm(desc_r, axis=1, keepdims=True) + 1e-8)

        # Compute distance matrix
        dists = np.linalg.norm(desc_q_norm[:, None] - desc_r_norm[None, :], axis=2)

        # Find nearest and second nearest neighbors
        sorted_indices = np.argsort(dists, axis=1)
        nn_dists = dists[np.arange(len(desc_q)), sorted_indices[:, 0]]
        nn2_dists = dists[np.arange(len(desc_q)), sorted_indices[:, 1]]

        # Lowe's ratio test
        ratio = nn_dists / (nn2_dists + 1e-8)
        valid = ratio < ratio_threshold

        query_indices = np.where(valid)[0]
        ref_indices = sorted_indices[valid, 0]

        mkpts_query = query_features['keypoints'][query_indices]
        mkpts_ref = ref_features['keypoints'][ref_indices]

        logger.debug(f"Fallback matcher found {len(mkpts_query)} matches")

        return mkpts_query, mkpts_ref