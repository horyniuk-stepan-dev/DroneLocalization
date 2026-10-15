import numpy as np
import torch

from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


class PatchifyRetrieval:
    """Multi-scale retrieval via DINOv2/DINOv3 patch descriptors.

    Splits the image into grid patches (1x1, 2x2, 3x3) = 14 patches total,
    extracts the global descriptor of each patch (the extractor's active one:
    CLS, VLAD or CESP), and retrieves the most similar frames via aggregated
    patch scores.
    """

    DEFAULT_GRIDS = [(1, 1), (2, 2), (3, 3)]  # 1 + 4 + 9 = 14 патчів

    def __init__(
        self,
        feature_extractor,
        descriptor_dim: int = 1024,
        grids: list[list[int]] | None = None,
        batch_size: int = 1,
        unique_coverage: bool = False,
    ):
        self.feature_extractor = feature_extractor
        self.descriptor_dim = descriptor_dim
        self.batch_size = max(1, batch_size)
        # localization.patchify_unique_coverage: score a frame by the best hit of
        # each query patch (coverage over distinct query patches, score <= 1)
        # instead of summing every reference-patch hit (see search()).
        self.unique_coverage = bool(unique_coverage)
        self.grids = [tuple(g) for g in grids] if grids else self.DEFAULT_GRIDS
        self.num_patches = sum(r * c for r, c in self.grids)

        # FAISS index (populated via build_index)
        self.patch_index = None
        # Mapping: linear_patch_idx → frame_id
        self.patch_frame_ids: np.ndarray | None = None

        logger.info(
            f"PatchifyRetrieval initialized: grids={self.grids}, "
            f"num_patches={self.num_patches}, batch_size={self.batch_size}, dim={descriptor_dim}"
        )

    # ── Patch extraction ─────────────────────────────────────────────────

    @staticmethod
    def extract_patches(image: np.ndarray, grids: list[tuple[int, int]]) -> list[np.ndarray]:
        """Crops image into grid patches.

        Args:
            image: (H, W, 3) RGB зображення
            grids: Список (rows, cols) сіток

        Returns:
            Список кропів у порядку: grid(1,1) → grid(2,2) → grid(3,3)
        """
        MIN_PATCH_PX = 32  # мінімальний розмір патча в пікселях

        h, w = image.shape[:2]
        patches = []

        for rows, cols in grids:
            ph, pw = h // rows, w // cols
            if ph < MIN_PATCH_PX or pw < MIN_PATCH_PX:
                raise ValueError(
                    f"Patch too small for grid {rows}×{cols}: "
                    f"{ph}×{pw}px (min={MIN_PATCH_PX}px). "
                    f"Image size: {w}×{h}."
                )

            for r in range(rows):
                for c in range(cols):
                    y1 = r * ph
                    x1 = c * pw
                    # Last patch takes the remainder (to avoid losing pixels)
                    y2 = h if r == rows - 1 else (r + 1) * ph
                    x2 = w if c == cols - 1 else (c + 1) * pw
                    patches.append(image[y1:y2, x1:x2].copy())

        return patches

    # ── Descriptor computation ───────────────────────────────────────────

    @torch.no_grad()
    def compute_patch_descriptors(self, image: np.ndarray) -> np.ndarray:
        """Extracts DINOv2 descriptor for each image patch.

        Args:
            image: (H, W, 3) RGB зображення

        Returns:
            (num_patches, descriptor_dim) float32 масив дескрипторів
        """
        patches = self.extract_patches(image, self.grids)
        descriptors = np.empty((len(patches), self.descriptor_dim), dtype=np.float32)

        if self.batch_size <= 1:
            # Sequential inference — minimal VRAM consumption
            for i, patch in enumerate(patches):
                descriptors[i] = self.feature_extractor.extract_global_descriptor(patch)
        else:
            # Batched inference — faster but uses more VRAM
            for start in range(0, len(patches), self.batch_size):
                batch = patches[start : start + self.batch_size]
                batch_descs = self._extract_batch_descriptors(batch)
                descriptors[start : start + len(batch)] = batch_descs

        return descriptors

    @torch.no_grad()
    def _extract_batch_descriptors(self, patches: list[np.ndarray]) -> np.ndarray:
        """Descriptors of a group of crops in one batched extractor call.

        Goes through ``FeatureExtractor.extract_global_descriptors_multi``, which
        resizes every crop to the DINO input size before stacking and applies the
        same preprocessing (``dino_cpu_resize``), aggregation (VLAD / CESP / CLS)
        and ``global_batch_max`` chunking as the sequential path. The previous
        own implementation stacked crops of different sizes (RuntimeError) and
        called the backbone directly (CLS instead of VLAD, no CPU resize).
        """
        fe = self.feature_extractor
        multi = getattr(fe, "extract_global_descriptors_multi", None)
        if multi is None:
            return np.stack([np.asarray(fe.extract_global_descriptor(p)) for p in patches])
        out = np.asarray(multi(list(patches)), dtype=np.float32)
        if out.shape != (len(patches), self.descriptor_dim):
            raise ValueError(
                f"Batched patch descriptors have shape {out.shape}, expected "
                f"({len(patches)}, {self.descriptor_dim})"
            )
        return out

    # ── Index management ─────────────────────────────────────────────────

    def build_index(self, patch_descriptors_all: np.ndarray, frame_ids: list[int]):
        """Builds FAISS index from all patch descriptors.

        Args:
            patch_descriptors_all: (N_frames, num_patches, D) — всі патч-дескриптори
            frame_ids: Список frame_id для кожного кадру
        """
        import faiss

        n_frames, n_patches, dim = patch_descriptors_all.shape
        assert n_patches == self.num_patches, (
            f"Expected {self.num_patches} patches per frame, got {n_patches}"
        )

        # Flatten to (N_frames × num_patches, D)
        flat = patch_descriptors_all.reshape(-1, dim).astype(np.float32)

        # L2-normalise for cosine similarity
        norms = np.linalg.norm(flat, axis=1, keepdims=True)
        flat = flat / (norms + 1e-8)

        # Mapping: each flat row → frame_id
        self.patch_frame_ids = np.repeat(np.array(frame_ids, dtype=np.int32), n_patches)

        # FAISS Inner Product index
        base_index = faiss.IndexFlatIP(dim)
        self.patch_index = base_index
        self.patch_index.add(flat)

        logger.info(
            f"Patchify FAISS index built: {self.patch_index.ntotal} vectors "
            f"({n_frames} frames × {n_patches} patches)"
        )

    def search(self, query_descriptors: np.ndarray, top_k: int = 10) -> list[tuple[int, float]]:
        """Searches top-K frames by aggregated patch scores.

        Args:
            query_descriptors: (num_patches, D) — патч-дескриптори query
            top_k: Кількість кандидатів

        Returns:
            Список (frame_id, aggregated_score) відсортований за зменшенням скору
        """
        if self.patch_index is None or self.patch_frame_ids is None:
            logger.warning("Patchify index not built, returning empty results")
            return []

        # Normalise query descriptors
        q = query_descriptors.astype(np.float32)
        norms = np.linalg.norm(q, axis=1, keepdims=True)
        q = q / (norms + 1e-8)

        # For each of the num_patches query patches, find top-K ref patches
        search_k = top_k * 3  # шукаємо більше для кращої агрегації

        # Guard: search_k cannot exceed the index size
        max_k = self.patch_index.ntotal
        if search_k > max_k:
            logger.debug(f"search_k={search_k} > index size={max_k}, clamping")
            search_k = max_k

        scores, indices = self.patch_index.search(q, search_k)

        # Aggregate: sum cosine scores and count hits per frame_id
        frame_scores: dict[int, float] = {}
        frame_hits: dict[int, int] = {}

        for patch_idx in range(len(q)):
            # unique_coverage: one hit per (query patch, frame) — its best one
            # (FAISS returns hits best-first). Legacy: every reference patch hit
            # counts, so one frame can collect several hits per query patch.
            seen: set[int] = set()
            for j in range(search_k):
                ref_idx = indices[patch_idx, j]
                if ref_idx < 0:
                    continue
                fid = int(self.patch_frame_ids[ref_idx])
                if self.unique_coverage:
                    if fid in seen:
                        continue
                    seen.add(fid)
                score = float(scores[patch_idx, j])

                frame_scores[fid] = frame_scores.get(fid, 0.0) + score
                frame_hits[fid] = frame_hits.get(fid, 0) + 1

        # Compute final score: coverage * avg_score (= sum of the hit scores /
        # num_patches). Legacy hits may exceed num_patches, so the score can be
        # > 1 (an identical frame scores min(search_k, patches per frame)); with unique_coverage
        # coverage <= 1 and the score is on the cosine scale.
        num_patches = len(q)
        final_scores: dict[int, float] = {}
        for fid in frame_scores:
            hits = frame_hits[fid]
            avg_score = frame_scores[fid] / hits  # якість патчів що знайшли
            coverage = hits / num_patches  # частка патчів що знайшли
            final_scores[fid] = coverage * avg_score

        # Sort and return top-K
        sorted_frames = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_frames[:top_k]
