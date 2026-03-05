import numpy as np
import torch

from src.geometry.coordinates import CoordinateConverter
from src.geometry.transformations import GeometryTransforms
from src.localization.matcher import FastRetrieval
from src.tracking.kalman_filter import TrajectoryFilter
from src.tracking.outlier_detector import OutlierDetector
from src.utils.logging_utils import get_logger

from typing import Any
from src.database.database_loader import DatabaseLoader
from src.models.wrappers.feature_extractor import FeatureExtractor
from src.localization.matcher import FeatureMatcher

logger = get_logger(__name__)


class Localizer:
    def __init__(
        self,
        database: DatabaseLoader,
        feature_extractor: FeatureExtractor,
        matcher: FeatureMatcher,
        calibration: Any,
        config: dict | None = None,
    ):
        self.database = database
        self.feature_extractor = feature_extractor
        self.matcher = matcher
        self.calibration = calibration
        self.config = config or {}

        self.min_matches = self.config.get("localization", {}).get("min_matches", 8)
        self.ransac_thresh = self.config.get("localization", {}).get("ransac_threshold", 5.0)
        self.enable_auto_rotation = self.config.get("localization", {}).get("auto_rotation", True)

        self.trajectory_filter = TrajectoryFilter(
            process_noise=self.config.get("tracking", {}).get("kalman_process_noise", 2.0),
            measurement_noise=self.config.get("tracking", {}).get("kalman_measurement_noise", 5.0),
            dt=1.0,
        )
        self.outlier_detector = OutlierDetector(
            window_size=self.config.get("tracking", {}).get("outlier_window", 10),
            threshold_std=self.config.get("tracking", {}).get("outlier_threshold_std", 3.0),
            max_speed_mps=self.config.get("tracking", {}).get("max_speed_mps", 1000.0),
        )

        # Створюємо FastRetrieval один раз — нормалізація дескрипторів відбувається лише тут
        self.retriever = FastRetrieval(self.database.global_descriptors)

        # Fallback: SuperPoint+LightGlue для складних сцен
        self.model_manager = self.config.get("_model_manager", None)
        self.fallback_enabled = self.config.get("localization", {}).get(
            "enable_lightglue_fallback", True
        )
        self.min_inliers_for_accept = self.config.get("localization", {}).get(
            "min_inliers_accept", 8
        )

    def localize_frame(
        self, query_frame: np.ndarray, static_mask: np.ndarray | None = None, dt: float = 1.0
    ) -> dict:
        height, width = query_frame.shape[:2]

        best_global_angle, best_global_candidates = self._find_best_rotation(query_frame)

        if not best_global_candidates:
            return {
                "success": False,
                "error": "No candidates found via global descriptor (DINOv2) in any rotation",
            }

        # 1.5 & 2. Локальний пошук (XFeat) для найкращого ракурсу
        best_inliers, best_candidate_id, best_mkpts_q, best_mkpts_r, best_total_matches = (
            self._perform_local_matching(
                query_frame, static_mask, best_global_angle, best_global_candidates
            )
        )

        # 2b. Fallback: SuperPoint+LightGlue
        if (
            best_inliers < self.min_inliers_for_accept
            and self.fallback_enabled
            and self.model_manager is not None
        ):
            fb_inliers, fb_candidate_id, fb_mkpts_q, fb_mkpts_r, fb_rot, fb_total = (
                self._run_lightglue_fallback(
                    query_frame,
                    static_mask,
                    best_global_angle,
                    best_global_candidates,
                    height,
                    width,
                )
            )
            if fb_inliers > best_inliers:
                best_inliers = fb_inliers
                best_candidate_id = fb_candidate_id
                best_mkpts_q = fb_mkpts_q
                best_mkpts_r = fb_mkpts_r
                best_total_matches = fb_total

        if best_inliers < self.min_matches or best_mkpts_r is None:
            return {
                "success": False,
                "error": f"Not enough valid inliers ({best_inliers} < {self.min_matches})",
            }

        # 3. Трансформація координат та Трекінг
        return self._calculate_tracking_update(
            best_mkpts_q,
            best_mkpts_r,
            best_candidate_id,
            best_global_angle,
            best_inliers,
            best_total_matches,
            width,
            height,
            dt,
        )

    def _find_best_rotation(self, query_frame: np.ndarray) -> tuple[int, list]:
        """Знаходить найкращий кут обертання та кандидатів через DINOv2 глобальний дескриптор."""
        angles_to_try = [0, 90, 180, 270] if self.enable_auto_rotation else [0]
        best_global_score = -1.0
        best_global_angle = 0
        best_global_candidates = []
        top_k = self.config.get("localization", {}).get("retrieval_top_k", 8)

        for angle in angles_to_try:
            k = angle // 90
            rotated_frame = np.rot90(query_frame, k=k).copy()
            global_desc = self.feature_extractor.extract_global_descriptor(rotated_frame)
            candidates = self.retriever.find_similar_frames(global_desc, top_k=top_k)

            if candidates:
                top_score = candidates[0][1]
                if top_score > best_global_score:
                    best_global_score = top_score
                    best_global_angle = angle
                    best_global_candidates = candidates

        if best_global_candidates:
            logger.info(
                f"Selected best rotation {best_global_angle}° with global score {best_global_score:.3f}"
            )

        return best_global_angle, best_global_candidates

    def _perform_local_matching(
        self,
        query_frame: np.ndarray,
        static_mask: np.ndarray | None,
        best_angle: int,
        candidates: list,
    ) -> tuple[int, int, np.ndarray | None, np.ndarray | None, int]:
        """Виконує XFeat локальний матчинг для найкращого кута."""
        k = best_angle // 90
        rotated_frame = np.rot90(query_frame, k=k).copy()
        rotated_mask = np.rot90(static_mask, k=k).copy() if static_mask is not None else None

        query_features = self.feature_extractor.extract_local_features(
            rotated_frame, static_mask=rotated_mask
        )

        best_inliers = 0
        best_candidate_id = -1
        best_mkpts_q_inliers = None
        best_mkpts_r_inliers = None
        best_total_matches = 0
        early_stop = self.config.get("localization", {}).get("early_stop_inliers", 30)

        for candidate_id, _score in candidates:
            ref_features = self.database.get_local_features(candidate_id)
            mkpts_q, mkpts_r = self.matcher.match(query_features, ref_features)

            if len(mkpts_q) >= self.min_matches:
                M, mask = GeometryTransforms.estimate_affine(
                    mkpts_q, mkpts_r, ransac_threshold=self.ransac_thresh
                )
                if M is not None:
                    inlier_mask = mask.ravel().astype(bool)
                    inliers = int(np.sum(inlier_mask))
                    if inliers > best_inliers:
                        best_inliers = inliers
                        best_candidate_id = candidate_id
                        best_mkpts_q_inliers = mkpts_q[inlier_mask]
                        best_mkpts_r_inliers = mkpts_r[inlier_mask]
                        best_total_matches = len(mkpts_q)

            if best_inliers >= early_stop:
                logger.info(
                    f"Early stop triggered with {best_inliers} inliers on candidate {best_candidate_id}"
                )
                break

        return (
            best_inliers,
            best_candidate_id,
            best_mkpts_q_inliers,
            best_mkpts_r_inliers,
            best_total_matches,
        )

    def _run_lightglue_fallback(
        self,
        query_frame: np.ndarray,
        static_mask: np.ndarray | None,
        best_angle: int,
        candidates: list,
        height: int,
        width: int,
    ) -> tuple[int, int, np.ndarray | None, np.ndarray | None, int, int]:
        """Запускає Fallback (SuperPoint+LightGlue)."""
        logger.info(f"Trying SuperPoint+LightGlue fallback on best angle {best_angle}°...")
        k = best_angle // 90
        rotated_frame = np.rot90(query_frame, k=k).copy()
        rotated_mask = np.rot90(static_mask, k=k).copy() if static_mask is not None else None

        return self._try_lightglue_fallback(rotated_frame, rotated_mask, candidates, height, width)

    def _calculate_tracking_update(
        self,
        mkpts_q: np.ndarray,
        mkpts_r: np.ndarray,
        candidate_id: int,
        global_angle: int,
        inliers: int,
        total_matches: int,
        width: int,
        height: int,
        dt: float,
    ) -> dict:
        """Перетворює координати, фільтрує викиди та розраховує кінцеві GPS координати."""
        # Фізичні розміри ПОВЕРНУТОГО кадру
        if global_angle in {90, 270}:
            rot_height, rot_width = width, height
        else:
            rot_height, rot_width = height, width

        centroid_ref = np.mean(mkpts_r, axis=0)
        centroid_query = np.mean(mkpts_q, axis=0)

        center_query = np.array([rot_width / 2.0, rot_height / 2.0], dtype=np.float32)
        offset_px = center_query - centroid_query

        affine_ref = self.database.get_frame_affine(candidate_id)
        if affine_ref is None:
            return {"success": False, "error": "No propagated calibration"}

        metric_centroid = GeometryTransforms.apply_affine(
            np.array([centroid_ref], dtype=np.float32), affine_ref
        )
        if metric_centroid is None or len(metric_centroid) == 0:
            return {"success": False, "error": "Projection failed"}
        metric_centroid = metric_centroid[0]

        # Витягуємо масштаб та матрицю повороту з афінної матриці
        scale_x = np.linalg.norm(affine_ref[0, :2])
        scale_y = np.linalg.norm(affine_ref[1, :2])

        # Матриця повороту базового кадру відносно карти
        R_ref = np.array(
            [
                [affine_ref[0, 0] / scale_x, affine_ref[0, 1] / scale_y],
                [affine_ref[1, 0] / scale_x, affine_ref[1, 1] / scale_y],
            ],
            dtype=np.float32,
        )

        # 1. Обертаємо вектор зсуву відносно центру рамки пошуку
        # Якщо кадр перевертали перед матчингом (global_angle),
        # offset_px треба покрутити у протилежний бік
        theta = np.deg2rad(-global_angle)
        R_inv_global = np.array(
            [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]], dtype=np.float32
        )

        offset_rotated_base = R_inv_global @ offset_px

        # 2. Множимо на масштаб та матрицю повороту референсного кадру
        # щоб перевести у глобальні осі карти
        metric_offset = R_ref @ np.array(
            [offset_rotated_base[0] * scale_x, offset_rotated_base[1] * scale_y], dtype=np.float32
        )

        metric_pt = metric_centroid + metric_offset

        if self.outlier_detector.is_outlier(metric_pt, dt):
            return {"success": False, "error": "Outlier detected — position jump filtered"}

        if hasattr(self.trajectory_filter, "update_with_dt"):
            filtered_pt = self.trajectory_filter.update_with_dt(metric_pt, dt)
        else:
            filtered_pt = self.trajectory_filter.update(metric_pt, dt=dt)

        self.outlier_detector.add_position(filtered_pt)
        lat, lon = CoordinateConverter.metric_to_gps(filtered_pt[0], filtered_pt[1])

        half_w, half_h = (rot_width / 2.0) * scale_x, (rot_height / 2.0) * scale_y
        gps_corners = self._calculate_fov_polygon(
            filtered_pt[0], filtered_pt[1], half_w, half_h, global_angle, R_ref
        )

        max_inliers = self.config.get("localization", {}).get("confidence_max_inliers", 50)
        confidence = min(
            1.0, 0.6 * min(1.0, inliers / max_inliers) + 0.4 * (inliers / max(total_matches, 1))
        )

        logger.success(
            f"Localization: ({lat:.6f}, {lon:.6f}), matched frame={candidate_id}, "
            f"rot={global_angle}°, confidence={confidence:.2f}"
        )

        return {
            "success": True,
            "lat": lat,
            "lon": lon,
            "confidence": confidence,
            "matched_frame": candidate_id,
            "inliers": inliers,
            "fov_polygon": gps_corners,
        }

    def _calculate_fov_polygon(
        self,
        cx: float,
        cy: float,
        half_w: float,
        half_h: float,
        global_angle: int,
        R_ref: np.ndarray,
    ) -> list[tuple[float, float]]:
        """Конвертує метричні кути FOV у GPS координати з урахуванням поворотів."""
        gps_corners = []

        # Кути прямокутника (відносно центру)
        corners = np.array(
            [
                [-half_w, -half_h],
                [half_w, -half_h],
                [half_w, half_h],
                [-half_w, half_h],
            ],
            dtype=np.float32,
        )

        # 1. Поворот назад (якщо DINOv2 повернув кадр перед матчингом)
        theta = np.deg2rad(-global_angle)
        R_inv_global = np.array(
            [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]], dtype=np.float32
        )

        corners_rotated = (R_inv_global @ corners.T).T

        # 2. Поворот відносно курсу самого референсного кадру
        corners_world = (R_ref @ corners_rotated.T).T

        from contextlib import suppress

        for mcx, mcy in corners_world:
            with suppress(Exception):
                # Додаємо глобальні координати центру до обчислених зсувів
                clat, clon = CoordinateConverter.metric_to_gps(cx + mcx, cy + mcy)
                gps_corners.append((clat, clon))
        return gps_corners

    def _try_lightglue_fallback(self, query_frame, static_mask, candidates, height, width):
        """Fallback: SuperPoint+LightGlue для складних сцен де XFeat не справився"""
        best_inliers = 0
        best_candidate_id = -1
        best_mkpts_q_inliers = None
        best_mkpts_r_inliers = None
        best_rot_angle = 0
        best_total_matches = 0

        try:
            sp_model = self.model_manager.load_superpoint()
            lg_model = self.model_manager.load_lightglue()
            device = self.model_manager.device
        except Exception as e:
            logger.warning(f"Cannot load SuperPoint/LightGlue for fallback: {e}")
            return (
                best_inliers,
                best_candidate_id,
                best_mkpts_q_inliers,
                best_mkpts_r_inliers,
                best_rot_angle,
                best_total_matches,
            )

        # Підготовка запиту для SuperPoint
        from lightglue.utils import numpy_image_to_torch

        query_tensor = numpy_image_to_torch(query_frame).to(device)

        with torch.no_grad():
            sp_query = sp_model.extract(query_tensor)

        # Фільтрація точок за маскою
        if static_mask is not None:
            kpts = sp_query["keypoints"][0].cpu().numpy()
            ix = np.round(kpts[:, 0]).astype(np.intp)
            iy = np.round(kpts[:, 1]).astype(np.intp)
            in_bounds = (
                (iy >= 0) & (iy < static_mask.shape[0]) & (ix >= 0) & (ix < static_mask.shape[1])
            )
            valid = np.zeros(len(kpts), dtype=bool)
            valid[in_bounds] = static_mask[iy[in_bounds], ix[in_bounds]] > 128
            if valid.any():
                valid_t = torch.from_numpy(valid).to(device)
                sp_query = {
                    "keypoints": sp_query["keypoints"][:, valid_t],
                    "descriptors": sp_query["descriptors"][:, valid_t],
                    "keypoint_scores": sp_query["keypoint_scores"][:, valid_t]
                    if "keypoint_scores" in sp_query
                    else None,
                }
                # Видаляємо None
                sp_query = {k: v for k, v in sp_query.items() if v is not None}

        # Перебір top-3 кандидатів (менше для швидкості)
        for candidate_id, _score in candidates[:3]:
            ref_features = self.database.get_local_features(candidate_id)

            # Підготовка ref для LightGlue
            ref_kpts = torch.from_numpy(ref_features["keypoints"]).float()[None].to(device)
            ref_desc = torch.from_numpy(ref_features["descriptors"]).float()[None].to(device)

            # Якщо ref дескриптори 64-dim (XFeat), fallback не підходить
            if ref_desc.shape[-1] != 256:
                logger.debug(
                    f"Skipping LightGlue fallback for frame {candidate_id}: ref desc dim={ref_desc.shape[-1]}"
                )
                continue

            try:
                with torch.no_grad():
                    data = {
                        "image0": sp_query,
                        "image1": {"keypoints": ref_kpts, "descriptors": ref_desc},
                    }
                    res = lg_model(data)
                    matches = res["matches"][0].cpu().numpy()

                if len(matches) >= self.min_matches:
                    q_kpts = sp_query["keypoints"][0].cpu().numpy()
                    mkpts_q = q_kpts[matches[:, 0]]
                    mkpts_r = ref_features["keypoints"][matches[:, 1]]

                    M, mask = GeometryTransforms.estimate_affine(
                        mkpts_q, mkpts_r, ransac_threshold=self.ransac_thresh
                    )
                    if M is not None:
                        inlier_mask = mask.ravel().astype(bool)
                        inliers = int(np.sum(inlier_mask))
                        if inliers > best_inliers:
                            best_inliers = inliers
                            best_candidate_id = candidate_id
                            best_mkpts_q_inliers = mkpts_q[inlier_mask]
                            best_mkpts_r_inliers = mkpts_r[inlier_mask]
                            best_rot_angle = 0
                            best_total_matches = len(matches)
                            logger.info(
                                f"LightGlue fallback: {inliers} inliers on frame {candidate_id}"
                            )
            except Exception as e:
                logger.debug(f"LightGlue fallback failed for frame {candidate_id}: {e}")
                continue

        return (
            best_inliers,
            best_candidate_id,
            best_mkpts_q_inliers,
            best_mkpts_r_inliers,
            best_rot_angle,
            best_total_matches,
        )
