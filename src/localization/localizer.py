import cv2
import numpy as np
from src.geometry.transformations import GeometryTransforms
from src.localization.matcher import FastRetrieval
from src.tracking.kalman_filter import TrajectoryFilter
from src.tracking.outlier_detector import OutlierDetector
from src.utils.logging_utils import get_logger
from src.geometry.coordinates import CoordinateConverter
import torch

logger = get_logger(__name__)

class Localizer:
    def __init__(self, database, feature_extractor, matcher, calibration, config=None):
        self.database = database
        self.feature_extractor = feature_extractor
        self.matcher = matcher
        self.calibration = calibration
        self.config = config or {}

        self.min_matches = self.config.get('localization', {}).get('min_matches', 8)
        self.ransac_thresh = self.config.get('localization', {}).get('ransac_threshold', 5.0)
        self.enable_auto_rotation = self.config.get('localization', {}).get('auto_rotation', True)

        self.trajectory_filter = TrajectoryFilter(
            process_noise=self.config.get('tracking', {}).get('kalman_process_noise', 2.0),
            measurement_noise=self.config.get('tracking', {}).get('kalman_measurement_noise', 5.0),
            dt=1.0
        )
        self.outlier_detector = OutlierDetector(
            window_size=self.config.get('tracking', {}).get('outlier_window', 10),
            threshold_std=self.config.get('tracking', {}).get('outlier_threshold_std', 3.0),
            max_speed_mps=self.config.get('tracking', {}).get('max_speed_mps', 1000.0)
        )

        # Створюємо FastRetrieval один раз — нормалізація дескрипторів відбувається лише тут
        self.retriever = FastRetrieval(self.database.global_descriptors)

        # Fallback: SuperPoint+LightGlue для складних сцен
        self.model_manager = self.config.get('_model_manager', None)
        self.fallback_enabled = self.config.get('localization', {}).get('enable_lightglue_fallback', True)
        self.min_inliers_for_accept = self.config.get('localization', {}).get('min_inliers_accept', 8)

    def localize_frame(self, query_frame: np.ndarray, static_mask: np.ndarray = None, dt: float = 1.0) -> dict:
        height, width = query_frame.shape[:2]

        angles_to_try = [0, 90, 180, 270] if self.enable_auto_rotation else [0]
        
        best_global_score = -1.0
        best_global_angle = 0
        best_global_candidates = []
        best_query_features = None

        top_k = self.config.get('localization', {}).get('retrieval_top_k', 8)

        # 1. Екстракція ознак для всіх дозволених кутів обертання та вибір найкращого ракурсу
        for angle in angles_to_try:
            k = angle // 90
            rotated_frame = np.rot90(query_frame, k=k).copy()
            
            # Витягуємо ТІЛЬКИ глобальний дескриптор DINOv2 для швидкого пошуку ракурсу
            global_desc = self.feature_extractor.extract_global_descriptor(rotated_frame)
            
            # Шукаємо кандидатів за допомогою DINOv2
            candidates = self.retriever.find_similar_frames(global_desc, top_k=top_k)
            
            if candidates:
                # Оцінкою ракурсу вважаємо скор найкращого кандидата
                top_score = candidates[0][1]
                if top_score > best_global_score:
                    best_global_score = top_score
                    best_global_angle = angle
                    best_global_candidates = candidates

        if not best_global_candidates:
            return {"success": False, "error": "No candidates found via global descriptor (DINOv2) in any rotation"}

        logger.info(f"Selected best rotation {best_global_angle}° with global score {best_global_score:.3f}")

        # 1.5. Локальна екстракція (XFeat) ТІЛЬКИ для НАЙКРАЩОГО ракурсу
        k = best_global_angle // 90
        best_rotated_frame = np.rot90(query_frame, k=k).copy()
        best_rotated_mask = np.rot90(static_mask, k=k).copy() if static_mask is not None else None
        
        # Обчислюємо ключові точки лише один раз для обраного кута!
        best_query_features = self.feature_extractor.extract_local_features(best_rotated_frame, static_mask=best_rotated_mask)

        best_inliers = 0
        best_candidate_id = -1
        best_mkpts_q_inliers = None
        best_mkpts_r_inliers = None
        best_total_matches = 0

        early_stop = self.config.get('localization', {}).get('early_stop_inliers', 30)

        # 2. Локальний пошук (XFeat) ТІЛЬКИ для найкращого знайденого ракурсу
        for candidate_id, score in best_global_candidates:
            ref_features = self.database.get_local_features(candidate_id)

            mkpts_q, mkpts_r = self.matcher.match(best_query_features, ref_features)

            if len(mkpts_q) >= self.min_matches:
                # Affine (6 DoF) замість Homography (8 DoF) — стабільніше для аерофото
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
                logger.info(f"Early stop triggered with {best_inliers} inliers on candidate {best_candidate_id}")
                break

        # 2b. Fallback: SuperPoint+LightGlue якщо XFeat дав мало inliers
        if (best_inliers < self.min_inliers_for_accept
                and self.fallback_enabled and self.model_manager is not None):
            logger.info(f"XFeat gave only {best_inliers} inliers, trying SuperPoint+LightGlue fallback on best angle {best_global_angle}°...")
            
            # Відправляємо вже ПОВЕРНУТИЙ кадр і маску на Fallback
            k = best_global_angle // 90
            rotated_frame = np.rot90(query_frame, k=k).copy()
            rotated_mask = np.rot90(static_mask, k=k).copy() if static_mask is not None else None
            
            fb_inliers, fb_candidate_id, fb_mkpts_q, fb_mkpts_r, fb_rot, fb_total = self._try_lightglue_fallback(
                rotated_frame, rotated_mask, best_global_candidates, height, width
            )
            
            # Оскільки fallback отримав вже повернутий кадр, кут відносно цього кадру 0°, 
            # але сумарний кут залишається best_global_angle.
            if fb_inliers > best_inliers:
                 best_inliers = fb_inliers
                 best_candidate_id = fb_candidate_id
                 best_mkpts_q_inliers = fb_mkpts_q
                 best_mkpts_r_inliers = fb_mkpts_r
                 best_total_matches = fb_total

        if best_inliers < self.min_matches or best_mkpts_r_inliers is None:
            return {"success": False, "error": f"Not enough valid inliers ({best_inliers} < {self.min_matches})"}

        # Фізичні розміри ПОВЕРНУТОГО кадру
        if best_global_angle in [90, 270]:
            rot_height, rot_width = width, height
        else:
            rot_height, rot_width = height, width

        # 3. Трансформація координат — через центроїд інлієрів (завжди в межах кадру!)
        centroid_ref = np.mean(best_mkpts_r_inliers, axis=0)  # Центр збігу в ref (в межах кадру)
        centroid_query = np.mean(best_mkpts_q_inliers, axis=0)

        # Зсув від центроїда збігу до центру кадру (в пікселях запиту)
        center_query = np.array([rot_width / 2.0, rot_height / 2.0], dtype=np.float32)
        offset_px = center_query - centroid_query  # Зсув в пікселях

        affine_ref = self.database.get_frame_affine(best_candidate_id)
        if affine_ref is None:
            return {"success": False, "error": "No propagated calibration"}

        # Центроїд ref → metric
        centroid_ref_arr = np.array([centroid_ref], dtype=np.float32)
        metric_centroid = GeometryTransforms.apply_affine(centroid_ref_arr, affine_ref)
        if metric_centroid is None or len(metric_centroid) == 0:
            return {"success": False, "error": "Projection failed"}
        metric_centroid = metric_centroid[0]

        # Перетворити піксельний зсув в метричний (через масштаб affine)
        scale_x = np.linalg.norm(affine_ref[0, :2])  # м/піксель по X
        scale_y = np.linalg.norm(affine_ref[1, :2])  # м/піксель по Y
        metric_offset = np.array([offset_px[0] * scale_x, offset_px[1] * scale_y], dtype=np.float32)

        metric_pt = metric_centroid + metric_offset

        # DEBUG: показати кожен крок
        logger.info(f"COORD DEBUG: centroid_ref=({centroid_ref[0]:.1f}, {centroid_ref[1]:.1f}), "
                     f"offset_px=({offset_px[0]:.1f}, {offset_px[1]:.1f}), frame={best_candidate_id}")
        logger.info(f"COORD DEBUG: metric_pt=({metric_pt[0]:.2f}, {metric_pt[1]:.2f})")

        # Перевіряємо чи нова точка — аномалія (стрибок координат)
        if self.outlier_detector.is_outlier(metric_pt, dt):
            return {"success": False, "error": "Outlier detected — position jump filtered"}

        # 4. Фільтр Калмана
        if hasattr(self.trajectory_filter, 'update_with_dt'):
            filtered_pt = self.trajectory_filter.update_with_dt(metric_pt, dt)
        else:
            filtered_pt = self.trajectory_filter.update(metric_pt, dt=dt)

        self.outlier_detector.add_position(filtered_pt)
        lat, lon = CoordinateConverter.metric_to_gps(filtered_pt[0], filtered_pt[1])

        # 5. Розрахунок FOV навколо метричної точки
        half_w = (rot_width / 2.0) * scale_x
        half_h = (rot_height / 2.0) * scale_y

        gps_corners = []
        fov_metric_corners = [
            (filtered_pt[0] - half_w, filtered_pt[1] - half_h),
            (filtered_pt[0] + half_w, filtered_pt[1] - half_h),
            (filtered_pt[0] + half_w, filtered_pt[1] + half_h),
            (filtered_pt[0] - half_w, filtered_pt[1] + half_h),
        ]
        for cx, cy in fov_metric_corners:
            try:
                clat, clon = CoordinateConverter.metric_to_gps(cx, cy)
                gps_corners.append((clat, clon))
            except Exception:
                pass

        # Покращена формула confidence: враховує inlier ratio + кількість
        max_inliers = self.config.get('localization', {}).get('confidence_max_inliers', 50)
        inlier_ratio = best_inliers / max(best_total_matches, 1)
        count_score = min(1.0, best_inliers / max_inliers)
        # Зважена комбінація: 60% від кількості inliers, 40% від їх частки
        confidence = min(1.0, 0.6 * count_score + 0.4 * inlier_ratio)

        logger.success(
            f"Localization: ({lat:.6f}, {lon:.6f}), matched frame={best_candidate_id}, "
            f"rot={best_global_angle}°, confidence={confidence:.2f}")

        return {
            "success": True, "lat": lat, "lon": lon,
            "confidence": confidence, "matched_frame": best_candidate_id,
            "inliers": best_inliers, "fov_polygon": gps_corners
        }

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
            return best_inliers, best_candidate_id, best_mkpts_q_inliers, best_mkpts_r_inliers, best_rot_angle, best_total_matches

        # Підготовка запиту для SuperPoint
        from lightglue.utils import numpy_image_to_torch
        query_tensor = numpy_image_to_torch(query_frame).to(device)

        with torch.no_grad():
            sp_query = sp_model.extract(query_tensor)

        # Фільтрація точок за маскою
        if static_mask is not None:
            kpts = sp_query['keypoints'][0].cpu().numpy()
            ix = np.round(kpts[:, 0]).astype(np.intp)
            iy = np.round(kpts[:, 1]).astype(np.intp)
            in_bounds = (iy >= 0) & (iy < static_mask.shape[0]) & (ix >= 0) & (ix < static_mask.shape[1])
            valid = np.zeros(len(kpts), dtype=bool)
            valid[in_bounds] = static_mask[iy[in_bounds], ix[in_bounds]] > 128
            if valid.any():
                valid_t = torch.from_numpy(valid).to(device)
                sp_query = {
                    'keypoints': sp_query['keypoints'][:, valid_t],
                    'descriptors': sp_query['descriptors'][:, valid_t],
                    'keypoint_scores': sp_query['keypoint_scores'][:, valid_t] if 'keypoint_scores' in sp_query else None,
                }
                # Видаляємо None
                sp_query = {k: v for k, v in sp_query.items() if v is not None}

        # Перебір top-3 кандидатів (менше для швидкості)
        for candidate_id, score in candidates[:3]:
            ref_features = self.database.get_local_features(candidate_id)

            # Підготовка ref для LightGlue
            ref_kpts = torch.from_numpy(ref_features['keypoints']).float()[None].to(device)
            ref_desc = torch.from_numpy(ref_features['descriptors']).float()[None].to(device)

            # Якщо ref дескриптори 64-dim (XFeat), fallback не підходить
            if ref_desc.shape[-1] != 256:
                logger.debug(f"Skipping LightGlue fallback for frame {candidate_id}: ref desc dim={ref_desc.shape[-1]}")
                continue

            try:
                with torch.no_grad():
                    data = {
                        'image0': sp_query,
                        'image1': {'keypoints': ref_kpts, 'descriptors': ref_desc}
                    }
                    res = lg_model(data)
                    matches = res['matches'][0].cpu().numpy()

                if len(matches) >= self.min_matches:
                    q_kpts = sp_query['keypoints'][0].cpu().numpy()
                    mkpts_q = q_kpts[matches[:, 0]]
                    mkpts_r = ref_features['keypoints'][matches[:, 1]]

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
                            logger.info(f"LightGlue fallback: {inliers} inliers on frame {candidate_id}")
            except Exception as e:
                logger.debug(f"LightGlue fallback failed for frame {candidate_id}: {e}")
                continue

        return best_inliers, best_candidate_id, best_mkpts_q_inliers, best_mkpts_r_inliers, best_rot_angle, best_total_matches