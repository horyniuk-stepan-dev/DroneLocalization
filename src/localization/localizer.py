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
        best_H_query_to_ref = None
        best_mkpts_q_inliers = None
        best_mkpts_r_inliers = None
        best_total_matches = 0

        early_stop = self.config.get('localization', {}).get('early_stop_inliers', 30)

        # 2. Локальний пошук (XFeat) ТІЛЬКИ для найкращого знайденого ракурсу
        for candidate_id, score in best_global_candidates:
            ref_features = self.database.get_local_features(candidate_id)

            mkpts_q, mkpts_r = self.matcher.match(best_query_features, ref_features)

            if len(mkpts_q) >= self.min_matches:
                # Використовуємо Full Affine (6 DoF: поворот, масштаб X/Y, shear, зсув)
                # Точніший варіант — підтримує анізотропний масштаб між query та ref
                M_eval, mask = GeometryTransforms.estimate_affine(
                    mkpts_q, mkpts_r, ransac_threshold=self.ransac_thresh
                )

                if M_eval is not None:
                    inlier_mask = mask.ravel().astype(bool)
                    inliers = int(np.sum(inlier_mask))
                    if inliers > best_inliers and inliers >= self.min_matches:
                        best_inliers = inliers
                        best_candidate_id = candidate_id
                        best_H_query_to_ref = M_eval  # Full Affine матриця (2x3)
                        best_mkpts_q_inliers = mkpts_q[inlier_mask]
                        best_mkpts_r_inliers = mkpts_r[inlier_mask]
                        best_total_matches = len(mkpts_q)
                        # Діагностика параметрів афінної матриці
                        sx = np.linalg.norm(M_eval[:, 0])
                        sy = np.linalg.norm(M_eval[:, 1])
                        angle_deg = np.degrees(np.arctan2(M_eval[1, 0], M_eval[0, 0]))
                        logger.debug(
                            f"Affine params for candidate {candidate_id}: "
                            f"scale_x={sx:.3f}, scale_y={sy:.3f}, "
                            f"rotation={angle_deg:.1f}°, "
                            f"tx={M_eval[0, 2]:.1f}, ty={M_eval[1, 2]:.1f}"
                        )

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
            
            fb_result = self._try_lightglue_fallback(
                rotated_frame, rotated_mask, best_global_candidates, height, width
            )

            if fb_result is None:
                logger.warning("LightGlue fallback returned None unexpectedly, skipping.")
            elif len(fb_result) == 7:
                fb_inliers, fb_candidate_id, fb_H_to_ref, fb_mkpts_q, fb_mkpts_r, fb_rot, fb_total = fb_result
                # Оскільки fallback отримав вже повернутий кадр, кут відносно цього кадру 0°, 
                # але сумарний кут залишається best_global_angle.
                if fb_inliers > best_inliers:
                     best_inliers = fb_inliers
                     best_candidate_id = fb_candidate_id
                     best_H_query_to_ref = fb_H_to_ref
                     best_mkpts_q_inliers = fb_mkpts_q
                     best_mkpts_r_inliers = fb_mkpts_r
                     best_total_matches = fb_total
            else:
                logger.warning(f"LightGlue fallback returned unexpected result shape: {fb_result}, skipping.")

        if best_inliers < self.min_matches or best_mkpts_r_inliers is None or best_H_query_to_ref is None:
            # Спробуємо фоллбек перед тим як повертати помилку.
            # Якщо ми не знайшли жодного відповідного кадру через Matching, беремо топ-1 з retrieval.
            target_id = best_candidate_id if (best_candidate_id != -1) else best_global_candidates[0][0]
            fallback_res = self._localize_by_reference_frame(target_id, best_global_score)
            if fallback_res:
                logger.info(f"Feature matching failed ({best_inliers} inliers), using retrieval-only fallback for frame {target_id} (score {best_global_score:.3f})")
                return fallback_res
            return {"success": False, "error": f"Not enough valid inliers ({best_inliers} < {self.min_matches})"}

        # 3. Отримуємо матрицю знайденого кадру з бази
        affine_ref = self.database.get_frame_affine(best_candidate_id)
        if affine_ref is None:
            target_id = best_candidate_id if (best_candidate_id != -1) else best_global_candidates[0][0]
            fallback_res = self._localize_by_reference_frame(target_id, best_global_score)
            if fallback_res:
                 logger.info(f"No propagated calibration for frame {target_id}, using retrieval-only fallback")
                 return fallback_res
            return {"success": False, "error": "No propagated calibration"}

        # 4. Рахуємо розміри ПОВЕРНУТОГО зображення
        if best_global_angle in [90, 270]:
            rot_height, rot_width = width, height
        else:
            rot_height, rot_width = height, width
            
        # 4. Багатоточкова локалізація більше не потрібна. Беремо ідеальний центр кадру
        proj_cfg = self.config.get('projection', {})
        
        # Використовуємо знайдений affine_partial
        M_query_to_ref = best_H_query_to_ref
        if M_query_to_ref is None:
            return {"success": False, "error": "Failed to compute affine transform"}

        # 5. Трансформуємо центральну точку: Query -> Reference -> Metric
        center_query = np.array([[rot_width / 2.0, rot_height / 2.0]], dtype=np.float32)
        pts_in_ref = GeometryTransforms.apply_affine(center_query, M_query_to_ref)
        if pts_in_ref is None or len(pts_in_ref) == 0:
            target_id = best_candidate_id if (best_candidate_id != -1) else best_global_candidates[0][0]
            fallback_res = self._localize_by_reference_frame(target_id, best_global_score)
            if fallback_res:
                logger.info(f"Affine transform failure, using retrieval-only fallback for frame {target_id} (score {best_global_score:.3f})")
                return fallback_res
            return {"success": False, "error": "Coordinate transformation error (affine failed)"}

        pts_metric = GeometryTransforms.apply_affine(pts_in_ref, affine_ref)
        
        # Оскільки ми взяли одну центральну точку, просто беремо її координати
        mx = float(pts_metric[0, 0])
        my = float(pts_metric[0, 1])
        metric_pt = np.array([mx, my], dtype=np.float32)

        # 6. Перевіряємо чи нова точка — аномалія (стрибок координат)
        if self.outlier_detector.is_outlier(metric_pt, dt):
            # Ми все одно логуємо спробу, але кажемо, що це аутлаєр
            logger.warning(f"Outlier filtered at frame {best_candidate_id}: jump from previous trajectory")
            return {"success": False, "error": "Outlier detected — position jump filtered"}

        # Оновлення Калмана (фільтрація шумів)
        if hasattr(self.trajectory_filter, 'update_with_dt'):
            filtered_pt = self.trajectory_filter.update_with_dt(metric_pt, dt)
        else:
            filtered_pt = self.trajectory_filter.update(metric_pt, dt=dt)

        self.outlier_detector.add_position(filtered_pt)
        lat, lon = CoordinateConverter.metric_to_gps(filtered_pt[0], filtered_pt[1])

        # Зсув для корекції FOV через фільтрацію
        dx, dy = filtered_pt[0] - metric_pt[0], filtered_pt[1] - metric_pt[1]

        # 7. Розрахунок поля зору (FOV)
        # Проектуємо кути QUERY-кадру через ланцюг: Query→Ref→Metric→GPS
        corners = np.array([
            [0, 0], [rot_width, 0], [rot_width, rot_height], [0, rot_height]
        ], dtype=np.float32)
        ref_corners = GeometryTransforms.apply_affine(corners, M_query_to_ref)

        gps_corners = []
        if ref_corners is not None:
            metric_corners = GeometryTransforms.apply_affine(ref_corners, affine_ref)
            if metric_corners is not None:
                # Діагностика розмірів FOV в метрах
                fov_w = np.linalg.norm(metric_corners[1] - metric_corners[0])
                fov_h = np.linalg.norm(metric_corners[3] - metric_corners[0])
                logger.debug(
                    f"FOV dimensions: {fov_w:.1f}m x {fov_h:.1f}m | "
                    f"Center metric: ({mx:.1f}, {my:.1f}) | "
                    f"Filtered: ({filtered_pt[0]:.1f}, {filtered_pt[1]:.1f})"
                )
                for cx, cy in metric_corners:
                    try:
                        clat, clon = CoordinateConverter.metric_to_gps(cx + dx, cy + dy)
                        gps_corners.append((clat, clon))
                    except Exception:
                        pass

        # 8. Confidence scoring (QA)
        max_inliers = proj_cfg.get('confidence_max_inliers', 80)
        inlier_score = min(1.0, best_inliers / max_inliers)
        
        # Full Affine підтримує анізотропний масштаб, стабільність завжди 1.0 для дрон-камери
        stability_score = 1.0
        
        confidence = float(np.clip(inlier_score * 0.7 + stability_score * 0.3, 0.05, 1.0))

        # ДІАГНОСТИКА
        logger.debug(f"Localize Frame {best_candidate_id}: Center transformed via Full Affine (6 DoF)")
        logger.debug(f"Sample Center METRIC: ({mx:.1f}, {my:.1f})")
        
        logger.success(
            f"Localized ({lat:.6f}, {lon:.6f}) | frame={best_candidate_id} | "
            f"metric=({mx:.1f}, {my:.1f}) | inliers={best_inliers} | conf={confidence:.2f}"
        )

        return {
            "success": True, 
            "lat": lat, "lon": lon,
            "confidence": confidence, 
            "matched_frame": int(best_candidate_id),
            "inliers": int(best_inliers), 
            "fov_polygon": gps_corners,
            "sample_spread_m": 0.0
        }


    def _try_lightglue_fallback(self, query_frame, static_mask, candidates, height, width):
        """Fallback: SuperPoint+LightGlue для складних сцен де XFeat не справився"""
        best_inliers = 0
        best_candidate_id = -1
        best_H_to_ref = None
        best_mkpts_q_inliers = None
        best_mkpts_r_inliers = None
        best_rot_angle = 0
        best_total_matches = 0

        fallback_type = self.config.get('localization', {}).get('fallback_extractor', 'aliked')

        try:
            import torch
            device = self.model_manager.device

            if fallback_type == 'aliked':
                # ═══ ALIKED (128-dim) fallback ═══
                aliked_model = self.model_manager.load_aliked()
                from lightglue.utils import numpy_image_to_torch

                query_tensor = numpy_image_to_torch(query_frame).to(device)
                with torch.no_grad():
                    aliked_query = aliked_model.extract(query_tensor)

                # Фільтрація точок за YOLO маскою
                if static_mask is not None:
                    kpts = aliked_query['keypoints'][0].cpu().numpy()
                    ix = np.round(kpts[:, 0]).astype(np.intp)
                    iy = np.round(kpts[:, 1]).astype(np.intp)
                    in_bounds = (iy >= 0) & (iy < static_mask.shape[0]) & (ix >= 0) & (ix < static_mask.shape[1])
                    valid = np.zeros(len(kpts), dtype=bool)
                    valid[in_bounds] = static_mask[iy[in_bounds], ix[in_bounds]] > 128
                    if valid.any():
                        valid_t = torch.from_numpy(valid).to(device)
                        aliked_query = {
                            'keypoints': aliked_query['keypoints'][:, valid_t],
                            'descriptors': aliked_query['descriptors'][:, valid_t],
                            'keypoint_scores': aliked_query.get('keypoint_scores',
                                               aliked_query['keypoints'].new_ones(1, int(valid_t.sum())))[:, :int(valid_t.sum())],
                        }
                        aliked_query = {k: v for k, v in aliked_query.items() if v is not None}

                # Перебір top-3 кандидатів
                for candidate_id, score in candidates[:3]:
                    ref_features = self.database.get_local_features(candidate_id)
                    ref_desc_dim = ref_features['descriptors'].shape[1] if len(ref_features['descriptors']) > 0 else 0

                    try:
                        q_kpts = aliked_query['keypoints'][0].cpu().numpy()
                        q_desc = aliked_query['descriptors'][0].cpu().numpy()

                        query_dict = {'keypoints': q_kpts, 'descriptors': q_desc}

                        if ref_desc_dim == 128:
                            # Ref також ALIKED → можна використати LightGlue(aliked) або Numpy L2
                            mkpts_q, mkpts_r = self.matcher.match(query_dict, ref_features)
                        else:
                            # Ref має XFeat (64-dim) → cross-descriptor через Numpy L2
                            # ALIKED 128-dim vs XFeat 64-dim — безпосередній матчинг неможливий
                            # Тому екстрагуємо ALIKED features з reference frame зображення
                            # Fallback: пропускаємо LightGlue, використовуємо Numpy L2
                            logger.debug(
                                f"ALIKED fallback: ref desc dim={ref_desc_dim} (XFeat), "
                                f"matching ALIKED query (128-dim) vs XFeat ref (64-dim) not possible directly. "
                                f"Skipping candidate {candidate_id}."
                            )
                            continue

                        if len(mkpts_q) >= self.min_matches:
                            M_eval, mask = GeometryTransforms.estimate_affine(
                                mkpts_q, mkpts_r, ransac_threshold=self.ransac_thresh
                            )
                            if M_eval is not None:
                                inlier_mask = mask.ravel().astype(bool)
                                inliers = int(np.sum(inlier_mask))
                                if inliers > best_inliers and inliers >= self.min_matches:
                                    best_inliers = inliers
                                    best_candidate_id = candidate_id
                                    best_H_to_ref = M_eval
                                    best_mkpts_q_inliers = mkpts_q[inlier_mask]
                                    best_mkpts_r_inliers = mkpts_r[inlier_mask]
                                    best_rot_angle = 0
                                    best_total_matches = len(mkpts_q)
                                    logger.info(f"ALIKED fallback: {inliers} inliers on frame {candidate_id}")
                    except Exception as e:
                        logger.debug(f"ALIKED fallback failed for frame {candidate_id}: {e}")
                        continue

            else:
                # ═══ SuperPoint (256-dim) fallback (legacy) ═══
                sp_model = self.model_manager.load_superpoint()
                lg_model = self.model_manager.load_lightglue()

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
                        sp_query = {k: v for k, v in sp_query.items() if v is not None}

                lg_compatible = False
                for candidate_id, score in candidates[:3]:
                    ref_features = self.database.get_local_features(candidate_id)
                    ref_kpts = torch.from_numpy(ref_features['keypoints']).float()[None].to(device)
                    ref_desc = torch.from_numpy(ref_features['descriptors']).float()[None].to(device)

                    if ref_desc.shape[-1] != 256:
                        logger.debug(f"Skipping LightGlue fallback for frame {candidate_id}: ref desc dim={ref_desc.shape[-1]}")
                        continue

                    lg_compatible = True
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

                            M_eval, mask = GeometryTransforms.estimate_affine(
                                mkpts_q, mkpts_r, ransac_threshold=self.ransac_thresh
                            )
                            if M_eval is not None:
                                inlier_mask = mask.ravel().astype(bool)
                                inliers = int(np.sum(inlier_mask))
                                if inliers > best_inliers and inliers >= self.min_matches:
                                    best_inliers = inliers
                                    best_candidate_id = candidate_id
                                    best_H_to_ref = M_eval
                                    best_mkpts_q_inliers = mkpts_q[inlier_mask]
                                    best_mkpts_r_inliers = mkpts_r[inlier_mask]
                                    best_rot_angle = 0
                                    best_total_matches = len(matches)
                                    logger.info(f"LightGlue fallback: {inliers} inliers on frame {candidate_id}")
                    except Exception as e:
                        logger.debug(f"LightGlue fallback failed for frame {candidate_id}: {e}")
                        continue

                if not lg_compatible:
                    logger.warning("LightGlue fallback: all candidates have XFeat (64-dim) descriptors — incompatible with SuperPoint. "
                                   "Consider switching to fallback_extractor='aliked' or re-index database.")

        except Exception as e:
            logger.warning(f"{fallback_type.upper()} fallback aborted with unexpected error: {e}", exc_info=True)

        return best_inliers, best_candidate_id, best_H_to_ref, best_mkpts_q_inliers, best_mkpts_r_inliers, best_rot_angle, best_total_matches

    def _localize_by_reference_frame(self, frame_id: int, score: float) -> dict:
        """Приблизна локалізація за центром опорного кадру (retrieval-only fallback)"""
        if frame_id == -1:
            return None
            
        threshold = self.config.get('localization', {}).get('retrieval_only_min_score', 0.90)
        if score < threshold:
            return None

        affine_ref = self.database.get_frame_affine(frame_id)
        if affine_ref is None:
            return None

        ref_h, ref_w = self.database.get_frame_size(frame_id)
        # Центр кадру в системі координат БД
        center_ref = np.array([[ref_w / 2, ref_h / 2]], dtype=np.float32)
        metric_pt = GeometryTransforms.apply_affine(center_ref, affine_ref)[0]
        
        lat, lon = CoordinateConverter.metric_to_gps(float(metric_pt[0]), float(metric_pt[1]))
        
        return {
            "success": True,
            "lat": lat, "lon": lon,
            "confidence": 0.3, # Низький confidence для retrieval-only
            "inliers": 0,
            "matched_frame": int(frame_id),
            "fallback_mode": "retrieval_only",
            "global_score": float(score),
            "fov_polygon": None
        }