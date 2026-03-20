import h5py
import numpy as np
import cv2
from pathlib import Path
from threading import Thread
from queue import Queue

from datetime import datetime
from src.utils.logging_utils import get_logger

from src.models.wrappers.yolo_wrapper import YOLOWrapper
from src.models.wrappers.feature_extractor import FeatureExtractor

logger = get_logger(__name__)


class DatabaseBuilder:
    """Builds HDF5 topometric database from reference video using XFeat & DINOv2"""

    def __init__(self, output_path, config=None):
        self.output_path = output_path
        self.config = config or {}
        self.descriptor_dim = self.config.get('dinov2', {}).get('descriptor_dim', 384)
        self.db_file = None

        logger.info(f"DatabaseBuilder initialized with output: {output_path}")
        logger.info(f"DINOv2 descriptor dimension: {self.descriptor_dim}")

    def build_from_video(self, video_path: str, model_manager, progress_callback=None,
                         save_keypoint_video: bool = True):
        """
        Process video and build database.

        Args:
            save_keypoint_video: якщо True — поруч з HDF5 збережеться відео
                                  *_keypoints.mp4 де на кожному кадрі намальовані
                                  точки XFeat (зелені) і маска YOLO (червона зона).
                                  Це допомагає визначити найкращі кадри для GPS калібрування.
        """
        logger.info(f"Starting database build from video: {video_path}")

        # Читаємо налаштування з конфігу (з дефолтом)
        frame_step = self.config.get('database', {}).get('frame_step', 3)
        if frame_step < 1:
            frame_step = 1

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error(f"Failed to open video: {video_path}")
            raise ValueError(f"Не вдалося відкрити відео: {video_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        original_fps = cap.get(cv2.CAP_PROP_FPS)

        # Обчислюємо скільки кадрів РЕАЛЬНО буде оброблено
        num_frames = (total_frames + frame_step - 1) // frame_step
        effective_fps = original_fps / frame_step

        if num_frames <= 0:
            cap.release()
            logger.error(f"Invalid frame count ({num_frames}). Video might be corrupted or uses unsupported codec.")
            raise ValueError(
                "OpenCV не зміг розпізнати відео. Файл пошкоджений або використовує непідтримуваний кодек. "
                "Спробуйте переконвертувати відео у стандартний MP4 (H.264)."
            )

        logger.info(f"Video properties: {width}x{height}, {total_frames} total frames, {original_fps:.2f} FPS")
        logger.info(f"Processing with step={frame_step} -> {num_frames} frames to process ({effective_fps:.2f} effective FPS)")

        # Ініціалізуємо запис відео з keypoints
        kp_video_path = None
        kp_writer = None
        kp_scale = 1.0
        if save_keypoint_video:
            # Зменшуємо роздільну здатність для великих відео (наприклад, 1080p і вище) щоб зекономити місце
            if width >= 1920:
                kp_scale = 0.5
                
            kp_width = int(width * kp_scale)
            kp_height = int(height * kp_scale)
            
            kp_video_path = str(Path(self.output_path).with_suffix('')) + '_keypoints.mp4'
            # Використовуємо H.264 (avc1) кодек для кращої компресії замість старого mp4v
            fourcc = cv2.VideoWriter_fourcc(*'avc1')
            kp_writer = cv2.VideoWriter(kp_video_path, fourcc, effective_fps, (kp_width, kp_height))
            
            if not kp_writer.isOpened():
                logger.warning("H.264 (avc1) codec not available, falling back to mp4v")
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                kp_writer = cv2.VideoWriter(kp_video_path, fourcc, effective_fps, (kp_width, kp_height))

            if kp_writer.isOpened():
                logger.info(f"Keypoint video will be saved to: {kp_video_path} at {kp_width}x{kp_height} (Scale: {kp_scale})")
            else:
                logger.warning("Failed to initialize keypoint video writer, skipping")
                kp_writer = None

        # Initialize neural network wrappers
        logger.info("Loading neural network models...")
        yolo_model = model_manager.load_yolo()
        yolo_wrapper = YOLOWrapper(yolo_model, model_manager.device)

        # ОНОВЛЕНО: Використовуємо XFeat замість SuperPoint
        local_model = model_manager.load_xfeat()
        nv_model = model_manager.load_dinov2()

        cesp = None
        if self.config.get('models', {}).get('cesp', {}).get('enabled', False):
            try:
                cesp = model_manager.load_cesp()
            except Exception:
                logger.warning("CESP loading failed during DB build, continuing without it")

        feature_extractor = FeatureExtractor(local_model, nv_model, model_manager.device, config=self.config, cesp_module=cesp)
        logger.success("All models loaded successfully")

        # Create empty database structure
        logger.info("Creating HDF5 database structure...")
        self.create_hdf5_structure(num_frames, width, height)

        # Накопичена поза: H(frame_i → frame_0), починаємо з identity для кадру 0
        current_pose = np.eye(3, dtype=np.float32)
        prev_features = None  # фічі попереднього кадру для матчингу

        # cuDNN benchmark для прискорення (input size не змінюється)
        import torch
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            logger.info("cuDNN benchmark ENABLED for consistent input sizes")

        # Тредовий prefetch відеокадрів — CPU декодує наступний кадр поки GPU обробляє поточний
        frame_queue = Queue(maxsize=4)

        def prefetch_frames():
            for i in range(total_frames):
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Пропускаємо кадри згідно з frame_step
                if i % frame_step != 0:
                    continue
                    
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # Зберігаємо "новий" індекс (0, 1, 2...) для бази
                db_index = i // frame_step
                frame_queue.put((db_index, (frame, frame_rgb)))
            
            frame_queue.put((-1, None))  # sentinel

        prefetch_thread = Thread(target=prefetch_frames, daemon=True)
        prefetch_thread.start()

        try:
            self.db_file = h5py.File(self.output_path, 'a')
            logger.info(f"Opened HDF5 file for writing: {self.output_path}")

            while True:
                idx, data = frame_queue.get()
                if idx == -1 or data is None:
                    if idx != -1:
                        logger.warning(f"Failed to read frame {idx}, stopping processing")
                    break

                frame, frame_rgb = data
                i = idx

                # Step 1: Get static mask (discard moving objects)
                logger.debug(f"Frame {i}/{num_frames}: Detecting dynamic objects...")
                static_mask, detections = yolo_wrapper.detect_and_mask(frame_rgb)
                logger.debug(f"Frame {i}: Found {len(detections)} objects")

                # Step 2: Extract local points and global description
                logger.debug(f"Frame {i}: Extracting features...")
                features = feature_extractor.extract_features(frame_rgb, static_mask)
                logger.debug(f"Frame {i}: Extracted {len(features['keypoints'])} keypoints")

                # Step 3: Transform coordinates
                features['coords_2d'] = features['keypoints']

                # Step 4: Малюємо keypoints на кадрі і записуємо у відео
                if kp_writer is not None:
                    kp_frame = self._draw_keypoints_frame(
                        frame, features['keypoints'], static_mask, i, num_frames
                    )
                    if kp_scale != 1.0:
                        kp_width = int(width * kp_scale)
                        kp_height = int(height * kp_scale)
                        kp_frame = cv2.resize(kp_frame, (kp_width, kp_height), interpolation=cv2.INTER_AREA)
                    kp_writer.write(kp_frame)

                # Step 5: Оновлюємо накопичену позу H(frame_i → frame_0)
                # Матчимо поточний кадр з попереднім і множимо гомографії
                if i == 0 or prev_features is None:
                    current_pose = np.eye(3, dtype=np.float64)
                else:
                    H_step = self._compute_inter_frame_H(prev_features, features)
                    if H_step is not None:
                        # current_pose = H(prev→frame_0) @ H(curr→prev) = H(curr→frame_0)
                        # Зберігаємо в float64 для мінімізації дрейфу в довгих ланцюгах
                        current_pose = current_pose @ H_step.astype(np.float64)
                    else:
                        # Якщо матч не вдався — залишаємо попередню позу як апроксимацію
                        logger.warning(f"Frame {i}: inter-frame match failed, reusing previous pose")

                prev_features = features

                # Step 5: Save data to HDF5
                self.save_frame_data(i, features, current_pose)

                progress_percent = int((i + 1) / num_frames * 100)

                if progress_callback:
                    progress_callback(progress_percent)

                if (i + 1) % 100 == 0:
                    logger.info(f"Processed {i + 1}/{num_frames} frames ({progress_percent}%)")

        except Exception as e:
            logger.error(f"Error during database building: {e}")
            raise
        finally:
            prefetch_thread.join(timeout=5)
            if kp_writer is not None:
                kp_writer.release()
                if kp_video_path:
                    logger.success(f"Keypoint video saved: {kp_video_path}")
            if self.db_file:
                self.db_file.close()
                logger.info("HDF5 file closed")
            cap.release()
            logger.info("Video capture released")

        logger.success(f"Database build completed successfully: {self.output_path}")

    def _draw_keypoints_frame(self, frame_bgr: np.ndarray, keypoints: np.ndarray,
                               static_mask: np.ndarray, frame_id: int,
                               total_frames: int) -> np.ndarray:
        """
        Малює на кадрі:
          - Червона напівпрозора зона: пікселі відфільтровані YOLO (рухомі об'єкти)
          - Зелені кола: XFeat keypoints на статичних зонах
          - Лічильник кадру та кількості точок у верхньому лівому куті

        Повертає BGR зображення для запису у VideoWriter.
        """
        vis = frame_bgr.copy()

        # 1. Червона маска: зони де YOLO знайшов рухомі об'єкти (mask == 0)
        if static_mask is not None:
            dynamic_zone = (static_mask == 0)
            if dynamic_zone.any():
                overlay = vis.copy()
                overlay[dynamic_zone] = (0, 0, 200)  # BGR червоний
                cv2.addWeighted(overlay, 0.35, vis, 0.65, 0, vis)

        # 2. Зелені точки: keypoints XFeat
        for x, y in keypoints:
            cx, cy = int(round(x)), int(round(y))
            cv2.circle(vis, (cx, cy), radius=3, color=(0, 255, 0), thickness=-1)
            cv2.circle(vis, (cx, cy), radius=4, color=(0, 180, 0), thickness=1)

        # 3. Інформаційна панель у верхньому лівому куті
        info_lines = [
            f"Frame: {frame_id:05d} / {total_frames:05d}",
            f"Keypoints: {len(keypoints)}",
            f"Dynamic mask: {'YES' if static_mask is not None else 'NO'}",
        ]
        panel_h = len(info_lines) * 28 + 14
        cv2.rectangle(vis, (0, 0), (340, panel_h), (0, 0, 0), -1)
        cv2.rectangle(vis, (0, 0), (340, panel_h), (80, 80, 80), 1)

        for idx, line in enumerate(info_lines):
            cv2.putText(
                vis, line,
                (8, 22 + idx * 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                (0, 255, 0), 1, cv2.LINE_AA
            )

        # 4. Легенда внизу
        legend_y = vis.shape[0] - 10
        cv2.circle(vis, (12, legend_y - 4), 5, (0, 255, 0), -1)
        cv2.putText(vis, "XFeat keypoint", (22, legend_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
        cv2.rectangle(vis, (200, legend_y - 10), (218, legend_y + 2), (0, 0, 200), -1)
        cv2.putText(vis, "YOLO dynamic zone", (224, legend_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 200), 1, cv2.LINE_AA)

        return vis

    def _compute_inter_frame_H(self, fa: dict, fb: dict,
                                min_matches: int = 15,
                                ransac_thresh: float = 3.0) -> np.ndarray | None:
        """
        Обчислює H(fb → fa): гомографію з поточного кадру в попередній.
        Використовує ШВИДКИЙ cv2.BFMatcher замість повільного Numpy argsort.
        """
        desc_a = fa['descriptors']  # (N, 64) для XFeat
        desc_b = fb['descriptors']  # (M, 64) для XFeat
        kpts_a = fa['keypoints']
        kpts_b = fb['keypoints']

        if len(kpts_a) < min_matches or len(kpts_b) < min_matches:
            return None

        # BFMatcher очікує L2-нормовані float32 дескриптори або просто float32
        # Нормалізація
        desc_a_n = desc_a / (np.linalg.norm(desc_a, axis=1, keepdims=True) + 1e-8)
        desc_b_n = desc_b / (np.linalg.norm(desc_b, axis=1, keepdims=True) + 1e-8)

        # Використовуємо L2 відстань (що еквівалентно косинусній для нормованих векторів)
        matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
        knn_matches = matcher.knnMatch(desc_b_n.astype(np.float32), desc_a_n.astype(np.float32), k=2)

        good_matches = []
        for match_pair in knn_matches:
            if len(match_pair) == 2:
                m, n = match_pair
                # Lowe's ratio test (ratio 0.8 ≈ cos_sim > 1.2)
                if m.distance < 0.8 * n.distance:
                    good_matches.append(m)
            elif len(match_pair) == 1:
                # Якщо знайшовся лише один сусід, ми не можемо застосувати ratio test, але можемо взяти його
                # якщо відстань дуже мала. Для надійності краще просто пропустити.
                pass

        if len(good_matches) < min_matches:
            return None

        # Cross-check manually to simulate MNN (хоча ratio test вже достатньо сильний)
        # Для швидкості залишаємо тільки ratio test, цього достатньо для RANSAC
        
        src_pts = np.float32([kpts_b[m.queryIdx] for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kpts_a[m.trainIdx] for m in good_matches]).reshape(-1, 1, 2)

        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, ransac_thresh)

        if H is None or int(np.sum(mask)) < min_matches:
            return None

        return H.astype(np.float32)

    def create_hdf5_structure(self, num_frames: int, width: int, height: int):
        """Create optimal HDF5 hierarchy with compression"""
        logger.info(f"Creating HDF5 structure for {num_frames} frames")

        with h5py.File(self.output_path, 'w') as f:
            # Group 1: Global data (kept in RAM)
            g1 = f.create_group('global_descriptors')
            g1.create_dataset('descriptors', shape=(num_frames, self.descriptor_dim),
                              dtype='float32', compression='gzip')
            g1.create_dataset('frame_poses', shape=(num_frames, 3, 3),
                              dtype='float64', compression='gzip')
            logger.debug("Created global_descriptors group")

            # Group 2: Local features (lazy loading)
            f.create_group('local_features')
            logger.debug("Created local_features group")

            # Group 3: Project metadata
            g3 = f.create_group('metadata')
            g3.attrs['num_frames'] = num_frames
            g3.attrs['creation_date'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            g3.attrs['frame_width'] = width
            g3.attrs['frame_height'] = height
            g3.attrs['descriptor_dim'] = self.descriptor_dim
            logger.debug("Created metadata group")

        logger.success("HDF5 structure created successfully")

    def save_frame_data(self, frame_id: int, features: dict, pose_2d: np.ndarray):
        """Save extracted data for a single frame"""
        logger.debug(f"Saving frame {frame_id} data to database")

        self.db_file['global_descriptors']['descriptors'][frame_id] = features['global_desc']
        self.db_file['global_descriptors']['frame_poses'][frame_id] = pose_2d

        frame_group = self.db_file['local_features'].create_group(f'frame_{frame_id}')
        frame_group.create_dataset('keypoints', data=features['keypoints'],
                                   dtype='float32', compression='gzip')
        frame_group.create_dataset('descriptors', data=features['descriptors'],
                                   dtype='float32', compression='gzip')
        frame_group.create_dataset('coords_2d', data=features['coords_2d'],
                                   dtype='float32', compression='gzip')

        logger.debug(f"Frame {frame_id}: Saved {len(features['keypoints'])} keypoints")