import h5py
import numpy as np
import cv2
from datetime import datetime
from src.utils.logging_utils import get_logger

from src.models.wrappers.yolo_wrapper import YOLOWrapper
from src.models.wrappers.feature_extractor import FeatureExtractor

logger = get_logger(__name__)


class DatabaseBuilder:
    """Builds HDF5 topometric database from reference video"""

    def __init__(self, output_path, config=None):
        self.output_path = output_path
        self.config = config or {}
        self.descriptor_dim = self.config.get('netvlad', {}).get('descriptor_dim', 32768)
        self.db_file = None

        logger.info(f"DatabaseBuilder initialized with output: {output_path}")
        logger.info(f"NetVLAD descriptor dimension: {self.descriptor_dim}")

    def build_from_video(self, video_path: str, model_manager, progress_callback=None):
        """Process video and build database"""
        logger.info(f"Starting database build from video: {video_path}")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error(f"Failed to open video: {video_path}")
            raise ValueError(f"Не вдалося відкрити відео: {video_path}")

        num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        if num_frames <= 0:
            cap.release()
            logger.error(f"Invalid frame count ({num_frames}). Video might be corrupted or uses unsupported codec.")
            raise ValueError(
                "OpenCV не зміг розпізнати відео. Файл пошкоджений або використовує непідтримуваний кодек. "
                "Спробуйте переконвертувати відео у стандартний MP4 (H.264)."
            )

        logger.info(f"Video properties: {width}x{height}, {num_frames} frames, {fps:.2f} FPS")

        # Initialize neural network wrappers
        logger.info("Loading neural network models...")
        yolo_model = model_manager.load_yolo()
        yolo_wrapper = YOLOWrapper(yolo_model, model_manager.device)

        sp_model = model_manager.load_superpoint()
        nv_model = model_manager.load_netvlad()
        feature_extractor = FeatureExtractor(sp_model, nv_model, model_manager.device)
        logger.success("All models loaded successfully")

        # Create empty database structure
        logger.info("Creating HDF5 database structure...")
        self.create_hdf5_structure(num_frames, width, height)

        # Initial homography (identity matrix 3x3) for first frame
        current_pose = np.eye(3, dtype=np.float32)

        try:
            self.db_file = h5py.File(self.output_path, 'a')
            logger.info(f"Opened HDF5 file for writing: {self.output_path}")

            for i in range(num_frames):
                ret, frame = cap.read()
                if not ret:
                    logger.warning(f"Failed to read frame {i}, stopping processing")
                    break

                # Convert BGR to RGB for neural networks
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

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

                # Step 4: Save data to HDF5
                self.save_frame_data(i, features, current_pose)

                # ВИПРАВЛЕНО: progress_percent тепер завжди визначена в цьому блоці
                progress_percent = int((i + 1) / num_frames * 100)

                if progress_callback:
                    progress_callback(progress_percent)

                if (i + 1) % 100 == 0:
                    logger.info(f"Processed {i + 1}/{num_frames} frames ({progress_percent}%)")

        except Exception as e:
            logger.error(f"Error during database building: {e}")
            raise
        finally:
            if self.db_file:
                self.db_file.close()
                logger.info("HDF5 file closed")
            cap.release()
            logger.info("Video capture released")

        logger.success(f"Database build completed successfully: {self.output_path}")

    def create_hdf5_structure(self, num_frames: int, width: int, height: int):
        """Create optimal HDF5 hierarchy with compression"""
        logger.info(f"Creating HDF5 structure for {num_frames} frames")

        with h5py.File(self.output_path, 'w') as f:
            # Group 1: Global data (kept in RAM)
            g1 = f.create_group('global_descriptors')
            g1.create_dataset('descriptors', shape=(num_frames, self.descriptor_dim),
                              dtype='float32', compression='gzip')
            g1.create_dataset('frame_poses', shape=(num_frames, 3, 3),
                              dtype='float32', compression='gzip')
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