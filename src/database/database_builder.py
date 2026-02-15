"""
Topometric database builder
"""

import h5py
import numpy as np


class DatabaseBuilder:
    """Builds HDF5 topometric database"""
    
    def __init__(self, output_path, config):
        self.output_path = output_path
        self.config = config
    
    def build_from_video(self, video_path, model_manager):
        """Build database from video"""
        # TODO: Open video
        # TODO: Process frames
        # TODO: Extract features
        # TODO: Compute topometric coordinates
        # TODO: Create HDF5 structure
        # TODO: Save data
        pass
    
    def create_hdf5_structure(self, num_frames):
        """Create HDF5 file structure"""
        # TODO: Create file
        # TODO: Create groups
        # TODO: Create datasets with proper dtypes
        # TODO: Set compression
        pass
    
    def save_frame_data(self, frame_id, features, pose_2d):
        """Save single frame data"""
        # TODO: Write to HDF5
        pass
