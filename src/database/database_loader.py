"""
Database loading and querying
"""

import h5py
import numpy as np


class DatabaseLoader:
    """Loads and queries HDF5 database"""
    
    def __init__(self, database_path):
        self.database_path = database_path
        self.db = None
        self.global_descriptors = None
    
    def load(self):
        """Load database"""
        # TODO: Open HDF5 file
        # TODO: Load global descriptors to RAM
        # TODO: Normalize vectors
        pass
    
    def find_similar_frames(self, query_descriptor, top_k=5):
        """Find top-K similar frames"""
        # TODO: Compute cosine similarity
        # TODO: Return top-K indices and scores
        pass
    
    def load_frame_features(self, frame_id):
        """Load features for specific frame"""
        # TODO: Read from HDF5
        # TODO: Return keypoints, descriptors, coords_2d
        pass
    
    def close(self):
        """Close database"""
        # TODO: Close HDF5 file
        pass
