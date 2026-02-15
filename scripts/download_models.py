#!/usr/bin/env python3
"""
Download pretrained models
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def download_models():
    """Download all required models"""
    models_dir = Path("data/models")
    models_dir.mkdir(parents=True, exist_ok=True)
    
    print("Downloading models...")
    
    # TODO: Download YOLOv8x-seg
    # TODO: Download SuperPoint
    # TODO: Download NetVLAD
    # TODO: Download LightGlue
    # TODO: Download Depth-Anything
    
    print("Models downloaded successfully!")


if __name__ == "__main__":
    download_models()
