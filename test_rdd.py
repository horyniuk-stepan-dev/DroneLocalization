import sys
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

try:
    from src.models.wrappers.rdd_wrapper import RDDWrapper
    print("Loading RDDWrapper...")
    wrapper = RDDWrapper(device="cuda")
    
    print(f"Wrapper initialized! Descriptor dim detected: {wrapper.desc_dim}")
    
    dummy_img = torch.randn(1, 3, 480, 640).cuda()
    print("Running extraction on 1x3x480x640 dummy image...")
    out = wrapper({"image": dummy_img})
    
    print("Extraction successful!")
    print(f"Keypoints shape: {out['keypoints'].shape}")
    print(f"Descriptors shape: {out['descriptors'].shape}")
except Exception as e:
    import traceback
    traceback.print_exc()
    print(f"Failed: {e}")
