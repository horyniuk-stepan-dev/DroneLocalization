#!/usr/bin/env python3
"""
Test GPU availability and CUDA setup
"""

import torch


def test_gpu():
    """Test GPU availability"""
    print("=" * 50)
    print("GPU Test Results")
    print("=" * 50)
    
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"Number of GPUs: {torch.cuda.device_count()}")
        
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"\nGPU {i}: {props.name}")
            print(f"  Memory: {props.total_memory / 1024**3:.2f} GB")
            print(f"  Compute capability: {props.major}.{props.minor}")
    else:
        print("WARNING: CUDA not available!")
        print("Please install CUDA 12.1 and PyTorch with CUDA support")
    
    print("=" * 50)


if __name__ == "__main__":
    test_gpu()
