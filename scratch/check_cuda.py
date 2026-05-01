import torch
if torch.cuda.is_available():
    prop = torch.cuda.get_device_properties(0)
    print(f"Device: {prop.name}")
    print(f"Compute Capability: {prop.major}.{prop.minor}")
else:
    print("CUDA not available")
