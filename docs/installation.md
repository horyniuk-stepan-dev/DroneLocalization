# Installation Guide

## Prerequisites

### Hardware
- NVIDIA GPU (RTX 3080 or better)
- 8GB+ VRAM
- 16GB+ RAM (32GB recommended)
- 50GB+ free disk space

### Software
- Windows 10/11
- Python 3.10 or 3.11
- CUDA 12.1+
- Git

## Installation Steps

### 1. Clone Repository

```powershell
git clone <repository-url>
cd DroneLocalization
```

### 2. Create Virtual Environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Install Dependencies

```powershell
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Download Models

```powershell
python scripts/download_models.py
```

### 5. Test GPU

```powershell
python scripts/test_gpu.py
```

### 6. Run Application

```powershell
python main.py
```

## Troubleshooting

### CUDA Not Available
- Verify NVIDIA drivers are installed
- Install CUDA Toolkit 12.1
- Reinstall PyTorch with CUDA support

### Import Errors
- Ensure virtual environment is activated
- Reinstall requirements: `pip install -r requirements.txt --force-reinstall`

### GUI Issues
- Update PyQt6: `pip install --upgrade PyQt6`
- Check Qt platform plugin
