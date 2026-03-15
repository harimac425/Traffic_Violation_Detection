# Installation Guide - Traffic Violation Detection

## Quick Install (CPU)

```bash
# Create virtual environment (recommended)
python -m venv venv
venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

## GPU Installation (NVIDIA Only)

### Option 1: CUDA 12.1 (Recommended)
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

### Option 2: CUDA 11.8
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

## Verify Installation

```python
import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
```

## Run Application

```bash
python main.py
```

## Build Executable

```bash
# Windows
build.bat

# Or manually
pyinstaller --onefile --windowed --name TrafficViolationDetector main.py
```

## Dependencies Summary

| Package | Purpose | Version |
|---------|---------|---------|
| torch | Deep Learning | >=2.0.0 |
| ultralytics | YOLO Detection | >=8.0.0 |
| opencv-python | Video Processing | >=4.8.0 |
| PyQt5 | GUI Framework | >=5.15.0 |
| easyocr | License Plate OCR | >=1.7.0 |
| numpy | Numerical Ops | >=1.24.0 |
| scipy | Scientific Ops | >=1.11.0 |
| lap | Tracking | >=0.4.0 |
| pyinstaller | Build EXE | >=6.0.0 |

## Troubleshooting

### CUDA Not Detected
- Ensure NVIDIA drivers are installed
- Check CUDA toolkit version matches PyTorch
- Run `nvidia-smi` to verify GPU

### DLL Errors on Windows
- Install Visual C++ Redistributable
- Reinstall PyTorch with CPU version

### Model Loading Slow
- First run downloads YOLO model (~40MB)
- Subsequent runs use cached model
