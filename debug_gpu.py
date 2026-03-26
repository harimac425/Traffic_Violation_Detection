import torch
import sys
from ultralytics import YOLO
import cv2

print("=" * 50)
print("  TVDS GPU DIAGNOSTIC TOOL")
print("=" * 50)

print(f"Python: {sys.version}")
print(f"PyTorch Version: {torch.__version__}")
print(f"PyTorch Path: {torch.__file__}")
print(f"CUDA Available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    device = "cuda"
else:
    print("!!! CUDA NOT DETECTED !!! PyTorch is running in CPU mode.")
    device = "cpu"

print("-" * 50)
print("Testing YOLO Model Load...")
try:
    model = YOLO("models/yolo11x.pt")
    model.to(device)
    print(f"Model loaded on: {model.device}")
    
    print("\nBenchmark (Standard Frame):")
    import numpy as np
    dummy = np.zeros((720, 1280, 3), dtype=np.uint8)
    import time
    start = time.time()
    for _ in range(5):
        model.predict(dummy, verbose=False)
    end = time.time()
    print(f"Average Inference Speed: {(end - start) / 5 * 1000:.1f}ms")
except Exception as e:
    print(f"Error: {e}")

print("=" * 50)
print("If CUDA is False or speed is > 100ms, your environment is broken.")
print("=" * 50)
