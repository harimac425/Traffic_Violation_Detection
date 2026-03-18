
import sys
import torch
import cv2
import PyQt5
import ultralytics
import easyocr
import numpy
import scipy

print(f"Python: {sys.version}")
print(f"Torch: {torch.__version__}, CUDA: {torch.cuda.is_available()}")
print(f"OpenCV: {cv2.__version__}")
print(f"PyQt5: {PyQt5.QtCore.PYQT_VERSION_STR}")
print(f"Ultralytics: {ultralytics.__version__}")
print(f"EasyOCR: {easyocr.__version__}")
print("All core dependencies found!")
