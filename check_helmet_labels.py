
from ultralytics import YOLO
import sys
import os

model_path = r"D:\Code\Traffic_Violation_Detection-main\models\helmet_yolov8n.pt"
if not os.path.exists(model_path):
    print(f"Model not found: {model_path}")
    sys.exit(1)

model = YOLO(model_path)
print("CLASS NAMES:", model.names)
