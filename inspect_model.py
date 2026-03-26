from ultralytics import YOLO
import sys

try:
    model = YOLO("models/helmet_yugank.pt")
    print("Class Names:", model.names)
except Exception as e:
    print(f"Error loading model: {e}")
