"""
Custom Model Training Script

This script helps you train custom YOLO models for helmet and plate detection.
"""
from ultralytics import YOLO


def train_helmet_model(dataset_path: str, epochs: int = 100):
    """
    Train a helmet detection model.
    
    Args:
        dataset_path: Path to dataset folder containing data.yaml
        epochs: Number of training epochs
    """
    # Load base model
    model = YOLO("yolo11n.pt")
    
    # Train
    results = model.train(
        data=f"{dataset_path}/data.yaml",
        epochs=epochs,
        imgsz=640,
        batch=16,
        name="helmet_detector",
        patience=20,
    )
    
    print(f"Training complete! Best model saved to: runs/detect/helmet_detector/weights/best.pt")
    print("Copy best.pt to models/helmet_yolov8n.pt")
    
    return results


def train_plate_model(dataset_path: str, epochs: int = 100):
    """
    Train a license plate detection model.
    
    Args:
        dataset_path: Path to dataset folder containing data.yaml
        epochs: Number of training epochs
    """
    # Load base model
    model = YOLO("yolo11n.pt")
    
    # Train
    results = model.train(
        data=f"{dataset_path}/data.yaml",
        epochs=epochs,
        imgsz=640,
        batch=16,
        name="plate_detector",
        patience=20,
    )
    
    print(f"Training complete! Best model saved to: runs/detect/plate_detector/weights/best.pt")
    print("Copy best.pt to models/plate_yolov8n.pt")
    
    return results


if __name__ == "__main__":
    print("Custom Model Training Script")
    print("=" * 40)
    print()
    print("Usage:")
    print("  1. Download dataset from Roboflow in YOLOv8 format")
    print("  2. Extract to a folder")
    print("  3. Run:")
    print("     train_helmet_model('path/to/helmet_dataset')")
    print("     train_plate_model('path/to/plate_dataset')")
