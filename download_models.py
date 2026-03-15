"""
Model Downloader - Downloads pre-trained helmet and license plate detection models

This script downloads publicly available YOLO models for specialized detection tasks.
"""
import os
import urllib.request
import zipfile
from pathlib import Path


def get_models_dir() -> Path:
    """Get the models directory path"""
    models_dir = Path(__file__).parent / "models"
    models_dir.mkdir(exist_ok=True)
    return models_dir


def download_file(url: str, destination: Path, description: str = "file"):
    """Download a file with progress indicator"""
    print(f"Downloading {description}...")
    
    def progress_hook(count, block_size, total_size):
        percent = int(count * block_size * 100 / total_size) if total_size > 0 else 0
        print(f"\r  Progress: {percent}%", end="", flush=True)
    
    try:
        urllib.request.urlretrieve(url, destination, progress_hook)
        print(f"\n  ✓ Downloaded to {destination}")
        return True
    except Exception as e:
        print(f"\n  ✗ Failed: {e}")
        return False


def download_helmet_model():
    """
    Download helmet detection model.
    Uses a YOLOv8 model trained on safety helmet dataset.
    """
    models_dir = get_models_dir()
    model_path = models_dir / "helmet_yolov8n.pt"
    
    if model_path.exists():
        print(f"Helmet model already exists: {model_path}")
        return model_path
    
    # Note: This is a placeholder URL. In production, you would:
    # 1. Train your own model using Roboflow dataset
    # 2. Use Roboflow's API to download: pip install roboflow
    # 3. Or use a publicly hosted model file
    
    print("=" * 50)
    print("HELMET DETECTION MODEL SETUP")
    print("=" * 50)
    print()
    print("Option 1: Train your own model (recommended for accuracy)")
    print("  1. Go to: https://universe.roboflow.com/search?q=helmet%20detection")
    print("  2. Select a dataset and download in YOLOv8 format")
    print("  3. Train using: yolo train data=data.yaml model=yolo11n.pt epochs=100")
    print("  4. Copy best.pt to: models/helmet_yolov8n.pt")
    print()
    print("Option 2: Use Roboflow API")
    print("  pip install roboflow")
    print("  from roboflow import Roboflow")
    print("  rf = Roboflow(api_key='YOUR_API_KEY')")
    print("  project = rf.workspace().project('helmet-detection')")
    print("  model = project.version(1).model")
    print()
    
    # Create a placeholder config file
    config_path = models_dir / "helmet_model_info.txt"
    with open(config_path, "w") as f:
        f.write("Helmet Detection Model Configuration\n")
        f.write("=" * 40 + "\n")
        f.write("Expected classes:\n")
        f.write("  0: helmet\n")
        f.write("  1: no_helmet OR head\n")
        f.write("\nRecommended datasets:\n")
        f.write("  - Roboflow: safety-helmet-dataset\n")
        f.write("  - Roboflow: helmet-detection (multiple versions)\n")
    
    print(f"Created config info at: {config_path}")
    return None


def download_plate_model():
    """
    Download license plate detection model.
    Uses a YOLOv8/v11 model trained on license plate dataset.
    """
    models_dir = get_models_dir()
    model_path = models_dir / "plate_yolov8n.pt"
    
    if model_path.exists():
        print(f"Plate model already exists: {model_path}")
        return model_path
    
    print("=" * 50)
    print("LICENSE PLATE DETECTION MODEL SETUP")
    print("=" * 50)
    print()
    print("Option 1: Use Hugging Face model")
    print("  Model: morsetechlab/yolo11-license-plate-detection")
    print("  Download .pt file from the repository")
    print()
    print("Option 2: Train your own model")
    print("  1. Go to: https://universe.roboflow.com/search?q=license%20plate")
    print("  2. Select a dataset (e.g., 'license-plate-recognition')")
    print("  3. Download in YOLOv8 format")
    print("  4. Train using: yolo train data=data.yaml model=yolo11n.pt epochs=100")
    print("  5. Copy best.pt to: models/plate_yolov8n.pt")
    print()
    
    # Create a placeholder config file
    config_path = models_dir / "plate_model_info.txt"
    with open(config_path, "w") as f:
        f.write("License Plate Detection Model Configuration\n")
        f.write("=" * 40 + "\n")
        f.write("Expected classes:\n")
        f.write("  0: license_plate\n")
        f.write("\nRecommended sources:\n")
        f.write("  - Hugging Face: morsetechlab/yolo11-license-plate-detection\n")
        f.write("  - Roboflow: license-plate-recognition\n")
    
    print(f"Created config info at: {config_path}")
    return None


def setup_roboflow_training():
    """Create a training script for custom models"""
    models_dir = get_models_dir()
    script_path = models_dir / "train_custom_model.py"
    
    script_content = '''"""
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
'''
    
    with open(script_path, "w") as f:
        f.write(script_content)
    
    print(f"Created training script: {script_path}")
    return script_path


def main():
    """Main setup function"""
    print("=" * 60)
    print("  TRAFFIC VIOLATION DETECTION - MODEL SETUP")
    print("=" * 60)
    print()
    
    models_dir = get_models_dir()
    print(f"Models directory: {models_dir}")
    print()
    
    # Setup helmet model
    helmet_path = download_helmet_model()
    print()
    
    # Setup plate model
    plate_path = download_plate_model()
    print()
    
    # Create training script
    setup_roboflow_training()
    print()
    
    print("=" * 60)
    print("SETUP COMPLETE")
    print("=" * 60)
    print()
    print("Next steps:")
    print("  1. Download or train helmet/plate models")
    print("  2. Place model files in:", models_dir)
    print("  3. Expected files:")
    print("     - helmet_yolov8n.pt (or helmet_yolo11n.pt)")
    print("     - plate_yolov8n.pt (or plate_yolo11n.pt)")
    print()
    print("The application will automatically use these models if present.")
    

if __name__ == "__main__":
    main()
