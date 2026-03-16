# Configuration for Traffic Violation Detection System
import os
from pathlib import Path

# ============================================================
# Model Paths
# ============================================================
MODELS_DIR = Path(__file__).parent / "models"

# Main detection model (general objects: person, vehicle, phone)
# Using YOLO11x (Extra Large) for better detection on CCTV/small objects
MODEL_PATH = MODELS_DIR / "yolo11x.pt" 

# CCTV Enhancement Settings
ENABLE_CLAHE = True  # Contrast Limited Adaptive Histogram Equalization for better visibility
CLAHE_CLIP_LIMIT = 2.0
CLAHE_GRID_SIZE = (8, 8)

# Specialized models (place in models/ folder)
HELMET_MODEL_PATH = MODELS_DIR / "helmet_yolov8n.pt"
PLATE_MODEL_PATH = MODELS_DIR / "plate_yolov8n.pt"

# Check if specialized models exist
USE_HELMET_MODEL = HELMET_MODEL_PATH.exists()
USE_PLATE_MODEL = PLATE_MODEL_PATH.exists()

# ============================================================
# Device Settings (GPU Acceleration)
# ============================================================
import torch

def get_available_devices():
    """Get list of available compute devices"""
    devices = [("cpu", "CPU")]
    
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        devices.append(("cuda", f"NVIDIA GPU ({gpu_name})"))
    
    return devices

def get_device(device_type: str = "auto"):
    """
    Get the device to use for inference.
    
    Args:
        device_type: "cpu", "cuda", or "auto"
    
    Returns:
        Tuple of (device, device_name)
    """
    if device_type == "cuda":
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            return "cuda", f"NVIDIA GPU ({gpu_name})"
        else:
            print("CUDA not available, falling back to CPU")
            return "cpu", "CPU"
    elif device_type == "cpu":
        return "cpu", "CPU"
    else:  # auto
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            return "cuda", f"NVIDIA GPU ({gpu_name})"
        return "cpu", "CPU"

# Default device setting
DEVICE_TYPE = "auto"  # Can be "cpu", "cuda", or "auto"
DEVICE, DEVICE_NAME = get_device(DEVICE_TYPE)
CUDA_AVAILABLE = torch.cuda.is_available()

# ============================================================
# Detection Settings
# ============================================================
# Detection Settings
CONFIDENCE_THRESHOLD = 0.5
IOU_THRESHOLD = 0.4  # Tightened from 0.45 for better deduplication

# Helmet detection specific
HELMET_CONFIDENCE = 0.5  # High confidence to avoid false positives
HELMET_CLASSES = {
    "helmet": ["helmet", "with_helmet", "Helmet", "With helmet", "with helmet", "0"],
    "no_helmet": ["no_helmet", "without_helmet", "head", "no-helmet", "No Helmet", "Without helmet", "  Without helmet", "1"]
}

# Plate detection specific
PLATE_CONFIDENCE = 0.4  # High confidence for clean UI
PLATE_CLASSES = ["license_plate", "plate", "number_plate", "License Plate", "Number Plate"]

# OCR Engine Settings
OCR_ENGINE = "paddleocr"  # Choices: "easyocr", "trocr", "paddleocr"
OCR_USE_GPU = True      # Leverages NVIDIA GPU if available
OCR_CLEAN_PLATES = True # Auto-formatting for Indian license plates


# ============================================================
# Detection classes (COCO dataset IDs for main model)
# ============================================================
CLASSES = {
    0: "person",
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
    67: "cell phone",
}

# ============================================================
# Violation Thresholds
# ============================================================
ENABLE_MISSING_PLATE_DETECTION = True  # Enabled with robust persistence check
SPEED_LIMIT_KMH = 50  # km/h

# --- SMART DETECTION IMPROVISATIONS ---
VIOLATION_PERSISTENCE_THRESHOLD = 3  # Frames consecutive needed to log
ENABLE_DYNAMIC_ROI = True            # Only process moving areas for FPS boost
LLM_VOTING_FRAMES = 3                # Number of crops for AI voting

# ============================================================
# Violation Toggle System (Dynamic Enable/Disable from UI)
# ============================================================
ENABLED_VIOLATIONS = {
    "NO_HELMET": True,
    "TRIPLE_RIDING": True,
    "OVERSPEED": True,
    "WRONG_WAY": True,
    "PHONE_USAGE": True,
    "RED_SIGNAL": True,
    "MISSING_PLATE": True,
    "ZEBRA_CROSSING": False,
}

# Red Signal Detection
ENABLE_RED_SIGNAL = True
STOP_LINE_Y = 450         # Default Y-coord of stop line
SIGNAL_ROI = [600, 100, 750, 250]  # Default [x1, y1, x2, y2]
RED_SIGNAL_COOLDOWN = 180  # Frames to wait before re-detecting same vehicle

# Triple Riding Detection
TRIPLE_RIDING_MIN_PERSONS = 3  # Detect if rider count is >= this value

# Phone Usage Detection
ENABLE_PHONE_USAGE_DETECTION = True
USE_MEDIAPIPE_PHONE = True  # Uses PoseLandmarker for calling gesture
PHONE_DETECTION_CONFIDENCE = 0.5
PHONE_NEAR_PERSON_MARGIN = 50  # Pixels to check around person for phone object

# Tracker Settings
USE_CUSTOM_TRACKER = True      # Use ConstraintAwareSORT
TRACKER_MAX_AGE = 30           # Reduced from 50 to minimize box ghosting after objects leave frame
TRACKER_MIN_HITS = 2           # Faster track initialization
TRACKER_IOU_THRESHOLD = 0.25   # Slightly more relaxed to handle fast motion
TRACKER_STOP_LINE_GATING = True


# Speed calculation (adjust based on camera calibration)
PIXELS_PER_METER = 20  # Approximate, needs calibration for each camera
FPS = 30  # Adjust based on video source

# Wrong-way detection (define allowed direction per lane)
# Format: {"lane_name": (min_angle, max_angle)} in degrees, 0 = right, 90 = up
LANE_DIRECTIONS = {
    "default": (0, 360),  # Allowed all directions (Disable Wrong Way pending calibration)
}

# Zebra Crossing Detection
# Format: [x1, y1, x2, y2] normalized coordinates (0.0 to 1.0)
# Default: Bottom 20% of the screen
ENABLE_ZEBRA_CROSSING = False # Disabled by default to prevent false positives
ZEBRA_CROSSING_BOX = [0.0, 0.8, 1.0, 1.0]

# LLM Settings
LLM_PROVIDER = "gemini"  # Added missing provider tracker
GEMINI_API_KEY = "" 
OPENAI_API_KEY = ""
SELECTED_MODEL = "gemini-2.0-flash" 
LLM_MAX_RPM = 15 

# Load from settings.json if exists
try:
    import json
    settings_path = "settings.json"
    if os.path.exists(settings_path):
        with open(settings_path, 'r') as f:
            width_data = json.load(f)
            LLM_PROVIDER = width_data.get("LLM_PROVIDER", "gemini")
            GEMINI_API_KEY = width_data.get("GEMINI_API_KEY", "")
            OPENAI_API_KEY = width_data.get("OPENAI_API_KEY", "")
            CUSTOM_MODEL_NAME = width_data.get("CUSTOM_MODEL_NAME", "Local LLM")
            CUSTOM_BASE_URL = width_data.get("CUSTOM_BASE_URL", "")
            CUSTOM_API_KEY = width_data.get("CUSTOM_API_KEY", "")
            SELECTED_MODEL = width_data.get("SELECTED_MODEL", "gemini-2.0-flash")
            LLM_MAX_RPM = int(width_data.get("LLM_MAX_RPM", 15))
except Exception as e:
    print(f"Error loading settings: {e}")

# ============================================================
# Database & Storage
# ============================================================
DATABASE_PATH = str(Path(__file__).parent / "violations.db")
EVIDENCE_DIR = str(Path(__file__).parent / "evidence")
EXPORT_DIR = str(Path(__file__).parent / "reports")

# Auto-create directories
os.makedirs(EVIDENCE_DIR, exist_ok=True)
os.makedirs(EXPORT_DIR, exist_ok=True)


# ============================================================

# ============================================================
# UI Settings
# ============================================================
WINDOW_TITLE = "Traffic Violation Detection System"
WINDOW_WIDTH = 1400
WINDOW_HEIGHT = 800


# ============================================================
# Startup Info
# ============================================================
def print_config_status():
    """Print configuration status on startup"""
    print("\n" + "=" * 50)
    print("  CONFIGURATION STATUS")
    print("=" * 50)
    print(f"  Device: {DEVICE_NAME}")
    print(f"  Main Model: {MODEL_PATH}")
    print(f"  Helmet Model: {'✓ Found' if USE_HELMET_MODEL else '✗ Not found'}")
    print(f"  Plate Model: {'✓ Found' if USE_PLATE_MODEL else '✗ Not found'}")
    if not USE_HELMET_MODEL or not USE_PLATE_MODEL:
        print("\n  Run 'python download_models.py' for setup instructions")
    print("=" * 50 + "\n")

