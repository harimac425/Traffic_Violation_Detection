# 🚦 Traffic Violation Detection System (TVDS)

> **AI-Powered Real-Time Traffic Enforcement Platform**  
> Version 2.5 — *"Premium Glass Slate"*

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://python.org)
[![YOLO](https://img.shields.io/badge/Detection-YOLOv11x-green?logo=pytorch)](https://ultralytics.com)
[![LLM](https://img.shields.io/badge/LLM-Gemini%202.0%20Flash-orange?logo=google)](https://ai.google.dev)
[![PyQt5](https://img.shields.io/badge/UI-PyQt5-purple)](https://riverbankcomputing.com/software/pyqt/)
[![License](https://img.shields.io/badge/License-MIT-lightgrey)](LICENSE)

---

## 📸 System Overview

TVDS is a complete end-to-end traffic violation enforcement pipeline that combines:

- **YOLO11x** — State-of-the-art object detection (CCTV-grade)
- **Constraint-Aware SORT** — Custom Kalman + Hungarian tracking
- **Gemini 2.0 Flash / GPT-4o** — Cognitive LLM verification brain
- **PaddleOCR / EasyOCR** — Indian license plate recognition
- **PyQt5 "Glass Slate" Dashboard** — Real-time operator interface
- **SQLite Database** — Forensic-grade violation storage with evidence JPEGs

---

## 🏗️ Architecture

```
Camera/Video Input
      ↓
CLAHE + Dynamic ROI Pre-Processing
      ↓
YOLO11x Detection (+ YOLOv8n Helmet & Plate sub-models)
      ↓
ConstraintAwareSORT Tracking
      ↓
OCR Pipeline (PaddleOCR → EasyOCR → LLM fallback)
+ Phone Detection (MediaPipe Pose + YOLO)
      ↓
Violation Logic Engine (6 violation types)
      ↓
Cognitive LLM Brain (Gemini 2.0 / GPT-4o multi-frame voting)
      ↓
SQLite DB + Evidence JPEG Storage
      ↓
Premium Glass Slate Dashboard (PyQt5)
```

---

## 🚗 Detected Violations (Verified)

| Violation | Method |
|---|---|
| 🪖 No Helmet | Head ROI extraction + YOLOv8n model |
| 👥 Excess Riding (Triple) | Person-centric spatial association |
| 📱 Phone Usage | MediaPipe Pose + YOLO dual-lock |
| 🪧 Missing Number Plate | Multi-frame persistence check |

---

## 🛠️ Requirements

### Hardware
- **GPU**: NVIDIA RTX (CUDA 11.8+ recommended) — runs on CPU but slower
- **RAM**: 8GB minimum, 16GB recommended

### Software
```bash
pip install ultralytics torch torchvision paddlepaddle paddleocr easyocr
pip install PyQt5 opencv-python mediapipe filterpy scipy
pip install google-generativeai openai python-docx python-pptx
```

---

## 📂 Project Structure

```
Traffic_violation_Detection/
├── main.py                    # Entry point
├── config.py                  # All configuration & parameters
├── src/
│   ├── detector.py            # Multi-model YOLO detection engine
│   ├── tracker.py             # ConstraintAwareSORT tracking
│   ├── violations.py          # Violation logic (all 6 types)
│   ├── ocr.py                 # License plate recognition engine
│   ├── llm.py                 # Cognitive LLM Brain (Gemini / GPT-4o)
│   ├── database.py            # SQLite violation storage
│   ├── phone_detection.py     # MediaPipe phone usage detection
│   ├── report.py              # CSV / HTML export engine
│   └── logger.py              # Session logging
├── ui/
│   ├── main_window.py         # PyQt5 main dashboard
│   ├── styles.qss             # Premium Glass Slate theme
│   └── llm_widgets.py         # AI Brain Wizard dialog
├── models/                    # Place YOLO model files here (not included)
│   ├── yolo11x.pt             # Main detection model (download separately)
│   ├── helmet_yolov8n.pt      # Helmet sub-model
│   └── plate_yolov8n.pt       # Plate sub-model
├── evidence/                  # Auto-created: violation JPEG snapshots
├── logs/                      # Auto-created: session log files
└── reports/                   # Auto-created: exported CSV/HTML reports
```

---

## 🚀 Quick Start

```bash
# 1. Clone the repo
git clone https://github.com/junkratroadhog/Traffic_violation_Detection.git
cd Traffic_violation_Detection

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download YOLO models (place in /models/)
#    yolo11x.pt: https://github.com/ultralytics/assets/releases
#    helmet_yolov8n.pt and plate_yolov8n.pt: custom trained models

# 4. (Optional) Configure LLM
#    Create settings.json with your API key:
#    { "gemini_api_key": "YOUR_KEY_HERE" }

# 5. Run the app
# Simply double-click Start_TVDS.bat (it will automatically install missing dependencies)
# Or run from command line:
python main.py
```

---

## ⚙️ Configuration

All parameters are in `config.py`. Key settings:

| Parameter | Default | Description |
|---|---|---|
| `MODEL_PATH` | `yolo11x.pt` | Main YOLO model path |
| `SPEED_LIMIT_KMH` | `50` | Speed violation threshold |
| `VIOLATION_PERSISTENCE_THRESHOLD` | `3` | Min consecutive frames to confirm |
| `LLM_MAX_RPM` | `15` | LLM API rate limit |
| `ENABLE_RED_SIGNAL` | `True` | Toggle red-signal detection |
| `ENABLE_DYNAMIC_ROI` | `True` | Toggle dynamic ROI processing |

---

## 🧠 LLM Setup

1. Open the app → click **"AI Brain Wizard"**
2. Select provider: **Gemini 2.0 Flash** or **OpenAI GPT-4o**
3. Enter your API key and click **Test Connection**

Get a free Gemini API key: [https://aistudio.google.com/apikey](https://aistudio.google.com/apikey)

---

## 📊 Dashboard Features

- **Live annotated video** with YOLO bounding boxes
- **Vehicle tracking table** with plate numbers & violation status
- **Real-time violation log** with LLM reasoning
- **Historical records browser** (SQLite-backed)
- **One-click CSV/HTML export** for law enforcement records
- **Stop-line & Signal ROI picker** — click directly on video to configure

---

## 📄 Documentation

| Document | Description |
|---|---|
| `TVDS_Architecture_Complete.docx` | Full technical architecture with component justifications |
| `Traffic_Violation_Demonstration.pptx` | Project demonstration presentation |

---

## ⚠️ Notes

- **Model files** (`.pt`) are NOT included due to size. Download from Ultralytics or contact the author.
- **`settings.json`** (API keys) is excluded from version control for security.
- **Evidence images** and **database** are gitignored — they are user-generated data.

---

## 📝 License

MIT License — see [LICENSE](LICENSE) for details.

---

*Built with ❤️ using YOLO11x + Gemini 2.0 Flash + PyQt5*
