@echo off
echo ================================================
echo   Traffic Violation Detection - Build Script
echo ================================================
echo.

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    pause
    exit /b 1
)

echo [1/3] Installing dependencies...
pip install -r requirements.txt
if errorlevel 1 (
    echo ERROR: Failed to install dependencies
    pause
    exit /b 1
)

echo.
echo [2/3] Downloading YOLO model (if not present)...
python -c "from ultralytics import YOLO; YOLO('yolo11m.pt')"
if errorlevel 1 (
    echo WARNING: Could not pre-download model. It will download on first run.
)

echo.
echo [3/3] Building executable...
pyinstaller --noconfirm --onedir --windowed ^
    --name "TrafficViolationDetector" ^
    --icon "app_icon.png" ^
    --add-data "ui/styles.qss;ui" ^
    --add-data "config.py;." ^
    --hidden-import "ultralytics" ^
    --hidden-import "easyocr" ^
    --hidden-import "cv2" ^
    --hidden-import "PyQt5" ^
    --hidden-import "mediapipe" ^
    --hidden-import "transformers" ^
    --collect-all "ultralytics" ^
    --collect-all "easyocr" ^
    --collect-all "mediapipe" ^
    main.py

if errorlevel 1 (
    echo ERROR: Build failed
    pause
    exit /b 1
)

echo.
echo ================================================
echo   BUILD COMPLETE!
echo   Executable: dist\TrafficViolationDetector\
echo ================================================
echo.
echo NOTE: Copy your YOLO model (yolo11m.pt) to the dist folder
echo       if you want it bundled with the exe.
echo.
pause
