@echo off
setlocal EnableDelayedExpansion

title TVDS - GPU Acceleration Fix

echo ============================================================
echo   TVDS - NVIDIA GPU ACCELERATION FIX
echo ============================================================
echo.
echo [*] This script will install the high-performance GPU version of AI Engine.
echo [!] REQUIRES: NVIDIA RTX/GTX GPU and latest drivers.
echo.

:: 1. Check for Administrator Privileges
net session >nul 2>&1
if %errorLevel% neq 0 (
    echo [!] ALERT: Please run this script as ADMINISTRATOR.
    echo     Right-click this file and select "Run as administrator".
    pause
    exit /b 1
)

:: 2. Search for Python 3.10
echo [*] Searching for verified Python 3.10...
set "PY_EXE="
for /f "tokens=*" %%i in ('where python') do (
    for /f "tokens=*" %%v in ('"%%i" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2^>nul') do (
        if "%%v"=="3.10" (
            set "PY_EXE=%%i"
            goto :FOUND_PY
        )
    )
)

:FOUND_PY
if "!PY_EXE!"=="" (
    echo [ERROR] Python 3.10 not found. 
    echo Please run 'install_python_310.bat' first.
    pause
    exit /b 1
)
echo [OK] Found Python 3.10: !PY_EXE!

:: 3. Check for NVIDIA GPU
echo [*] Verifying NVIDIA Hardware...
nvidia-smi >nul 2>&1
if %errorlevel% neq 0 (
    echo [WARNING] No NVIDIA GPU detected via nvidia-smi.
    echo If you have a GPU, please update your drivers from NVIDIA's website.
    echo.
    set /p "CHOICE=Attempt GPU install anyway? (y/n): "
    if /i "!CHOICE!" neq "y" exit /b 1
) else (
    echo [OK] NVIDIA GPU detected.
)

:: 4. Installation
echo.
echo [*] Uninstalling existing CPU versions...
"!PY_EXE!" -m pip uninstall torch torchvision torchaudio -y

echo.
echo [*] Installing CUDA-enabled PyTorch (High Performance)...
echo [!] This will download approx 2GB of AI drivers. Please wait...
echo.

"!PY_EXE!" -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

if %errorlevel% equ 0 (
    echo.
    echo ============================================================
    echo   [SUCCESS] GPU ACCELERATION ENABLED!
    echo ============================================================
    echo [*] Now, open the App and select "GPU" in the Brain Selector.
) else (
    echo.
    echo [ERROR] GPU installation failed (Code: %errorlevel%).
    echo Try installing CUDA 11.8 version instead?
    set /p "CHOICE=Install CUDA 11.8 version? (y/n): "
    if /i "!CHOICE!"=="y" (
        "!PY_EXE!" -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    )
)

echo.
pause
exit /b 0
