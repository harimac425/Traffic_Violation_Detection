@echo off
setlocal

:: Set title and color
title TVDS Application Launcher
color 0B

echo ===================================================
echo     Traffic Violation Detection System (TVDS)
echo ===================================================
echo.

set PYTHON_CMD=python

:: 1. Check if Python is installed globally
python --version >nul 2>&1
if %errorlevel% equ 0 goto check_deps

:: If 'python' fails, try 'py' (Windows Python Launcher)
py --version >nul 2>&1
if %errorlevel% equ 0 (
    set PYTHON_CMD=py
    goto check_deps
)

:: If 'py' fails, check common Anaconda/Miniconda paths
if exist "C:\ProgramData\miniconda3\python.exe" (
    set PYTHON_CMD="C:\ProgramData\miniconda3\python.exe"
    goto check_deps
)
if exist "%USERPROFILE%\miniconda3\python.exe" (
    set PYTHON_CMD="%USERPROFILE%\miniconda3\python.exe"
    goto check_deps
)
if exist "%USERPROFILE%\Anaconda3\python.exe" (
    set PYTHON_CMD="%USERPROFILE%\Anaconda3\python.exe"
    goto check_deps
)

color 0E
echo [WARNING] Python was not found in your system PATH.
echo [*] The script will now attempt to download and install Python 3.10.11...
echo [*] Downloading from python.org...
curl -L -o python_installer.exe https://www.python.org/ftp/python/3.10.11/python-3.10.11-amd64.exe

if exist python_installer.exe (
    echo [*] Installing Python 3.10 silently... (This takes about 1-2 minutes)
    start /wait python_installer.exe /quiet InstallAllUsers=0 PrependPath=1 Include_test=0
    del python_installer.exe
    
    echo [*] Python installed. We need to restart the script to refresh the system PATH.
    echo Press any key to restart...
    pause >nul
    start "" "%~dpnx0"
    exit /b
) else (
    color 0C
    echo [ERROR] Failed to download Python. Please check your internet connection or install manually.
    pause
    exit /b
)

:check_deps
echo [*] Successfully located Python: %PYTHON_CMD%

echo [*] Checking default dependencies...
if exist requirements.txt (
    %PYTHON_CMD% -m pip install -r requirements.txt
    if %errorlevel% neq 0 (
        echo [WARNING] Some dependencies threw a warning during install.
        echo The app will now attempt to launch anyway.
        echo If it crashes, please open Command Prompt and run:
        echo    %PYTHON_CMD:"=% -m pip install -r requirements.txt
        echo.
        echo Starting in 3 seconds...
        timeout /t 3 /nobreak >nul
    ) else (
        echo [OK] All dependencies are satisfied.
    )
) else (
    color 0E
    echo [WARNING] requirements.txt not found. Skipping dependency check.
)
echo.

:: NEW: Advanced Setup Options
echo [*] Press 'G' to force install the GPU version of AI Models (Requires NVIDIA card).
echo [*] Press 'C' to force install the CPU-only version (Fixes c10.dll/CUDA errors on non-NVIDIA PCs).
echo [*] Press 'D' to install Microsoft C++ Fixes (If you get DLL load failed errors).
echo [*] Press ANY OTHER KEY to continue normally without changing anything.
choice /c GCD1 /n /t 3 /d 1 /m "> "

if errorlevel 4 goto launch_app
if errorlevel 3 goto install_dll
if errorlevel 2 goto install_cpu
if errorlevel 1 goto install_gpu

:install_gpu
echo [*] Installing GPU version of PyTorch...
%PYTHON_CMD% -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
goto launch_app

:install_cpu
echo [*] Installing CPU-only version of PyTorch (This will fix CUDA DLL errors)...
%PYTHON_CMD% -m pip uninstall torch torchvision torchaudio -y
%PYTHON_CMD% -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
goto launch_app

:install_dll
echo [*] Downloading Microsoft Visual C++ Redistributable...
curl -L -o vc_redist.exe https://aka.ms/vs/17/release/vc_redist.x64.exe
if exist vc_redist.exe (
    echo [*] Installing C++ Dependencies... (Please click YES if Windows asks for permission)
    start /wait vc_redist.exe /install /quiet /norestart
    del vc_redist.exe
    echo [*] C++ Installation complete!
)
goto launch_app

:launch_app
:: 3. Launch Application
echo [*] Launching TVDS Dashboard...
echo Close this console window to completely stop the application.
echo.
%PYTHON_CMD% main.py

if %errorlevel% neq 0 (
    color 0C
    echo.
    echo [ERROR] The application crashed or was closed unexpectedly.
    echo Check the error messages above for details.
    echo.
    pause
)

endlocal
