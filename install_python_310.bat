@echo off
setlocal EnableDelayedExpansion

:: This title helps identify the window
title TVDS Python 3.10 Installer - DEBUG MODE

echo ============================================================
echo   TVDS - Python 3.10 ROBUST INSTALLER (Debug Version)
echo ============================================================
echo.
echo [!] THIS WINDOW WILL NOT CLOSE AUTOMATICALLY.
echo [!] YOU MUST MANUALLY CLOSE IT AFTER READING ANY ERRORS.
echo.

:: 1. Check for Administrator Privileges
echo [*] Step 1: Checking for Administrator Privileges...
net session >nul 2>&1
if %errorLevel% == 0 (
    echo [OK] Running with Administrator privileges.
) else (
    echo [!] WARNING: Not running as Administrator.
    echo [!] If this fails, please Right-Click -> "Run as Administrator".
    echo.
)

:: 2. Check for winget
echo [*] Step 2: Checking for Windows Package Manager (winget)...
winget --version
if %errorlevel% neq 0 (
    echo [!] winget is missing or failed to report version.
    echo [!] Error Code: %errorlevel%
    echo.
    goto :MANUAL_FALLBACK
)
echo [OK] winget detected.

:: 3. Attempt Automated Install
echo [*] Step 3: Installing Python 3.10.11 via winget...
echo [!] This may take several minutes. Please do not close this window.
echo.
winget install Python.Python.3.10 --version 3.10.11 --silent --show-progress --accept-package-agreements --accept-source-agreements

if %errorlevel% equ 0 (
    echo.
    echo [SUCCESS] Python 3.10.11 installed via winget!
    echo.
    goto :FINISHED
) else (
    echo.
    echo [!] ERROR: winget install failed with code: %errorlevel%
    echo [!] Potential causes: No internet, disk full, or winget database corrupt.
    echo.
    goto :MANUAL_FALLBACK
)

:MANUAL_FALLBACK
echo.
echo ============================================================
echo   MANUAL INSTALLATION FALLBACK
echo ============================================================
echo [!] Automated installation failed.
echo.
echo [1] Attempting to open the official download link...
start https://www.python.org/ftp/python/3.10.11/python-3.10.11-amd64.exe
echo.
echo [2] If the browser didn't open, copy/paste this URL:
echo     https://www.python.org/ftp/python/3.10.11/python-3.10.11-amd64.exe
echo.
echo [3] CRITICAL INSTALLATION STEPS:
echo     a) Run the downloaded .exe file.
echo     b) CHECK THE BOX: "Add Python 3.10 to PATH" (Very Important!)
echo     c) Click "Install Now".
echo.

:FINISHED
echo ============================================================
echo   PROCESS COMPLETE - STATUS CHECK
echo ============================================================
echo.
echo [*] If you saw [SUCCESS] above, please RESTART YOUR COMPUTER.
echo [*] If you saw [ERROR], please follow the MANUAL steps above.
echo.
echo [!] WINDOW PERSISTENCE: Press any key to exit.
pause
exit /b 0
