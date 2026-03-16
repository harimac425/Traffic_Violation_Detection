"""
Traffic Violation Detection System
Main Entry Point

Run this file to start the application.
"""
# CRITICAL: Import torch before any other library (cv2, PyQt5) to avoid DLL initialization errors (WinError 1114)
try:
    import torch
except ImportError:
    pass

import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Fix for OpenMP/MKL DLL conflicts
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# Support for portable python environment DLL discovery
if getattr(sys, 'frozen', False):
    # This logic only applies when running as a compiled EXE
    python_env_dir = project_root.parent / "python_env"
    if python_env_dir.exists():
        os.add_dll_directory(str(python_env_dir))

from ui.main_window import run_app


if __name__ == "__main__":
    print("=" * 50)
    print("  Traffic Violation Detection System")
    print("  Starting application...")
    print("=" * 50)
    
    run_app()
