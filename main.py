"""
Traffic Violation Detection System
Main Entry Point

Run this file to start the application.
"""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from ui.main_window import run_app


if __name__ == "__main__":
    print("=" * 50)
    print("  Traffic Violation Detection System")
    print("  Starting application...")
    print("=" * 50)
    
    run_app()
