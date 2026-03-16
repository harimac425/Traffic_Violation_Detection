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

def check_compatibility():
    """Verify system environment before launching"""
    print("=" * 60)
    print("  Traffic Violation Detection System - Compatibility Doctor")
    print("=" * 60)
    
    # 1. Check Python Version
    major, minor = sys.version_info[:2]
    print(f"[*] Checking Python version: {major}.{minor}...", end=" ")
    if major == 3 and minor == 10:
        print("[OK]")
    else:
        print("[ERROR]")
        print(f"\n[!] CRITICAL: You are running Python {major}.{minor}.")
        print("    This application strictly requires Python 3.10 for AI stability.")
        print("    Please run 'install_python_310.bat' from the app folder.")
        print("    Then restart the application.\n")
    
    # 2. Check Core Dependencies
    dependencies = {
        "cv2": "opencv-python",
        "PyQt5": "PyQt5",
        "ultralytics": "ultralytics",
        "torch": "torch (AI Engine)"
    }
    
    print("[*] Verifying core AI modules...")
    missing = []
    for module, package in dependencies.items():
        try:
            __import__(module)
            print(f"    [OK] {package}")
        except ImportError:
            print(f"    [MISSING] {package}")
            missing.append(package)
    
    if missing:
        print("\n" + "!" * 60)
        print("  CRITICAL ERROR: Dependencies not installed!")
        print("!" * 60)
        print("  The following modules are missing:")
        for m in missing:
            print(f"  - {m}")
        
        print("\n  [!] SELF-HEAL: Would you like to attempt automatic installation? (y/n): ", end="")
        choice = input().lower().strip()
        if choice == 'y':
            print("\n[*] Starting emergency installation...")
            import subprocess
            try:
                subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], check=True)
                print("\n[SUCCESS] Installation complete! Please restart the app.")
            except Exception as e:
                print(f"\n[ERROR] Auto-install failed: {e}")
                print("  FIX: Run 'install_python_310.bat' to install the verified environment.")
            
        print("\n  Press Enter to close...")
        input()
        sys.exit(1)
        
    print("\n[*] Environment verified. Launching System...\n")

if __name__ == "__main__":
    check_compatibility()
    from ui.main_window import run_app
    run_app()
