import sys, os
from pathlib import Path

# Add project root and DLL search paths
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

if sys.platform == 'win32' and hasattr(os, 'add_dll_directory'):
    print("[*] Registering DLL search paths...")
    # Add root and site-packages/torch/lib etc
    site_packages = project_root / "python_env" / "Lib" / "site-packages"
    dll_paths = [
        project_root / "python_env",
        site_packages / "torch" / "lib",
        site_packages / "cv2"
    ]
    for path in dll_paths:
        if path.exists():
            os.add_dll_directory(str(path))
            print(f"    [OK] Linked: {path.name}")

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
        
        import subprocess, os
        try:
            # Isolated environment
            env = os.environ.copy()
            for var in ['PYTHONPATH', 'PYTHONHOME', 'PYTHONEXECUTABLE', 'PYTHONNOUSERSITE']:
                env.pop(var, None)
            
            subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], check=True, env=env)
            print("\n[SUCCESS] Installation complete! Restarting app...")
            return True
        except Exception as e:
            print(f"\n[ERROR] Auto-install failed: {e}")
            print("  FIX: Run 'install_python_310.bat' to install the verified environment.")
            sys.exit(1)
        
    print("\n[*] Environment verified. Launching System...\n")
    return True

if __name__ == "__main__":
    check_compatibility()
    from ui.main_window import run_app
    run_app()
