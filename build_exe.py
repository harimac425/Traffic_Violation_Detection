import subprocess
import os
import sys
from pathlib import Path

def build():
    launcher_script = "launcher_pro.py"
    exe_name = "TVDS"
    
    # Optional: If we had an icon
    icon_path = Path("resources/app_icon.ico")
    icon_arg = ["--icon", str(icon_path)] if icon_path.exists() else []

    print(f"[*] Starting professional build process for {exe_name}.exe...")
    
    # 1. Clean previous builds
    for folder in ["build", "dist"]:
        if os.path.exists(folder):
            shutil_rmtree(folder)
            
    # 2. Run PyInstaller
    # --onefile: Bundles everything into a single EXE
    # --name: Specific name for the EXE
    # --clean: Clean cache
    # --noconsole: We keep the console for now so users see the self-healing bootstrap logs
    
    cmd = [
        "pyinstaller",
        "--onefile",
        "--name", exe_name,
        "--clean",
        *icon_arg,
        launcher_script
    ]
    
    print(f"[*] Executing: {' '.join(cmd)}")
    try:
        subprocess.run(cmd, check=True)
        print(f"\n[SUCCESS] {exe_name}.exe generated in the 'dist' folder.")
        print("[*] You can now move this EXE to your release folder.")
    except subprocess.CalledProcessError as e:
        print(f"\n[ERROR] Build failed: {e}")

def shutil_rmtree(path):
    import shutil
    try:
        shutil.rmtree(path)
    except Exception:
        pass

if __name__ == "__main__":
    build()
