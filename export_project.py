import os
import shutil
from pathlib import Path

def export_project():
    src_dir = Path(r"d:\Code\Traffic_violation_Detection")
    export_dir = Path(r"d:\Code\TVDS_Professional_v1.0")
    dist_dir = src_dir / "dist"
    
    print(f"[*] Preparing professional distribution: {export_dir}")

    # Remove existing export dir if it exists
    if export_dir.exists():
        print("[*] Cleaning previous distribution files...")
        shutil.rmtree(export_dir, ignore_errors=True)
        
    export_dir.mkdir(parents=True, exist_ok=True)

    # 1. Copy the Professional EXE (The Heart of the launch)
    exe_file = dist_dir / "TVDS.exe"
    if exe_file.exists():
        shutil.copy2(exe_file, export_dir / "TVDS.exe")
        print("[OK] Copied TVDS.exe (Professional Launcher)")
    else:
        print("[ERROR] TVDS.exe not found in 'dist'! Run build_exe.py first.")
        return

    # 2. Copy application core files
    files_to_copy = [
        "main.py", 
        "config.py", 
        "requirements.txt", 
        "README.md"
    ]
    
    for f in files_to_copy:
        src_file = src_dir / f
        if src_file.exists():
            shutil.copy2(src_file, export_dir / f)
            print(f"Copied core file: {f}")

    # 3. Copy resource directories
    dirs_to_copy = ["src", "ui", "models", "data", "resources"]
    
    for d in dirs_to_copy:
        src_path = src_dir / d
        if src_path.exists():
            shutil.copytree(
                src_path, 
                export_dir / d, 
                ignore=shutil.ignore_patterns('__pycache__', '*.pyc', '.git', 'dist', 'build')
            )
            print(f"Copied resource: {d}/")

    # 4. Initialize operational directories
    operational_dirs = ["evidence", "logs", "reports"]
    for d in operational_dirs:
        (export_dir / d).mkdir(exist_ok=True)
        print(f"Initialized directory: {d}/")

    # 5. Create ZIP Archive
    zip_name = export_dir.name
    zip_path = export_dir.parent / zip_name
    print(f"\n[*] Creating ZIP archive: {zip_path}.zip")
    shutil.make_archive(str(zip_path), 'zip', export_dir)

    print("\n" + "="*40)
    print("[SUCCESS] Professional TVDS Distribution Created!")
    print(f"Folder: {export_dir}")
    print(f"Archive: {zip_path}.zip")
    print("="*40)
    
if __name__ == "__main__":
    export_project()
