import os
import sys
import urllib.request
import zipfile
import subprocess
import shutil
from pathlib import Path

# Configuration
GITHUB_REPO_ZIP = "https://github.com/harimac425/Traffic_Violation_Detection/archive/refs/heads/main.zip"
PYTHON_URL = "https://www.python.org/ftp/python/3.10.11/python-3.10.11-embed-amd64.zip"
GET_PIP_URL = "https://bootstrap.pypa.io/get-pip.py"

APP_DIR = Path("Traffic_Violation_Detection-main")
ENV_DIR = Path("python_env")
PYTHON_EXE = ENV_DIR / "python.exe"

def print_step(msg):
    print(f"\n[{'*'*10}] {msg} [{'*'*10}]")

def fetch_source_code():
    if APP_DIR.exists() and (APP_DIR / "main.py").exists():
        return
        
    print_step("First run setup: Downloading Application Source Code from GitHub...")
    zip_path = Path("app_source.zip")
    
    print("Downloading Application Data...")
    urllib.request.urlretrieve(GITHUB_REPO_ZIP, zip_path)
    
    print("Extracting Application Data...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(".")
        
    os.remove(zip_path)
    
def setup_python():
    if PYTHON_EXE.exists():
        return

    print_step("First run setup: Downloading isolated Python 3.10 environment...")
    ENV_DIR.mkdir(exist_ok=True)
    
    zip_path = ENV_DIR / "python.zip"
    
    # Download Python
    print("Downloading Python engine...")
    urllib.request.urlretrieve(PYTHON_URL, zip_path)
    
    # Extract
    print("Extracting Python engine...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(ENV_DIR)
        
    os.remove(zip_path)
    
    # Fix python._pth to allow pip
    pth_file = ENV_DIR / "python310._pth"
    if pth_file.exists():
        with open(pth_file, 'r') as f:
            lines = f.readlines()
        with open(pth_file, 'w') as f:
            for line in lines:
                if line.startswith('#import site'):
                    f.write('import site\n')
                else:
                    f.write(line)
                    
    # Download and install pip
    print_step("Installing Package Manager (pip)...")
    pip_script = ENV_DIR / "get-pip.py"
    urllib.request.urlretrieve(GET_PIP_URL, pip_script)
    
    subprocess.run([str(PYTHON_EXE), str(pip_script)], check=True)
    os.remove(pip_script)

def install_requirements():
    print_step("Checking and installing dependencies (Downloading AI models may take a while)...")
    req_file = APP_DIR / "requirements.txt"
    if req_file.exists():
        # Remove paddlepaddle and paddleocr as they are incompatible / not strictly needed
        with open(req_file, 'r') as f:
            lines = f.readlines()
        with open(req_file, 'w') as f:
            for line in lines:
                if "paddlepaddle" not in line and "paddleocr" not in line:
                    f.write(line)
        
        subprocess.run([str(PYTHON_EXE.resolve()), "-m", "pip", "install", "-r", str(req_file.resolve())], check=True)

def run_app():
    print_step("Starting Traffic Violation Detection System...")
    main_script = APP_DIR / "main.py"
    
    if not main_script.exists():
        print(f"Error: {main_script} not found after download.")
        input("Press Enter to exit...")
        sys.exit(1)
        
    # Change working directory to the app folder so relative paths work properly
    os.chdir(APP_DIR)
    subprocess.run([str(PYTHON_EXE.resolve()), str(main_script.name)])

if __name__ == "__main__":
    try:
        fetch_source_code()
        setup_python()
        install_requirements()
        run_app()
    except Exception as e:
        print(f"\nAn error occurred during startup:\n{e}")
        input("Press Enter to exit...")
