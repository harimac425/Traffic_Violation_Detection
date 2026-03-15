import os
import sys
import subprocess
import logging
import platform
import shutil
import time
import json
import winreg # For deep registry lookup
from pathlib import Path

# --- Constants ---
IS_FROZEN = getattr(sys, 'frozen', False)
EXE_PATH = Path(sys.executable)
APP_DIR = EXE_PATH.parent if IS_FROZEN else Path(__file__).parent

# --- Configuration ---
APP_NAME = "Traffic Violation Detection System"
APP_SHORT_NAME = "TVDS"
VERSION = "1.0.0"
LOG_DIR = Path("logs")
BOOTSTRAP_LOG = LOG_DIR / "bootstrap.log"
REQUIREMENTS_FILE = Path("requirements.txt")
MAIN_SCRIPT = Path("main.py")
MODELS_DIR = Path("models")

# --- Logging Setup ---
os.makedirs(LOG_DIR, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(BOOTSTRAP_LOG, mode='a'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("Bootstrap")

def log_header():
    logger.info("=" * 60)
    logger.info(f"  {APP_NAME} v{VERSION}")
    logger.info(f"  SYSTEM BOOTSTRAP: {datetime_now()}")
    logger.info("=" * 60)

def datetime_now():
    from datetime import datetime
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# --- Utilities ---

def get_python_from_registry():
    """Queries the Windows Registry for installed Python interpreters."""
    found_paths = []
    
    # We check both User and Local Machine hives
    hives = [winreg.HKEY_CURRENT_USER, winreg.HKEY_LOCAL_MACHINE]
    base_key = r"Software\Python\PythonCore"
    
    for hive in hives:
        try:
            with winreg.OpenKey(hive, base_key) as root_key:
                # Iterate through version keys (e.g., 3.12)
                i = 0
                while True:
                    try:
                        version = winreg.EnumKey(root_key, i)
                        i += 1
                        
                        # Dive into InstallPath
                        try:
                            install_path_key = f"{base_key}\\{version}\\InstallPath"
                            with winreg.OpenKey(hive, install_path_key) as ipk:
                                # 1. Try to get ExecutablePath (preferred)
                                try:
                                    exe_path, _ = winreg.QueryValueEx(ipk, "ExecutablePath")
                                    if exe_path and os.path.exists(exe_path):
                                        found_paths.append(exe_path)
                                except:
                                    pass
                                    
                                # 2. Try Default value (Install directory)
                                try:
                                    folder = winreg.QueryValue(ipk, None) # Returns string, not tuple
                                    if folder:
                                        exe_path = os.path.join(folder, "python.exe")
                                        if os.path.exists(exe_path):
                                            found_paths.append(exe_path)
                                except:
                                    pass
                        except:
                            continue
                    except OSError: # End of keys
                        break
        except FileNotFoundError:
            continue
            
    return found_paths

def get_python_path():
    """Searches for a valid Python interpreter in the system."""
    potential_interpreters = []

    # 1. Registry Lookup (Most consistent as requested)
    logger.info("[*] Searching Windows Registry for Python installations...")
    registry_paths = get_python_from_registry()
    potential_interpreters.extend(registry_paths)

    # 2. Try 'python' and 'python3' commands from PATH
    logger.info("[*] Scanning System PATH for Python...")
    for cmd in ["python", "python3", "py -3", "py"]:
        try:
            check_cmd = ["where", cmd.split()[0]] if os.name == 'nt' else ["which", cmd]
            res = subprocess.run(check_cmd, capture_output=True, text=True, check=False)
            if res.returncode == 0:
                paths = res.stdout.strip().split('\n')
                for p in paths:
                    potential_interpreters.append(p.strip())
        except:
            continue

    # 3. Hardcoded fallback paths
    common_paths = [
        os.path.expandvars(r"%LocalAppData%\Programs\Python\Python312\python.exe"),
        os.path.expandvars(r"%LocalAppData%\Programs\Python\Python311\python.exe"),
        os.path.expandvars(r"%LocalAppData%\Programs\Python\Python310\python.exe"),
        os.path.expandvars(r"%ProgramFiles%\Python312\python.exe"),
        os.path.expandvars(r"%ProgramFiles%\Python311\python.exe"),
        os.path.expandvars(r"%ProgramFiles%\Python310\python.exe"),
    ]
    potential_interpreters.extend(common_paths)

    # Validate all found paths
    visited_paths = set()
    for p in potential_interpreters:
        p = p.strip()
        if not p or p in visited_paths: continue
        visited_paths.add(p)
        
        if not os.path.exists(p): continue
        if not "python" in p.lower(): continue
        
        # CRITICAL: Skip if it's the EXE itself (recursion protection)
        if IS_FROZEN and str(EXE_PATH).lower() in p.lower():
            continue
            
        # Skip Windows Store stubs (WindowsApps)
        if "windowsapps" in p.lower():
            continue
            
        # SANITY CHECK - Does it actually work?
        try:
            test = subprocess.run([p, "-c", "import sys; print('ready')"], capture_output=True, text=True, timeout=2, check=False)
            if test.returncode == 0 and "ready" in test.stdout:
                return p
        except:
            continue

    # 4. Last ditch: If not frozen, use current
    if not IS_FROZEN:
        return sys.executable
        
    return None

PYTHON_EXE = get_python_path()

def run_command(cmd, wait=True, stream=False, use_system_python=False):
    """Safely runs a command and returns the exit code and output."""
    if use_system_python and cmd[0] == sys.executable:
        cmd[0] = PYTHON_EXE
        
    try:
        if wait:
            if stream:
                # Direct piping to console for real-time progress (pip install etc)
                result = subprocess.run(cmd, check=False)
                return result.returncode, "", ""
            else:
                result = subprocess.run(cmd, capture_output=True, text=True, check=False)
                return result.returncode, result.stdout, result.stderr
        else:
            subprocess.Popen(cmd)
            return 0, "", ""
    except Exception as e:
        return -1, "", str(e)

# --- Environment Checks ---

def check_python():
    """Verifies Python installation and PATH."""
    logger.info("[*] Verifying Python Environment...")
    major, minor = sys.version_info[:2]
    logger.info(f"[*] Native Python detected: {major}.{minor}")
    
    # We want 3.10-3.12 for maximum compatibility with PyTorch/MediaPipe
    if major != 3 or minor < 10 or minor > 12:
        logger.warning(f"[!] Warning: Python {major}.{minor} is outside recommended range (3.10-3.12).")
        logger.warning("[!] If you encounter DLL errors, please downgrade to Python 3.10.")
    return True

def check_gpu():
    """Checks if a compatible NVIDIA GPU is available."""
    logger.info("[*] Searching for NVIDIA GPU acceleration hardware...")
    rc, out, err = run_command(["nvidia-smi"])
    if rc == 0:
        logger.info("[OK] NVIDIA GPU detected.")
        return True
    else:
        logger.info("[INFO] No NVIDIA GPU detected or drivers missing. Switching to CPU optimization.")
        return False

def verify_models():
    """Checks if required YOLO models are present."""
    logger.info("[*] Integrity check: AI Models...")
    required_models = ["yolo11x.pt", "helmet_yolov8n.pt", "plate_yolov8n.pt", "pose_landmarker_lite.task"]
    models_path = APP_DIR / "models"
    
    if not models_path.exists():
        logger.warning(f"[!] Models directory missing at {models_path}.")
        return False
    
    missing = []
    for m in required_models:
        if not (models_path / m).exists():
            missing.append(m)
    
    if missing:
        logger.warning(f"[!] Missing models: {', '.join(missing)}")
        return False
    
    logger.info("[OK] All required AI models cataloged.")
    return True

# --- Self-Healing ---

def repair_torch(cpu_only=True):
    """Automatically repairs the AI engine if it's broken or incompatible."""
    logger.info(f"[*] REPAIR INITIATED: Reinstalling PyTorch engine (CPU-Only: {cpu_only})...")
    
    # 1. Uninstall broken versions
    logger.info("[*] Removing existing PyTorch files...")
    run_command([PYTHON_EXE, "-m", "pip", "uninstall", "torch", "torchvision", "torchaudio", "-y"])
    
    # 2. Reinstall correct version
    if cpu_only:
        logger.info("[*] Installing stable CPU version...")
        rc, out, err = run_command([
            PYTHON_EXE, "-m", "pip", "install", 
            "torch", "torchvision", "torchaudio", 
            "--index-url", "https://download.pytorch.org/whl/cpu"
        ], stream=True)
    else:
        logger.info("[*] Installing GPU/CUDA optimized version...")
        rc, out, err = run_command([
            PYTHON_EXE, "-m", "pip", "install", 
            "torch", "torchvision", "torchaudio"
        ], stream=True)
    
    if rc == 0:
        logger.info("[SUCCESS] AI Engine repaired.")
        return True
    else:
        logger.error(f"[ERROR] Repair failed: {err}")
        return False

def install_dependencies():
    """Force-installs all required dependencies from requirements.txt."""
    if not REQUIREMENTS_FILE.exists():
        logger.error("[ERROR] Missing requirements.txt. Cannot verify environment.")
        return False
        
    logger.info("[*] Syncing application dependencies... (this may take a minute)")
    logger.info("[INFO] DETAILED INSTALL LOGS WILL BE SHOWN BELOW:")
    logger.info("-" * 40)
    rc, out, err = run_command([PYTHON_EXE, "-m", "pip", "install", "-r", str(REQUIREMENTS_FILE)], stream=True)
    logger.info("-" * 40)
    if rc == 0:
        logger.info("[OK] Dependencies synchronized.")
        return True
    else:
        logger.warning(f"[!] Warning during dependency sync: {err}")
        return True # Continue anyway, app might still work

# --- Start Sequence ---

def launch_application():
    """Launches the main window and monitors for environmental crashes."""
    if not MAIN_SCRIPT.exists():
        logger.critical(f"[FATAL] Entry point {MAIN_SCRIPT} not found. Reinstall the application.")
        return

    logger.info("[*] Handing control to Main Application...")
    
    # We run in a managed subprocess so we can catch WinError 1114
    try:
        process = subprocess.Popen(
            [PYTHON_EXE, str(MAIN_SCRIPT)],
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' and False else 0 
        )
        
        # Monitor stderr for the famous DLL crash
        while True:
            line = process.stderr.readline()
            if not line and process.poll() is not None:
                break
            
            if line:
                print(line.strip(), file=sys.stderr) # Keep printing to console
                
                # Check for the specific PyTorch/CUDA crash
                if "WinError 1114" in line or "c10.dll" in line:
                    logger.error("[CRITICAL] Detected AI Engine crash (DLL load failure).")
                    process.terminate()
                    
                    if repair_torch(cpu_only=True):
                        logger.info("[*] Self-heal successful. Restarting application...")
                        return launch_application() # Re-launch
                    else:
                        logger.error("[FATAL] Auto-repair failed. Please contact engineering support.")
                        sys.exit(1)
                        
        if process.returncode != 0:
            logger.warning(f"[*] Application closed with exit code {process.returncode}")
            
    except Exception as e:
        logger.error(f"[ERROR] Launcher failed to start process: {e}")

def main():
    # Loop Protection
    if IS_FROZEN and len(sys.argv) > 1:
        if "-m" in sys.argv or str(MAIN_SCRIPT) in sys.argv:
            print("[CRITICAL ERROR] Recursive launcher loop detected. Stopping.")
            sys.exit(1)

    log_header()
    
    # Check if we even found a Python interpreter
    if not PYTHON_EXE:
        logger.error("[FATAL] A valid Python interpreter could not be found.")
        logger.error("Please ensure Python 3.10-3.12 is installed and added to your System PATH.")
        logger.error("Download from: https://www.python.org/downloads/")
        input("Press Enter to exit...")
        sys.exit(1)

    logger.info(f"[*] Targeting Verified Python: {PYTHON_EXE}")
    
    # Phase 1: Environment Integrity
    check_python()
    gpu_available = check_gpu()
    
    # Phase 2: Dependency Sync
    install_dependencies()
    
    # Phase 3: Assets Check
    verify_models()
    
    # Phase 4: Launch & Monitor
    launch_application()
    
    # Keep window open if something failed or finished
    logger.info("[*] System operations completed.")
    input("Press Enter to close this window...")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("[*] Bootstrap interrupted by user.")
    except Exception as e:
        logger.critical(f"[FATAL UNCAUGHT ERROR] {e}")
