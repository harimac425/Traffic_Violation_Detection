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
REPAIR_ATTEMPTS = 0
MAX_REPAIR_ATTEMPTS = 2

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
    logger.info("[*] Filtering for STRICT Python 3.10 environment...")
    
    # First pass: Look for 3.10 explicitly
    for p in potential_interpreters:
        p = p.strip()
        if not p or p in visited_paths: continue
        visited_paths.add(p)
        
        if not os.path.exists(p): continue
        if not "python" in p.lower(): continue
        
        # Check version
        try:
            v_check = subprocess.run([p, "-c", "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"], 
                                   capture_output=True, text=True, timeout=2, check=False)
            if v_check.returncode == 0 and v_check.stdout.strip() == "3.10":
                logger.info(f"    [MATCH] Found verified Python 3.10: {p}")
                return p
        except:
            continue

    logger.warning("[!] No Python 3.10 installation found. The app requires 3.10 for stability.")
    
    # Second pass: Fallback to any working python but warn
    for p in visited_paths:
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
                logger.warning(f"    [FALLBACK] Using alternate version: {p}")
                return p
        except:
            continue

    # 4. Last ditch: If not frozen, use current
    if not IS_FROZEN:
        return sys.executable
        
    # 5. Filter for STABLE versions only (Ignore 3.13, 3.14+ for AI apps)
    logger.info("[*] Filtering for compatible AI environments (3.10 to 3.12)...")
    stable_paths = []
    for p in visited_paths:
        try:
            v_check = subprocess.run([p, "-c", "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"], 
                                   capture_output=True, text=True, timeout=2, check=False)
            if v_check.returncode == 0:
                v_str = v_check.stdout.strip()
                if v_str in ["3.10", "3.11", "3.12"]:
                    stable_paths.append(p)
                    logger.info(f"    [MATCH] Found stable {v_str}: {p}")
        except:
            continue
            
    if stable_paths:
        return stable_paths[0] # Prefer the first stable one found
    
    # If NO stable versions found, but we have ANY python, warn the user
    if potential_interpreters:
        best_bad_bet = potential_interpreters[0]
        logger.warning(f"[!] No stable versions (3.10-3.12) found. Using best available: {best_bad_bet}")
        return best_bad_bet

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
    
    # We strictly want 3.10 for the "Self-Healing" release
    if major != 3 or minor != 10:
        logger.error(f"[!] ERROR: Python {major}.{minor} detected. Python 3.10 is REQUIRED.")
        logger.error("[!] Please run 'install_python_310.bat' from the app folder.")
        return False
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

def download_file(url, target_path):
    """Downloads a file from a URL with progress logging."""
    import requests
    try:
        logger.info(f"[*] Downloading: {url}")
        # Add headers to avoid some basic bot blocks
        headers = {'User-Agent': 'TVDS-Bootstrapper/1.0'}
        response = requests.get(url, stream=True, headers=headers)
        response.raise_for_status()
        
        with open(target_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        return True
    except Exception as e:
        logger.error(f"[ERROR] Download failed: {e}")
        return False

def recover_source_code():
    """Downloads and extracts the application source code from the master repository."""
    import zipfile
    source_url = "https://github.com/harimac425/Traffic_Violation_Detection/archive/refs/heads/momcodebase.zip"
    zip_tmp = APP_DIR / "source_temp.zip"
    
    logger.info("\n" + "*" * 60)
    logger.info("  STANDALONE BOOTSTRAP: Application source code missing.")
    logger.info("  Recovering from repository: momcodebase branch...")
    logger.info("*" * 60)
    
    if download_file(source_url, zip_tmp):
        try:
            logger.info("[*] Extracting source code...")
            with zipfile.ZipFile(zip_tmp, 'r') as zip_ref:
                # GitHub ZIPs usually have a top-level folder like 'RepoName-BranchName'
                zip_ref.extractall(APP_DIR)
            
            # Find the extracted folder
            extracted_dirs = [d for d in APP_DIR.iterdir() if d.is_dir() and "Traffic_Violation_Detection-momcodebase" in d.name]
            if extracted_dirs:
                root_src = extracted_dirs[0]
                logger.info(f"[*] Moving files from {root_src.name} to application root...")
                for item in root_src.iterdir():
                    dest = APP_DIR / item.name
                    if dest.exists():
                        if dest.is_dir(): shutil.rmtree(dest)
                        else: os.remove(dest)
                    shutil.move(str(item), str(dest))
                
                # Cleanup
                shutil.rmtree(root_src)
                if zip_tmp.exists(): os.remove(zip_tmp)
                
                logger.info("[SUCCESS] Application source code recovered.")
                return True
            else:
                logger.error("[ERROR] Could not find extracted source directory.")
                return False
        except Exception as e:
            logger.error(f"[ERROR] Extraction failed: {e}")
            return False
    return False

def verify_models():
    """Checks if required YOLO models are present and attempts download if missing."""
    logger.info("[*] Integrity check: AI Models...")
    
    # Model Map: File -> Download URL (Using official/verified mirrors)
    # Note: These are placeholder URLs for custom models; main YOLO ones auto-download via Ultralytics
    models_to_verify = {
        "yolo11x.pt": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11x.pt",
        "helmet_yolov8n.pt": None, # Custom model
        "plate_yolov8n.pt": None,  # Custom model
        "pose_landmarker_lite.task": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
    }
    
    models_path = APP_DIR / "models"
    models_path.mkdir(exist_ok=True)
    
    missing_critical = []
    for m, url in models_to_verify.items():
        if not (models_path / m).exists():
            if url:
                logger.info(f"[!] Missing: {m}. Attempting automatic recovery...")
                if not download_file(url, models_path / m):
                    missing_critical.append(m)
            else:
                logger.warning(f"[!] Missing custom model: {m}. Please copy manually.")
                missing_critical.append(m)
    
    if missing_critical:
        # If we are missing custom models, we can't auto-fix them easily unless hosted
        logger.warning(f"[!] Missing assets: {', '.join(missing_critical)}")
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

def trigger_python_310_install():
    """Automatically installs Python 3.10 using winget if verified version is missing."""
    logger.warning("\n" + "!" * 60)
    logger.warning("  STABILITY ALERT: Python 3.10 is required but not found.")
    logger.warning("!" * 60)
    print("\n[*] Would you like to automatically install the verified Python 3.10 environment? (y/n): ", end="")
    choice = input().lower().strip()
    if choice != 'y':
        logger.error("[FATAL] Manual installation required. Please use 'install_python_310.bat'.")
        return False
        
    logger.info("[*] Launching Automated Python 3.10 Installer...")
    # We use winget directly for the most seamless experience
    cmd = [
        "winget", "install", "Python.Python.3.10", 
        "--version", "3.10.11", "--silent", 
        "--accept-package-agreements", "--accept-source-agreements"
    ]
    
    rc, out, err = run_command(cmd, stream=True)
    if rc == 0:
        logger.info("[SUCCESS] Python 3.10.11 installed. Please restart the application.")
        return True
    else:
        logger.error(f"[ERROR] Automated install failed (Code: {rc}).")
        logger.error("Please run 'install_python_310.bat' manually as Administrator.")
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
                
                # --- SELF-HEALING TRIGGERS ---
                global REPAIR_ATTEMPTS
                
                # Check if we should even attempt repair (Protect against loops on incompatible Python)
                major, minor = sys.version_info[:2]
                can_repair = (major == 3 and 10 <= minor <= 12)
                
                if not can_repair and ("ModuleNotFoundError" in line or "No module named" in line or "AttributeError" in line):
                    logger.error("[CRITICAL] Environment Incompatibility: Auto-repair disabled for Python 3.13+")
                    logger.error("Please switch to Python 3.11 for a stable experience.")
                    process.terminate()
                    sys.exit(1)

                # 1. Check for missing modules
                if "ModuleNotFoundError" in line or "No module named" in line:
                    if REPAIR_ATTEMPTS >= MAX_REPAIR_ATTEMPTS:
                        logger.error("[FATAL] Multiple repair attempts failed. Stopping to prevent loop.")
                        process.terminate()
                        sys.exit(1)
                        
                    REPAIR_ATTEMPTS += 1
                    logger.error(f"[SELF-HEAL] Detected missing dependency: {line.strip()}")
                    process.terminate()
                    logger.info(f"[*] Attempting to auto-install (Attempt {REPAIR_ATTEMPTS}/{MAX_REPAIR_ATTEMPTS})...")
                    if install_dependencies():
                        logger.info("[OK] Dependencies repaired. Restarting application...")
                        time.sleep(2)
                        return launch_application()
                
                # 2. Check for the specific PyTorch/CUDA crash (DLL load failure)
                if "WinError 1114" in line or "c10.dll" in line or "AttributeError: module 'torch' has no attribute 'save'" in line:
                    if REPAIR_ATTEMPTS >= MAX_REPAIR_ATTEMPTS:
                        logger.error("[FATAL] AI Engine repair failed multiple times. Stopping.")
                        process.terminate()
                        sys.exit(1)
                        
                    REPAIR_ATTEMPTS += 1
                    logger.error("[CRITICAL] Detected AI Engine crash (DLL load failure or broken Torch).")
                    process.terminate()
                    
                    if repair_torch(cpu_only=True):
                        logger.info(f"[*] Self-heal successful (Attempt {REPAIR_ATTEMPTS}). Restarting...")
                        return launch_application() # Re-launch
                    else:
                        logger.error("[FATAL] Auto-repair failed. Please contact engineering support.")
                        sys.exit(1)
                        
        if process.poll() is not None and process.returncode != 0:
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
    global PYTHON_EXE
    if not PYTHON_EXE:
        if trigger_python_310_install():
            input("Install complete. Press Enter to exit and re-launch...")
            sys.exit(0)
        else:
            sys.exit(1)

    logger.info(f"[*] Targeting Verified Python: {PYTHON_EXE}")
    
    # Phase 0: Source Recovery (Standalone Bootstrapper)
    if not MAIN_SCRIPT.exists():
        if not recover_source_code():
            logger.critical("[FATAL] Source recovery failed. Cannot continue.")
            input("Press Enter to close...")
            sys.exit(1)
            
    # Phase 1: Environment Integrity
    if not check_python():
        input("Press Enter to close...")
        sys.exit(1)
        
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
