import os
import sys
import urllib.request
import urllib.error
import zipfile
import subprocess
import shutil
import time
import ssl
from pathlib import Path

# ─── Configuration ───────────────────────────────────────────────────────────

# Multiple mirrors for the Python 3.10.11 embeddable zip (tried in order)
PYTHON_MIRRORS = [
    "https://www.python.org/ftp/python/3.10.11/python-3.10.11-embed-amd64.zip",
    # Fallback: NuGet-hosted copy (reliable CDN)
    "https://www.nuget.org/api/v2/package/python/3.10.11",
]

GITHUB_REPO_ZIP = "https://github.com/harimac425/Traffic_Violation_Detection/archive/refs/heads/main.zip"
GET_PIP_URL = "https://bootstrap.pypa.io/get-pip.py"

APP_DIR = Path("Traffic_Violation_Detection-main")
ENV_DIR = Path("python_env")
PYTHON_EXE = ENV_DIR / "python.exe"

# Download settings
DOWNLOAD_TIMEOUT = 30        # seconds per connection attempt
DOWNLOAD_RETRIES = 3         # retries per mirror
CHUNK_SIZE = 64 * 1024       # 64 KB chunks for progress reporting

# ─── Helpers ─────────────────────────────────────────────────────────────────

def print_step(msg):
    print(f"\n[{'*'*10}] {msg} [{'*'*10}]")

def format_size(size_bytes):
    """Human-readable file size."""
    for unit in ("B", "KB", "MB", "GB"):
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} TB"

def download_with_progress(url, dest_path, description="file", timeout=DOWNLOAD_TIMEOUT):
    """
    Download a URL to a local file with:
      - Connection timeout
      - Chunk-based progress reporting
      - Automatic cleanup on failure
    Returns True on success, raises on failure.
    """
    # Create an unverified SSL context as a fallback for corporate proxies
    ctx = ssl.create_default_context()
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "TVDS-Launcher/1.0"})
        response = urllib.request.urlopen(req, timeout=timeout, context=ctx)
    except (urllib.error.URLError, ssl.SSLError):
        # Retry with no SSL verification (helps behind corporate firewalls)
        ctx = ssl._create_unverified_context()
        req = urllib.request.Request(url, headers={"User-Agent": "TVDS-Launcher/1.0"})
        response = urllib.request.urlopen(req, timeout=timeout, context=ctx)

    total = int(response.headers.get("Content-Length", 0))
    downloaded = 0
    start_time = time.time()

    with open(dest_path, "wb") as f:
        while True:
            chunk = response.read(CHUNK_SIZE)
            if not chunk:
                break
            f.write(chunk)
            downloaded += len(chunk)

            # Progress bar
            elapsed = time.time() - start_time
            speed = downloaded / max(elapsed, 0.001)
            if total:
                pct = downloaded / total * 100
                bar_len = 30
                filled = int(bar_len * downloaded / total)
                bar = "█" * filled + "░" * (bar_len - filled)
                print(f"\r  {description}: [{bar}] {pct:5.1f}%  {format_size(speed)}/s", end="", flush=True)
            else:
                print(f"\r  {description}: {format_size(downloaded)}  {format_size(speed)}/s", end="", flush=True)

    print()  # newline after progress bar

    if total and downloaded < total:
        os.remove(dest_path)
        raise RuntimeError(f"Incomplete download: got {downloaded}/{total} bytes")

    return True

def download_with_retries(urls, dest_path, description="file"):
    """
    Try downloading from a list of mirror URLs with retries.
    Returns True on first success; raises RuntimeError if all fail.
    """
    errors = []
    for i, url in enumerate(urls if isinstance(urls, list) else [urls]):
        mirror_label = f"mirror {i+1}/{len(urls)}" if isinstance(urls, list) and len(urls) > 1 else ""
        for attempt in range(1, DOWNLOAD_RETRIES + 1):
            try:
                if mirror_label:
                    print(f"  Trying {mirror_label} (attempt {attempt}/{DOWNLOAD_RETRIES})...")
                elif attempt > 1:
                    print(f"  Retry {attempt}/{DOWNLOAD_RETRIES}...")
                download_with_progress(url, dest_path, description)
                return True
            except Exception as e:
                err_msg = f"{url} attempt {attempt}: {e}"
                errors.append(err_msg)
                print(f"  ⚠ Download failed: {e}")
                # Clean up partial file
                if os.path.exists(dest_path):
                    try:
                        os.remove(dest_path)
                    except OSError:
                        pass
                if attempt < DOWNLOAD_RETRIES:
                    wait = 2 ** attempt
                    print(f"  Waiting {wait}s before retry...")
                    time.sleep(wait)

    raise RuntimeError(
        f"All download attempts failed for {description}.\n"
        + "\n".join(f"  - {e}" for e in errors)
        + "\n\nPlease check your internet connection and firewall settings."
    )

# ─── Setup Steps ─────────────────────────────────────────────────────────────

def fetch_source_code():
    if APP_DIR.exists() and (APP_DIR / "main.py").exists():
        return

    print_step("First run setup: Downloading Application Source Code from GitHub...")
    zip_path = Path("app_source.zip")

    download_with_retries(GITHUB_REPO_ZIP, zip_path, "Application source code")

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

    # Download Python from mirrors
    print("Downloading Python engine...")
    download_with_retries(PYTHON_MIRRORS, zip_path, "Python 3.10.11")

    # Validate the zip before extracting
    print("Validating download...")
    try:
        with zipfile.ZipFile(zip_path, 'r') as zf:
            bad = zf.testzip()
            if bad:
                raise zipfile.BadZipFile(f"Corrupt entry: {bad}")
            entry_count = len(zf.namelist())
        print(f"  ✓ Valid archive ({entry_count} files)")
    except zipfile.BadZipFile as e:
        os.remove(zip_path)
        raise RuntimeError(
            f"Downloaded file is corrupt: {e}\n"
            "This usually means the download was interrupted or blocked.\n"
            "Please check your internet connection and try again."
        )

    # Extract
    print("Extracting Python engine...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(ENV_DIR)

    os.remove(zip_path)

    # Verify python.exe actually exists after extraction
    if not PYTHON_EXE.exists():
        raise RuntimeError(
            f"Python extraction failed: {PYTHON_EXE} not found.\n"
            "The zip may have a different structure than expected."
        )

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
    download_with_retries(GET_PIP_URL, pip_script, "pip installer")

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

# ─── Main ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    try:
        fetch_source_code()
        setup_python()
        install_requirements()
        run_app()
    except Exception as e:
        print(f"\n{'='*50}")
        print(f"  SETUP ERROR")
        print(f"{'='*50}")
        print(f"\n{e}\n")
        input("Press Enter to exit...")
