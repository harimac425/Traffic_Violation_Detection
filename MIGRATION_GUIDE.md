# Migration Guide: Setting up on NVIDIA GPU System

This guide explains how to move the **Traffic Violation Detection System** to a new computer and enable full GPU acceleration for maximum performance.

## 1. Transfer Files
1.  Copy the entire `Traffic_violation_Detection` folder to your new machine (e.g., using a USB drive or zip file).
    *   **Important**: Include the `models/` folder so you don't have to download the large models again.

## 2. Prerequisites
On the new NVIDIA system, ensure you have:
*   **Python 3.10+**: Installed and added to PATH.
*   **NVIDIA Drivers**: Latest drivers for your GPU detected.
*   **CUDA Toolkit** (Optional but recommended): Version 11.8 or 12.1 is best for PyTorch.

## 3. Install Dependencies with GPU Support
This is the most critical step. Standard `pip install` often installs the CPU-only version of PyTorch.

1.  Open a terminal/command prompt in the `Traffic_violation_Detection` folder.
2.  **Uninstall existing PyTorch** (if any):
    ```powershell
    pip uninstall torch torchvision torchaudio
    ```
3.  **Install PyTorch with CUDA 12.1 support**:
    ```powershell
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    ```
    *(If you use CUDA 11.8, use `cu118` instead of `cu121`)*.

4.  **Install remaining requirements**:
    ```powershell
    pip install -r requirements.txt
    ```

## 4. Verify GPU Installation
1.  Run the application:
    ```powershell
    python main.py
    ```
2.  Look at the console output on startup. You should see:
    ```text
    Using device: cuda
    Device name: NVIDIA GeForce RTX ...
    ```
3.  In the Application UI > Bottom Bar, check the **Device** dropdown. It should auto-select **CUDA**.

## 5. Performance Note
*   With an NVIDIA GPU (e.g., RTX 3060 or better), the **YOLO11x** (Extra Large) model should run smoothly at 15-30+ FPS.
*   The **Cascade Detection** (cropping) will be significantly faster on GPU.

## Troubleshooting
*   **Error: "Torch not compiled with CUDA enabled"**: This means you have the CPU version. Repeat Step 3 carefully.
*   **Out of Memory (OOM)**: If the YOLO11x model is too big for your GPU VRAM (needs ~6-8GB), switch back to `yolo11m.pt` in `config.py`.
