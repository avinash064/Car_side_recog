@echo off
REM GPU Training Environment Setup Script
REM This script downloads Miniconda and sets up TensorFlow with GPU support

echo ================================================================================
echo GPU Training Environment Setup
echo ================================================================================
echo.

REM Check if Miniconda installer exists
if not exist "Miniconda3-latest-Windows-x86_64.exe" (
    echo [1/5] Downloading Miniconda...
    curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-Windows-x86_64.exe
    echo Download complete!
) else (
    echo [1/5] Miniconda installer already downloaded
)

echo.
echo [2/5] Please install Miniconda manually:
echo.
echo    1. Run: Miniconda3-latest-Windows-x86_64.exe
echo    2. Check "Add Miniconda3 to PATH" during installation
echo    3. Complete the installation
echo    4. Close this window and open a NEW Command Prompt
echo.
echo Press any key when installation is complete...
pause

echo.
echo [3/5] Creating GPU environment with Python 3.9...
call conda create -n tf_gpu python=3.9 -y

echo.
echo [4/5] Installing TensorFlow 2.10 with GPU support...
call conda activate tf_gpu
pip install tensorflow==2.10.0

echo.
echo [5/5] Installing CUDA libraries and dependencies...
call conda install -c conda-forge cudatoolkit=11.2 cudnn=8.1.0 -y
pip install numpy pandas scikit-learn matplotlib Pillow

echo.
echo ================================================================================
echo Setup Complete!
echo ================================================================================
echo.
echo To verify GPU is working:
echo    conda activate tf_gpu
echo    python check_gpu.py
echo.
echo To start training:
echo    conda activate tf_gpu
echo    python train_viewpoint_classifier_fast.py
echo.
echo Press any key to exit...
pause
