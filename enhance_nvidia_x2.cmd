@echo off
echo Installing dependencies from requirements.txt...
pip install -r requirements.txt
if errorlevel 1 (
    echo Error installing dependencies.
    pause
    exit /b 1
)
echo Running image enhancement with EDSR_x2.pb on NVIDIA GPU (CUDA)...
python enhance.py --model x2 --use_gpu
pause 