@echo off
echo Installing dependencies from requirements.txt...
pip install -r requirements.txt
if errorlevel 1 (
    echo Error installing dependencies.
    pause
    exit /b 1
)
echo Running video enhancement with EDSR_x3.pb on CPU...
python enhance_video.py input_videos/ output_videos/ --model x3
pause 