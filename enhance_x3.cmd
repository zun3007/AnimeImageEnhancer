@echo off
echo Running image enhancement with EDSR_x3.pb on GPU...
python enhance.py --model x3 --use_gpu
pause