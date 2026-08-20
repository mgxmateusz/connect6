@echo off
setlocal

py -m venv .venv
call .venv\Scripts\activate
python -m pip install --upgrade pip

REM RTX 50 / Blackwell requires a PyTorch build with CUDA 12.8 or newer.
REM If the current PyTorch selector recommends a newer CUDA build, use that instead.
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt

python check_gpu.py
pause
