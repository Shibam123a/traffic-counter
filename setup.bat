@echo off
REM ===============================
REM Setup + Run Traffic Counter
REM ===============================

SETLOCAL ENABLEDELAYEDEXPANSION
set PYTHON=python

REM 1) Create venv if missing
if not exist venv (
  %PYTHON% -m venv venv
)

REM 2) Activate venv
call venv\Scripts\activate

REM 3) Upgrade pip tooling
python -m pip install --upgrade pip wheel setuptools

REM 4) Install CPU Torch (change to CUDA wheel if you have NVIDIA GPU)
REM For GPU, replace the next line with:
REM   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

REM 5) Install the rest
pip install ultralytics opencv-python yt-dlp pandas

REM 6) Run
set VIDEO_URL=https://www.youtube.com/watch?v=MNn9qKG2UFI

echo.
echo Starting Traffic Counter Program...
python traffic_counter.py --youtube_url "%VIDEO_URL%" --output_csv results.csv --output_video overlay_output.mp4 --line_pos 0.60

echo.
echo Program finished. Output files generated:
echo   - results.csv
echo   - overlay_output.mp4
pause
