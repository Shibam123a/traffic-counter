@echo off
REM ===============================
REM Install Dependencies
REM ===============================

echo Creating virtual environment (if not already present)...
python -m venv venv

echo Activating virtual environment...
call venv\Scripts\activate

echo Upgrading pip...
python -m pip install --upgrade pip

echo Installing required packages...
pip install -r requirements.txt

echo.
echo Dependencies installed successfully.
pause
