@echo off
chcp 65001 >nul
echo Starting AI VTuber...

if not exist "venv\Scripts\python.exe" (
    echo [ERROR] Virtual environment not found. Run install_dependencies.bat first.
    pause
    exit /b 1
)

call venv\Scripts\activate.bat
python Aivtuber.py
pause
