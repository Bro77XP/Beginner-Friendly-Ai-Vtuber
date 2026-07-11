@echo off
title Installing AI VTuber Dependencies...
chcp 65001 >nul

echo ============================================
echo  AI VTuber - Dependency Installer
echo ============================================
echo.

REM Check if Python 3.10 is available via py launcher
py -3.10 --version >nul 2>&1
if %errorlevel% equ 0 (
    set PYTHON_CMD=py -3.10
    goto :found_python
)

REM Fallback: check if default python is 3.10
python --version 2>&1 | findstr /C:"3.10" >nul
if %errorlevel% equ 0 (
    set PYTHON_CMD=python
    goto :found_python
)

echo [ERROR] Python 3.10 is required but not found.
echo Install Python 3.10 from https://www.python.org/downloads/release/python-31011/
echo Make sure to check "Add Python to PATH" and install the "py" launcher.
pause
exit /b 1

:found_python
%PYTHON_CMD% --version
echo.

REM Create virtual environment if it doesn't exist
if not exist "venv\Scripts\python.exe" (
    echo [INFO] Creating virtual environment with Python 3.10...
    %PYTHON_CMD% -m venv venv
    if %errorlevel% neq 0 (
        echo [ERROR] Failed to create virtual environment.
        pause
        exit /b 1
    )
    echo [OK] Virtual environment created.
) else (
    echo [OK] Virtual environment already exists.
)
echo.

REM Activate virtual environment
echo [INFO] Activating virtual environment...
call venv\Scripts\activate.bat
if %errorlevel% neq 0 (
    echo [ERROR] Failed to activate virtual environment.
    pause
    exit /b 1
)
echo [OK] Virtual environment activated.
echo.

REM Upgrade pip
echo [INFO] Upgrading pip...
python -m pip install --upgrade pip
echo.

REM Install dependencies
echo [INFO] Installing dependencies from requirements.txt...
pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo.
    echo [WARNING] Some dependencies failed to install.
    echo Attempting alternative installation for common problematic packages...
    echo.

    REM Try installing PyAudio via alternative method
    echo [INFO] Installing PyAudio...
    pip install pipwin
    if %errorlevel% equ 0 (
        pipwin install pyaudio
    ) else (
        pip install pyaudio
    )
)

echo.
echo ============================================
echo  Installation Complete!
echo ============================================
echo.
echo To run the AI VTuber, use:
echo     venv\Scripts\activate ^&^& python Aivtuber.py
echo.
echo Or simply double-click: run_vtuber.bat (if available)
echo.
pause
