@echo off
chcp 65001 >nul

echo ============================================
echo  AI VTuber - Dependency Installer
echo ============================================
echo.

REM Check if Python 3.10 is already available
py -3.10 --version >nul 2>&1
if %errorlevel% equ 0 goto :install_deps

echo [INFO] Python 3.10 not found. Downloading...

REM Download Python 3.10.11 installer
set "PYTHON_URL=https://www.python.org/ftp/python/3.10.11/python-3.10.11-amd64.exe"
set "INSTALLER=%TEMP%\python-3.10.11-amd64.exe"

echo [INFO] Downloading Python 3.10.11...
curl -# -L -o "%INSTALLER%" "%PYTHON_URL%"

if %errorlevel% neq 0 (
    echo [ERROR] Download failed. Try manually installing from:
    echo https://www.python.org/downloads/release/python-31011/
    pause
    exit /b 1
)

echo [INFO] Installing Python 3.10.11 (silent)...
"%INSTALLER%" /quiet InstallAllUsers=0 PrependPath=1 Include_launcher=1
del "%INSTALLER%"

REM Refresh PATH for the current session
for /f "tokens=2*" %%a in ('reg query "HKCU\Environment" /v PATH 2^>nul') do set "USER_PATH=%%b"
if defined USER_PATH set "PATH=%USER_PATH%;%PATH%"
set "PATH=%LOCALAPPDATA%\Programs\Python\Python310\;%LOCALAPPDATA%\Programs\Python\Python310\Scripts\;%PATH%"

REM Verify installation
echo [INFO] Verifying installation...
py -3.10 --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python 3.10 installation may have failed. Try manual install.
    pause
    exit /b 1
)

:install_deps
if not exist "venv\Scripts\python.exe" (
    echo [INFO] Creating virtual environment with Python 3.10...
    py -3.10 -m venv venv
)

echo [INFO] Activating virtual environment...
call venv\Scripts\activate.bat

echo [INFO] Upgrading pip...
python -m pip install --upgrade pip

echo [INFO] Installing dependencies...
pip install -r requirements.txt

echo.
echo ============================================
echo  Done! Run the VTuber: run_vtuber.bat
echo ============================================
pause
