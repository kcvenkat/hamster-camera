@echo off
setlocal

set "SCRIPT_DIR=%~dp0"
cd /d "%SCRIPT_DIR%"

set "VENV_DIR=%SCRIPT_DIR%.venv"
set "PYTHON_CMD="

where py >nul 2>nul
if not errorlevel 1 (
    set "PYTHON_CMD=py -3"
) else (
    where python >nul 2>nul
    if errorlevel 1 (
        echo Python 3 was not found. Install Python 3.10+ from https://www.python.org/downloads/windows/ and try again.
        exit /b 1
    )
    set "PYTHON_CMD=python"
)

echo Creating virtual environment...
%PYTHON_CMD% -m venv "%VENV_DIR%"

call "%VENV_DIR%\Scripts\activate.bat"

python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt pyinstaller

echo Building Windows executable...
pyinstaller --noconfirm --onefile --windowed ^
  --add-data "images;images" ^
  --add-data "hand_landmarker.task;." ^
  --add-data "face_landmarker.task;." ^
  hamster_thing.py

if exist "dist\hamster_thing.exe" (
    echo.
    echo Build complete: dist\hamster_thing.exe
) else (
    echo.
    echo Build failed. Check the output above for details.
    exit /b 1
)

endlocal
