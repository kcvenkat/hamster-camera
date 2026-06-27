@echo off
setlocal EnableDelayedExpansion

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
        echo.
        echo Python 3.10+ was not found.
        echo Install Python from https://www.python.org/downloads/windows/
        pause
        exit /b 1
    )
    set "PYTHON_CMD=python"
)

echo.
echo ===============================
echo Creating virtual environment...
echo ===============================

%PYTHON_CMD% -m venv "%VENV_DIR%"

call "%VENV_DIR%\Scripts\activate.bat"

echo.
echo ===============================
echo Installing dependencies...
echo ===============================

python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
python -m pip install pyinstaller

echo.
echo ===============================
echo Cleaning previous build...
echo ===============================

if exist build rmdir /s /q build
if exist dist rmdir /s /q dist
if exist hamster_thing.spec del /f /q hamster_thing.spec

echo.
echo ===============================
echo Building executable...
echo ===============================

pyinstaller ^
    --noconfirm ^
    --clean ^
    --onefile ^
    --windowed ^
    --collect-all mediapipe ^
    --hidden-import mediapipe.tasks ^
    --hidden-import mediapipe.tasks.python ^
    --hidden-import mediapipe.tasks.c ^
    --add-data "images;images" ^
    --add-data "hand_landmarker.task;." ^
    --add-data "face_landmarker.task;." ^
    hamster_thing.py

echo.

if exist "dist\hamster_thing.exe" (
    echo ============================================
    echo Build successful!
    echo.
    echo Executable:
    echo     dist\hamster_thing.exe
    echo ============================================
) else (
    echo ============================================
    echo Build FAILED.
    echo Check the errors above.
    echo ============================================
    pause
    exit /b 1
)

pause
endlocal