@echo off
echo ===================================================
echo Starting Scheduled MLOps Sync: %date% %time%
echo ===================================================

:: Step 1: Switch to the D drive and navigate to the project directory
D:
cd "D:\ml projects\mlops_time_series_modeling"

:: Step 2: Sync Git repository via rebase
echo [INFO] Executing git pull --rebase...
git pull --rebase
if %errorlevel% neq 0 (
    echo [ERROR] Git pull failed with exit code %errorlevel%. Exiting...
    exit /b %errorlevel%
)

:: Step 3: Fetch updated heavy binaries via DVC
echo [INFO] Executing dvc pull...
dvc pull
if %errorlevel% neq 0 (
    echo [ERROR] DVC pull failed with exit code %errorlevel%. Exiting...
    exit /b %errorlevel%
)

echo ===================================================
echo MLOps pipeline synchronization complete!
echo ===================================================