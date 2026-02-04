@echo off
cd /d "%~dp0"
echo Starting LoRA Lens Community Edition...
echo.
python run_lens.py
if errorlevel 1 (
    echo.
    echo ERROR: App failed to start. Check above for details.
)
pause
