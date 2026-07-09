@echo off
cd /d "%~dp0"
set "VENV_PYTHON=%~dp0..\.venv\Scripts\python.exe"
echo.
echo  Intelligent Call Prioritization System
echo  Setting up (first run only) and starting the server...
echo.
if exist "%VENV_PYTHON%" (
  "%VENV_PYTHON%" run.py
) else (
  python run.py
)
if errorlevel 1 (
  echo.
  echo  Something went wrong. Try running these manually:
  echo    python -m pip install -r requirements.txt
  echo    python run.py
  echo.
  pause
)
