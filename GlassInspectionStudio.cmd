@echo off
rem Glass Inspection Studio - main entry: web studio (project / capture / labelling / analysis / report).
rem Opens http://127.0.0.1:8800 in the default browser. Close this window to stop the studio and release the camera.
rem Optional: pass a project folder as the first argument (otherwise the most recent project is opened).
setlocal
set "ROOT=%~dp0"
set "PY=%ROOT%.venv\Scripts\python.exe"
if not exist "%PY%" (
  echo Virtual environment not found. Run setup.ps1 first:
  echo   powershell -ExecutionPolicy Bypass -File "%ROOT%setup.ps1"
  pause
  exit /b 1
)
set "HF_HUB_DISABLE_TELEMETRY=1"
cd /d "%ROOT%"
title Glass Inspection Studio (close to stop)
if "%~1"=="" (
  "%PY%" -m gistudio.cli studio --port 8800 --open-browser
) else (
  "%PY%" -m gistudio.cli studio --port 8800 --open-browser --project "%~1"
)
pause
endlocal
