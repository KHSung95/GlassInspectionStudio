@echo off
rem Glass Inspection Studio - desktop GUI launcher (double-click).
rem Optional: drag a project folder onto this file, or pass it as the first argument.
setlocal
set "ROOT=%~dp0"
set "PYW=%ROOT%.venv\Scripts\pythonw.exe"
if not exist "%PYW%" (
  echo Virtual environment not found. Run setup.ps1 first:
  echo   powershell -ExecutionPolicy Bypass -File "%ROOT%setup.ps1"
  pause
  exit /b 1
)
set "HF_HUB_DISABLE_TELEMETRY=1"
cd /d "%ROOT%"
start "" "%PYW%" -m gistudio %*
endlocal
