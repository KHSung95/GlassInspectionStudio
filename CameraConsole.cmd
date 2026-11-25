@echo off
rem Glass Inspection Studio - web camera console (live feed + capture) on http://127.0.0.1:8765
rem Usage: double-click (uses C:\dev\gis_pilot) or pass a project folder as the first argument.
rem Close this window to release the camera.
setlocal
set "ROOT=%~dp0"
set "PY=%ROOT%.venv\Scripts\python.exe"
set "PROJECT=%~1"
if "%PROJECT%"=="" set "PROJECT=C:\dev\gis_pilot"
if not exist "%PY%" (
  echo Virtual environment not found. Run setup.ps1 first.
  pause
  exit /b 1
)
cd /d "%ROOT%"
start "" http://127.0.0.1:8765
rem 8192-byte GigE packets need Jumbo Frame (9014) on the camera NIC; 5 fps keeps ~50% link headroom on 1 GbE.
"%PY%" -m gistudio.cli console "%PROJECT%" --port 8765 --packet-size 8192 --fps 5
pause
endlocal
