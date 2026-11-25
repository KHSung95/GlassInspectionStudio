@echo off
rem Glass Inspection Studio - browser draft labelling on http://127.0.0.1:8770
rem Usage: double-click (uses C:\dev\gis_real2d) or pass a project folder as the first argument.
setlocal
set "ROOT=%~dp0"
set "PY=%ROOT%.venv\Scripts\python.exe"
set "PROJECT=%~1"
if "%PROJECT%"=="" set "PROJECT=C:\dev\gis_real2d"
cd /d "%ROOT%"
start "" http://127.0.0.1:8770
"%PY%" -m gistudio.cli label "%PROJECT%" --port 8770
pause
endlocal
