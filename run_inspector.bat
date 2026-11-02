@echo off
set SCRIPT_DIR=%~dp0
cd /d "%SCRIPT_DIR%"
echo Starting VisionInspect AI Server from %SCRIPT_DIR%...

:: Kill any existing process on port 8000
for /f "tokens=5" %%a in ('netstat -aon ^| findstr :8000 ^| findstr LISTENING') do taskkill /f /pid %%a >nul 2>&1

start "VisionInspect AI Server" cmd /k "cd /d "%SCRIPT_DIR%" && venv\Scripts\uvicorn.exe backend.app:app --host 0.0.0.0 --port 8000 --reload"
timeout /t 2 /nobreak >nul
start http://localhost:8000
echo Inspection Web UI: http://localhost:8000
echo Swagger API Docs:  http://localhost:8000/docs
echo CLI Tool:          python inspect_cli.py --help
