# VisionInspect AI Launcher
$Root = $PSScriptRoot
Write-Host "Starting VisionInspect AI Server from $Root..." -ForegroundColor Cyan

# Free port 8000 if an old background server is still lingering
$Existing = Get-NetTCPConnection -LocalPort 8000 -ErrorAction SilentlyContinue
if ($Existing) {
    Write-Host "Stopping previous server instance on port 8000..." -ForegroundColor Yellow
    $Existing | ForEach-Object { Stop-Process -Id $_.OwningProcess -Force -ErrorAction SilentlyContinue }
    Start-Sleep -Seconds 1
}

# Start uvicorn server in a dedicated window
Start-Process powershell -WorkingDirectory $Root -ArgumentList "-NoExit", "-Command", "& '$Root\venv\Scripts\uvicorn.exe' backend.app:app --host 0.0.0.0 --port 8000 --reload"

Start-Sleep -Seconds 2
Start-Process "http://localhost:8000"

Write-Host "Inspection Web UI: http://localhost:8000" -ForegroundColor Green
Write-Host "Swagger API Docs:  http://localhost:8000/docs" -ForegroundColor Green
Write-Host "CLI Tool:          python inspect_cli.py --help" -ForegroundColor Yellow
