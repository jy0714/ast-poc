# AST PoC - Backend Server
# Usage: .\start-backend.ps1

$ErrorActionPreference = "Stop"

if (Test-Path ".venv\Scripts\Activate.ps1") {
    & .venv\Scripts\Activate.ps1
} else {
    Write-Host "[ERROR] .venv not found. Create it first:" -ForegroundColor Red
    Write-Host "  python -m venv .venv" -ForegroundColor Yellow
    Write-Host "  .venv\Scripts\Activate.ps1" -ForegroundColor Yellow
    Write-Host "  pip install -e '.[dev]'" -ForegroundColor Yellow
    exit 1
}

Write-Host "[Backend] Starting uvicorn (http://localhost:8000)" -ForegroundColor Green
Write-Host "[Backend] API docs: http://localhost:8000/docs" -ForegroundColor Cyan
Write-Host ""

uvicorn src.api.main:app --reload --port 8000
