# AST PoC - Backend Server
# Usage: .\start-backend.ps1

$ErrorActionPreference = "Stop"

# 스크립트 위치로 이동 (어디서 실행해도 동작)
Set-Location $PSScriptRoot

# venv 폴더 탐색 (.venv와 venv 둘 다 지원)
$VenvPath = $null
if (Test-Path ".\.venv\Scripts\python.exe") {
    $VenvPath = ".\.venv"
} elseif (Test-Path ".\venv\Scripts\python.exe") {
    $VenvPath = ".\venv"
} else {
    Write-Host "[ERROR] Virtual environment not found." -ForegroundColor Red
    Write-Host "Expected one of: .venv or venv" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "Create it with:" -ForegroundColor Yellow
    Write-Host "  C:\Python313\python.exe -m venv venv" -ForegroundColor Cyan
    Write-Host "  .\venv\Scripts\python.exe -m pip install -e '.[dev]'" -ForegroundColor Cyan
    exit 1
}

$PythonExe = "$VenvPath\Scripts\python.exe"

# Python 버전 출력 (디버깅용)
Write-Host "[Backend] Using virtual environment: $VenvPath" -ForegroundColor Cyan
Write-Host "[Backend] Python version:" -NoNewline -ForegroundColor Cyan
& $PythonExe --version

# FastAPI 설치 확인
$FastAPICheck = & $PythonExe -c "import fastapi; print(fastapi.__version__)" 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Host "[ERROR] FastAPI not installed in $VenvPath" -ForegroundColor Red
    Write-Host "Install dependencies first:" -ForegroundColor Yellow
    Write-Host "  $PythonExe -m pip install -e '.[dev]'" -ForegroundColor Cyan
    exit 1
}
Write-Host "[Backend] FastAPI version: $FastAPICheck" -ForegroundColor Cyan

Write-Host ""
Write-Host "[Backend] Starting uvicorn (http://localhost:8000)" -ForegroundColor Green
Write-Host "[Backend] API docs: http://localhost:8000/docs" -ForegroundColor Green
Write-Host "[Backend] Press Ctrl+C to stop" -ForegroundColor Yellow
Write-Host ""

# uvicorn 실행 (venv의 python을 직접 호출)
& $PythonExe -m uvicorn src.api.main:app --reload --port 8000