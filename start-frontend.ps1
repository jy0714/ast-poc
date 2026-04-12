# AST PoC - Frontend Dev Server
# Usage: .\start-frontend.ps1

$ErrorActionPreference = "Stop"

# frontend 폴더가 존재하는지 먼저 확인
if (-not (Test-Path "$PSScriptRoot\frontend")) {
    Write-Host "[ERROR] frontend folder not found." -ForegroundColor Red
    exit 1
}

# Node.js 설치 확인
try {
    $NodeVersion = node --version 2>&1
    $NpmVersion = npm --version 2>&1
    Write-Host "[Frontend] Node.js: $NodeVersion" -ForegroundColor Cyan
    Write-Host "[Frontend] npm: $NpmVersion" -ForegroundColor Cyan
} catch {
    Write-Host "[ERROR] Node.js is not installed or not in PATH." -ForegroundColor Red
    Write-Host "Download from: https://nodejs.org/" -ForegroundColor Yellow
    exit 1
}

Push-Location "$PSScriptRoot\frontend"

try {
    if (-not (Test-Path "node_modules")) {
        Write-Host "[Frontend] Running npm install..." -ForegroundColor Yellow
        cmd /c npm install
        if ($LASTEXITCODE -ne 0) { throw "npm install failed" }
    }

    Write-Host ""
    Write-Host "[Frontend] Starting Vite (http://localhost:5173)" -ForegroundColor Green
    Write-Host "[Frontend] Press Ctrl+C to stop" -ForegroundColor Yellow
    Write-Host ""

    cmd /c npm run dev
} finally {
    Pop-Location
}