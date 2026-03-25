# AST PoC - Frontend Dev Server
# Usage: .\start-frontend.ps1

$ErrorActionPreference = "Stop"

Push-Location "$PSScriptRoot\frontend"

try {
    if (-not (Test-Path "node_modules")) {
        Write-Host "[Frontend] Running npm install..." -ForegroundColor Yellow
        cmd /c npm install
        if ($LASTEXITCODE -ne 0) { throw "npm install failed" }
    }

    Write-Host "[Frontend] Starting Vite (http://localhost:5173)" -ForegroundColor Green
    Write-Host ""

    cmd /c npm run dev
} finally {
    Pop-Location
}
