# Start khalij-watch on this machine and open the console.
#
# Requires uv (https://docs.astral.sh/uv/), which installs Python 3.12 itself.
# The built console in web\dist is served by the API, so Node is not needed.

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    Write-Host "uv is not installed. Install it, then run this again:"
    Write-Host "  winget install --id astral-sh.uv -e"
    exit 1
}

if (-not (Test-Path "web\dist\index.html")) {
    Write-Host "web\dist is missing. Build it once on a machine that has Node:"
    Write-Host "  cd web; npm ci; npm run build"
    exit 1
}

if ((Get-Command Get-NetTCPConnection -ErrorAction SilentlyContinue) -and
    (Get-NetTCPConnection -LocalPort 8000 -State Listen -ErrorAction SilentlyContinue)) {
    Write-Host "Port 8000 is already in use. Close the other program and try again."
    exit 1
}

if (-not (Test-Path ".env")) {
    # No credentials: show the bundled snapshot instead of retrying live feeds.
    $env:AIS_AUTOSTART_ENABLED = "0"
    $env:OPENWATERS_ENABLED = "0"
    Write-Host "No .env found: live AIS is off, showing the bundled snapshot."
}

Write-Host "Syncing Python dependencies (first run only, needs internet)..."
uv sync

Start-Job { Start-Sleep -Seconds 4; Start-Process "http://127.0.0.1:8000" } | Out-Null
Write-Host "Console: http://127.0.0.1:8000  (Ctrl+C stops it)"
uv run --no-sync uvicorn services.api.main:app --host 127.0.0.1 --port 8000
