<#
.SYNOPSIS
    Build a clean, transferable copy of khalij-watch without touching the working tree.

.DESCRIPTION
    Copies the console, the training code, the final report PDF and the data the
    console reads into a staging folder, refuses to include secrets or
    machine-specific environments, then zips the result.

    Never copied: .env, .venv*, web/node_modules, .git, data/qa, data/kaggle,
    data/datasets (unless -IncludeTrainingData), models/runs (unless requested).

.PARAMETER Out
    Destination zip. Defaults to the Desktop.

.PARAMETER IncludeTrainingData
    Also copy data/datasets (training chips) and models/runs (checkpoints), which
    are needed to retrain or reproduce the report comparisons.

.EXAMPLE
    .\clean_ship.ps1
    .\clean_ship.ps1 -Out D:\handover\khalij-watch.zip -IncludeTrainingData
#>
[CmdletBinding()]
param(
    [string]$Out = (Join-Path ([Environment]::GetFolderPath("Desktop")) "khalij-watch-ship.zip"),
    [switch]$IncludeTrainingData
)

$ErrorActionPreference = "Stop"
$root = $PSScriptRoot
$staging = Join-Path ([System.IO.Path]::GetTempPath()) ("khalij-watch-ship-" + [guid]::NewGuid().ToString("N").Substring(0, 8))

function Copy-Tree {
    param([string]$From, [string]$To)
    if (-not (Test-Path -LiteralPath $From)) {
        Write-Warning "missing, skipped: $From"
        return
    }
    robocopy $From $To /E /XJ /XD __pycache__ .pytest_cache .ruff_cache /NFL /NDL /NJH /NJS /NP | Out-Null
    # robocopy: 0-7 are success (0 nothing to do, 1+ copied), 8+ is a real error.
    if ($LASTEXITCODE -ge 8) { throw "robocopy failed for $From (exit $LASTEXITCODE)" }
}

try {
    Write-Host "Staging in $staging"
    New-Item -ItemType Directory -Path $staging | Out-Null

    # Root files the launcher and the build need.
    foreach ($f in @("pyproject.toml", "uv.lock", "start.ps1", "start.cmd", ".env.example", "README.md", ".gitignore")) {
        $src = Join-Path $root $f
        if (Test-Path -LiteralPath $src) { Copy-Item -LiteralPath $src -Destination $staging }
    }

    # Code: application, pipeline, training, tests, docs.
    foreach ($d in @("core", "services", "scripts", "training", "tests", "notebooks", "docs")) {
        Copy-Tree (Join-Path $root $d) (Join-Path $staging $d)
    }

    # Frontend: sources for a rebuild, dist for running without Node.
    foreach ($d in @("web\src", "web\dist")) {
        Copy-Tree (Join-Path $root $d) (Join-Path $staging $d)
    }
    foreach ($f in @("package.json", "package-lock.json", "index.html", "tsconfig.json", "vite.config.ts")) {
        $src = Join-Path $root "web\$f"
        if (Test-Path -LiteralPath $src) { Copy-Item -LiteralPath $src -Destination (Join-Path $staging "web") }
    }

    # Data the console actually reads.
    New-Item -ItemType Directory -Path (Join-Path $staging "data") | Out-Null
    Copy-Item -LiteralPath (Join-Path $root "data\khalij.db") -Destination (Join-Path $staging "data")
    foreach ($d in @("chips", "reference")) {
        Copy-Tree (Join-Path $root "data\$d") (Join-Path $staging "data\$d")
    }

    # The product model (not the superseded v1).
    $onnx = Join-Path $root "models\point_detector_kaggle-v2.onnx"
    if (Test-Path -LiteralPath $onnx) {
        New-Item -ItemType Directory -Path (Join-Path $staging "models") | Out-Null
        Copy-Item -LiteralPath $onnx -Destination (Join-Path $staging "models")
    }

    # The final report: PDF only.
    $pdf = Join-Path $root "reports\report-final.pdf"
    if (Test-Path -LiteralPath $pdf) {
        New-Item -ItemType Directory -Path (Join-Path $staging "reports") | Out-Null
        Copy-Item -LiteralPath $pdf -Destination (Join-Path $staging "reports")
    } else {
        Write-Warning "reports\report-final.pdf not found; the zip will have no report."
    }

    # Recipient guide. It lives outside the project on purpose: it ships in the
    # zip only, never in the working tree.
    $guide = Join-Path (Split-Path -Parent $root) "khalij-watch-handover\START-HERE.pdf"
    if (Test-Path -LiteralPath $guide) {
        Copy-Item -LiteralPath $guide -Destination (Join-Path $staging "START-HERE.pdf")
    } else {
        Write-Warning "recipient guide not found: $guide"
    }

    if ($IncludeTrainingData) {
        Copy-Tree (Join-Path $root "data\datasets") (Join-Path $staging "data\datasets")
        Copy-Tree (Join-Path $root "models\runs") (Join-Path $staging "models\runs")
    }

    # Safety checks: a shipped copy must never carry secrets or a foreign environment.
    $forbidden = @(
        (Join-Path $staging ".env"),
        (Join-Path $staging "web\node_modules"),
        (Join-Path $staging ".venv")
    )
    foreach ($p in $forbidden) {
        if (Test-Path -LiteralPath $p) { throw "refusing to ship forbidden path: $p" }
    }
    if (Get-ChildItem $staging -Force -Directory | Where-Object { $_.Name -like ".venv*" }) {
        throw "refusing to ship: a .venv* folder reached the staging area"
    }
    if (Get-ChildItem $staging -Recurse -Force -Directory -ErrorAction SilentlyContinue |
        Where-Object { $_.Name -eq "node_modules" -and $_.FullName -ne $staging }) {
        throw "refusing to ship: a node_modules folder reached the staging area"
    }

    # Required pieces, so a silent copy failure cannot produce a broken zip.
    $required = @(
        "start.ps1", "pyproject.toml", "uv.lock",
        "web\dist\index.html", "data\khalij.db", "data\chips", "data\reference",
        "services\api\main.py", "core\db\session.py"
    )
    foreach ($r in $required) {
        if (-not (Test-Path -LiteralPath (Join-Path $staging $r))) { throw "missing from staging: $r" }
    }

    if (Test-Path -LiteralPath $Out) { Remove-Item -LiteralPath $Out -Force }
    New-Item -ItemType Directory -Force -Path (Split-Path -Parent $Out) | Out-Null

    Write-Host "Compressing..."
    $tar = Get-Command tar.exe -ErrorAction SilentlyContinue
    if ($tar) {
        # Name the top-level entries explicitly. Archiving "." would prefix every
        # entry with "./", and Windows Explorer then shows the zip as empty.
        $names = Get-ChildItem -LiteralPath $staging -Force | Select-Object -ExpandProperty Name
        & $tar.Source -a -c -f $Out -C $staging -- @names
        if ($LASTEXITCODE -ne 0) { throw "tar failed (exit $LASTEXITCODE)" }
    } else {
        Compress-Archive -Path (Join-Path $staging "*") -DestinationPath $Out -CompressionLevel Optimal
    }

    $files = (Get-ChildItem $staging -Recurse -File -Force | Measure-Object).Count
    $size = [math]::Round((Get-Item -LiteralPath $Out).Length / 1MB, 1)
    Write-Host ""
    Write-Host "Wrote $Out"
    Write-Host "  $size MB, $files files"
    if (-not $IncludeTrainingData) {
        Write-Host "  (no data\datasets or models\runs; add -IncludeTrainingData for those)"
    }
}
finally {
    if (Test-Path -LiteralPath $staging) { Remove-Item -LiteralPath $staging -Recurse -Force -ErrorAction SilentlyContinue }
}
