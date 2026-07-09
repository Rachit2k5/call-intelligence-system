<#
Usage (run from repo root PowerShell):
  .\scripts\package_models.ps1 -OutFile .\models.tar.gz

This creates a gzipped tarball containing the `models/` directory so it can be uploaded
somewhere and referenced by the `MODEL_ARTIFACTS_URL` GitHub secret.
#>
param(
    [string]$OutFile = "models.tar.gz"
)

$repoRoot = Split-Path -Parent $PSScriptRoot
$modelsDir = Join-Path $repoRoot "project\models"

if (-not (Test-Path $modelsDir)) {
    Write-Error "models directory not found at $modelsDir"
    exit 1
}

# Use tar if available (Windows 10+ includes bsdtar/tar)
if (Get-Command tar -ErrorAction SilentlyContinue) {
    Write-Host "Creating $OutFile from $modelsDir..."
    Push-Location $repoRoot
    & tar -czf $OutFile -C project models
    Pop-Location
    Write-Host "Created $OutFile"
    exit 0
}

Write-Error "'tar' not found on PATH. Install tar (or use WSL) to create a .tar.gz archive."
exit 1
