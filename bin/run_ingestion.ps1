<#
.SYNOPSIS
    Runs the document ingestion pipeline on Windows.
#>

$ErrorActionPreference = "Stop"

# Determine the project root directory (parent of the 'bin' directory)
$ProjectRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)

# Virtual Environment Setup using uv
$VenvPath = Join-Path $ProjectRoot ".venv"
if (-Not (Test-Path $VenvPath)) {
    Write-Host "Virtual environment not found in $ProjectRoot. Creating one using uv..."
    if (-Not (Get-Command "uv" -ErrorAction SilentlyContinue)) {
        Write-Error "'uv' is not installed. Please install it first (e.g., pip install uv)."
        exit 1
    }
    
    Push-Location $ProjectRoot
    try {
        uv venv
        # Activate environment
        . .\.venv\Scripts\Activate.ps1
        Write-Host "Installing project dependencies from pyproject.toml..."
        uv pip install -e .
    } finally {
        Pop-Location
    }
} else {
    # Activate existing environment
    . "$ProjectRoot\.venv\Scripts\Activate.ps1"
}

# Add the 'src' directory to PYTHONPATH
$SrcDir = Join-Path $ProjectRoot "src"
if ($env:PYTHONPATH) {
    $env:PYTHONPATH = "$SrcDir;$env:PYTHONPATH"
} else {
    $env:PYTHONPATH = $SrcDir
}

Write-Host "=========================================="
Write-Host "Starting Document Ingestion Pipeline..."
Write-Host "=========================================="

# Run the python pipeline script
python -m pipeline.store
