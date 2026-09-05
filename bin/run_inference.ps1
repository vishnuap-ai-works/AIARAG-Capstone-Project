$ErrorActionPreference = "Stop"

# Ensure a query is provided
if (-not $args[0]) {
    Write-Host "Usage: .\bin\run_inference.ps1 `"<your_question>`""
    exit 1
}

$Query = $args[0]

# Determine the root directory of the project (parent of the 'bin' directory)
$ProjectRoot = Split-Path (Split-Path $MyInvocation.MyCommand.Path -Parent) -Parent

# Set current location to project root
Push-Location $ProjectRoot

try {
    # Virtual Environment Setup using uv
    if (-not (Test-Path "$ProjectRoot\.venv")) {
        Write-Host "Virtual environment not found in $ProjectRoot. Creating one using uv..."
        if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
            Write-Host "Error: 'uv' is not installed. Please install it first (e.g., pip install uv)." -ForegroundColor Red
            exit 1
        }
        
        uv venv
        
        # Activate environment
        if (Test-Path "$ProjectRoot\.venv\Scripts\Activate.ps1") {
            . "$ProjectRoot\.venv\Scripts\Activate.ps1"
        } elseif (Test-Path "$ProjectRoot\.venv\bin\Activate.ps1") {
            . "$ProjectRoot\.venv\bin\Activate.ps1"
        }
        
        Write-Host "Installing project dependencies from pyproject.toml..."
        uv pip install -e .
    } else {
        # Activate existing environment
        if (Test-Path "$ProjectRoot\.venv\Scripts\Activate.ps1") {
            . "$ProjectRoot\.venv\Scripts\Activate.ps1"
        } elseif (Test-Path "$ProjectRoot\.venv\bin\Activate.ps1") {
            . "$ProjectRoot\.venv\bin\Activate.ps1"
        }
    }

    # Add the 'src' directory to PYTHONPATH
    $SrcDir = Join-Path $ProjectRoot "src"
    if ($env:PYTHONPATH) {
        $env:PYTHONPATH = "$SrcDir;" + $env:PYTHONPATH
    } else {
        $env:PYTHONPATH = "$SrcDir"
    }

    # Run the inference script
    python src/pipeline/inference.py $Query

} finally {
    Pop-Location
}
