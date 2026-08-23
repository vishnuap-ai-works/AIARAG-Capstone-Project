# PowerShell script to install requirements (via pyproject.toml) using uv, and run Docker deployment.
# Run via: .\bin\setup_env.ps1
if (-Not (Test-Path ".env")) {
    Write-Host "Creating .env from .env.example..."
    Copy-Item ".env.example" -Destination ".env"
}
if (-Not (Test-Path ".venv")) {
    Write-Host "Syncing project dependencies and creating uv.lock..."
    uv sync
} else {
    Write-Host "Virtual environment already exists, skipping sync."
}
Write-Host "Activating virtual environment..."
.\.venv\Scripts\Activate.ps1
Write-Host "Starting Docker deployment..."
docker-compose up -d --build
