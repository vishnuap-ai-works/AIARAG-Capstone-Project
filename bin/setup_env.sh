#!/bin/bash
# Shell script to install requirements (via pyproject.toml) using uv, and run Docker deployment.
# Run via: bash bin/setup_env.sh
if [ ! -f ".env" ]; then
    echo "Creating .env from .env.example..."
    cp .env.example .env
fi
if [ ! -d ".venv" ]; then
    echo "Syncing project dependencies and creating uv.lock..."
    uv sync
else
    echo "Virtual environment already exists, skipping sync."
fi
echo "Activating virtual environment..."
source .venv/bin/activate
echo "Starting Docker deployment..."
docker-compose up -d --build
