#!/usr/bin/env bash

# Exit immediately if a command exits with a non-zero status
set -e

# Determine the root directory of the project (parent of the 'bin' directory)
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Virtual Environment Setup using uv
if [ ! -d "${PROJECT_ROOT}/.venv" ]; then
    echo "Virtual environment not found in ${PROJECT_ROOT}. Creating one using uv..."
    if ! command -v uv &> /dev/null; then
        echo "Error: 'uv' is not installed. Please install it first (e.g., pip install uv)."
        exit 1
    fi
    
    # Temporarily cd into project root to install dependencies properly
    cd "${PROJECT_ROOT}"
    uv venv
    source .venv/bin/activate
    echo "Installing project dependencies from pyproject.toml..."
    uv pip install -e .
    cd - > /dev/null # Return to original directory
else
    # Activate existing environment
    source "${PROJECT_ROOT}/.venv/bin/activate"
fi

# Add the 'src' directory to PYTHONPATH so python can find the 'rag' and 'pipeline' modules
export PYTHONPATH="${PROJECT_ROOT}/src:${PYTHONPATH}"

# Check if an argument is provided
##if [ -z "$1" ]; then
 #   echo "Usage: $0 <path_to_file_or_directory>"
 #   echo "Example: $0 data/test"
 #   exit 1
#fi

TARGET_PATH="$1"

echo "=========================================="
echo "Starting Document Ingestion Pipeline..."
echo "Target: $TARGET_PATH"
echo "=========================================="

# Run the python pipeline script
python -m pipeline.store
