#!/bin/bash

# MedVision AI - Jupyter Lab Setup Script
# This script prepares the environment and launches Jupyter Lab on port 11006

echo "--- Initializing MedVision-AI Environment ---"

# Check if virtualenv exists, if not create it
if [ ! -d "venv_jupyter" ]; then
    echo "Creating virtual environment 'venv_jupyter'..."
    python3 -m venv venv_jupyter
fi

# Activate environment
source venv_jupyter/bin/activate

# Install core dependencies
echo "Installing/Updating dependencies..."
pip install --upgrade pip
pip install jupyterlab tensorflow opencv-python-headless numpy matplotlib fpdf2 axios-python pandas

# Install notebook specific tools
pip install ipywidgets

# Check for GPU (NVIDIA)
if command -v nvidia-smi &> /dev/null; then
    echo "NVIDIA GPU detected. Ensuring tensorflow-gpu compatibility..."
    # Usually handled by standard tensorflow in newer versions, but specific for older if needed
fi

echo "--- Launching Jupyter Lab on Port 11006 ---"
echo "If you are on a remote server, use: ssh -L 11006:localhost:11006 user@host"

# Launch Jupyter Lab
# --no-browser: Don't open a browser automatically
# --port 11006: Use the requested port
# --ip 0.0.0.0: Listen on all interfaces (useful for DGX/Spark environments)
# --allow-root: Often needed on cluster environments
python3 -m jupyter lab --ip=0.0.0.0 --port=11006 --no-browser --allow-root
