#!/bin/bash

# M-BEIR Evaluation Installation Script

echo "Installing M-BEIR Evaluation Framework..."

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is not installed. Please install Python 3.8 or higher."
    exit 1
fi

# Create virtual environment if it doesn't exist
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv .venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source .venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# Install package in development mode
echo "Installing package in development mode..."
pip install -e .

echo ""
echo "Installation completed successfully!"
echo ""
echo "To activate the virtual environment, run:"
echo "source .venv/bin/activate"
echo ""
echo "To run the example evaluation, run:"
echo "python example_usage.py"
echo ""
echo "To run the full evaluation, run:"
echo "python run_evaluation.py --config configs/default.yaml"