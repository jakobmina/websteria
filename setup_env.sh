#!/bin/bash

# --- 1. Environment Configuration ---
# Regla 1.3: Prohibición de Singularidades (No se permiten sistemas puramente conservativos ni disipativos)
# Este script inicializa el entorno dinámico para el desarrollo de Websteria.

VENV_NAME="venv"
PYTHON_BIN=$(which python3)

echo "--- Initializing Websteria Environment ---"

if [ -z "$PYTHON_BIN" ]; then
    echo "Error: Python 3 not found. Please install it."
    exit 1
fi

# Create virtual environment if it doesn't exist
if [ ! -d "$VENV_NAME" ]; then
    echo "Creating virtual environment: $VENV_NAME..."
    $PYTHON_BIN -m venv $VENV_NAME
else
    echo "Virtual environment $VENV_NAME already exists."
fi

# Activate environment
source $VENV_NAME/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install requirements
if [ -f "requirements.txt" ]; then
    echo "Installing dependencies from requirements.txt..."
    # We ignore errors for editable installs that might not exist on the current system
    # but we inform the user.
    pip install -r requirements.txt || echo "Warning: Some dependencies could not be installed. Check requirements.txt for absolute paths."
else
    echo "Error: requirements.txt not found."
    exit 1
fi

# Install pytest for testing (Regla 4)
echo "Installing pytest..."
pip install pytest pytest-mock

# --- 2. Finalizing ---
echo "--- Environment Setup Complete ---"
echo "To activate the environment, run:"
echo "source $VENV_NAME/bin/activate"
