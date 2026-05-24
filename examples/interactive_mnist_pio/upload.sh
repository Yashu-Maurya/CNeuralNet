#!/bin/bash
# Unofficial bash strict mode
set -eo pipefail

echo "================================================="
echo "      ESP32 CNeuralNet PlatformIO Uploader       "
echo "================================================="

# Add PlatformIO default path to PATH if not already present
if ! command -v pio &> /dev/null; then
    if [ -f "$HOME/.platformio/penv/bin/pio" ]; then
        export PATH="$HOME/.platformio/penv/bin:$PATH"
    fi
fi

# Check if PlatformIO is installed
if ! command -v pio &> /dev/null; then
    echo "Error: PlatformIO CLI (pio) is not installed."
    echo "Please install PlatformIO Core: https://docs.platformio.org/en/latest/core/installation.html"
    exit 1
fi

# Check if Python 3 is installed
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is not installed but is required to run the bridge."
    exit 1
fi

# Navigate to the script's directory
cd "$(dirname "$0")"

echo -e "\n---> Step 1: Compiling & Uploading to ESP32..."
pio run --target upload

echo -e "\n---> Step 2: Checking/Installing Python dependencies..."
python3 -m pip install pyserial || {
    echo "Warning: Failed to install pyserial automatically. Please run 'pip install pyserial' manually."
}

echo -e "\n---> Step 3: Launching MNIST Serial Bridge Monitor..."
python3 mnist_bridge.py
