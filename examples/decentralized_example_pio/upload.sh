#!/bin/bash
# Unofficial bash strict mode
set -eo pipefail

echo "================================================="
# Clean print
echo "  ESP32 Split NN PlatformIO Uploader             "
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

# Check arguments
if [ "$1" != "node_a" ] && [ "$1" != "node_b" ]; then
    echo "Usage: ./upload.sh [node_a|node_b]"
    echo "  node_a - Upload Layer 1 (Feature Extractor) node"
    echo "  node_b - Upload Layer 2 (Classifier) node"
    exit 1
fi

ENV_NAME=$1

# Navigate to the script's directory
cd "$(dirname "$0")"

echo -e "\n---> Compiling & Uploading to ESP32 (Environment: ${ENV_NAME})..."
pio run -e "$ENV_NAME" --target upload

echo -e "\n---> Upload complete!"
echo "To monitor execution, use the Serial monitor command:"
echo "  pio device monitor -e ${ENV_NAME}"
