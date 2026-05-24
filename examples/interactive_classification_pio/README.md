# Interactive Hotdog Classifier (PlatformIO / ESP32)

This example runs a 3-layer neural network directly on the ESP32 microcontroller to classify images as "Hotdog" or "Not-Hotdog". Training and testing images are processed one by one on your computer, normalized, and streamed dynamically to the board over a Serial connection.

## System Architecture
- **Microcontroller (ESP32)**: Holds the network weights, runs forward and backward propagation, evaluates accuracy, and saves/loads weights to flash memory using LittleFS.
- **Computer (Python Bridge)**: Scans your local dataset directory, loads JPEG/PNG images, resizes them to 32x32 grayscale using Pillow, and streams them on-demand as hex-encoded pixel arrays.

---

## Step-by-Step Run Guide

### 1. Prerequisites
- **PlatformIO CLI**: Install [PlatformIO Core](https://docs.platformio.org/en/latest/core/installation/index.html) or use the VS Code extension.
- **Python 3**: Ensure Python 3 is installed.
- **Python Libraries**: Install the serial and image processing libraries:
  ```bash
  pip install pyserial Pillow
  ```
- **ESP32 Board**: Connect your development board to your computer via USB.
- **Dataset**: Ensure you have downloaded the Hotdog Not Hotdog dataset inside `examples/classification_example/` as described in its README.

### 2. Flash and Run the Project

#### On macOS / Linux:
Use the provided automation script:
```bash
cd examples/interactive_classification_pio
./upload.sh
```
This script adds default PlatformIO Core installation folders to the PATH, compiles the code, uploads the firmware to your board, and starts the Python bridge monitor.

#### On Windows:
Open Command Prompt or PowerShell and run the steps manually:
1. Navigate to the folder:
   ```cmd
   cd examples\interactive_classification_pio
   ```
2. Build and upload the firmware:
   ```cmd
   pio run --target upload
   ```
3. Start the Python serial bridge:
   ```cmd
   python classification_bridge.py
   ```

### 3. Using the Interactive Console
Once the Python bridge runs, it will scan for connected serial ports. Select the index of the port connected to your ESP32. After connecting, the ESP32 will display the command menu:

- `train 100`  - Request 100 random training images, run training on the ESP32, and print the loss.
- `test 50`    - Test accuracy on 50 random test samples.
- `show 10`    - Fetch test image #10, print a 32x32 ASCII representation to the terminal, make a prediction, and show the confidence score.
- `save`       - Save the current model weights to the ESP32's LittleFS partition.
- `load`       - Restore previously saved weights from LittleFS.
- `info`       - Output network layer dimensions and monitor ESP32 free heap memory.
- `reset`      - Re-randomize the network weights.
- `exit_bridge` - Exit the Python script and close the serial monitor.
