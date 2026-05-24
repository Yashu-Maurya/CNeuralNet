# Interactive MNIST Digit Classifier (PlatformIO / ESP32)

This example runs a 3-layer neural network directly on the ESP32 microcontroller to classify MNIST handwritten digits (0–9). Training and testing data are streamed dynamically from your computer to the board over a Serial connection using a Python bridge script.

## System Architecture
- **Microcontroller (ESP32)**: Holds the network weights, runs forward and backward propagation, evaluates accuracy, and saves/loads weights to flash memory using LittleFS.
- **Computer (Python Bridge)**: Downloads the MNIST dataset, acts as a serial console, and feeds images to the ESP32 on-demand as hex-encoded pixel streams.

---

## Step-by-Step Run Guide

### 1. Prerequisites
- **PlatformIO CLI**: Install [PlatformIO Core](https://docs.platformio.org/en/latest/core/installation/index.html) or use the VS Code extension.
- **Python 3**: Ensure Python 3 is installed.
- **Python Serial Library**: Install the required package:
  ```bash
  pip install pyserial
  ```
- **ESP32 Board**: Connect your development board to your computer via USB.

### 2. Flash and Run the Project

#### On macOS / Linux:
Use the provided automation script:
```bash
cd examples/interactive_mnist_pio
./upload.sh
```
This script adds default PlatformIO Core installation folders to the PATH, compiles the code, uploads the firmware to your board, and starts the Python bridge monitor.

#### On Windows:
Open Command Prompt or PowerShell and run the steps manually:
1. Navigate to the folder:
   ```cmd
   cd examples\interactive_mnist_pio
   ```
2. Build and upload the firmware:
   ```cmd
   pio run --target upload
   ```
3. Start the Python serial bridge:
   ```cmd
   python mnist_bridge.py
   ```

### 3. Using the Interactive Console
Once the Python bridge runs, it will scan for connected serial ports. Select the index of the port connected to your ESP32. After connecting, the ESP32 will display the command menu:

- `train 1000` - Train the network on 1,000 random MNIST samples.
- `test 100`  - Test accuracy on 100 random test samples.
- `show 42`   - Fetch test image #42, print a 28x28 ASCII representation to the terminal, make a prediction, and show the confidence score.
- `save`      - Save the current model weights to the ESP32's LittleFS partition.
- `load`      - Restore previously saved weights from LittleFS.
- `info`      - Output network layer dimensions and monitor ESP32 free heap memory.
- `reset`     - Re-randomize the network weights.
- `exit_bridge` - Exit the Python script and close the serial monitor.
