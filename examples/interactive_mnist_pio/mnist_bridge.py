#!/usr/bin/env python3
"""
MNIST Serial Bridge for ESP32 C NeuralNet
Feeds MNIST dataset samples from the computer to the ESP32 over serial.
Acts as a bidirectional serial monitor.
"""

import os
import sys
import time
import struct
import gzip
import random
import threading
import urllib.request

# Check for pyserial
try:
    import serial
    import serial.tools.list_ports
except ImportError:
    print("Error: 'pyserial' library is required.")
    print("Please install it by running: pip install pyserial")
    sys.exit(1)

# MNIST dataset file names
MNIST_FILES = {
    "train_img": "train-images-idx3-ubyte.gz",
    "train_lbl": "train-labels-idx1-ubyte.gz",
    "test_img": "t10k-images-idx3-ubyte.gz",
    "test_lbl": "t10k-labels-idx1-ubyte.gz"
}

# Reliable mirrors to download MNIST dataset
MIRRORS = [
    "https://storage.googleapis.com/cvdf-datasets/mnist/",
    "https://ossci-datasets.s3.amazonaws.com/mnist/",
    "https://github.com/mkolod/MNIST/raw/master/"
]

DATA_DIR = "mnist_data"

def download_mnist():
    """Downloads the MNIST dataset files if not already present."""
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
        
    for key, filename in MNIST_FILES.items():
        filepath = os.path.join(DATA_DIR, filename)
        if os.path.exists(filepath):
            continue
            
        print(f"File '{filename}' not found. Downloading...")
        downloaded = False
        for mirror in MIRRORS:
            url = mirror + filename
            print(f"Trying mirror: {url}")
            try:
                # 20 second timeout
                urllib.request.urlretrieve(url, filepath)
                print(f"Successfully downloaded {filename}")
                downloaded = True
                break
            except Exception as e:
                print(f"Failed to download from {mirror}: {e}")
                
        if not downloaded:
            print(f"\nCRITICAL ERROR: Could not download {filename} from any mirror.")
            print(f"Please manually download this file and place it in the '{DATA_DIR}' folder.")
            sys.exit(1)

class MNISTDataset:
    """Helper to parse and hold MNIST images and labels from gzip files."""
    def __init__(self, img_filename, lbl_filename):
        self.img_path = os.path.join(DATA_DIR, img_filename)
        self.lbl_path = os.path.join(DATA_DIR, lbl_filename)
        self.images = []
        self.labels = []
        self.load()

    def load(self):
        print(f"Loading {os.path.basename(self.img_path)}...")
        try:
            with gzip.open(self.img_path, 'rb') as f:
                magic, num, rows, cols = struct.unpack(">IIII", f.read(16))
                if magic != 2051:
                    raise ValueError(f"Invalid magic number in images: {magic}")
                raw_data = f.read()
                # Slice into 784-byte chunks
                self.images = [raw_data[i * 784 : (i + 1) * 784] for i in range(num)]
        except Exception as e:
            print(f"Error reading images {self.img_path}: {e}")
            sys.exit(1)

        print(f"Loading {os.path.basename(self.lbl_path)}...")
        try:
            with gzip.open(self.lbl_path, 'rb') as f:
                magic, num = struct.unpack(">II", f.read(8))
                if magic != 2049:
                    raise ValueError(f"Invalid magic number in labels: {magic}")
                self.labels = list(f.read())
        except Exception as e:
            print(f"Error reading labels {self.lbl_path}: {e}")
            sys.exit(1)

        if len(self.images) != len(self.labels):
            print(f"Error: Number of images ({len(self.images)}) does not match labels ({len(self.labels)}).")
            sys.exit(1)
            
        print(f"Successfully loaded {len(self.images)} samples.")

def select_serial_port():
    """Lists available serial ports and prompts the user to select one."""
    ports = list(serial.tools.list_ports.comports())
    if not ports:
        print("No serial ports found. Please connect your ESP32.")
        return None
        
    print("\n--- Available Serial Ports ---")
    for idx, port in enumerate(ports):
        print(f" [{idx}] {port.device} - {port.description}")
        
    while True:
        try:
            choice = input("\nSelect port index (or type path directly): ").strip()
            if not choice:
                continue
            if choice.isdigit():
                idx = int(choice)
                if 0 <= idx < len(ports):
                    return ports[idx].device
                else:
                    print("Index out of range.")
            else:
                return choice
        except (KeyboardInterrupt, SystemExit):
            print("\nExiting.")
            sys.exit(0)

def main():
    print("====================================================")
    print("      MNIST Serial Bridge for ESP32 C NeuralNet     ")
    print("====================================================")
    
    # 1. Download and load dataset
    download_mnist()
    train_dataset = MNISTDataset(MNIST_FILES["train_img"], MNIST_FILES["train_lbl"])
    test_dataset = MNISTDataset(MNIST_FILES["test_img"], MNIST_FILES["test_lbl"])
    
    # 2. Setup Serial port
    port = None
    if len(sys.argv) > 1:
        port = sys.argv[1]
    else:
        port = select_serial_port()
        
    if not port:
        print("No serial port selected. Exiting.")
        sys.exit(1)
        
    print(f"\nConnecting to ESP32 on {port} at 115200 baud...")
    try:
        ser = serial.Serial(port, 115200, timeout=1)
        # Flush serial buffers
        ser.reset_input_buffer()
        ser.reset_output_buffer()
    except Exception as e:
        print(f"Failed to open serial port {port}: {e}")
        sys.exit(1)
        
    print("Connected! Rebooting ESP32 (toggling DTR/RTS)...")
    # Toggle DTR/RTS to reset the board
    ser.setDTR(False)
    ser.setRTS(False)
    time.sleep(0.1)
    ser.setDTR(True)
    ser.setRTS(True)
    time.sleep(1.0)
    
    print("\nReady! Bidirectional communication established.")
    print("Type commands (like 'train 500', 'test 100', 'show 42', 'info') below.")
    print("----------------------------------------------------\n")
    
    running = True
    
    # 3. Serial command handler
    def handle_command(cmd_line):
        """Processes request commands from ESP32 and replies with MNIST data."""
        parts = cmd_line.strip().split(" ")
        cmd_type = parts[0]
        
        if cmd_type == "CMD:TRAIN_REQ":
            # Pick a random training sample
            idx = random.randint(0, len(train_dataset.images) - 1)
            img = train_dataset.images[idx]
            lbl = train_dataset.labels[idx]
            hex_data = img.hex()
            response = f"RESP:TRAIN:{lbl}:{hex_data}\n"
            ser.write(response.encode('utf-8'))
            
        elif cmd_type == "CMD:TEST_REQ":
            if len(parts) < 2:
                print("[Bridge Error] TEST_REQ missing index argument.")
                return
            try:
                idx = int(parts[1])
                if 0 <= idx < len(test_dataset.images):
                    img = test_dataset.images[idx]
                    lbl = test_dataset.labels[idx]
                    hex_data = img.hex()
                    response = f"RESP:TEST:{idx}:{lbl}:{hex_data}\n"
                    ser.write(response.encode('utf-8'))
                else:
                    print(f"[Bridge Error] Index {idx} out of range [0, 9999].")
            except ValueError:
                print(f"[Bridge Error] Invalid index '{parts[1]}'.")
                
        elif cmd_type == "CMD:TEST_RAND":
            idx = random.randint(0, len(test_dataset.images) - 1)
            img = test_dataset.images[idx]
            lbl = test_dataset.labels[idx]
            hex_data = img.hex()
            response = f"RESP:TEST:{idx}:{lbl}:{hex_data}\n"
            ser.write(response.encode('utf-8'))
            
        else:
            print(f"[Bridge Warning] Unknown request command: {cmd_line}")

    # 4. Background thread to read from ESP32 serial
    def read_from_serial():
        nonlocal running
        while running:
            try:
                if ser.in_waiting > 0:
                    line = ser.readline()
                    if not line:
                        continue
                    try:
                        decoded = line.decode('utf-8', errors='ignore').rstrip('\r\n')
                    except Exception:
                        continue
                        
                    if decoded.startswith("CMD:"):
                        handle_command(decoded)
                    else:
                        print(decoded)
                else:
                    time.sleep(0.001)
            except Exception as e:
                print(f"\nSerial read thread error: {e}")
                running = False
                break

    serial_thread = threading.Thread(target=read_from_serial, daemon=True)
    serial_thread.start()

    # 5. Main thread accepts keyboard input and sends to ESP32
    try:
        while running:
            user_input = sys.stdin.readline()
            if not user_input:
                break
            
            cleaned = user_input.strip()
            if cleaned == "exit_bridge":
                print("Exiting bridge.")
                break
                
            try:
                ser.write(user_input.encode('utf-8'))
            except Exception as e:
                print(f"Failed to write to serial: {e}")
                break
    except KeyboardInterrupt:
        print("\nDisconnecting and exiting.")
    finally:
        running = False
        ser.close()

if __name__ == "__main__":
    main()
