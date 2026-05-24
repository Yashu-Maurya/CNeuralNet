#!/usr/bin/env python3
"""
Hotdog/Not-Hotdog Serial Bridge for ESP32 C NeuralNet
Feeds training and testing images from the computer to the ESP32 over serial.
"""

import os
import sys
import time
import random
import threading

# Check for PIL (Pillow) and serial dependencies
try:
    from PIL import Image
except ImportError:
    print("Error: 'Pillow' library is required to process images.")
    print("Please install it by running: pip install Pillow")
    sys.exit(1)

try:
    import serial
    import serial.tools.list_ports
except ImportError:
    print("Error: 'pyserial' library is required.")
    print("Please install it by running: pip install pyserial")
    sys.exit(1)

# Dataset configuration
DATA_BASE = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "classification_example",
        "Hotdog Not Hotdog Archive",
        "hotdog-nothotdog",
        "hotdog-nothotdog"
    )
)

IMG_SIZE = 32  # Resize dimensions for the ESP32

def load_image_paths(dir_path):
    """Recursively search for JPG/JPEG images in a directory."""
    paths = []
    if not os.path.exists(dir_path):
        return paths
    for root, _, files in os.walk(dir_path):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                paths.append(os.path.join(root, file))
    return paths

def process_image_to_hex(filepath, target_size=IMG_SIZE):
    """Loads, grayscales, resizes, and encodes an image into a hex string."""
    try:
        with Image.open(filepath) as img:
            # Convert to grayscale
            img_gray = img.convert("L")
            # Resize
            img_resized = img_gray.resize((target_size, target_size), Image.Resampling.LANCZOS)
            # Convert pixels to hex representation
            pixel_bytes = bytearray(img_resized.tobytes())
            return pixel_bytes.hex()
    except Exception as e:
        print(f"[Bridge Error] Failed to process image {os.path.basename(filepath)}: {e}")
        return None

def select_serial_port():
    """Lists available serial ports and prompts selection."""
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
    print("    Hotdog Classifier Serial Bridge for ESP32       ")
    print("====================================================")
    
    # 1. Scan dataset directories
    print(f"Scanning dataset in: {DATA_BASE}")
    train_hd = load_image_paths(os.path.join(DATA_BASE, "train", "hotdog"))
    train_nhd = load_image_paths(os.path.join(DATA_BASE, "train", "nothotdog"))
    test_hd = load_image_paths(os.path.join(DATA_BASE, "test", "hotdog"))
    test_nhd = load_image_paths(os.path.join(DATA_BASE, "test", "nothotdog"))
    
    if not train_hd or not train_nhd:
        print("\nCRITICAL ERROR: No training images found.")
        print(f"Please ensure the dataset is downloaded inside:\n{DATA_BASE}")
        sys.exit(1)
        
    print(f"  Found {len(train_hd)} hotdog and {len(train_nhd)} not-hotdog training images.")
    print(f"  Found {len(test_hd)} hotdog and {len(test_nhd)} not-hotdog testing images.")
    
    # Structure parallel lists for train and test
    train_samples = [(p, 1) for p in train_hd] + [(p, 0) for p in train_nhd]
    test_samples = [(p, 1) for p in test_hd] + [(p, 0) for p in test_nhd]
    
    # Shuffle lists
    random.shuffle(train_samples)
    random.shuffle(test_samples)
    
    # 2. Setup Serial Port
    port = sys.argv[1] if len(sys.argv) > 1 else select_serial_port()
    if not port:
        print("No serial port selected. Exiting.")
        sys.exit(1)
        
    print(f"\nConnecting to ESP32 on {port} at 115200 baud...")
    try:
        ser = serial.Serial(port, 115200, timeout=1)
        ser.reset_input_buffer()
        ser.reset_output_buffer()
    except Exception as e:
        print(f"Failed to open serial port {port}: {e}")
        sys.exit(1)
        
    print("Connected! Rebooting ESP32...")
    ser.setDTR(False)
    ser.setRTS(False)
    time.sleep(0.1)
    ser.setDTR(True)
    ser.setRTS(True)
    time.sleep(1.0)
    
    print("\nReady! Bidirectional communication established.")
    print("Type commands (like 'train 200', 'test 50', 'show 10', 'info') below.")
    print("----------------------------------------------------\n")
    
    running = True
    
    # 3. Serial request command handler
    def handle_command(cmd_line):
        parts = cmd_line.strip().split(" ")
        cmd_type = parts[0]
        
        if cmd_type == "CMD:TRAIN_REQ":
            # Pick a random training sample
            filepath, label = random.choice(train_samples)
            hex_pixels = process_image_to_hex(filepath)
            if hex_pixels:
                response = f"RESP:TRAIN:{label}:{hex_pixels}\n"
                ser.write(response.encode('utf-8'))
                
        elif cmd_type == "CMD:TEST_REQ":
            if len(parts) < 2:
                print("[Bridge Error] TEST_REQ missing index argument.")
                return
            try:
                idx = int(parts[1])
                if 0 <= idx < len(test_samples):
                    filepath, label = test_samples[idx]
                    hex_pixels = process_image_to_hex(filepath)
                    if hex_pixels:
                        response = f"RESP:TEST:{idx}:{label}:{hex_pixels}\n"
                        ser.write(response.encode('utf-8'))
                else:
                    print(f"[Bridge Error] Index {idx} out of range [0, {len(test_samples)-1}].")
            except ValueError:
                print(f"[Bridge Error] Invalid index '{parts[1]}'.")
                
        elif cmd_type == "CMD:TEST_RAND":
            idx = random.randint(0, len(test_samples) - 1)
            filepath, label = test_samples[idx]
            hex_pixels = process_image_to_hex(filepath)
            if hex_pixels:
                response = f"RESP:TEST:{idx}:{label}:{hex_pixels}\n"
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
