# ESP32-CAM Live Conv2D Filter (PlatformIO)

This example demonstrates real-time 2D convolution (Sobel vertical edge detection) on a live camera feed from an ESP32-CAM (AI-Thinker). The camera captures VGA grayscale frames and applies a Conv2D layer with a vertical Sobel kernel, then streams the filtered result over HTTP.

## Requirements

- ESP32-CAM board (AI-Thinker pinout)
- ESP32 development environment with [PlatformIO](https://docs.platformio.org/)
- PSRAM-enabled board (recommended for VGA buffers)

## How It Works

1. The ESP32-CAM captures 640x480 grayscale frames.
2. A `Conv2d` layer with a 3x3 vertical Sobel kernel is applied using `layer_forward_conv2d_into()`.
3. The filtered grayscale frame is JPEG-encoded and served via an HTTP web server.
4. Open the ESP32's IP address in a browser to see side-by-side raw and filtered video streams.

## Endpoints

- `/` - HTML page with side-by-side raw and filtered streams
- `/capture` - Raw JPEG snapshot
- `/stream` - Raw JPEG stream
- `/filtered` - Sobel-filtered JPEG stream

## Build & Upload

1. Edit `src/main.cpp` and set your WiFi credentials (`ssid`, `password`).
2. Build and upload:
   ```bash
   cd examples/interactive_cam_pio
   pio run --target upload
   ```
3. Open the Serial Monitor to see the assigned IP address, then navigate to it in a browser.
