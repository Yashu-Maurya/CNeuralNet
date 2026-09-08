# CNeuralNet

**Version 1.0.0**

A lightweight, dependency-free neural network library written in pure C. Designed for both desktop environments and resource-constrained embedded systems (like the ESP32), CNeuralNet provides matrix arithmetic operations, polymorphic layer abstractions including 2D convolution, high-level network training APIs, image preprocessing utilities, and binary model serialization.

## Features

- **Matrix Operations**: Core linear algebra operations including matrix multiplication, transpositions, addition, subtraction, scalar modifications, and activation mapping.
- **Polymorphic Layer Abstraction**: Easily construct feed-forward architectures using Dense (fully connected), Conv2D, Sigmoid, and ReLU layers with forward/backward propagation.
- **High-Level Network API**: Train networks using forward/backward passes and perform inference with simple, high-level functions.
- **Image Preprocessing**: Built-in support for reading common image formats (JPEG, PNG, etc.) via a single-header dependency (`stb_image.h`), converting pixels to grayscale, scaling/downsizing, and extracting directory listings.
- **Model Serialization**: Save trained model structures and parameters to disk in a compact, custom binary format, and load them back for instant inference. Supports both standard filesystem I/O and Arduino/LittleFS.
- **Embedded Ready**: Designed to compile out-of-the-box on standard systems (via CMake) and on embedded microcontrollers (via PlatformIO/Arduino frameworks).
- **Memory Safe**: Engineered with strict allocation verification and standard deallocation routines to avoid memory leaks.

---

## Project Structure

```text
CNeuralNet/
├── include/
│   ├── matrix.h          # Matrix data structures & algebraic operations
│   ├── layer.h           # Polymorphic Layer interface (Dense, Conv2D, Sigmoid, ReLU)
│   ├── network.h         # High-level Network builder & training runtime
│   ├── math_functions.h  # Math utilities (sigmoid, relu)
│   ├── image.h           # Image loading, resizing, & gray-scaling
│   └── stb_image.h       # Single-header image reader (STB library)
├── src/
│   ├── matrix.c
│   ├── layer.c
│   ├── network.c
│   ├── math_functions.c
│   ├── image.c
│   ├── cnn_platform.c
│   └── network_storage.cpp  # Arduino/LittleFS serialization helpers
└── examples/
    ├── simple_net.c                    # Basic regression without high-level abstraction
    ├── layers_example.c                # Manual training using only the Layer API
    ├── network_example.c               # Training & testing using the high-level Network API
    ├── conv2d_example.c                # Conv2D forward/backward with custom kernels
    ├── mnist/                          # Loading & training on MNIST CSV datasets
    ├── classification_example/         # Image classification (e.g., Hotdog / Not Hotdog)
    ├── save_weights/                   # Save and load network model to/from binary files
    ├── interactive_mnist_pio/          # ESP32-compatible interactive MNIST app
    ├── interactive_classification_pio/ # ESP32-compatible interactive Hotdog classifier
    └── interactive_cam_pio/            # ESP32-CAM live camera inference
```

---

## Building & Running

### Desktop (CMake)

To build the library and example executables on desktop environments (Linux, macOS, Windows):

```bash
mkdir build && cd build
cmake ..
make
```

This compiles a static library `libc_neural_net_lib.a` and produces example executables under the `build` directory.

#### Available Targets

| Executable | Source | Description |
|---|---|---|
| `simple_net_example` | `examples/simple_net.c` | Basic regression with manual gradient descent |
| `layers_example` | `examples/layers_example.c` | Training using only the Layer API |
| `network_example` | `examples/network_example.c` | Training using the high-level Network API |
| `conv2d_example` | `examples/conv2d_example.c` | Conv2D forward/backward with custom kernels |
| `mnist_example` | `examples/mnist/mnist_example.c` | MNIST digit classification |
| `classification_example` | `examples/classification_example/classification_example.c` | Hotdog / Not-Hotdog image classification |
| `save_and_load_weights_example` | `examples/save_weights/save_and_load_weights.c` | Model serialization demo |

### Embedded Systems (PlatformIO / ESP32)

CNeuralNet compiles and runs directly on embedded microcontrollers like the ESP32. We provide three interactive examples:

1. **Interactive MNIST** (`examples/interactive_mnist_pio/`): 28x28 handwritten digit recognition.
2. **Interactive Hotdog Classifier** (`examples/interactive_classification_pio/`): 32x32 binary image classification.
3. **ESP32-CAM Live Filter** (`examples/interactive_cam_pio/`): Real-time Conv2D (Sobel edge detection) on live camera feed with HTTP streaming.

#### 1. Running the Interactive Examples

**Prerequisites:**
- [PlatformIO Core CLI](https://docs.platformio.org/en/latest/core/installation/index.html) (or PlatformIO IDE extension in VS Code)
- Python 3 with dependencies installed:
  ```bash
  pip install pyserial Pillow
  ```
- An ESP32 development board connected via USB.

**Step-by-Step Instructions:**
1. Connect your ESP32 board to your computer.
2. Navigate to the interactive example directory:
   - For MNIST: `cd examples/interactive_mnist_pio`
   - For Hotdog: `cd examples/interactive_classification_pio`
   - For ESP32-CAM: `cd examples/interactive_cam_pio`
3. Compile, upload the firmware, and launch the bridge (MNIST/Hotdog only) using:
   ```bash
   ./upload.sh
   ```
   *(Alternatively, run `pio run --target upload` to flash the board). For interactive training examples, follow with `python3 mnist_bridge.py` or `python3 classification_bridge.py` to start the bridge manually.*
4. For MNIST/Hotdog: Select the serial port index when prompted by the Python script, then enter commands:
   - `train <num>` - Request `<num>` random images, run training on the ESP32, and print the loss.
   - `test <num>`  - Request `<num>` random test samples and evaluate accuracy.
   - `show <index>` - Fetch test image at `<index>`, render ASCII art, and predict.
   - `save`        - Save weights to the ESP32's internal flash (LittleFS).
   - `load`        - Reload previously saved weights from flash.
   - `info`        - Print layer dimensions and free heap memory.
   - `reset`       - Re-initialize and randomize the weights.
   - `exit_bridge` - Exit the Python bridge script.
5. For ESP32-CAM (`interactive_cam_pio`): Open a browser to the ESP32's IP address (printed in the Serial Monitor) to view the live filtered camera stream.

---

#### 2. Using CNeuralNet in Your Own PlatformIO Projects

To include the library in your custom microcontroller project:

##### A. Configure `platformio.ini`
Include the library directory under `lib_deps`. You can use a relative symlink if the library is stored locally, or reference it via git:
```ini
[env:esp32dev]
platform = espressif32
board = esp32dev
framework = arduino
monitor_speed = 115200

# Reference the library locally (using relative path to library root)
lib_deps =
    symlink://../CNeuralNet
```

##### B. C++ Compatibility wrapper
Since PlatformIO primarily compiles C++ (`.cpp`) files and CNeuralNet is pure C, wrap your headers in `extern "C"` when writing your C++ application code:
```cpp
#include <Arduino.h>

extern "C" {
#include <network.h>
#include <layer.h>
#include <matrix.h>
}
```

##### C. Setup Custom Logging
Redirect internal library print logs to the Arduino Serial Monitor using a logging callback:
```cpp
void myLogger(const char *message) {
  Serial.print(message);
}

void setup() {
  Serial.begin(115200);
  
  // Set the logger callback
  cnn_set_log_callback(myLogger);
}
```

##### D. Save/Load Weights on Flash Memory
When compiling on Arduino-supported platforms, the library exposes filesystem helper functions (`save_network_fs` and `load_network_fs`) allowing you to save models directly onto local filesystems such as `LittleFS` or `SPIFFS`:
```cpp
#include <LittleFS.h>

// Save model to flash
save_network_fs(my_network, LittleFS, "/my_model.bin");

// Load model from flash
load_network_fs(my_network, LittleFS, "/my_model.bin");
```

---

## API Reference

### 1. Math Functions (`math_functions.h`)

```c
float sigmoid(float x);
float relu(float x);
```

Scalar activation functions used internally by layers and matrix operations.

---

### 2. Matrix API (`matrix.h`)

Matrices are represented as a flat dynamic array of floats with dimension properties:

```c
typedef struct {
  int rows;
  int columns;
  float *data;
} Matrix;
```

#### Construction & Destruction:
* `Matrix* create_matrix(int rows, int columns)`: Allocates a new rows × columns matrix. Returns `NULL` if allocation fails. **(Caller must free)**
* `void free_matrix(Matrix* m)`: Safely releases the matrix and its data.

#### Basic Operations:
* `Matrix* copy_matrix(Matrix* m)`: Returns a duplicate of the matrix. **(Caller must free)**
* `void randomize_matrix(Matrix* m)`: Populates the matrix with random floats in the range `[0, 1]`.
* `void zero_matrix(Matrix* m)`: Fills the matrix with `0.0f`.
* `void print_matrix(Matrix* m)`: Formats and prints the matrix values to standard output.
* `size_t get_matrix_size(Matrix* m)`: Returns the total number of elements (`rows * columns`).

#### Arithmetic:
* `Matrix* multiply_mat(Matrix* m1, Matrix* m2)`: Multiplies `m1` × `m2`. Returns a new matrix. **(Caller must free)**
* `Matrix* transpose_mat(Matrix* m)`: Computes the transpose. Returns a new matrix. **(Caller must free)**
* `Matrix* subtract_matrix(Matrix* m1, Matrix* m2)`: Subtracts `m2` from `m1`. Returns a new matrix. **(Caller must free)**
* `void add_matrix(Matrix* m1, Matrix* m2)`: Performs in-place element-wise addition (`m1 = m1 + m2`).
* `void add_scaler(Matrix* m, float scaler)`: Adds a scalar value to all elements in-place.
* `void subtract_scaler(Matrix* m, float scaler)`: Subtracts a scalar value from all elements in-place.
* `void scale_matrix(Matrix* m, float scaler)`: Multiplies all elements by a scalar in-place.

#### Activation:
* `void matrix_sigmoid(Matrix* m)`: Applies the sigmoid activation function to all elements in-place.
* `int argmax(Matrix* m)`: Returns the index of the highest value in a column vector (useful for classification).

#### Serialization:
* `int save_matrix(Matrix* m, FILE* fp)`: Writes matrix dimensions and data to a file stream. Returns `0` on success, `-1` on failure.
* `int load_matrix(Matrix* m, FILE* fp)`: Reads matrix dimensions and data from a file stream, validating shape against the pre-allocated matrix. Returns `0` on success, `-1` on failure.

---

### 3. Layer API (`layer.h`)

Construct layers using custom behaviors (Dense, Conv2D, Sigmoid, ReLU).

```c
typedef Matrix* (*ForwardFunction)(struct Layer *l, Matrix *input);
typedef Matrix* (*BackwardFunction)(struct Layer *l, Matrix *error_gradient, float learning);

struct Layer {
  ForwardFunction forward;
  BackwardFunction backward;
  Matrix *inputs;
  Matrix *weights;
  Matrix *bias;
  Matrix *output;
  Matrix *d_weight;
  Matrix *d_bias;
  int input_n;
  int output_n;
  int input_rows;
  int input_columns;
  int output_rows;
  int output_columns;
  const char *name; // "Dense", "Conv2d", "Sigmoid", "ReLU"
};
```

#### Layer Construction:
* `Layer* layer_create_dense(int input_n, int output_n)`: Creates a fully connected layer with weights initialized using Xavier/Glorot scaling.
* `Layer* layer_create_sigmoid()`: Creates a Sigmoid activation layer.
* `Layer* layer_create_relu()`: Creates a Rectified Linear Unit (ReLU) activation layer.
* `Layer* layer_create_conv2d(int input_size, int kernel_size)`: Creates a 2D convolution layer with a square input and square kernel (stride=1, no padding). Weights are randomly initialized.
* `Layer* layer_create_conv2d_shape(int input_rows, int input_columns, int kernel_rows, int kernel_columns)`: Creates a 2D convolution layer with arbitrary rectangular input and kernel dimensions.
* `void free_layer(Layer* layer)`: Safely deallocates a layer, including weights, biases, and gradients.

#### Run Functions:
* `Matrix* layer_forward(Layer* l, Matrix* input)`: Performs a forward pass through the layer. Returns output matrix. **(Caller must free)**
* `Matrix* layer_backward(Layer* l, Matrix* error_gradient, float learning_rate)`: Computes backpropagation gradients, updates weights/biases in-place, and returns the gradient of the loss with respect to the layer's input. **(Caller must free)**

#### Conv2D-Specific Functions:
* `Matrix* _layer_forward_conv2d_with_kernel(Layer* l, Matrix* input, Matrix* kernel)`: Performs a forward pass using a custom kernel without modifying the layer's own weights. Returns output matrix. **(Caller must free)**
* `int layer_forward_conv2d_into(Layer* l, Matrix* input, Matrix* output)`: Performs a forward pass and writes directly into a pre-allocated output matrix. Returns `0` on success, `-1` on failure. Useful for zero-allocation streaming pipelines.
* `int layer_forward_conv2d_with_kernel_into(Layer* l, Matrix* input, Matrix* kernel, Matrix* output)`: Same as above but uses a custom kernel.

#### Utility:
* `void print_layer_info(Layer* l)`: Prints layer type, name, input/output dimensions, and weight/bias shapes to standard output.

#### Serialization:
* `int save_layer(Layer* l, FILE* fp)`: Writes a single layer (type, dimensions, weights, biases) to a file stream. Returns `0` on success.
* `Layer* load_layer(FILE* fp)`: Reads and reconstructs a layer from a file stream. Returns the layer pointer on success, `NULL` on failure. **(Caller must free via free_layer or add_layer)**

---

### 4. Network API (`network.h`)

Manages sequential layers as a cohesive neural network model.

```c
struct Network {
  Layer **layers;
  int layer_count;
};
```

#### Key Functions:
* `Network* create_network()`: Instantiates an empty neural network.
* `void add_layer(Network* n, Layer* l)`: Appends a layer to the network. **(The network takes ownership of the layer's lifetime)**
* `void free_network(Network* n)`: Releases the network and all registered layers.
* `Matrix* predict_network(Network* n, Matrix* input)`: Feeds input through all layers sequentially. Returns prediction matrix. **(Caller must free)**
* `void train_network(Network* n, Matrix* inputs, Matrix* targets, float learning_rate)`: Executes forward and backward propagation through the entire network, adjusting weights.
* `void print_network_info(Network* n)`: Logs the network configuration and architecture to standard output.

#### Serialization:
* `int save_network(Network* n, const char* filename)`: Saves the entire network (layers, weights, biases) to a binary file. Returns `0` on success.
* `void load_network(Network* n, const char* filename)`: Loads and reconstructs a network from a binary file. The network must be empty (just created via `create_network`).

---

### 5. Image Preprocessing API (`image.h`)

Allows reading, resizing, and converting images to matrix inputs.

```c
typedef struct {
  float *data;
  char *type;
  char *name;
  int width;
  int height;
  int channel;
} Image;
```

#### Key Functions:
* `Image* read_image(const char* path)`: Loads JPEG/PNG image data from disk. Returns `NULL` if failed. **(Caller must free)**
* `void free_image(Image* img)`: Frees the image structural allocations.
* `void print_image_info(Image* img)`: Prints image dimensions, type, and name to standard output.
* `Matrix* image_to_matrix(Image* img, int target_size)`: Down-samples the image to a `target_size × target_size` matrix and normalizes pixel values to the range `[0, 1]`. Converts RGB inputs to grayscale automatically. **(Caller must free)**
* `int list_image_paths(const char* dir_path, char*** paths_out)`: Scans a directory and compiles a list of absolute file paths to JPEG images. Returns the total count found. **(Caller must free the array and path strings)**

---

### 6. Logging & Platform API (`cnn_platform.h`)

CNeuralNet exposes a logging abstraction layer that can be redirected for embedded environments:

```c
typedef void (*CnnLogCallback)(const char *message);

void cnn_set_log_callback(CnnLogCallback callback);
void cnn_log(const char *message);
void cnn_logf(const char *format, ...);
void cnn_vlogf(const char *format, va_list args);
void *cnn_alloc(size_t size);
void cnn_free(void *ptr);
```

By default, logs are printed to `stdout`. Call `cnn_set_log_callback` to redirect output (e.g., to Arduino `Serial`). All internal memory allocation goes through `cnn_alloc`/`cnn_free`.

---

## Model Serialization

CNeuralNet features binary serialization to easily save and load trained models without losing learned weights and biases.

```c
// Standard file I/O
int save_network(Network* n, const char* filename);
void load_network(Network* n, const char* filename);

// Per-layer serialization (for custom workflows)
int save_layer(Layer* l, FILE* fp);
Layer* load_layer(FILE* fp);

// Arduino/LittleFS (see embedded section above)
int save_network_fs(Network* n, fs::FS &filesystem, const char* path);
void load_network_fs(Network* n, fs::FS &filesystem, const char* path);
```

### Binary Serialization Format
The library serializes data into a lightweight, custom binary format structured as follows:

1. **Network Header**:
   - `int layer_count`: Number of layers in the network.
2. **Layer Data Blocks** (Repeated `layer_count` times sequentially):
   - `int type`: The layer identifier (`0` for Dense, `1` for Sigmoid, `2` for ReLU, `3` for Conv2d).
   - *If `type == 0` (Dense Layer)*:
     - `int input_n`: Dimension of input vector.
     - `int output_n`: Dimension of output vector.
     - **Weights Matrix**:
       - `int rows`, `int columns` (Shape of weight matrix)
       - `float data[]` (Raw weight data array of size `rows * columns`)
     - **Biases Matrix**:
       - `int rows`, `int columns` (Shape of bias matrix)
       - `float data[]` (Raw bias data array of size `rows * columns`)
   - *If `type == 3` (Conv2d Layer)*:
     - `int input_rows`: Height of input feature map.
     - `int input_columns`: Width of input feature map.
     - `int output_rows`: Height of output feature map.
     - `int output_columns`: Width of output feature map.
     - **Weights Matrix** (kernel): Shape and data.
     - **Biases Matrix** (bias): Shape and data.
   - *If `type == 1` (Sigmoid) or `type == 2` (ReLU)*:
     - No additional parameters are stored, as these activation layers do not contain learnable parameters.

---

## Memory Ownership Guidelines

As a pure C library, memory management is deterministic but requires adherence to simple guidelines:

1. **Returned Matrices**: Functions like `create_matrix`, `copy_matrix`, `multiply_mat`, `transpose_mat`, `subtract_matrix`, `predict_network`, `layer_forward`, and `layer_backward` return freshly allocated matrices. The caller is responsible for calling `free_matrix()` on these pointers when they are no longer needed.
2. **Layer Ownership**: Once a `Layer` is added to a `Network` via `add_layer(network, layer)`, the `Network` takes ownership. You do not need to call `free_layer()` on the individual layers; calling `free_network(network)` will safely release all of them.
3. **Image Lifecycle**: Call `free_image()` on the `Image` struct pointer returned by `read_image()`. When you convert an image via `image_to_matrix()`, the returned matrix is owned by the caller and must be freed using `free_matrix()`.

---

## Code Examples

### 1. Simple Network API Training (Regression)

This example constructs a minimal network to learn a simple function: `y = 2x`.

```c
#include "network.h"
#include <stdio.h>

#define LEARNING_RATE 0.1f
#define EPOCHS 1000

int main() {
    // 1. Initialize Network and Layers
    Network* network = create_network();
    Layer* dense1 = layer_create_dense(1, 1);
    add_layer(network, dense1);

    // 2. Prepare single-element inputs and targets
    Matrix* inputs = create_matrix(1, 1);
    Matrix* targets = create_matrix(1, 1);

    // 3. Train network (learns to map x -> 2x)
    for (int i = 0; i < EPOCHS; i++) {
        for (int j = 0; j < 10; j++) {
            inputs->data[0] = j / 20.0f;
            targets->data[0] = (j * 2.0f) / 20.0f;
            train_network(network, inputs, targets, LEARNING_RATE);
        }
    }

    // 4. Test Inference
    float test_val = 123.0f;
    inputs->data[0] = test_val / 20.0f;
    Matrix* output = predict_network(network, inputs);
    float prediction = output->data[0] * 20.0f;
    printf("Input: %.2f | Predicted Output: %.2f (Expected: %.2f)\n", test_val, prediction, test_val * 2.0f);

    // 5. Cleanup
    free_matrix(output);
    free_matrix(inputs);
    free_matrix(targets);
    free_network(network); // Frees dense1 automatically

    return 0;
}
```

### 2. MNIST Digit Classification

Check the code in `examples/mnist/mnist_example.c`. It loads CSV datasets containing digit pixels, trains a sequential classifier (`784` input -> `128` Hidden with ReLU -> `10` Output with Sigmoid), and uses the `argmax` function to calculate predictions.

### 3. Binary Image Classification (Hotdog / Not Hotdog)

Located in `examples/classification_example/classification_example.c`, this example demonstrates scanning directories for JPG images, loading and normalizing them to `32x32` grayscale matrices using the image preprocessing utilities, and training a binary classifier.

```c
// Example snippet loading and classifying an image
Image* img = read_image("sample.jpg");
if (img != NULL) {
    Matrix* input = image_to_matrix(img, 32); // Convert to 32x32 normalized grayscale matrix
    free_image(img);

    Matrix* prediction = predict_network(network, input);
    printf("Probability: %.4f\n", prediction->data[0]);

    free_matrix(prediction);
    free_matrix(input);
}
```

### 4. Conv2D Layer (Sobel Edge Detection)

Located in `examples/conv2d_example.c`, this example demonstrates creating a Conv2D layer, running forward passes with built-in and custom kernels (Sobel edge detector, averaging blur), and performing a backward pass.

```c
#include "layer.h"
#include "matrix.h"

// Create a 5x5 input with a 3x3 convolution kernel
Layer *conv = layer_create_conv2d(5, 3);

// Forward pass with layer's own random kernel
Matrix *output = layer_forward(conv, input);

// Forward pass with a custom Sobel kernel
Matrix *sobel = create_matrix(3, 3);
float sobel_data[] = {-1, 0, 1, -2, 0, 2, -1, 0, 1};
for (int i = 0; i < 9; i++) sobel->data[i] = sobel_data[i];
Matrix *edges = _layer_forward_conv2d_with_kernel(conv, input, sobel);

// Cleanup
free_matrix(output);
free_matrix(edges);
free_matrix(sobel);
free_layer(conv);
```

### 5. Model Save & Load

Located in `examples/save_weights/save_and_load_weights.c`, this example demonstrates creating a network, saving it to a binary file, and loading it back.

```c
Network *n = create_network();
add_layer(n, layer_create_dense(10, 20));
add_layer(n, layer_create_dense(20, 10));
add_layer(n, layer_create_sigmoid());

save_network(n, "model.cnet");
free_network(n);

Network *n2 = create_network();
load_network(n2, "model.cnet");
print_network_info(n2);
free_network(n2);
```
