# CNeuralNet

A lightweight neural network library written in pure C. Provides matrix operations and layer abstractions for building and training simple neural networks from scratch.

## Features

- **Matrix Operations** - Create, manipulate, and perform math on matrices
- **Polymorphic Layers** - Dense (fully connected) and Sigmoid activation layers with forward/backward pass
- **Memory Safe** - Proper allocation checks, cleanup functions, and no memory leaks
- **No Dependencies** - Pure C with only standard library

## Project Structure

```
CNeuralNet/
├── include/
│   ├── matrix.h         # Matrix struct and operations
│   ├── layer.h          # Layer struct and layer types (Dense, Sigmoid)
│   ├── network.h        # Network struct for managing multiple layers
│   └── math_functions.h # Activation functions (sigmoid)
├── src/
│   ├── matrix.c
│   ├── layer.c
│   ├── network.c
│   └── math_functions.c
└── examples/
    ├── simple_net.c     # Manual neural net implementation
    ├── layers_example.c # Using Layer abstraction
    └── network_example.c # Using Network API
    └── mnist_example.c # Using Network API for MNIST Dataset
```

## Building

```bash
mkdir build && cd build
cmake ..
make
```

## API Reference

### Matrix

```c
// Create a rows x columns matrix (caller must free)
Matrix* create_matrix(int rows, int columns);

// Free matrix and its data
void free_matrix(Matrix* m);

// Copy a matrix (returns new matrix, caller must free)
Matrix* copy_matrix(Matrix* m);

// Fill matrix with random values [0, 1]
void randomize_matrix(Matrix* m);

// Set all elements to zero
void zero_matrix(Matrix* m);

// Print matrix to stdout
void print_matrix(Matrix* m);

// Matrix multiplication: result = m1 × m2 (returns new matrix)
Matrix* multiply_mat(Matrix* m1, Matrix* m2);

// Transpose matrix (returns new matrix)
Matrix* transpose_mat(Matrix* m);

// Element-wise addition (modify m1 in-place)
void add_matrix(Matrix* m1, Matrix* m2);

// Element-wise subtraction (returns new matrix, caller must free)
Matrix* subtract_matrix(Matrix* m1, Matrix* m2);

// Scalar operations (modify matrix in-place)
void add_scaler(Matrix* m, float scaler);
void subtract_scaler(Matrix* m, float scaler);
void scale_matrix(Matrix* m, float scaler);  // m[i] *= scaler

// Apply sigmoid activation in-place
void matrix_sigmoid(Matrix* m);

// Find index of maximum value (useful for classification)
int argmax(Matrix* m);
```

### Layer

The library provides a polymorphic layer system using function pointers:

```c
// Create dense layer: input_size → output_size
// Weights shape: (output_size × input_size)
Layer* layer_create_dense(int input_size, int output_size);

// Create sigmoid activation layer
Layer* layer_create_sigmoid();

// Create ReLU activation layer
Layer* layer_create_relu();

// Free layer and all its matrices
void free_layer(Layer* layer);

// Forward pass: compute layer output (returns new matrix, caller must free)
Matrix* layer_forward(Layer* l, Matrix* input);

// Backward pass: compute gradients and update weights (returns new matrix, caller must free)
Matrix* layer_backward(Layer* l, Matrix* error_gradient, float learning_rate);

// Print layer configuration
void print_layer_info(Layer *l);
```

### Network

High-level API for building and training multi-layer networks:

```c
// Create an empty network
Network* create_network();

// Add a layer to the network (network takes ownership)
void add_layer(Network* n, Layer* l);

// Free network and all its layers
void free_network(Network* n);

// Forward pass through all layers (returns new matrix, caller must free)
Matrix* predict_network(Network* n, Matrix* input);

// Train network: forward pass, compute loss gradient, backward pass
void train_network(Network* n, Matrix* inputs, Matrix* targets, float learning_rate);

// Print network architecture and layer details
void print_network_info(Network* n);
```

## Examples

### Simple Regression

Learns to multiply numbers by 2.

```c
#include "network.h"

#define LEARNING_RATE 0.1f
#define EPOCHS 1000

int main() {
    // Create network
    Network* network = create_network();
    Layer* dense1 = layer_create_dense(1, 1);
    add_layer(network, dense1);

    // Training data
    Matrix* inputs = create_matrix(1, 1);
    Matrix* targets = create_matrix(1, 1);

    // Training loop (learns y = 2x)
    for (int i = 0; i < EPOCHS; i++) {
        for (int j = 0; j < 10; j++) {
            inputs->data[0] = j / 20.0f;
            targets->data[0] = (j * 2.0f) / 20.0f;
            train_network(network, inputs, targets, LEARNING_RATE);
        }
    }

    // Inference
    float test_input = 123.0f;
    inputs->data[0] = test_input / 20.0f;
    Matrix* output = predict_network(network, inputs);
    float result = output->data[0] * 20.0f;
    printf("Input: %.2f, Output: %.2f\n", test_input, result);  // ~246.0

    // Cleanup
    free_matrix(output);
    free_network(network);
    free_matrix(inputs);
    free_matrix(targets);

    return 0;
}
```

### MNIST Digit Classification

A complete example of training a network on the MNIST dataset is available in `examples/mnist/`.

**Structure:**

- `mnist_example.c`: Loads CSV data, trains a 784 -> 128 -> 10 network.
- Uses `layer_create_relu()` for hidden layers and `layer_create_sigmoid()` for output.
- Demonstrates `argmax()` for interpreting classification results.

To run the MNIST example:

1. Download `mnist_train.csv` and `mnist_test.csv` (e.g. from Kaggle) into `build/`
2. Build the project
3. Run `./build/mnist_example`

## Memory Ownership

- **Matrix functions**: `create_matrix`, `copy_matrix`, `multiply_mat`, `transpose_mat`, and `subtract_matrix` return new matrices that the **caller must free**
- **Layer functions**: `layer_forward` and `layer_backward` return new matrices that the **caller must free**
- **Network**: When you call `add_layer`, the network takes ownership of the layer. Call `free_network` to free all layers.
- **predict_network**: Returns a new matrix that the **caller must free**

---

_This documentation was analyzed and developed with LLM assistance. Verified by [Yashu-Maurya](https://github.com/Yashu-Maurya)._
