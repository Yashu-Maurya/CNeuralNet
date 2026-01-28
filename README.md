# CNeuralNet

A lightweight neural network library written in pure C. Provides matrix operations and layer abstractions for building and training simple neural networks from scratch.

## Features

- **Matrix Operations** - Create, manipulate, and perform math on matrices
- **Neural Network Layers** - Dense (fully connected) layer with forward/backward pass
- **Memory Safe** - Proper allocation checks and cleanup functions
- **No Dependencies** - Pure C with only standard library

## Project Structure

```
CNeuralNet/
├── include/
│   ├── matrix.h        # Matrix struct and operations
│   ├── layer.h         # Dense layer struct and functions
│   └── math_functions.h # Activation functions (sigmoid)
├── src/
│   ├── matrix.c
│   ├── layer.c
│   └── math_functions.c
└── examples/
    ├── simple_net.c    # Manual neural net implementation
    └── layers_example.c # Using Layer abstraction
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

// Element-wise operations (modify m1 in-place)
void add_matrix(Matrix* m1, Matrix* m2);
void subtract_matrix(Matrix* m1, Matrix* m2);

// Scalar operations (modify matrix in-place)
void add_scaler(Matrix* m, float scaler);
void subtract_scaler(Matrix* m, float scaler);
void scale_matrix(Matrix* m, float scaler);  // m[i] *= scaler

// Apply sigmoid activation in-place
void matrix_sigmoid(Matrix* m);
```

### Layer

```c
// Create dense layer: input_size → neurons
Layer* layer_create(int input_size, int neurons);

// Free layer and all its matrices
void free_layer(Layer* layer);

// Forward pass: compute output = weights × input + bias
Matrix* layer_forward(Layer* l, Matrix* input);

// Backward pass: compute gradients and update weights
Matrix* layer_backward(Layer* l, Matrix* error_gradient, float learning_rate);
```

## Usage Example

```c
#include "layer.h"

int main() {
    // Create a 1→1 dense layer (learns y = 2x)
    Layer* dense = layer_create(1, 1);
    
    Matrix* input = create_matrix(1, 1);
    Matrix* loss_grad = create_matrix(1, 1);
    
    // Training loop
    for (int epoch = 0; epoch < 1000; epoch++) {
        for (int i = 0; i < 10; i++) {
            input->data[0] = i / 20.0f;
            float target = (i * 2.0f) / 20.0f;
            
            Matrix* pred = layer_forward(dense, input);
            loss_grad->data[0] = pred->data[0] - target;
            
            Matrix* grad = layer_backward(dense, loss_grad, 0.1f);
            free_matrix(grad);
        }
    }
    
    // Inference
    input->data[0] = 5.0f / 20.0f;
    Matrix* out = layer_forward(dense, input);
    printf("Prediction: %f\n", out->data[0] * 20.0f);  // ~10.0
    
    free_matrix(input);
    free_matrix(loss_grad);
    free_layer(dense);
    return 0;
}
```

# This documentation was analyzed and developed by LLMs. This has been verified by me (Yashu-Maurya).
