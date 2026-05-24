# MNIST Digit Classification Example

This example demonstrates how to build and train a sequential neural network to classify handwritten digits (0–9) using the MNIST dataset in CSV format.

## Network Architecture
- **Input Layer**: 784 neurons (28x28 flattened grayscale pixels)
- **Hidden Layer**: 128 neurons (ReLU activation)
- **Output Layer**: 10 neurons (Sigmoid activation)
- **Loss Function**: Mean Squared Error (MSE)
- **Optimizer**: Stochastic Gradient Descent (SGD)

---

## Step-by-Step Run Guide

### 1. Download the Dataset
The CNeuralNet library does not include the raw MNIST CSV files due to size. You can download `mnist_train.csv` and `mnist_test.csv` from public sources such as:
- [Kaggle MNIST Dataset in CSV](https://www.kaggle.com/datasets/oddrationale/mnist-in-csv)

Extract and place the downloaded `mnist_train.csv` and `mnist_test.csv` files inside the `examples/mnist/` directory.

### 2. Build the Library and Examples
From the root directory of the `CNeuralNet` repository, compile the project using CMake:

#### On Linux / macOS:
```bash
mkdir -p build
cd build
cmake ..
make
```

#### On Windows (using Command Prompt & MinGW/MSBuild):
```cmd
mkdir build
cd build
cmake ..
cmake --build .
```

This compiles the executable `mnist_example` inside the `build` directory.

### 3. Run the Example
The example expects the CSV files to be present in the current working directory. Run the executable from the `examples/mnist/` directory:

#### On Linux / macOS:
```bash
cd examples/mnist
../../build/mnist_example
```

#### On Windows:
```cmd
cd examples\mnist
..\..\build\Debug\mnist_example.exe
```

---

## What the Program Does
1. **Initializes the Network**: Allocates a network structure, registers the dense and activation layers, and randomizes weights.
2. **Trains the Model**: Iterates through 10 epochs. In each epoch, it parses 5,000 samples from `mnist_train.csv` one by one, runs forward and backward propagation, updates weights, and logs progress.
3. **Tests the Model**: Processes 1,000 samples from `mnist_test.csv` to calculate overall test accuracy.
4. **Digit-by-Digit Accuracy**: Outputs a confusion matrix showing accuracy for each individual digit (0–9).
5. **Single Sample Inference**: Runs quick inference on the first 5 samples of the test set and prints predictions with confirmation icons (`✓` or `✗`).
