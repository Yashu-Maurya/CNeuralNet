# Save and Load Weights Example

This example demonstrates how to serialize (save) and deserialize (load) a neural network model to and from a binary file.

---

## Step-by-Step Run Guide

### 1. Build the Examples
Compile the project using CMake from the repository root:

#### On Linux / macOS:
```bash
mkdir -p build
cd build
cmake ..
make
```

#### On Windows:
```cmd
mkdir build
cd build
cmake ..
cmake --build .
```

This compiles the executable `save_and_load_weights_example` inside the `build` directory.

### 2. Run the Example
You can run this executable from any directory. For example, run it from the `build` directory:

#### On Linux / macOS:
```bash
cd build
./save_and_load_weights_example
```

#### On Windows:
```cmd
cd build
Debug\save_and_load_weights_example.exe
```

---

## What the Program Does
1. **Creates a Network**: Instantiates a sequential model with:
   - A Dense layer (10 $\rightarrow$ 20)
   - A Dense layer (20 $\rightarrow$ 10)
   - A Sigmoid activation layer
2. **Saves the Network**: Saves the model to a binary file named `model.cnet` in the current working directory using `save_network()`.
3. **Frees the Network**: Destroys the active network instance from memory.
4. **Loads the Network**: Creates a new, blank network, reads the structure and weights from `model.cnet` using `load_network()`, and populates the model.
5. **Prints Model Layout**: Prints the loaded network layers and dimensions to verify the loading succeeded without data corruption.
