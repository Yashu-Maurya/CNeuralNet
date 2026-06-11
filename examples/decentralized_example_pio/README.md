# CNeuralNet - Decentralized Split Neural Network Example

This example demonstrates how to implement a **Split Neural Network** running cooperatively across two separate ESP32 microcontrollers using ESP-NOW for low-latency wireless communication.

## Theoretical Overview

In standard centralized machine learning, the entire network is evaluated and updated on a single processor. In **Split Learning**, the network layers are divided between two or more physical nodes:

*   **Node A (Feature Extractor)**: Holds the input layer and the first hidden layer. It takes raw inputs, computes the initial activations, and transmits these activations to Node B.
*   **Node B (Classifier)**: Receives the activations from Node A, processes them through the remaining hidden layers and the output layer to make a prediction.

### Split Backpropagation (Training)
For training, the error backpropagation is also split:
1.  **Node B** calculates the loss relative to the target label.
2.  **Node B** backpropagates the loss through its local layers to compute the error gradient w.r.t its input (which corresponds to Node A's output activations).
3.  **Node B** transmits these gradients back to **Node A** over ESP-NOW.
4.  **Node A** uses these gradients to continue backpropagation through its local layers and update its own weights.

This allows cooperative training of a single centralized neural network across two physically separated, low-power microcontrollers!

---

## Network Architecture

We train a non-linear classifier to solve the **Circle Boundary Problem** (predicting whether a random 2D point $(x, y) \in [-1.0, 1.0]$ lies inside a circle of radius $\sqrt{0.5} \approx 0.707$):

$$x^2 + y^2 < 0.5$$

The overall centralized network is:
*   **Input**: 2 units (representing $x, y$ coordinates)
*   **Layer 1 (Dense)**: 2 inputs $\rightarrow$ 8 outputs
*   **Activation (ReLU)**: 8 units
*   **Layer 2 (Dense)**: 8 inputs $\rightarrow$ 1 output
*   **Activation (Sigmoid)**: 1 unit (probability output)

### Distributed Layout
*   **Node A (Sender)**: `Dense(2 -> 8)` + `ReLU(8)`
*   **Node B (Receiver)**: `Dense(8 -> 1)` + `Sigmoid(1)`

---

## Auto-Pairing Protocol

The example includes a plug-and-play **wireless pairing protocol** so you do not need to manually hardcode MAC addresses:
1.  **Unpaired Phase**: Upon booting, Node B periodically broadcasts a wireless beacon (`MSG_PING`).
2.  **Pairing Phase**: When Node A receives a beacon, it extracts Node B's MAC address, adds it as a peer, and replies with a unicast `MSG_PAIR_REQ`.
3.  **Completion**: Node B receives the request, registers Node A's MAC address as a peer, sends a `MSG_PAIR_ACK` back, and transitions to the running state.

Both nodes then transition from broadcast to unicast communication, enabling automatic hardware retransmissions and acknowledgments for high reliability.

---

## Setup & Execution

### 1. Prerequisites
Ensure you have the [PlatformIO Core CLI](https://docs.platformio.org/en/latest/core/installation.html) installed.

### 2. Connect Your Hardware
Connect both of your ESP32 boards to your computer via USB.

### 3. Flash Node A (Sender)
Specify the port or let PlatformIO detect it automatically:
```bash
./upload.sh node_a
```

### 4. Flash Node B (Receiver)
Unplug Node A or specify the target port if uploading simultaneously, then run:
```bash
./upload.sh node_b
```

### 5. Monitor Output
Open two terminal windows to watch the live cooperative training process:

*   **To monitor Node A**:
    ```bash
    pio device monitor -e node_a
    ```
*   **To monitor Node B**:
    ```bash
    pio device monitor -e node_b
    ```

---

## Expected Output Logs

Once both boards are running, you will see them perform auto-pairing, proceed with cooperative training (500 steps), and switch to inference.

### Node A Console:
```text
=================================================
  CNeuralNet Decentralized Example: NODE A (Sender) 
=================================================

[NN] Initialized Layer 1: Dense(2->8) + ReLU(8)
[System] MAC Address: E0:5A:1B:A2:89:C4
[System] WiFi Channel: 1
[Pairing] Waiting for Node B ping...
[Pairing] Received Beacon Ping from Node B.
[Pairing] Registered peer: E0:5A:1B:A2:8D:1C
[Pairing] Sent Pairing Request to Node B.
[Pairing] Received Pairing ACK. Pairing complete!
[Training] Sample 25/500 completed | RTT: 4 ms | Weights updated
[Training] Sample 50/500 completed | RTT: 3 ms | Weights updated
...
[Training] Sample 500/500 completed | RTT: 4 ms | Weights updated
[Inference] Pt 501 | Pred: 0.9812 (INSIDE ) | Target: 1.0 | ✓ MATCH | RTT: 3 ms
[Inference] Pt 502 | Pred: 0.1042 (OUTSIDE) | Target: 0.0 | ✓ MATCH | RTT: 3 ms
```

### Node B Console:
```text
=================================================
  CNeuralNet Decentralized Example: NODE B (Receiver) 
=================================================

[NN] Initialized Layer 2: Dense(8->1) + Sigmoid(1)
[System] MAC Address: E0:5A:1B:A2:8D:1C
[System] WiFi Channel: 1
[Pairing] Sent Beacon Ping. Waiting for Node A...
[Pairing] Received Pairing Request from Node A.
[Pairing] Registered peer: E0:5A:1B:A2:89:C4
[Pairing] Sent Pairing ACK. Pairing complete!
[Training] Sample 25 | Pred: 0.4412 | Target: 1.0 | Avg Loss: 0.23190
[Training] Sample 50 | Pred: 0.5891 | Target: 1.0 | Avg Loss: 0.20120
...
[Training] Sample 500 | Pred: 0.9122 | Target: 1.0 | Avg Loss: 0.04123
[Inference] Sample 501 | Pred: 0.9812 (INSIDE ) | Target: 1.0 | ✓ MATCH
[Inference] Sample 502 | Pred: 0.1042 (OUTSIDE) | Target: 0.0 | ✓ MATCH
```
