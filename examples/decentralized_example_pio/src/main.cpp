#include <Arduino.h>
#include <WiFi.h>
#include <esp_now.h>
#include <esp_wifi.h>

// Wrap original C library headers in extern "C" so C++ can link with them
extern "C" {
#include <layer.h>
#include <matrix.h>
#include <network.h>
#include <cnn_platform.h>
}

// Ensure both nodes are configured for NODE_A or NODE_B
#if !defined(NODE_A) && !defined(NODE_B)
#error "Please define either NODE_A or NODE_B in build flags"
#endif

// Communication state machine
enum State {
  STATE_UNPAIRED,
  STATE_PAIRED_IDLE,
  STATE_AWAITING_BACKWARD,
  STATE_AWAITING_RESULT
};

volatile State current_state = STATE_UNPAIRED;

// ESP-NOW packet definitions
enum MsgType {
  MSG_PING,
  MSG_PAIR_REQ,
  MSG_PAIR_ACK,
  MSG_FORWARD,
  MSG_BACKWARD,
  MSG_RESULT
};

struct PacketHeader {
  uint8_t type;         // MsgType
  uint32_t sample_id;   // Unique ID to match forward/backward/result passes
};

struct ForwardPacket {
  PacketHeader header;
  float activations[8]; // Layer 1 output (ReLU activations)
  float target;         // Target label (1.0 or 0.0)
  bool is_training;     // Flag to toggle training mode
};

struct BackwardPacket {
  PacketHeader header;
  float gradients[8];   // Error gradients backpropagated from Layer 2 to Layer 1
};

struct ResultPacket {
  PacketHeader header;
  float prediction;     // Final prediction probability
  float target;         // Actual target label
};

// Peer MAC Address
uint8_t peer_mac[6] = {0};
bool peer_registered = false;

// Shared volatile variables for ESP-NOW RX
volatile bool packet_received = false;
uint8_t rx_buffer[250];
int rx_len = 0;
uint8_t rx_mac[6];

// System configuration
const int TRAIN_SAMPLES = 500;
const int INFERENCE_DELAY = 1000; // Delay in ms between inference steps
const float LEARNING_RATE = 0.05f;

// Neural Network Layers (allocated globally)
#ifdef NODE_A
Layer *dense1 = NULL;
Layer *relu1 = NULL;
#else // NODE_B
Layer *dense2 = NULL;
Layer *sigmoid2 = NULL;
#endif

// Log helper callback
static void neuralNetLog(const char *message) { Serial.print(message); }

// Helper to register ESP-NOW peer
void register_peer(const uint8_t *mac) {
  if (peer_registered) {
    esp_now_del_peer(peer_mac);
  }
  
  esp_now_peer_info_t peerInfo = {};
  memcpy(peerInfo.peer_addr, mac, 6);
  peerInfo.channel = 1; // Explicit channel
  peerInfo.encrypt = false;
  
  if (esp_now_add_peer(&peerInfo) != ESP_OK) {
    Serial.println("[Pairing] Failed to register peer!");
  } else {
    memcpy(peer_mac, mac, 6);
    peer_registered = true;
    Serial.printf("[Pairing] Registered peer: %02X:%02X:%02X:%02X:%02X:%02X\n",
                  mac[0], mac[1], mac[2], mac[3], mac[4], mac[5]);
  }
}

// ESP-NOW Receive Callback (macro matches ESP32 Arduino Core version)
#if ESP_ARDUINO_VERSION >= ESP_ARDUINO_VERSION_VAL(3, 0, 0)
void OnDataRecv(const esp_now_recv_info_t *recv_info, const uint8_t *data, int len) {
  if (len > 250) return;
  memcpy((void*)rx_buffer, data, len);
  rx_len = len;
  memcpy((void*)rx_mac, recv_info->src_addr, 6);
  packet_received = true;
}
#else
void OnDataRecv(const uint8_t *mac, const uint8_t *data, int len) {
  if (len > 250) return;
  memcpy((void*)rx_buffer, data, len);
  rx_len = len;
  memcpy((void*)rx_mac, mac, 6);
  packet_received = true;
}
#endif

// ESP-NOW Send Callback (useful for monitoring link status)
void OnDataSent(const uint8_t *mac_addr, esp_now_send_status_t status) {
  if (status != ESP_NOW_SEND_SUCCESS) {
    Serial.println("[ESP-NOW] Warning: Packet transmission failed!");
  }
}

void setup() {
  Serial.begin(115200);
  delay(1500);

  Serial.println("\n=================================================");
  #ifdef NODE_A
  Serial.println("  CNeuralNet Decentralized Example: NODE A (Sender) ");
  #else
  Serial.println("  CNeuralNet Decentralized Example: NODE B (Receiver) ");
  #endif
  Serial.println("=================================================\n");

  cnn_set_log_callback(neuralNetLog);
  srand((unsigned int)micros());

  // Initialize WiFi
  WiFi.mode(WIFI_STA);
  WiFi.disconnect();
  
  // Force WiFi to channel 1 to guarantee mutual visibility
  esp_wifi_set_promiscuous(true);
  esp_wifi_set_channel(1, WIFI_SECOND_CHAN_NONE);
  esp_wifi_set_promiscuous(false);

  Serial.printf("[System] MAC Address: %s\n", WiFi.macAddress().c_str());
  Serial.printf("[System] WiFi Channel: %d\n", WiFi.channel());

  // Initialize ESP-NOW
  if (esp_now_init() != ESP_OK) {
    Serial.println("[ESP-NOW] Critical Error: Failed to initialize ESP-NOW!");
    return;
  }

  esp_now_register_recv_cb(OnDataRecv);
  esp_now_register_send_cb(OnDataSent);

  // Initialize CNeuralNet local layers
  #ifdef NODE_A
  dense1 = layer_create_dense(2, 8);
  relu1 = layer_create_relu();
  if (dense1 == NULL || relu1 == NULL) {
    Serial.println("[NN] Error: Failed to initialize layers!");
  } else {
    Serial.println("[NN] Initialized Layer 1: Dense(2->8) + ReLU(8)");
  }
  #else
  dense2 = layer_create_dense(8, 1);
  sigmoid2 = layer_create_sigmoid();
  if (dense2 == NULL || sigmoid2 == NULL) {
    Serial.println("[NN] Error: Failed to initialize layers!");
  } else {
    Serial.println("[NN] Initialized Layer 2: Dense(8->1) + Sigmoid(1)");
  }
  #endif

  current_state = STATE_UNPAIRED;
}

// Variables for training / execution progress
uint32_t sample_counter = 0;
unsigned long last_ping_time = 0;
unsigned long send_time = 0;

// Tracking loss metrics on Node B
#ifdef NODE_B
float running_loss = 0.0f;
int training_count = 0;
#endif

// Math check: Circle classifier target function
float check_circle_label(float x, float y) {
  // Return 1.0 if inside a circle of radius sqrt(0.5) ~ 0.707
  return (x*x + y*y < 0.5f) ? 1.0f : 0.0f;
}

void loop() {
  // 1. Process ESP-NOW packets safely
  if (packet_received) {
    noInterrupts();
    uint8_t data[250];
    int len = rx_len;
    uint8_t mac[6];
    memcpy(data, (const void*)rx_buffer, len);
    memcpy(mac, (const void*)rx_mac, 6);
    packet_received = false;
    interrupts();

    if (len >= sizeof(PacketHeader)) {
      PacketHeader *header = (PacketHeader *)data;

      if (header->type == MSG_PING) {
        #ifdef NODE_A
        if (current_state == STATE_UNPAIRED) {
          Serial.println("[Pairing] Received Beacon Ping from Node B.");
          register_peer(mac);
          
          PacketHeader reply;
          reply.type = MSG_PAIR_REQ;
          reply.sample_id = 0;
          esp_now_send(peer_mac, (uint8_t *)&reply, sizeof(PacketHeader));
          Serial.println("[Pairing] Sent Pairing Request to Node B.");
        }
        #endif
      } 
      else if (header->type == MSG_PAIR_REQ) {
        #ifdef NODE_B
        if (current_state == STATE_UNPAIRED) {
          Serial.println("[Pairing] Received Pairing Request from Node A.");
          register_peer(mac);
          
          PacketHeader reply;
          reply.type = MSG_PAIR_ACK;
          reply.sample_id = 0;
          esp_now_send(peer_mac, (uint8_t *)&reply, sizeof(PacketHeader));
          current_state = STATE_PAIRED_IDLE;
          Serial.println("[Pairing] Sent Pairing ACK. Pairing complete!");
        }
        #endif
      } 
      else if (header->type == MSG_PAIR_ACK) {
        #ifdef NODE_A
        if (current_state == STATE_UNPAIRED) {
          Serial.println("[Pairing] Received Pairing ACK. Pairing complete!");
          register_peer(mac);
          current_state = STATE_PAIRED_IDLE;
        }
        #endif
      } 
      else if (header->type == MSG_FORWARD) {
        #ifdef NODE_B
        if (current_state == STATE_PAIRED_IDLE && len >= sizeof(ForwardPacket)) {
          ForwardPacket *f_pkt = (ForwardPacket *)data;
          
          // Re-create activation matrix from incoming packet
          Matrix *a1_matrix = create_matrix(8, 1);
          memcpy(a1_matrix->data, f_pkt->activations, 8 * sizeof(float));

          // Run Layer 2 Forward Pass
          Matrix *out2 = layer_forward(dense2, a1_matrix);
          Matrix *a2 = layer_forward(sigmoid2, out2);

          float prediction = a2->data[0];
          float target = f_pkt->target;

          if (f_pkt->is_training) {
            // Compute MSE Loss Gradient: dL/da2 = a2 - y
            float error = prediction - target;
            running_loss += error * error;
            training_count++;

            Matrix *loss_grad_matrix = create_matrix(1, 1);
            loss_grad_matrix->data[0] = error;

            // Run Layer 2 Backward Pass
            Matrix *grad_sig = layer_backward(sigmoid2, loss_grad_matrix, LEARNING_RATE);
            Matrix *grad_dense2 = layer_backward(dense2, grad_sig, LEARNING_RATE);

            // grad_dense2 (8x1) contains gradients w.r.t dense2 inputs (Node A's activations)
            // Transmit these gradients back to Node A
            BackwardPacket b_pkt;
            b_pkt.header.type = MSG_BACKWARD;
            b_pkt.header.sample_id = f_pkt->header.sample_id;
            memcpy(b_pkt.gradients, grad_dense2->data, 8 * sizeof(float));
            
            esp_now_send(peer_mac, (uint8_t *)&b_pkt, sizeof(BackwardPacket));

            // Log details
            if (training_count % 25 == 0) {
              Serial.printf("[Training] Sample %d | Pred: %.4f | Target: %.1f | Avg Loss: %.5f\n", 
                            f_pkt->header.sample_id, prediction, target, running_loss / training_count);
            }

            // Cleanup matrices
            free_matrix(loss_grad_matrix);
            free_matrix(grad_sig);
            free_matrix(grad_dense2);
          } else {
            // Inference phase: send back prediction result
            ResultPacket r_pkt;
            r_pkt.header.type = MSG_RESULT;
            r_pkt.header.sample_id = f_pkt->header.sample_id;
            r_pkt.prediction = prediction;
            r_pkt.target = target;
            
            esp_now_send(peer_mac, (uint8_t *)&r_pkt, sizeof(ResultPacket));

            Serial.printf("[Inference] Sample %d | Pred: %.4f (%s) | Target: %.1f | %s\n",
                          f_pkt->header.sample_id, prediction, 
                          (prediction >= 0.5f) ? "INSIDE " : "OUTSIDE", target,
                          ((prediction >= 0.5f) == (target >= 0.5f)) ? "✓ MATCH" : "✗ MISMATCH");
          }

          free_matrix(a1_matrix);
          free_matrix(out2);
          free_matrix(a2);
        }
        #endif
      } 
      else if (header->type == MSG_BACKWARD) {
        #ifdef NODE_A
        if (current_state == STATE_AWAITING_BACKWARD && len >= sizeof(BackwardPacket) && header->sample_id == sample_counter) {
          BackwardPacket *b_pkt = (BackwardPacket *)data;
          unsigned long rtt = millis() - send_time;

          // Re-create the error gradient matrix w.r.t ReLU activations
          Matrix *grad_from_b = create_matrix(8, 1);
          memcpy(grad_from_b->data, b_pkt->gradients, 8 * sizeof(float));

          // Run Layer 1 Backward Pass (updates local weights & bias)
          Matrix *grad_relu = layer_backward(relu1, grad_from_b, LEARNING_RATE);
          Matrix *grad_dense = layer_backward(dense1, grad_relu, LEARNING_RATE);

          // Cleanup
          free_matrix(grad_from_b);
          free_matrix(grad_relu);
          free_matrix(grad_dense);

          if ((sample_counter + 1) % 25 == 0 || (sample_counter + 1) == TRAIN_SAMPLES) {
            Serial.printf("[Training] Sample %d/%d completed | RTT: %lu ms | Weights updated\n", 
                          sample_counter + 1, TRAIN_SAMPLES, rtt);
          }

          sample_counter++;
          current_state = STATE_PAIRED_IDLE;
        }
        #endif
      } 
      else if (header->type == MSG_RESULT) {
        #ifdef NODE_A
        if (current_state == STATE_AWAITING_RESULT && len >= sizeof(ResultPacket) && header->sample_id == sample_counter) {
          ResultPacket *r_pkt = (ResultPacket *)data;
          unsigned long rtt = millis() - send_time;

          Serial.printf("[Inference] Pt %d | Pred: %.4f (%s) | Target: %.1f | %s | RTT: %lu ms\n",
                        sample_counter, r_pkt->prediction,
                        (r_pkt->prediction >= 0.5f) ? "INSIDE " : "OUTSIDE", r_pkt->target,
                        ((r_pkt->prediction >= 0.5f) == (r_pkt->target >= 0.5f)) ? "✓ MATCH" : "✗ MISMATCH",
                        rtt);

          sample_counter++;
          current_state = STATE_PAIRED_IDLE;
        }
        #endif
      }
    }
  }

  // 2. Perform periodic state machine updates
  if (current_state == STATE_UNPAIRED) {
    #ifdef NODE_B
    // Node B beacons pairing pings periodically
    unsigned long now = millis();
    if (now - last_ping_time >= 1000) {
      last_ping_time = now;
      PacketHeader ping;
      ping.type = MSG_PING;
      ping.sample_id = 0;
      
      const uint8_t broadcastAddress[] = {0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF};
      register_peer(broadcastAddress);
      esp_now_send(peer_mac, (uint8_t *)&ping, sizeof(PacketHeader));
      Serial.println("[Pairing] Sent Beacon Ping. Waiting for Node A...");
    }
    #endif
  } 
  else if (current_state == STATE_PAIRED_IDLE) {
    #ifdef NODE_A
    // Wait a brief period between samples
    if (sample_counter < TRAIN_SAMPLES) {
      delay(10); // Swift training
    } else {
      delay(INFERENCE_DELAY); // Visual readability for inference
    }

    // Generate random 2D coordinate point
    float rx = ((float)rand() / (float)RAND_MAX) * 2.0f - 1.0f; // Range [-1.0, 1.0]
    float ry = ((float)rand() / (float)RAND_MAX) * 2.0f - 1.0f; // Range [-1.0, 1.0]
    float target = check_circle_label(rx, ry);

    // Create 2D coordinate input matrix (2x1)
    Matrix *x = create_matrix(2, 1);
    x->data[0] = rx;
    x->data[1] = ry;

    // Run Layer 1 Forward Pass
    Matrix *out1 = layer_forward(dense1, x);
    Matrix *a1 = layer_forward(relu1, out1);

    // Prepare forward packet to transmit activations (8 floats)
    ForwardPacket pkt;
    pkt.header.type = MSG_FORWARD;
    pkt.header.sample_id = sample_counter;
    memcpy(pkt.activations, a1->data, 8 * sizeof(float));
    pkt.target = target;
    pkt.is_training = (sample_counter < TRAIN_SAMPLES);

    send_time = millis();
    esp_now_send(peer_mac, (uint8_t *)&pkt, sizeof(ForwardPacket));

    if (pkt.is_training) {
      current_state = STATE_AWAITING_BACKWARD;
    } else {
      current_state = STATE_AWAITING_RESULT;
    }

    // Cleanup local matrices
    free_matrix(x);
    free_matrix(out1);
    free_matrix(a1);
    #endif
  } 
  else if (current_state == STATE_AWAITING_BACKWARD || current_state == STATE_AWAITING_RESULT) {
    #ifdef NODE_A
    // Timeout handler to prevent locking up if packets are dropped
    if (millis() - send_time > 1000) {
      Serial.println("[System] Timeout waiting for reply! Retrying step...");
      current_state = STATE_PAIRED_IDLE;
    }
    #endif
  }
}
