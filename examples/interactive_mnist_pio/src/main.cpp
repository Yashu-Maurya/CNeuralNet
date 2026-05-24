#include <Arduino.h>
#include <LittleFS.h>
#include <cnn_platform.h>

// Wrap original C library headers in extern "C" so C++ can link with them
extern "C" {
#include <network.h>
#include <layer.h>
#include <matrix.h>
}

// Declare the filesystem helper functions inline since they are implemented 
// in network_storage.cpp but not present in the original unmodified network.h
#include <FS.h>
int save_network_fs(Network *network, fs::FS &filesystem, const char *path);
void load_network_fs(Network *network, fs::FS &filesystem, const char *path);

#define INPUT_SIZE 784     // 28x28 pixels
#define HIDDEN_SIZE 16     // Hidden layer neurons
#define OUTPUT_SIZE 10     // Digit classes 0-9
#define LEARNING_RATE 0.10f
#define MODEL_PATH "/mnist_model.cnet"

Network *network = NULL;
Matrix *input_matrix = NULL;
Matrix *target_matrix = NULL;

static void neuralNetLog(const char *message) {
  Serial.print(message);
}

// Clear incoming serial buffer
void clear_serial_buffer() {
  while (Serial.available()) {
    Serial.read();
  }
}

// Request data from python bridge and read the response
String get_bridge_response(const String &request) {
  clear_serial_buffer();
  Serial.println(request);
  
  // Wait up to 3 seconds for the response
  unsigned long start = millis();
  while (millis() - start < 3000) {
    if (Serial.available()) {
      String resp = Serial.readStringUntil('\n');
      resp.trim();
      if (resp.startsWith("RESP:")) {
        return resp;
      }
    }
    yield();
  }
  return ""; // Timeout
}

// Parse Response in format:
// RESP:TRAIN:<label>:<hex_pixels>
// RESP:TEST:<index>:<label>:<hex_pixels>
bool parse_response(const String &resp, String &type, int &index, int &label, String &hex_data) {
  if (!resp.startsWith("RESP:")) {
    return false;
  }
  
  int first_colon = resp.indexOf(':'); // RESP
  if (first_colon == -1) return false;
  
  int second_colon = resp.indexOf(':', first_colon + 1); // TYPE (TRAIN/TEST)
  if (second_colon == -1) return false;
  
  type = resp.substring(first_colon + 1, second_colon);
  
  if (type == "TRAIN") {
    int third_colon = resp.indexOf(':', second_colon + 1); // LABEL
    if (third_colon == -1) return false;
    
    label = resp.substring(second_colon + 1, third_colon).toInt();
    hex_data = resp.substring(third_colon + 1);
    index = -1;
    return (hex_data.length() >= INPUT_SIZE * 2);
  } 
  else if (type == "TEST") {
    int third_colon = resp.indexOf(':', second_colon + 1); // INDEX
    if (third_colon == -1) return false;
    
    int fourth_colon = resp.indexOf(':', third_colon + 1); // LABEL
    if (fourth_colon == -1) return false;
    
    index = resp.substring(second_colon + 1, third_colon).toInt();
    label = resp.substring(third_colon + 1, fourth_colon).toInt();
    hex_data = resp.substring(fourth_colon + 1);
    return (hex_data.length() >= INPUT_SIZE * 2);
  }
  
  return false;
}

// Helper to convert hex string into float array [0.0, 1.0]
bool parse_hex_pixels(const String &hex, float *data, int len) {
  if (hex.length() < len * 2) {
    return false;
  }
  for (int i = 0; i < len; i++) {
    char h_char = hex[i * 2];
    char l_char = hex[i * 2 + 1];
    
    uint8_t h_val = (h_char >= 'a') ? (h_char - 'a' + 10) :
                    (h_char >= 'A') ? (h_char - 'A' + 10) : (h_char - '0');
    uint8_t l_val = (l_char >= 'a') ? (l_char - 'a' + 10) :
                    (l_char >= 'A') ? (l_char - 'A' + 10) : (l_char - '0');
                    
    uint8_t val = (h_val << 4) | l_val;
    data[i] = (float)val / 255.0f;
  }
  return true;
}

// Initialize / reset the network structure
bool reset_network() {
  if (network != NULL) {
    free_network(network);
  }
  
  network = create_network();
  if (network == NULL) {
    Serial.println("Error: Failed to create network.");
    return false;
  }
  add_layer(network, layer_create_dense(INPUT_SIZE, HIDDEN_SIZE));
  add_layer(network, layer_create_relu());
  add_layer(network, layer_create_dense(HIDDEN_SIZE, OUTPUT_SIZE));
  add_layer(network, layer_create_sigmoid());
  
  if (network->layer_count < 4) {
    Serial.println("CRITICAL ERROR: Failed to allocate all layers (Out of Memory).");
    Serial.printf("Layer count is %d / 4. Try reducing hidden layer sizes.\n", network->layer_count);
    return false;
  }
  
  Serial.println("Network initialized / weights randomized.");
  return true;
}

// Perform training loop
void run_training(int num_samples) {
  Serial.printf("[Training] Starting training of %d samples...\n", num_samples);
  unsigned long start_time = millis();
  float total_loss = 0.0f;
  int success_count = 0;
  
  for (int i = 0; i < num_samples; i++) {
    String resp_str = get_bridge_response("CMD:TRAIN_REQ");
    if (resp_str.length() == 0) {
      Serial.println("[Training Error] Timeout waiting for bridge response.");
      return;
    }
    
    String type;
    int index;
    int label;
    String hex_data;
    
    if (!parse_response(resp_str, type, index, label, hex_data) || type != "TRAIN") {
      Serial.println("[Training Error] Failed to parse bridge response.");
      return;
    }
    
    if (!parse_hex_pixels(hex_data, input_matrix->data, INPUT_SIZE)) {
      Serial.println("[Training Error] Failed to parse hex pixels.");
      return;
    }
    
    // Setup target output
    zero_matrix(target_matrix);
    target_matrix->data[label] = 1.0f;
    
    // Evaluate MSE loss before the weight update
    Matrix *prediction = predict_network(network, input_matrix);
    if (prediction == NULL) {
      Serial.println("[Training Error] Prediction returned NULL.");
      return;
    }
    float sample_loss = 0.0f;
    for (int k = 0; k < OUTPUT_SIZE; k++) {
      float diff = prediction->data[k] - target_matrix->data[k];
      sample_loss += diff * diff;
    }
    total_loss += sample_loss;
    free_matrix(prediction);
    
    // Backpropagation and weight updates
    train_network(network, input_matrix, target_matrix, LEARNING_RATE);
    success_count++;
    
    // Print progress status
    if ((i + 1) % 50 == 0 || (i + 1) == num_samples) {
      Serial.printf("[Training] Progress: %d/%d | Avg MSE: %.4f | Free Heap: %u\n",
                    i + 1, num_samples, total_loss / (float)(i + 1), ESP.getFreeHeap());
    }
    
    yield();
  }
  
  unsigned long duration = millis() - start_time;
  float avg_time = (float)duration / (float)success_count;
  Serial.printf("[Training] Completed %d samples in %lu ms (%.1f ms/sample).\n",
                success_count, duration, avg_time);
}

// Evaluate performance accuracy on testing dataset
void run_testing(int num_samples) {
  Serial.printf("[Testing] Starting evaluation on %d random test samples...\n", num_samples);
  unsigned long start_time = millis();
  int correct_count = 0;
  int success_count = 0;
  
  for (int i = 0; i < num_samples; i++) {
    String resp_str = get_bridge_response("CMD:TEST_RAND");
    if (resp_str.length() == 0) {
      Serial.println("[Testing Error] Timeout waiting for bridge response.");
      return;
    }
    
    String type;
    int index;
    int label;
    String hex_data;
    
    if (!parse_response(resp_str, type, index, label, hex_data) || type != "TEST") {
      Serial.println("[Testing Error] Failed to parse bridge response.");
      return;
    }
    
    if (!parse_hex_pixels(hex_data, input_matrix->data, INPUT_SIZE)) {
      Serial.println("[Testing Error] Failed to parse hex pixels.");
      return;
    }
    
    Matrix *prediction = predict_network(network, input_matrix);
    if (prediction == NULL) {
      Serial.println("[Testing Error] Prediction returned NULL.");
      return;
    }
    int pred_class = argmax(prediction);
    free_matrix(prediction);
    
    if (pred_class == label) {
      correct_count++;
    }
    success_count++;
    
    yield();
  }
  
  unsigned long duration = millis() - start_time;
  float accuracy = ((float)correct_count / (float)success_count) * 100.0f;
  Serial.printf("[Testing] Completed! Accuracy: %.2f%% (%d/%d correct) in %lu ms.\n",
                accuracy, correct_count, success_count, duration);
}

// Request specific test sample, render ASCII art, and perform prediction
void run_show_sample(int sample_idx) {
  Serial.printf("[Inference] Requesting test image %d...\n", sample_idx);
  
  String req_str = "CMD:TEST_REQ " + String(sample_idx);
  String resp_str = get_bridge_response(req_str);
  if (resp_str.length() == 0) {
    Serial.println("[Inference Error] Timeout waiting for bridge response.");
    return;
  }
  
  String type;
  int index;
  int label;
  String hex_data;
  
  if (!parse_response(resp_str, type, index, label, hex_data) || type != "TEST") {
    Serial.println("[Inference Error] Failed to parse bridge response.");
    return;
  }
  
  if (!parse_hex_pixels(hex_data, input_matrix->data, INPUT_SIZE)) {
    Serial.println("[Inference Error] Failed to parse hex pixels.");
    return;
  }
  
  // Draw ASCII art representation of the 28x28 digit
  Serial.println("\n--- Image Render (28x28) ---");
  for (int r = 0; r < 28; r++) {
    String line = "";
    for (int c = 0; c < 28; c++) {
      float pixel = input_matrix->data[r * 28 + c];
      if (pixel > 0.8f) line += "@";
      else if (pixel > 0.6f) line += "#";
      else if (pixel > 0.4f) line += "*";
      else if (pixel > 0.2f) line += ".";
      else line += " ";
    }
    Serial.println(line);
  }
  Serial.println("-----------------------------\n");
  
  // Perform network prediction
  Matrix *prediction = predict_network(network, input_matrix);
  if (prediction == NULL) {
    Serial.println("[Inference Error] Prediction returned NULL.");
    return;
  }
  int pred_class = argmax(prediction);
  float confidence = prediction->data[pred_class];
  
  Serial.printf("Prediction: %d (Confidence: %.4f)\n", pred_class, confidence);
  Serial.printf("Actual Label: %d (%s)\n", label, (pred_class == label) ? "MATCH!" : "MISMATCH");
  
  Serial.println("Confidence distribution:");
  for (int k = 0; k < OUTPUT_SIZE; k++) {
    Serial.printf("  Digit %d: %.4f\n", k, prediction->data[k]);
  }
  
  free_matrix(prediction);
}

// Save model weights to LittleFS
void save_model() {
  if (network == NULL || network->layer_count == 0) {
    Serial.println("Error: Network is not initialized.");
    return;
  }
  
  int res = save_network_fs(network, LittleFS, MODEL_PATH);
  if (res == 0) {
    Serial.printf("Model saved successfully to LittleFS '%s'!\n", MODEL_PATH);
  } else {
    Serial.println("Error: Failed to save model to LittleFS.");
  }
}

// Load model weights from LittleFS
void load_model() {
  if (!LittleFS.exists(MODEL_PATH)) {
    Serial.printf("Error: Model file '%s' not found on LittleFS.\n", MODEL_PATH);
    return;
  }
  
  if (network != NULL) {
    free_network(network);
  }
  
  network = create_network();
  load_network_fs(network, LittleFS, MODEL_PATH);
  
  // Validate the loaded network's dimensions
  if (network->layer_count > 0) {
    int first_dense_input = -1;
    for (int i = 0; i < network->layer_count; i++) {
      if (strcmp(network->layers[i]->name, "Dense") == 0) {
        first_dense_input = network->layers[i]->input_n;
        break;
      }
    }
    
    int last_dense_output = -1;
    for (int i = network->layer_count - 1; i >= 0; i--) {
      if (strcmp(network->layers[i]->name, "Dense") == 0) {
        last_dense_output = network->layers[i]->output_n;
        break;
      }
    }
    
    if (first_dense_input != INPUT_SIZE || last_dense_output != OUTPUT_SIZE) {
      Serial.printf("Error: Loaded model dimensions (%d -> ... -> %d) do not match expected (%d -> ... -> %d).\n",
                    first_dense_input, last_dense_output, INPUT_SIZE, OUTPUT_SIZE);
      Serial.println("Resetting network to default architecture.");
      reset_network();
      return;
    }
  } else {
    Serial.println("Error: Loaded model is empty.");
    reset_network();
    return;
  }
  
  Serial.println("Model loaded successfully from LittleFS!");
}

// Show help menu
void print_menu() {
  Serial.println("\n=== Interactive MNIST C NeuralNet Shell ===");
  Serial.println("Commands:");
  Serial.println("  train <num>  - Train on <num> random MNIST images (e.g. train 1000)");
  Serial.println("  test <num>   - Test overall accuracy on <num> random test images (e.g. test 100)");
  Serial.println("  show <index> - Render image at index (0-9999) in ASCII, predict, and show details");
  Serial.println("  info         - Print layers architecture and ESP32 free heap memory");
  Serial.println("  save         - Save current model weights to LittleFS");
  Serial.println("  load         - Load model weights from LittleFS");
  Serial.println("  reset        - Randomize network weights");
  Serial.println("  help         - Print this command list");
  Serial.println("  <number>     - Shortcut for 'show <number>' (e.g. 42)");
  Serial.println("------------------------------------------");
}

void setup() {
  Serial.begin(115200);
  delay(1000);
  
  // Increase Serial read buffer/timeout for large packets
  Serial.setTimeout(3000);
  
  cnn_set_log_callback(neuralNetLog);
  srand((unsigned int)micros());
  
  if (!LittleFS.begin(true)) {
    Serial.println("Warning: LittleFS mount failed. Model save/load features disabled.");
  }
  
  // Allocate static matrices
  input_matrix = create_matrix(INPUT_SIZE, 1);
  target_matrix = create_matrix(OUTPUT_SIZE, 1);
  
  // Create first network weights
  if (!reset_network()) {
    Serial.println("Warning: Network initialization failed.");
  }
  
  // Print commands menu
  print_menu();
}

void loop() {
  if (!Serial.available()) {
    return;
  }
  
  String command = Serial.readStringUntil('\n');
  command.trim();
  if (command.length() == 0) {
    return;
  }
  
  if (command == "help") {
    print_menu();
  } 
  else if (command == "info") {
    print_network_info(network);
    Serial.printf("ESP32 Free Heap: %u bytes\n", ESP.getFreeHeap());
    if (LittleFS.begin()) {
      Serial.printf("LittleFS Storage: %u / %u bytes used\n", 
                    LittleFS.usedBytes(), LittleFS.totalBytes());
    }
  } 
  else if (command == "save") {
    save_model();
  } 
  else if (command == "load") {
    load_model();
  } 
  else if (command == "reset") {
    reset_network();
  } 
  else if (command.startsWith("train ")) {
    int num = command.substring(6).toInt();
    if (num > 0) {
      run_training(num);
    } else {
      Serial.println("Usage: train <number_of_samples>");
    }
  } 
  else if (command.startsWith("test ")) {
    int num = command.substring(5).toInt();
    if (num > 0) {
      run_testing(num);
    } else {
      Serial.println("Usage: test <number_of_samples>");
    }
  } 
  else if (command.startsWith("show ")) {
    int idx = command.substring(5).toInt();
    if (idx >= 0 && idx < 10000) {
      run_show_sample(idx);
    } else {
      Serial.println("Usage: show <index> (where 0 <= index < 10000)");
    }
  } 
  else if (command.length() > 0 && command[0] >= '0' && command[0] <= '9') {
    // Treat raw numeric input as shortcut for show
    int idx = command.toInt();
    if (idx >= 0 && idx < 10000) {
      run_show_sample(idx);
    } else {
      Serial.println("Error: Index must be in range [0, 9999]");
    }
  } 
  else {
    Serial.printf("Unknown command: '%s'. Type 'help' to see list.\n", command.c_str());
  }
}
