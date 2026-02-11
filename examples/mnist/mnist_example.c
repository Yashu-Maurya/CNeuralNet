#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "../../include/network.h"

#define INPUT_SIZE 784 
#define HIDDEN_SIZE 128 
#define OUTPUT_SIZE 10  

#define LEARNING_RATE 0.01f
#define TRAIN_SAMPLES 5000
#define TEST_SAMPLES 1000
#define EPOCHS 10

int parse_csv_line(FILE *file, Matrix *input, Matrix *target) {
  char line[5000];

  if (fgets(line, sizeof(line), file) == NULL) {
    return -1;
  }

  if (strlen(line) <= 1) {
    return -1;
  }

  char *token = strtok(line, ",");
  if (token == NULL)
    return -1;

  int label = atoi(token);

  for (int i = 0; i < OUTPUT_SIZE; i++) {
    target->data[i] = (i == label) ? 1.0f : 0.0f;
  }

  int pixel_idx = 0;
  while ((token = strtok(NULL, ",")) != NULL && pixel_idx < INPUT_SIZE) {
    input->data[pixel_idx] = atof(token) / 255.0f;
    pixel_idx++;
  }

  return label;
}

int main() {
  srand((unsigned int)time(NULL));

  printf("=== MNIST Neural Network Training ===\n\n");
  printf("Network Architecture: %d -> %d -> %d\n", INPUT_SIZE, HIDDEN_SIZE,
         OUTPUT_SIZE);
  printf("Learning Rate: %.4f\n", LEARNING_RATE);
  printf("Training Samples: %d, Test Samples: %d\n", TRAIN_SAMPLES,
         TEST_SAMPLES);
  printf("Epochs: %d\n\n", EPOCHS);

  Network *network = create_network();
  if (network == NULL) {
    fprintf(stderr, "Failed to create network\n");
    return -1;
  }

  Layer *dense1 = layer_create_dense(INPUT_SIZE, HIDDEN_SIZE);
  Layer *relu1 = layer_create_relu();
  Layer *dense2 = layer_create_dense(HIDDEN_SIZE, OUTPUT_SIZE);
  Layer *sigmoid2 = layer_create_sigmoid();

  if (dense1 == NULL || relu1 == NULL || dense2 == NULL || sigmoid2 == NULL) {
    fprintf(stderr, "Failed to create layers\n");
    free_network(network);
    return -1;
  }

  add_layer(network, dense1);
  add_layer(network, relu1);
  add_layer(network, dense2);
  add_layer(network, sigmoid2);

  Matrix *input = create_matrix(INPUT_SIZE, 1);
  Matrix *target = create_matrix(OUTPUT_SIZE, 1);

  if (input == NULL || target == NULL) {
    fprintf(stderr, "Failed to create input/target matrices\n");
    free_network(network);
    return -1;
  }

  printf("--- Training Phase ---\n");

  for (int epoch = 0; epoch < EPOCHS; epoch++) {
    FILE *train_file = fopen("mnist_train.csv", "r");
    if (train_file == NULL) {
      fprintf(stderr, "Error: Could not open mnist_train.csv\n");
      fprintf(stderr,
              "Make sure you run this from the examples/mnist directory\n");
      free_matrix(input);
      free_matrix(target);
      free_network(network);
      return -1;
    }

    int correct = 0;
    float total_loss = 0.0f;

    for (int sample = 0; sample < TRAIN_SAMPLES; sample++) {
      int label = parse_csv_line(train_file, input, target);
      if (label < 0)
        break;

      Matrix *prediction = predict_network(network, input);
      int predicted_label = argmax(prediction);

      if (predicted_label == label) {
        correct++;
      }

      for (int i = 0; i < OUTPUT_SIZE; i++) {
        float diff = prediction->data[i] - target->data[i];
        total_loss += diff * diff;
      }

      free_matrix(prediction);

      train_network(network, input, target, LEARNING_RATE);

      if ((sample + 1) % 200 == 0) {
        printf("  Epoch %d: Processed %d/%d samples...\n", epoch + 1,
               sample + 1, TRAIN_SAMPLES);
      }
    }

    fclose(train_file);

    float accuracy = (float)correct / TRAIN_SAMPLES * 100.0f;
    float avg_loss = total_loss / TRAIN_SAMPLES;
    printf("Epoch %d/%d - Train Accuracy: %.2f%% - Avg Loss: %.4f\n", epoch + 1,
           EPOCHS, accuracy, avg_loss);
  }

  printf("\n--- Testing Phase ---\n");

  FILE *test_file = fopen("mnist_test.csv", "r");
  if (test_file == NULL) {
    fprintf(stderr, "Error: Could not open mnist_test.csv\n");
    free_matrix(input);
    free_matrix(target);
    free_network(network);
    return -1;
  }

  int test_correct = 0;
  int confusion_matrix[10][10] = {0};

  for (int sample = 0; sample < TEST_SAMPLES; sample++) {
    int label = parse_csv_line(test_file, input, target);
    if (label < 0)
      break;

    Matrix *prediction = predict_network(network, input);
    int predicted_label = argmax(prediction);

    confusion_matrix[label][predicted_label]++;

    if (predicted_label == label) {
      test_correct++;
    }

    free_matrix(prediction);
  }

  fclose(test_file);

  float test_accuracy = (float)test_correct / TEST_SAMPLES * 100.0f;
  printf("\n=== Results ===\n");
  printf("Test Accuracy: %d/%d = %.2f%%\n", test_correct, TEST_SAMPLES,
         test_accuracy);

  printf("\nPer-digit accuracy:\n");
  for (int digit = 0; digit < 10; digit++) {
    int total = 0;
    int correct = confusion_matrix[digit][digit];
    for (int j = 0; j < 10; j++) {
      total += confusion_matrix[digit][j];
    }
    if (total > 0) {
      printf("  Digit %d: %d/%d (%.1f%%)\n", digit, correct, total,
             (float)correct / total * 100.0f);
    }
  }

  printf("\n--- Demo: Single Sample Predictions ---\n");

  FILE *demo_file = fopen("mnist_test.csv", "r");
  if (demo_file != NULL) {
    for (int i = 0; i < 5; i++) {
      int label = parse_csv_line(demo_file, input, target);
      if (label < 0)
        break;

      Matrix *prediction = predict_network(network, input);
      int predicted = argmax(prediction);

      printf("  Sample %d: True Label = %d, Predicted = %d %s\n", i + 1, label,
             predicted, (label == predicted) ? "✓" : "✗");

      free_matrix(prediction);
    }
    fclose(demo_file);
  }

  print_network_info(network);

  free_matrix(input);
  free_matrix(target);
  free_network(network);

  printf("\nDone!\n");
  return 0;
}