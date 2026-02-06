#include "../include/network.h"
#include <stdio.h>

#define LEARNING_RATE 0.1
#define EPOCHS 1000

int main() {
  Network *network = create_network();
  if (network == NULL) {
    return -1;
  }

  Layer *dense1 = layer_create_dense(1, 1);

  Matrix *inputs = create_matrix(1, 1);
  Matrix *targets = create_matrix(1, 1);

  add_layer(network, dense1);

  for (int i = 0; i < EPOCHS; i += 1) {
    float epoch_error = 0.0f;

    for (int j = 0; j < 10; j += 1) {
      inputs->data[0] = j / 20.0f;
      float target_val = (j * 2.0f) / 20.0f;
      targets->data[0] = target_val;

      Matrix *pred = predict_network(network, inputs);
      float error = pred->data[0] - target_val;
      epoch_error += error * error;
      free_matrix(pred);

      train_network(network, inputs, targets, LEARNING_RATE);
    }

    if (i % 100 == 0) {
      printf("EPOCH %d | MSE: %f\n", i, epoch_error / 10.0f);
    }
  }
  printf("\n--- Final Inference Test ---\n");
  float test_input = 100.0f;
  inputs->data[0] = test_input / 20.0f;
  Matrix *output_matrix = predict_network(network, inputs);
  float output = output_matrix->data[0];
  float result = output * 20.0f;

  printf("Input: %.2f, Output: %f\n", test_input, result);

  free_matrix(output_matrix);

  free_network(network);
  free_matrix(inputs);
  free_matrix(targets);

  return 0;
}