#include <stdio.h>
#include <stdlib.h>

#include "../include/layer.h"
#include "../include/matrix.h"

#define LEARNING_RATE 0.1
#define EPOCHS 1000

int main() {

  Layer *dense = layer_create_dense(1, 1);

  Matrix *inputs = create_matrix(1, 1);
  Matrix *targets = create_matrix(1, 1);
  Matrix *loss_gradient = create_matrix(1, 1);

  printf("Training Started (Target: f(x) = 2x)...\n");

  for (int j = 0; j < EPOCHS; j++) {
    float epoch_error = 0;

    for (int i = 0; i < 10; i++) {
      inputs->data[0] = i / 20.0f;
      float target_val = (i * 2.0f) / 20.0f;
      targets->data[0] = target_val;

      Matrix *pred = layer_forward(dense, inputs);
      loss_gradient->data[0] = pred->data[0] - targets->data[0];

      epoch_error += (loss_gradient->data[0] * loss_gradient->data[0]);

      Matrix *upstream_grad =
          layer_backward(dense, loss_gradient, LEARNING_RATE);

      free_matrix(pred);
      free_matrix(upstream_grad);
    }

    if (j % 100 == 0) {
      printf("EPOCH %d | MSE: %f\n", j, epoch_error / 10.0f);
    }
  }

  printf("\n--- Final Inference Test ---\n");
  float test_input = 123.0f;
  inputs->data[0] = test_input / 20.0f;

  Matrix *out = layer_forward(dense, inputs);
  float result = out->data[0] * 20.0f;

  printf("Input: %.0f\n", test_input);
  printf("Predicted: %.4f (Expected: %.4f)\n", result, test_input * 2.0f);

  free_matrix(out);

  if (dense->weights != NULL) {
    printf("Learned Weight: %.4f\n", dense->weights->data[0]);
    printf("Learned Bias: %.4f\n", dense->bias->data[0]);
  }

  free_matrix(inputs);
  free_matrix(targets);
  free_matrix(loss_gradient);

  free_layer(dense);

  return 0;
}