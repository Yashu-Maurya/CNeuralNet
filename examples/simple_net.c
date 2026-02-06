#include <stdio.h>
#include <stdlib.h>

#include "../include/matrix.h"

#define LEARNING_RATE 0.1
#define EPOCHS 1000

Matrix *run_model(Matrix *inputs, Matrix *weights, Matrix *bias);
void train_model(Matrix *out, float target, Matrix *inputs, Matrix *weights,
                 Matrix *bias);

int main() {
  // f(x) = 2x
  Matrix *inputs = create_matrix(1, 1);  // a 1x1 input matrix.
  Matrix *weights = create_matrix(1, 1); // 1x1 weight matrix.
  Matrix *bias = create_matrix(1, 1);    // 1x1 bias matrix

  // Activation(InputxWeights+Bias)

  randomize_matrix(weights);
  randomize_matrix(bias);

  for (int j = 0; j < EPOCHS; j++) {
    for (int i = 0; i < 10; i++) {
      inputs->data[0] = i / 20.0f;
      float target = i * 2.0f / 20.f;

      Matrix *out;
      out = run_model(inputs, weights, bias);
      train_model(out, target, inputs, weights, bias);

      free_matrix(out);
    }

    printf("\nEPOCH %i DONE \n", j);
  }
  inputs->data[0] = 15 / 20.0f;
  Matrix *out;
  out = run_model(inputs, weights, bias);
  printf("\n\nFinal Inference Test for 15: ");
  out->data[0] = out->data[0] * 20.0f;
  print_matrix(out);

  free_matrix(out);
  free_matrix(inputs);
  free_matrix(weights);
  free_matrix(bias);

  return 0;
}

Matrix *run_model(Matrix *inputs, Matrix *weights, Matrix *bias) {
  Matrix *out;
  out = multiply_mat(weights, inputs);
  add_matrix(out, bias);
  return out;
}

void train_model(Matrix *out, float target, Matrix *inputs, Matrix *weights,
                 Matrix *bias) {
  float error = target - out->data[0];
  print_matrix(out);

  float derivative = 1.0f; // linear activation

  float gradient = error * derivative * inputs->data[0];

  float bias_gradient = error * derivative * 1;

  add_scaler(weights, (LEARNING_RATE * gradient));
  add_scaler(bias, (LEARNING_RATE * bias_gradient));
}