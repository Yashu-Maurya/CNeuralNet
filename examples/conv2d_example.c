#include <stdio.h>
#include <stdlib.h>

#include "../include/layer.h"
#include "../include/matrix.h"

void print_matrix_shape(Matrix *m) {
  printf("(%d x %d)\n", m->rows, m->columns);
}

int main() {
  printf("=== Conv2d Layer Example ===\n\n");

  // Create a conv2d layer with a 3x3 kernel, input size 5x5
  Layer *conv = layer_create_conv2d(5, 3);
  if (conv == NULL) {
    fprintf(stderr, "Failed to create conv2d layer\n");
    return 1;
  }
  printf("Layer: %s\n", conv->name);
  printf("Input size: %d\n", conv->input_n);
  printf("Output size: %d\n", conv->output_n);
  printf("Kernel shape: ");
  print_matrix_shape(conv->weights);
  printf("\n");

  // Create a simple 5x5 input
  Matrix *input = create_matrix(5, 5);
  float val = 0.0f;
  for (int i = 0; i < 5; i++) {
    for (int j = 0; j < 5; j++) {
      input->data[i * 5 + j] = val;
      val += 1.0f;
    }
  }
  printf("Input:\n");
  print_matrix(input);
  printf("\n");

  // ---- Using the layer's own (random-initialized) kernel ----
  printf("--- Forward with layer's own kernel ---\n");
  Matrix *out = layer_forward(conv, input);
  if (out == NULL) {
    fprintf(stderr, "Forward pass failed\n");
    free_layer(conv);
    free_matrix(input);
    return 1;
  }
  printf("Output:\n");
  print_matrix(out);
  print_matrix_shape(out);
  free_matrix(out);

  // ---- Using a custom kernel (Sobel horizontal edge detector) ----
  Matrix *sobel = create_matrix(3, 3);
  float sobel_data[] = {-1, 0, 1, -2, 0, 2, -1, 0, 1};
  for (int i = 0; i < 9; i++) {
    sobel->data[i] = sobel_data[i];
  }

  printf("\n--- Forward with custom Sobel kernel (detect horizontal edges) ---\n");
  Matrix *edges = _layer_forward_conv2d_with_kernel(conv, input, sobel);
  if (edges == NULL) {
    fprintf(stderr, "Forward with custom kernel failed\n");
    free_matrix(sobel);
    free_layer(conv);
    free_matrix(input);
    return 1;
  }
  printf("Sobel kernel:\n");
  print_matrix(sobel);
  printf("Output (edges):\n");
  print_matrix(edges);
  print_matrix_shape(edges);
  free_matrix(edges);

  // ---- Using a custom averaging kernel ----
  Matrix *avg = create_matrix(3, 3);
  for (int i = 0; i < 9; i++) {
    avg->data[i] = 1.0f / 9.0f;
  }

  printf("\n--- Forward with custom averaging kernel ---\n");
  Matrix *blurred = _layer_forward_conv2d_with_kernel(conv, input, avg);
  if (blurred == NULL) {
    fprintf(stderr, "Forward with averaging kernel failed\n");
    free_matrix(avg);
    free_matrix(sobel);
    free_layer(conv);
    free_matrix(input);
    return 1;
  }
  printf("Averaging kernel:\n");
  print_matrix(avg);
  printf("Output (blurred):\n");
  print_matrix(blurred);
  print_matrix_shape(blurred);
  free_matrix(blurred);

  // ---- Backward pass demonstration ----
  printf("\n--- Backward pass ---\n");
  // Re-run forward to store fresh inputs/outputs
  Matrix *fwd = layer_forward(conv, input);
  if (fwd == NULL) {
    fprintf(stderr, "Forward pass failed\n");
    free_matrix(avg);
    free_matrix(sobel);
    free_layer(conv);
    free_matrix(input);
    return 1;
  }
  free_matrix(fwd);

  // Create a dummy error gradient (same shape as output: 3x3)
  Matrix *error_grad = create_matrix(3, 3);
  for (int i = 0; i < 9; i++) {
    error_grad->data[i] = 0.1f;
  }

  Matrix *upstream = layer_backward(conv, error_grad, 0.01f);
  if (upstream != NULL) {
    printf("Input gradient shape: ");
    print_matrix_shape(upstream);
    printf("Input gradient:\n");
    print_matrix(upstream);
    free_matrix(upstream);
  }

  printf("\nUpdated kernel (after backward):\n");
  print_matrix(conv->weights);

  // Cleanup
  free_matrix(error_grad);
  free_matrix(avg);
  free_matrix(sobel);
  free_matrix(input);
  free_layer(conv);

  printf("\nDone.\n");
  return 0;
}
