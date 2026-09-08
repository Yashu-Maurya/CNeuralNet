#include <string.h>
#include "../include/layer.h"

static int conv2d_compute(Layer *l, Matrix *input, Matrix *kernel,
                          Matrix *out) {
  if (l == NULL || input == NULL || kernel == NULL || out == NULL ||
      l->bias == NULL || input->data == NULL || kernel->data == NULL ||
      out->data == NULL || l->bias->data == NULL) {
    return -1;
  }
  if (input->rows < kernel->rows || input->columns < kernel->columns) {
    return -1;
  }

  int output_rows = input->rows - kernel->rows + 1;
  int output_columns = input->columns - kernel->columns + 1;
  if (out->rows != output_rows || out->columns != output_columns) {
    return -1;
  }

  for (int i = 0; i < output_rows; i++) {
    for (int j = 0; j < output_columns; j++) {
      float sum = 0.0f;
      for (int ki = 0; ki < kernel->rows; ki++) {
        for (int kj = 0; kj < kernel->columns; kj++) {
          int input_idx = (i + ki) * input->columns + (j + kj);
          int kernel_idx = ki * kernel->columns + kj;
          sum += input->data[input_idx] * kernel->data[kernel_idx];
        }
      }
      out->data[i * out->columns + j] = sum + l->bias->data[0];
    }
  }

  l->input_rows = input->rows;
  l->input_columns = input->columns;
  l->output_rows = output_rows;
  l->output_columns = output_columns;
  l->input_n = input->rows;
  l->output_n = output_rows;

  return 0;
}

Matrix *_layer_forward_conv2d_with_kernel(Layer *l, Matrix *input, Matrix *kernel) {
  
  // default stride = 1, padding = 0;

  if(l == NULL || input == NULL || kernel == NULL || l->bias == NULL ||
     input->data == NULL || kernel->data == NULL || l->bias->data == NULL) {
    return NULL;
  }
  if (input->rows < kernel->rows || input->columns < kernel->columns) {
    return NULL;
  }

  if (l->inputs == NULL || l->inputs->rows != input->rows || l->inputs->columns != input->columns) {
    if (l->inputs != NULL) free_matrix(l->inputs);
    l->inputs = create_matrix(input->rows, input->columns);
  }
  if (l->inputs == NULL) {
    return NULL;
  }
  memcpy(l->inputs->data, input->data, input->rows * input->columns * sizeof(float));

  int output_rows = input->rows - kernel->rows + 1;
  int output_columns = input->columns - kernel->columns + 1;
  if (l->output == NULL || l->output->rows != output_rows ||
      l->output->columns != output_columns) {
    if (l->output != NULL) free_matrix(l->output);
    l->output = create_matrix(output_rows, output_columns);
  }
  Matrix *out = l->output;
  if (out == NULL) {
    return NULL;
  }

  if (conv2d_compute(l, input, kernel, out) != 0) {
    return NULL;
  }

  return copy_matrix(out);
}

Matrix *_layer_forward_conv2d(Layer *l, Matrix *input) {
  return _layer_forward_conv2d_with_kernel(l, input, l->weights);
}

int layer_forward_conv2d_with_kernel_into(Layer *l, Matrix *input,
                                          Matrix *kernel, Matrix *output) {
  return conv2d_compute(l, input, kernel, output);
}

int layer_forward_conv2d_into(Layer *l, Matrix *input, Matrix *output) {
  if (l == NULL) {
    return -1;
  }
  return layer_forward_conv2d_with_kernel_into(l, input, l->weights, output);
}

Matrix *_layer_backward_conv2d(Layer *l, Matrix *error_gradient,
                               float learning_rate) {
  if (l == NULL || error_gradient == NULL || l->inputs == NULL ||
      l->weights == NULL || l->bias == NULL) {
    return NULL;
  }

  if (error_gradient->rows != l->output_rows ||
      error_gradient->columns != l->output_columns) {
    return NULL;
  }

  int kernel_rows = l->weights->rows;
  int kernel_columns = l->weights->columns;
  int input_rows = l->inputs->rows;
  int input_columns = l->inputs->columns;
  int output_rows = error_gradient->rows;
  int output_columns = error_gradient->columns;

  // 1. Compute kernel gradient: dW[ki][kj] = sum_i sum_j X[i+ki][j+kj] * dY[i][j]
  Matrix *d_weights = create_matrix(kernel_rows, kernel_columns);
  if (d_weights == NULL) {
    return NULL;
  }
  zero_matrix(d_weights);

  for (int ki = 0; ki < kernel_rows; ki++) {
    for (int kj = 0; kj < kernel_columns; kj++) {
      float sum = 0.0f;
      for (int i = 0; i < output_rows; i++) {
        for (int j = 0; j < output_columns; j++) {
          sum += l->inputs->data[(i + ki) * l->inputs->columns + (j + kj)] *
                 error_gradient->data[i * error_gradient->columns + j];
        }
      }
      d_weights->data[ki * d_weights->columns + kj] = sum;
    }
  }

  // 2. Compute bias gradient: db = sum of all error elements
  float bias_grad = 0.0f;
  for (int i = 0; i < output_rows * output_columns; i++) {
    bias_grad += error_gradient->data[i];
  }

  // 3. Compute input gradient: dX[i][j] = sum_ki sum_kj W[ki][kj] * dY[i-ki][j-kj]
  Matrix *input_grad = create_matrix(input_rows, input_columns);
  if (input_grad == NULL) {
    free_matrix(d_weights);
    return NULL;
  }
  zero_matrix(input_grad);

  for (int i = 0; i < input_rows; i++) {
    for (int j = 0; j < input_columns; j++) {
      float sum = 0.0f;
      for (int ki = 0; ki < kernel_rows; ki++) {
        for (int kj = 0; kj < kernel_columns; kj++) {
          int out_i = i - ki;
          int out_j = j - kj;
          if (out_i >= 0 && out_i < output_rows && out_j >= 0 &&
              out_j < output_columns) {
            sum += l->weights
                       ->data[ki * l->weights->columns + kj] *
                   error_gradient->data[out_i * error_gradient->columns + out_j];
          }
        }
      }
      input_grad->data[i * input_grad->columns + j] = sum;
    }
  }

  // 4. Update weights: W -= lr * dW
  scale_matrix(d_weights, -learning_rate);
  add_matrix(l->weights, d_weights);
  free_matrix(d_weights);

  // 5. Update bias: b -= lr * db
  l->bias->data[0] -= learning_rate * bias_grad;

  return input_grad;
}

Layer *layer_create_conv2d(int input_size, int kernel_size) {
  return layer_create_conv2d_shape(input_size, input_size, kernel_size,
                                   kernel_size);
}

Layer *layer_create_conv2d_shape(int input_rows, int input_columns,
                                 int kernel_rows, int kernel_columns) {
  if (input_rows <= 0 || input_columns <= 0 || kernel_rows <= 0 ||
      kernel_columns <= 0 || kernel_rows > input_rows ||
      kernel_columns > input_columns) {
    return NULL;
  }

  Layer *l = (Layer *)malloc(sizeof(Layer));
  if (l == NULL) {
    perror("Could not allocate memory for conv2d layer");
    return NULL;
  }

  l->forward = _layer_forward_conv2d;
  l->backward = _layer_backward_conv2d;

  l->weights = create_matrix(kernel_rows, kernel_columns);
  l->bias = create_matrix(1, 1);

  if (l->weights == NULL || l->bias == NULL) {
    fprintf(stderr, "Error: Failed to allocate weights or bias for conv2d\n");
    free_matrix(l->weights);
    free_matrix(l->bias);
    free(l);
    return NULL;
  }

  float scale = sqrtf(2.0f / (float)(kernel_rows * kernel_columns));
  for (int i = 0; i < kernel_rows * kernel_columns; i++) {
    l->weights->data[i] =
        ((float)rand() / (float)RAND_MAX * 2.0f - 1.0f) * scale;
  }
  zero_matrix(l->bias);

  l->d_weight = NULL;
  l->d_bias = NULL;
  l->inputs = NULL;
  l->output = NULL;

  l->input_n = input_rows;
  l->output_n = input_rows - kernel_rows + 1;
  l->input_rows = input_rows;
  l->input_columns = input_columns;
  l->output_rows = input_rows - kernel_rows + 1;
  l->output_columns = input_columns - kernel_columns + 1;

  l->name = "Conv2d";
  return l;
}

Matrix *_layer_forward_dense(Layer *l, Matrix *input) {
  if (l == NULL || input == NULL || l->weights == NULL || l->bias == NULL) {
    return NULL;
  }
  // Free previous inputs to prevent memory leak
  if (l->inputs == NULL || l->inputs->rows != input->rows || l->inputs->columns != input->columns) {
    if (l->inputs != NULL) free_matrix(l->inputs);
    l->inputs = create_matrix(input->rows, input->columns);
  }
  if (l->inputs != NULL) {
    memcpy(l->inputs->data, input->data, input->rows * input->columns * sizeof(float));
  }

  // Multiply handles creating the output matrix, but we can reuse l->output if we want,
  // however multiply_mat always allocates. Let's let it allocate and just free the old one.
  if (l->output != NULL) {
    free_matrix(l->output);
  }

  Matrix *out = multiply_mat(l->weights, input);
  if (out == NULL) {
    fprintf(stderr,
            "Error: multiply_mat failed in dense forward. weights: (%d, %d), "
            "input: (%d, %d)\n",
            l->weights->rows, l->weights->columns, input->rows, input->columns);
    return NULL;
  }
  add_matrix(out, l->bias);
  l->output = out;

  // Return a copy so caller owns it
  return copy_matrix(out);
}

Matrix *_layer_backward_dense(Layer *l, Matrix *error_gradient,
                              float learning_rate) {
  if (l == NULL || l->inputs == NULL || error_gradient == NULL) {
    fprintf(stderr, "Error: NULL input to backward_dense\n");
    return NULL;
  }

  // 1. Calculate input gradient first (using old weights)
  Matrix *weights_t = transpose_mat(l->weights);
  if (weights_t == NULL) {
    fprintf(stderr, "Error: transpose_mat of weights failed in backward_dense\n");
    return NULL;
  }

  Matrix *input_gradient = multiply_mat(weights_t, error_gradient);
  if (input_gradient == NULL) {
    fprintf(stderr,
            "Error: input_gradient multiply failed. weights_t: (%d,%d), "
            "error_grad: (%d,%d)\n",
            weights_t->rows, weights_t->columns, error_gradient->rows,
            error_gradient->columns);
  }
  free_matrix(weights_t);

  // 2. Update weights and bias
  Matrix *input_t = transpose_mat(l->inputs);
  if (input_t == NULL) {
    fprintf(stderr, "Error: transpose_mat of inputs failed in backward_dense\n");
    return input_gradient;
  }

  Matrix *d_weights = multiply_mat(error_gradient, input_t);
  if (d_weights == NULL) {
    fprintf(stderr,
            "Error: d_weights multiply failed. error_grad: (%d,%d), input_t: "
            "(%d,%d)\n",
            error_gradient->rows, error_gradient->columns, input_t->rows,
            input_t->columns);
    free_matrix(input_t);
    return input_gradient;
  }

  // W = w - lr*dW
  scale_matrix(d_weights, -learning_rate);
  add_matrix(l->weights, d_weights);

  // Create a copy of error_gradient for bias update (don't mutate input)
  Matrix *d_bias = copy_matrix(error_gradient);
  if (d_bias != NULL) {
    scale_matrix(d_bias, -learning_rate);
    add_matrix(l->bias, d_bias);
    free_matrix(d_bias);
  }

  free_matrix(input_t);
  free_matrix(d_weights);

  return input_gradient;
}

Layer *layer_create_dense(int input_n, int output_n) {
  Layer *l = (Layer *)malloc(sizeof(Layer));

  if (l == NULL) {
    perror("Could Not allocate memory for layer. NULL");
    return NULL;
  }

  l->forward = _layer_forward_dense;
  l->backward = _layer_backward_dense;

  // Weights: (output_n × input_n) for multiplication with input (input_n × 1)
  l->weights = create_matrix(output_n, input_n);
  l->bias = create_matrix(output_n, 1);
  l->d_weight = NULL;
  l->d_bias = NULL;

  if (l->weights == NULL || l->bias == NULL) {
    fprintf(stderr, "Error: Failed to allocate weights or bias in layer_create_dense\n");
    if (l->weights != NULL) {
      free_matrix(l->weights);
    }
    if (l->bias != NULL) {
      free_matrix(l->bias);
    }
    free(l);
    return NULL;
  }

  // Xavier initialization: scale by sqrt(2 / (fan_in + fan_out)), centered at 0
  float scale = sqrtf(2.0f / (float)(input_n + output_n));
  int weight_count = output_n * input_n;
  for (int i = 0; i < weight_count; i++) {
    // Random value in [-scale, scale]
    l->weights->data[i] =
        ((float)rand() / (float)RAND_MAX * 2.0f - 1.0f) * scale;
  }
  zero_matrix(l->bias);

  l->inputs = NULL;
  l->output = NULL;

  l->input_n = input_n;
  l->output_n = output_n;
  l->input_rows = input_n;
  l->input_columns = 1;
  l->output_rows = output_n;
  l->output_columns = 1;

  l->name = "Dense";
  return l;
}

Matrix *_layer_forward_sigmoid(Layer *l, Matrix *input) {
  if (l == NULL || input == NULL) {
    return NULL;
  }
  // Free previous output to prevent memory leak
  if (l->output != NULL) {
    free_matrix(l->output);
  }

  // Sigmoid forward implementation
  Matrix *out = copy_matrix(input);

  for (int i = 0; i < out->rows * out->columns; i++) {
    out->data[i] = sigmoid(out->data[i]);
  }

  l->output = out;

  // Return a copy so caller owns it
  return copy_matrix(out);
}

Matrix *_layer_backward_sigmoid(Layer *l, Matrix *error_gradient,
                                float learning) {
  if (l == NULL || error_gradient == NULL || l->output == NULL) {
    return NULL;
  }
  Matrix *input_grad = copy_matrix(error_gradient);
  if (input_grad == NULL) {
    return NULL;
  }

  // returns derivative
  for (int i = 0; i < input_grad->columns * input_grad->rows; i += 1) {
    float s = l->output->data[i];
    input_grad->data[i] *= (s * (1.0f - s));
  }

  return input_grad;
}

Matrix *_layer_forward_relu(Layer *l, Matrix *input) {
  if (l == NULL || input == NULL) {
    return NULL;
  }
  // Free previous output to prevent memory leak
  if (l->output != NULL) {
    free_matrix(l->output);
  }

  // ReLU forward implementation
  Matrix *out = copy_matrix(input);

  for (int i = 0; i < out->rows * out->columns; i++) {
    out->data[i] = relu(out->data[i]);
  }

  l->output = out;

  // Return a copy so caller owns it
  return copy_matrix(out);
}

Matrix *_layer_backward_relu(Layer *l, Matrix *error_gradient, float learning) {
  if (l == NULL || error_gradient == NULL || l->output == NULL) {
    return NULL;
  }
  Matrix *input_grad = copy_matrix(error_gradient);
  if (input_grad == NULL) {
    return NULL;
  }

  // returns derivative
  for (int i = 0; i < input_grad->columns * input_grad->rows; i += 1) {
    float out_val = l->output->data[i];
    input_grad->data[i] *= (out_val > 0) ? 1.0f : 0.0f;
  }

  return input_grad;
}

Layer *layer_create_sigmoid() {
  Layer *l = (Layer *)malloc(sizeof(Layer));

  if (l == NULL) {
    perror("Could Not allocate memory for layer. NULL");
    return NULL;
  }

  l->forward = _layer_forward_sigmoid;
  l->backward = _layer_backward_sigmoid;

  l->weights = NULL;
  l->bias = NULL;
  l->d_weight = NULL;
  l->d_bias = NULL;

  l->inputs = NULL;
  l->output = NULL;

  l->input_n = 0;
  l->output_n = 0;
  l->input_rows = 0;
  l->input_columns = 0;
  l->output_rows = 0;
  l->output_columns = 0;

  l->name = "Sigmoid";
  return l;
}

Layer *layer_create_relu() {
  Layer *l = (Layer *)malloc(sizeof(Layer));

  if (l == NULL) {
    perror("Could Not allocate memory for layer. NULL");
    return NULL;
  }

  l->forward = _layer_forward_relu;
  l->backward = _layer_backward_relu;

  l->weights = NULL;
  l->bias = NULL;
  l->d_weight = NULL;
  l->d_bias = NULL;

  l->inputs = NULL;
  l->output = NULL;

  l->input_n = 0;
  l->output_n = 0;
  l->input_rows = 0;
  l->input_columns = 0;
  l->output_rows = 0;
  l->output_columns = 0;

  l->name = "ReLU";
  return l;
}

void free_layer(Layer *layer) {
  if (layer == NULL) {
    return;
  }

  free_matrix(layer->weights);
  free_matrix(layer->bias);
  free_matrix(layer->d_weight);
  free_matrix(layer->d_bias);

  // Note: layer->name points to string literals ("Dense", "Sigmoid")
  // which are in read-only memory and must NOT be freed

  if (layer->inputs != NULL) {
    free_matrix(layer->inputs);
  }
  if (layer->output != NULL) {
    free_matrix(layer->output);
  }
  free(layer);
}

// Wrapper functions that call the layer's function pointers
Matrix *layer_forward(Layer *l, Matrix *input) {
  if (l == NULL || l->forward == NULL) {
    return NULL;
  }
  return l->forward(l, input);
}

Matrix *layer_backward(Layer *l, Matrix *error_gradient, float learning_rate) {
  if (l == NULL || l->backward == NULL) {
    return NULL;
  }
  return l->backward(l, error_gradient, learning_rate);
}

void print_layer_info(Layer *l) {
  if (l == NULL) {
    printf("Layer: NULL\n");
    return;
  }
  printf("Layer: %s, Input: %dx%d, Output: %dx%d\n", l->name,
         l->input_rows, l->input_columns, l->output_rows, l->output_columns);
  size_t layer_size = sizeof(Layer);
  layer_size += get_matrix_size(l->inputs);
  layer_size += get_matrix_size(l->weights); 
  layer_size += get_matrix_size(l->bias); 
  layer_size += get_matrix_size(l->output);

  layer_size += get_matrix_size(l->d_weight); 
  layer_size += get_matrix_size(l->d_bias); 

  printf("Total space taken by layer: %lu bytes\n", layer_size);
}

#define LAYER_DENSE 0
#define LAYER_SIGMOID 1
#define LAYER_RELU 2
#define LAYER_CONV2D 3

int save_layer(Layer *l, FILE *fp) {
  if (fp == NULL || l == NULL) {
    return -1;
  }
  
  int type = -1;
  if (l->forward == _layer_forward_dense) type = LAYER_DENSE;
  else if (l->forward == _layer_forward_sigmoid) type = LAYER_SIGMOID;
  else if (l->forward == _layer_forward_relu) type = LAYER_RELU;
  else if (l->forward == _layer_forward_conv2d) type = LAYER_CONV2D;

  fwrite(&type, sizeof(int), 1, fp);
  
  if (type == LAYER_DENSE) {
    fwrite(&l->input_n, sizeof(int), 1, fp);
    fwrite(&l->output_n, sizeof(int), 1, fp);
    save_matrix(l->weights, fp);
    save_matrix(l->bias, fp);
  } else if (type == LAYER_CONV2D) {
    fwrite(&l->input_rows, sizeof(int), 1, fp);
    fwrite(&l->input_columns, sizeof(int), 1, fp);
    fwrite(&l->output_rows, sizeof(int), 1, fp);
    fwrite(&l->output_columns, sizeof(int), 1, fp);
    save_matrix(l->weights, fp);
    save_matrix(l->bias, fp);
  }
  
  return 0;
}

Layer *load_layer(FILE *fp) {
  if (fp == NULL) {
    return NULL;
  }
  
  int type = -1;
  if (fread(&type, sizeof(int), 1, fp) != 1) {
    return NULL; // Failed to read type
  }

  if (type == LAYER_DENSE) {
    int input_n = 0, output_n = 0;
    if (fread(&input_n, sizeof(int), 1, fp) != 1 ||
        fread(&output_n, sizeof(int), 1, fp) != 1) {
      return NULL;
    }
    Layer *l = layer_create_dense(input_n, output_n);
    if (l == NULL) {
      return NULL;
    }
    if (load_matrix(l->weights, fp) != 0 ||
        load_matrix(l->bias, fp) != 0) {
      free_layer(l);
      return NULL;
    }
    return l;
  } else if (type == LAYER_CONV2D) {
    int input_rows = 0, input_columns = 0, output_rows = 0, output_columns = 0;
    if (fread(&input_rows, sizeof(int), 1, fp) != 1 ||
        fread(&input_columns, sizeof(int), 1, fp) != 1 ||
        fread(&output_rows, sizeof(int), 1, fp) != 1 ||
        fread(&output_columns, sizeof(int), 1, fp) != 1) {
      return NULL;
    }
    int kernel_rows = input_rows - output_rows + 1;
    int kernel_columns = input_columns - output_columns + 1;
    Layer *l = layer_create_conv2d_shape(input_rows, input_columns,
                                         kernel_rows, kernel_columns);
    if (l == NULL) {
      return NULL;
    }
    if (load_matrix(l->weights, fp) != 0 ||
        load_matrix(l->bias, fp) != 0) {
      free_layer(l);
      return NULL;
    }
    return l;
  } else if (type == LAYER_SIGMOID) {
    return layer_create_sigmoid();
  } else if (type == LAYER_RELU) {
    return layer_create_relu();
  }
  
  return NULL;
}
