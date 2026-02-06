#include "../include/layer.h"

Matrix *_layer_forward_dense(Layer *l, Matrix *input) {
  // Free previous inputs to prevent memory leak
  if (l->inputs != NULL) {
    free_matrix(l->inputs);
  }
  // Store a copy of input (not just pointer) for use in backward pass
  l->inputs = copy_matrix(input);

  // Free previous output to prevent memory leak
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

  Matrix *input_t = transpose_mat(l->inputs);
  if (input_t == NULL) {
    fprintf(stderr, "Error: transpose_mat failed in backward_dense\n");
    return NULL;
  }

  Matrix *d_weights = multiply_mat(error_gradient, input_t);
  if (d_weights == NULL) {
    fprintf(stderr,
            "Error: d_weights multiply failed. error_grad: (%d,%d), input_t: "
            "(%d,%d)\n",
            error_gradient->rows, error_gradient->columns, input_t->rows,
            input_t->columns);
    free_matrix(input_t);
    return NULL;
  }

  // W = w - lr*dW
  scale_matrix(d_weights, -learning_rate);
  add_matrix(l->weights, d_weights);

  // Create a copy of error_gradient for bias update (don't mutate input)
  Matrix *d_bias = copy_matrix(error_gradient);
  if (d_bias == NULL) {
    free_matrix(input_t);
    free_matrix(d_weights);
    return NULL;
  }
  scale_matrix(d_bias, -learning_rate);
  add_matrix(l->bias, d_bias);

  // B = b - lr*dB

  Matrix *weights_t = transpose_mat(l->weights);
  if (weights_t == NULL) {
    free_matrix(input_t);
    free_matrix(d_weights);
    free_matrix(d_bias);
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

  free_matrix(input_t);
  free_matrix(d_weights);
  free_matrix(d_bias);
  free_matrix(weights_t);

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
  l->d_weight = create_matrix(output_n, input_n);
  l->d_bias = create_matrix(output_n, 1);

  // Xavier initialization: scale by sqrt(2 / (fan_in + fan_out)), centered at 0
  float scale = sqrtf(2.0f / (float)(input_n + output_n));
  int weight_count = output_n * input_n;
  for (int i = 0; i < weight_count; i++) {
    // Random value in [-scale, scale]
    l->weights->data[i] =
        ((float)rand() / (float)RAND_MAX * 2.0f - 1.0f) * scale;
  }
  zero_matrix(l->bias);

  zero_matrix(l->d_weight);
  zero_matrix(l->d_bias);

  l->inputs = NULL;
  l->output = NULL;

  l->input_n = input_n;
  l->output_n = output_n;

  l->name = "Dense";
  return l;
}

Matrix *_layer_forward_sigmoid(Layer *l, Matrix *input) {
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
  Matrix *input_grad = copy_matrix(error_gradient);

  // returns derivative
  for (int i = 0; i < input_grad->columns * input_grad->rows; i += 1) {
    float s = l->output->data[i];
    input_grad->data[i] *= (s * (1.0f - s));
  }

  return input_grad;
}

Matrix *_layer_forward_relu(Layer *l, Matrix *input) {
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
  Matrix *input_grad = copy_matrix(error_gradient);

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