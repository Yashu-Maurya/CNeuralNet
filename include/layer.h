#ifndef LAYER_H
#define LAYER_H

#include "matrix.h"

typedef struct Layer Layer;

typedef Matrix *(*ForwardFunction)(struct Layer *l, Matrix *input);
typedef Matrix *(*BackwardFunction)(struct Layer *l, Matrix *error_gradient,
                                    float learning);

struct Layer {

  ForwardFunction forward;
  BackwardFunction backward;

  Matrix *inputs;
  Matrix *weights;
  Matrix *bias;
  Matrix *output;

  Matrix *d_weigh
  ;
  Matrix *d_bias;

  int input_n;
  int output_n;
  int input_rows;
  int input_columns;
  int output_rows;
  int output_columns;

  const char *name; // FOR REFERENCE ONLY
};

// Layer* layer_create(int input_n, int output_n);
Layer *layer_create_dense(int input_n, int output_n);
Layer *layer_create_sigmoid();
Layer *layer_create_relu();
Layer *layer_create_conv2d(int input_size, int kernel_size);
Layer *layer_create_conv2d_shape(int input_rows, int input_columns,
                                 int kernel_rows, int kernel_columns);
Matrix *_layer_forward_conv2d_with_kernel(Layer *l, Matrix *input, Matrix *kernel);
int layer_forward_conv2d_into(Layer *l, Matrix *input, Matrix *output);
int layer_forward_conv2d_with_kernel_into(Layer *l, Matrix *input,
                                          Matrix *kernel, Matrix *output);

void free_layer(Layer *layer);
Matrix *layer_forward(Layer *l, Matrix *input);
Matrix *layer_backward(Layer *l, Matrix *error_gradient, float learning_rate);

void print_layer_info(Layer *l);

int save_layer(Layer *l, FILE *fp);
Layer *load_layer(FILE *fp);


#endif
