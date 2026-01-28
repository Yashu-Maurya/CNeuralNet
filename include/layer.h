#ifndef LAYER_H
#define LAYER_H

#include "matrix.h"

typedef struct {
    Matrix *inputs;
    Matrix *weights;
    Matrix *bias;
    Matrix *output;

    Matrix *d_weight;
    Matrix *d_bias;

    int input_n;
    int output_n;
} Layer;

Layer* layer_create(int input_size, int neurons);
void free_layer(Layer *layer);
Matrix* layer_forward(Layer *l, Matrix *input);
Matrix* layer_backward(Layer* l, Matrix* error_gradient, float learning_rate);
#endif