#ifndef LAYER_H
#define LAYER_H

#include "matrix.h"

typedef struct Layer Layer;

typedef Matrix* (*ForwardFunction)(struct Layer *l, Matrix *input);
typedef Matrix* (*BackwardFunction)(struct Layer* l, Matrix* error_gradient, float learning);


struct Layer{

    ForwardFunction forward;
    BackwardFunction backward;

    Matrix *inputs;
    Matrix *weights;
    Matrix *bias;
    Matrix *output;

    Matrix *d_weight;
    Matrix *d_bias;

    int input_n;
    int output_n;

    char *name; // FOR REFERENCE ONLY
};

// Layer* layer_create(int input_n, int output_n);
Layer* layer_create_dense(int input_n, int output_n);
Layer* layer_create_sigmoid();
void free_layer(Layer *layer);
Matrix* layer_forward(Layer *l, Matrix *input);
Matrix* layer_backward(Layer* l, Matrix* error_gradient, float learning_rate);
#endif