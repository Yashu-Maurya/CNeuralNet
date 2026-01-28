#include "../include/layer.h"

Layer* layer_create(int input_n, int output_n) {
    Layer* l = (Layer*)malloc(sizeof(Layer));

    if (l == NULL) {
        perror("Could Not allocate memory for layer. NULL");
        return NULL;
    }

    l->weights = create_matrix(input_n, output_n);
    l->bias = create_matrix(output_n, 1);
    l->d_weight = create_matrix(input_n, output_n);
    l->d_bias = create_matrix(output_n, 1);

    randomize_matrix(l->weights);
    zero_matrix(l->bias);

    zero_matrix(l->d_weight);
    zero_matrix(l->d_bias);

    l->inputs = NULL;
    l->output = NULL;

    l->input_n = input_n;
    l->output_n = output_n;

    return l;
}

void free_layer(Layer* layer) {
    if (layer == NULL) {
        return;
    }

    free_matrix(layer->weights);
    free_matrix(layer->bias);
    free_matrix(layer->d_weight);
    free_matrix(layer->d_bias);

    if (layer->output != NULL) {
        free_matrix(layer->output);
    }
    free(layer);
}

Matrix* layer_forward(Layer *l, Matrix *input) {
    l->inputs = input;
    

    if (l->output != NULL) {
        free_matrix(l->output);
    }
    
    l->output = multiply_mat(l->weights, l->inputs);
    add_matrix(l->output, l->bias);
    return l->output;
}
Matrix* layer_backward(Layer* l, Matrix* error_gradient, float learning_rate) {
    Matrix* input_t = transpose_mat(l->inputs);
    
    Matrix* d_weights = multiply_mat(error_gradient, input_t);
    

    scale_matrix(d_weights, -learning_rate);
    add_matrix(l->weights, d_weights);

    Matrix* d_bias = create_matrix(error_gradient->rows, error_gradient->columns);
    
    scale_matrix(error_gradient, -learning_rate);
    *d_bias->data = *error_gradient->data;
    add_matrix(l->bias, d_bias);
    
    Matrix* weights_t = transpose_mat(l->weights);
    Matrix* input_gradient = multiply_mat(weights_t, error_gradient);

    free_matrix(input_t);
    free_matrix(d_weights);
    free_matrix(d_bias);
    free_matrix(weights_t);

    return input_gradient;
}