#include "../include/layer.h"

Matrix* _layer_forward_dense(Layer* l, Matrix* input) {
    l->inputs = input;
    
    // Free previous output to prevent memory leak
    if (l->output != NULL) {
        free_matrix(l->output);
    }
    
    Matrix* out = multiply_mat(l->weights, input);
    add_matrix(out, l->bias);
    l->output = out;
    return out;
}

Matrix* _layer_backward_dense(Layer* l, Matrix* error_gradient, float learning_rate) {
    Matrix* input_t = transpose_mat(l->inputs);

    Matrix* d_weights = multiply_mat(error_gradient, input_t);

    // W= w - lr*dW

    scale_matrix(d_weights, -learning_rate);
    add_matrix(l->weights, d_weights);

    Matrix* d_bias =
        create_matrix(error_gradient->rows, error_gradient->columns);

    scale_matrix(error_gradient, -learning_rate);
    // Copy all elements, not just the first one
    for (int i = 0; i < error_gradient->rows * error_gradient->columns; i++) {
        d_bias->data[i] = error_gradient->data[i];
    }
    add_matrix(l->bias, d_bias);

    // B = b- lr*dB

    Matrix* weights_t = transpose_mat(l->weights);
    Matrix* input_gradient = multiply_mat(weights_t, error_gradient);

    free_matrix(input_t);
    free_matrix(d_weights);
    free_matrix(d_bias);
    free_matrix(weights_t);

    return input_gradient;
}

Layer* layer_create_dense(int input_n, int output_n) {
    Layer* l = (Layer*)malloc(sizeof(Layer));

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

    randomize_matrix(l->weights);
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

Matrix* _layer_forward_sigmoid(Layer* l, Matrix* input) {
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
    return out;
}

Matrix* _layer_backward_sigmoid(Layer* l, Matrix* error_gradient, float learning) {
        Matrix* input_grad = copy_matrix(error_gradient);

        // returns derivative
        for(int i = 0;i < input_grad->columns*input_grad->rows;i +=1 ) {
            float s = l->output->data[i];
            input_grad->data[i] *= (s * (1.0f - s));
        } 

        return input_grad;
}

Layer* layer_create_sigmoid() {
    Layer* l = (Layer*)malloc(sizeof(Layer));

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

void free_layer(Layer* layer) {
    if (layer == NULL) {
        return;
    }

    free_matrix(layer->weights);
    free_matrix(layer->bias);
    free_matrix(layer->d_weight);
    free_matrix(layer->d_bias);

    // Note: layer->name points to string literals ("Dense", "Sigmoid")
    // which are in read-only memory and must NOT be freed

    if (layer->output != NULL) {
        free_matrix(layer->output);
    }
    free(layer);
}

// Wrapper functions that call the layer's function pointers
Matrix* layer_forward(Layer* l, Matrix* input) {
    if (l == NULL || l->forward == NULL) {
        return NULL;
    }
    return l->forward(l, input);
}

Matrix* layer_backward(Layer* l, Matrix* error_gradient, float learning_rate) {
    if (l == NULL || l->backward == NULL) {
        return NULL;
    }
    return l->backward(l, error_gradient, learning_rate);
}