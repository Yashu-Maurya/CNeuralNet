#include "../include/network.h"
#include <stdio.h>
#include <stdlib.h>

Network* create_network() {
    Network* n = (Network*)malloc(sizeof(Network));
    if (n == NULL) {
        perror("Error Assigning Memory for Network.\n");
        return NULL;
    }
    n->layers = NULL;
    n->layer_count = 0;
    return n;
}

void free_network(Network* n) {
    if (n == NULL) {
        perror("Can't Find Network to free from memory\n");
        return;
    }

    for (int i = 0; i < n->layer_count; i += 1) {
        if (n->layers[i] != NULL) {
            free_layer(n->layers[i]);
        }
    }

    if (n->layers != NULL) {
        free(n->layers);
    }

    free(n);
    return;
}

void add_layer(Network* n, Layer* l) {
    if (l == NULL) {
        perror("Error Adding Layer, Layer is NULL\n");
        return;
    }

    if (n == NULL) {
        perror("Error Adding Layer, Network is NULL\n");
        return;
    }

    int nc = n->layer_count + 1;
    Layer** temp = realloc(n->layers, nc * sizeof(Layer*));
    if (temp == NULL) {
        perror("Error Reallocating memory to the layer pointer.\n");
        return;
    }

    n->layers = temp;
    n->layers[n->layer_count] = l;
    n->layer_count = nc;

    return;
}

Matrix* predict_network(Network* n, Matrix* input) {
    if (n == NULL) {
        perror("Network is NULL, Can't predict. \n");
        return NULL;
    }
    if (input == NULL) {
        perror("Input is Empty \n");
        return NULL;
    }
    
    if (n->layer_count == 0) {
        return copy_matrix(input);
    }

    Matrix* out = layer_forward(n->layers[0], input);

    for (int i = 1; i < n->layer_count; i++) {
        Matrix* next_out = layer_forward(n->layers[i], out);
        free_matrix(out);  // Free the previous intermediate result
        out = next_out;
    }

    return out;
}

void train_network(Network* n, Matrix* input, Matrix* target, float learning_rate) {
    if (n == NULL || input == NULL || target == NULL) return;

    Matrix* prediction = predict_network(n, input);
    Matrix* loss_gradient = subtract_matrix(prediction, target);
    Matrix* current_gradient = loss_gradient;

    for (int i = n->layer_count - 1; i >= 0; i--) {
        Matrix* next_gradient = layer_backward(n->layers[i], current_gradient, learning_rate);
        
        if (i < n->layer_count - 1) {
             free_matrix(current_gradient);
        }
        
        current_gradient = next_gradient;
    }

    free_matrix(current_gradient);
    free_matrix(loss_gradient);
    free_matrix(prediction);
}