#ifndef NETWORK_H
#define NETWORK_H

#include "layer.h"

typedef struct Network Network;

struct Network {
    Layer **layers;
    int layer_count;
};

Network* create_network();
void add_layer(Network *n, Layer *l);
void free_network(Network *n);
void train_network(Network *n, Matrix *inputs, Matrix* targets, float learning_rate);
Matrix* predict_network(Network *n, Matrix *input);
void print_network_info(Network *n);

#endif