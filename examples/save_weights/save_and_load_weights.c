#include <stdio.h>
#include <stdlib.h>

#include "../../include/layer.h"
#include "../../include/network.h"

int main() {
  Network *n = create_network();
  Layer *l1 = layer_create_dense(10, 20);
  Layer *l2 = layer_create_dense(20, 10);
  add_layer(n, l1);
  add_layer(n, l2);
  add_layer(n, layer_create_sigmoid());
  save_network(n, "model.cnet");
  free_network(n);

  Network *n2 = create_network();
  load_network(n2, "model.cnet");
  print_network_info(n2);
  free_network(n2);

  return 0;
}