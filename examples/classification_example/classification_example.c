#include "network.h"
#include "image.h"

int main() {

    Network* n = create_network();

    Image *img = read_image("examples/classification_example/Hotdog Not Hotdog Archive/hotdog-nothotdog/hotdog-nothotdog/train/hotdog/1423.jpg");

    return 0;
}