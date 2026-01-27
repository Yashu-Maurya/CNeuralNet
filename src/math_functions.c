#include "math_functions.h"

float sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}