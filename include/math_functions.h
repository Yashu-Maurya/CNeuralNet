#ifndef MATH_FUNCTIONS_H
#define MATH_FUNCTIONS_H

#include <math.h>

float sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

#endif