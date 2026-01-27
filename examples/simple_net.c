#include <stdio.h>
#include <stdlib.h>

#include "../include/matrix.h"

#define LEARNING_RATE 0.1
#define EPOCHS 1000

int main() {
    // f(x) = 2x
    Matrix* inputs = create_matrix(1, 1);   // a 1x1 input matrix.
    Matrix* weights = create_matrix(1, 1);  // 1x1 weight matrix.
    Matrix* bias = create_matrix(1, 1);     // 1x1 bias matrix

    // Activation(InputxWeights+Bias)

    randomize_matrix(weights);
    randomize_matrix(bias);

    for (int j = 0; j < EPOCHS; j++) {
        for (int i = 0; i < 10; i++) {
            inputs->data[0] = i / 20.0f;
            float target = i * 2.0f / 20.f;

            Matrix* out;
            out = multiply_mat(weights, inputs);
            add_matrix(out, bias);
            matrix_sigmoid(out);

            float error = target - out->data[0];
            print_matrix(out);
            printf("Error %i: %f\t", i, error);

            float derivative = out->data[0] * (1 - out->data[0]);
            printf("Derivative %i: %f\t", i, derivative);

            float gradient = error * derivative * inputs->data[0];
            printf("Gradient %i: %f\t", i, gradient);

            float bias_gradient = error * derivative * 1;
            printf("Bias Gradient %i: %f\t", i, bias_gradient);

            add_scaler(weights, (LEARNING_RATE * gradient));
            add_scaler(bias, (LEARNING_RATE * bias_gradient));

            free_matrix(out);
        }
        
        printf("\nEPOCH %i DONE \n", j);
    }
        inputs->data[0] = 5 / 20.0f;
        Matrix* out;
        out = multiply_mat(weights, inputs);
        add_matrix(out, bias);
        matrix_sigmoid(out);
        printf("\n\nFinal Inference Test for 5: ");
        out->data[0] = out->data[0] * 20.0f;
        print_matrix(out);
        free_matrix(out);

    return 0;
}