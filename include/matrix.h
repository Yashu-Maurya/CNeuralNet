#ifndef MATRIX_H
#define MATRIX_H

#include <stdio.h>
#include <stdlib.h>

#include "math_functions.h"

typedef struct {
  int rows;
  int columns;
  float *data;
} Matrix;

Matrix *create_matrix(int rows, int columns);
void free_matrix(Matrix *m);
void randomize_matrix(Matrix *m);
void print_matrix(Matrix *m);
Matrix *multiply_mat(Matrix *m1, Matrix *m2);
void add_scaler(Matrix *m, float scaler);
void subtract_scaler(Matrix *m, float scaler);
void add_matrix(Matrix *m1, Matrix *m2);
Matrix* subtract_matrix(Matrix *m1, Matrix *m2);
void matrix_sigmoid(Matrix *m);
void zero_matrix(Matrix *m);
void scale_matrix(Matrix *m, float scaler);
Matrix *copy_matrix(Matrix *m);
Matrix *transpose_mat(Matrix *m);
#endif