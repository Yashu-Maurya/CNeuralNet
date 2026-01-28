#include "../include/matrix.h"

Matrix* create_matrix(int rows, int columns) {
    Matrix* m = malloc(sizeof(Matrix));
    m->rows = rows;
    m->columns = columns;
    m->data = malloc(sizeof(float) * rows * columns);

    return m;
}

void free_matrix(Matrix* m) {
    free(m->data);
    free(m);
    m = NULL;
}

void randomize_matrix(Matrix* m) {
    int n = m->rows * m->columns;
    for (int i = 0; i < n; i++) {
        (m->data[i]) = (float)rand() / (float)RAND_MAX;
    }
}

void print_matrix(Matrix* m) {
    int n = m->rows * m->columns;
    for (int i = 0; i < n; i++) {
        if (i % m->columns == 0) {
            printf("\n");
        }
        printf("%f\t", m->data[i]);
    }
}

Matrix* multiply_mat(Matrix *m1, Matrix *m2) {
if (m1->columns != m2->rows) {
        printf("Error: Incompatible dimensions for multiplication\n");
        return NULL;
    }

    Matrix *result = create_matrix(m1->rows, m2->columns);

    for (int i = 0; i < m1->rows; i++) {
        for (int j = 0; j < m2->columns; j++) {
            
            float sum = 0;
            
            for (int k = 0; k < m2->rows; k++) {
                int a = i * m1->columns + k;
                int b = k * m2->columns + j;
                
                sum += m1->data[a] * m2->data[b];
            }
            
            int r = i * result->columns + j;
            result->data[r] = sum;
        }
    }

    return result;
}

void add_scaler(Matrix *m, float scaler) {
    int n = m->rows * m->columns;
    for (int i = 0; i < n; i++) {
        m->data[i] += scaler;
    }
}

void subtract_scaler(Matrix *m, float scaler) {
    int n = m->rows * m->columns;
    for (int i = 0; i < n; i++) {
        m->data[i] -= scaler;
    }
}

void add_matrix(Matrix *m1, Matrix *m2) {
    if (m1->rows != m2->rows || m1->columns != m2->columns) {
        printf("Error: Incompatible dimensions for addition\n");
        return;
    }

    int n = m1->rows * m1->columns;
    for (int i = 0; i < n; i++) {
        m1->data[i] += m2->data[i];
    }
}

void subtract_matrix(Matrix *m1, Matrix *m2) {
    if (m1->rows != m2->rows || m1->columns != m2->columns) {
        printf("Error: Incompatible dimensions for subtraction\n");
        return;
    }

    int n = m1->rows * m1->columns;
    for (int i = 0; i < n; i++) {
        m1->data[i] -= m2->data[i];
    }
}

void matrix_sigmoid(Matrix *m) {
     int n = m->rows * m->columns;
    for(int i = 0; i < n; i++) {
        m->data[i] = sigmoid(m->data[i]);
    }
}

void zero_matrix(Matrix *m) {
    int n = m->rows * m->columns;
    for(int i = 0; i < n; i++) {
        m->data[i] = 0;
    }
}