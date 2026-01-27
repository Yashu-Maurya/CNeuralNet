#include <stdio.h>
#include "../include/matrix.h"

int main() {

    Matrix *m1 = create_matrix(3,2);
    Matrix *m2 = create_matrix(2,3);
    Matrix *r;
    randomize_matrix(m1);
    randomize_matrix(m2);
    
    print_matrix(m1);
    puts("");
    print_matrix(m2);
    puts("");
    r = multiply_mat(m1,m2);
    print_matrix(r);
    puts("");

    free_matrix(m1);
    free_matrix(m2);
    free_matrix(r);

    return 0;
}