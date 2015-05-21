#include <stdio.h>
#include "data.h"

int main()
{
    float mat_C[SIZE_M*SIZE_P];
    for (int i = 0; i < SIZE_M; i++) {
        for (int j = 0; j < SIZE_P; j++) {
            float tmp = 0.0f;
            for (int k = 0; k < SIZE_N; k++) {
                tmp += mat_A[i * SIZE_N + k] * mat_B[k * SIZE_P + j];
            }
            mat_C[i * SIZE_P + j] = tmp;
        }
    }

    FILE *fp = fopen("normal.out", "w");
    for (int i = 0; i < SIZE_M; i++) {
        for (int j = 0; j < SIZE_P; j++) {
            fprintf(fp, "%.2f ", mat_C[i * SIZE_P + j]);
        }
        fprintf(fp, "\n");
    }
}

