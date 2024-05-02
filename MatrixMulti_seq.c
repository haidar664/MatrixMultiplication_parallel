#include <stdio.h>
#include <stdlib.h>

void matrixMul(float* A, float* B, float* C, int M, int N, int P) {
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < P; ++j) {
            float sum = 0.0;
            for (int k = 0; k < N; ++k) {
                sum += A[i * N + k] * B[k * P + j];
            }
            C[i * P + j] = sum;
        }
    }
}

int main() {
    int M, N, P;
    printf("Enter the dimensions of matrices M, N, P (space-separated): ");
    scanf("%d %d %d", &M, &N, &P);

    float *A = (float*)malloc(M * N * sizeof(float));
    float *B = (float*)malloc(N * P * sizeof(float));
    float *C = (float*)malloc(M * P * sizeof(float));

    for (int i = 0; i < M * N; ++i) {
        A[i] = 1.0f; 
    }
    for (int i = 0; i < N * P; ++i) {
        B[i] = 2.0f;
    }

    matrixMul(A, B, C, M, N, P);

    printf("Result matrix C:\n");
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < P; ++j) {
            printf("%.2f ", C[i * P + j]);
        }
        printf("\n");
    }

    free(A);
    free(B);
    free(C);

    return 0;
}
