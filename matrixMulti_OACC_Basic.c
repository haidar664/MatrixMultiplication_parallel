#include <stdio.h>
#include <stdlib.h>

void computeAcc(float *P, const float *M, const float *N, int Mh, int Mw, int Nw) {
    #pragma acc parallel loop copyin(M[0:Mh*Mw]) copyin(N[0:Mw*Nw]) copyout(P[0:Mh*Nw]) 
    for (int i = 0; i < Mh; i++) {
        #pragma acc loop 
        for (int j = 0; j < Nw; j++) {
            float sum = 0;
            for (int k = 0; k < Mw; k++) {
                float a = M[i * Mw + k];
                float b = N[k * Nw + j];
                sum += a * b;
            }
            P[i * Nw + j] = sum;
        }
    }
}

int main() {
    int Mh, Mw, Nw;
    printf("Enter the number of rows for matrix M: ");
    scanf("%d", &Mh);
    printf("Enter the number of columns for matrix M: ");
    scanf("%d", &Mw);
    printf("Enter the number of columns for matrix N: ");
    scanf("%d", &Nw);
    float *M = (float *)malloc(Mh * Mw * sizeof(float));
    float *N = (float *)malloc(Mw * Nw * sizeof(float));
    float *P = (float *)malloc(Mh * Nw * sizeof(float));
    for (int i = 0; i < Mh * Mw; i++)
        M[i] = (float)(i + 1);
    for (int i = 0; i < Mw * Nw; i++)
        N[i] = (float)(i + 1);
    computeAcc(P, M, N, Mh, Mw, Nw);
    printf("Resultant matrix P:\n");
    for (int i = 0; i < Mh; i++) {
        for (int j = 0; j < Nw; j++) {
            printf("%.2f ", P[i * Nw + j]);
        }
        printf("\n");
    }
    free(M);
    free(N);
    free(P);

    return 0;
}
