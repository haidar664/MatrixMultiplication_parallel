#include <stdio.h>

__global__ void MatrixMulKernel(float* M, float* N, float* P, int MWidth, int NWidth, int PWidth) {
    int Row = blockIdx.y * blockDim.y + threadIdx.y;
    int Col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if ((Row < MWidth) && (Col < NWidth)) {
        float Pvalue = 0;
        for (int k = 0; k < PWidth; ++k) {
            Pvalue += M[Row * PWidth + k] * N[k * NWidth + Col];
        }
        P[Row * NWidth + Col] = Pvalue;
    }
}

int main() {
    int MWidth, NWidth, PWidth;
    printf("Enter the dimensions of matrices M, N, P (space-separated): ");
    scanf("%d %d %d", &MWidth, &NWidth, &PWidth);
    
    float *h_M, *h_N, *h_P; 
    float *d_M, *d_N, *d_P; 
    
    h_M = (float*)malloc(MWidth * PWidth * sizeof(float));
    h_N = (float*)malloc(PWidth * NWidth * sizeof(float));
    h_P = (float*)malloc(MWidth * NWidth * sizeof(float));
    
   
    cudaMalloc((void**)&d_M, MWidth * PWidth * sizeof(float));
    cudaMalloc((void**)&d_N, PWidth * NWidth * sizeof(float));
    cudaMalloc((void**)&d_P, MWidth * NWidth * sizeof(float));
    
   
    for (int i = 0; i < MWidth * PWidth; ++i) {
        h_M[i] = 1.0f; 
    }
    for (int i = 0; i < PWidth * NWidth; ++i) {
        h_N[i] = 2.0f;
    }
    
    cudaMemcpy(d_M, h_M, MWidth * PWidth * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_N, h_N, PWidth * NWidth * sizeof(float), cudaMemcpyHostToDevice);
    
    dim3 threadsPerBlock(16, 16); 
    dim3 blocksPerGrid((NWidth + threadsPerBlock.x - 1) / threadsPerBlock.x, (MWidth + threadsPerBlock.y - 1) / threadsPerBlock.y);
    
    MatrixMulKernel<<<blocksPerGrid, threadsPerBlock>>>(d_M, d_N, d_P, MWidth, NWidth, PWidth);
    
    cudaMemcpy(h_P, d_P, MWidth * NWidth * sizeof(float), cudaMemcpyDeviceToHost);
    
    cudaFree(d_M);
    cudaFree(d_N);
    cudaFree(d_P);
    
    free(h_M);
    free(h_N);
    free(h_P);
    
    return 0;
}
