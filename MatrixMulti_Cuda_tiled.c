#include <stdio.h>

#define TILE_WIDTH 16 

__global__ void MatrixMulKernel(float* M, float* N, float* P, int Width) {
    __shared__ float ds_M[TILE_WIDTH][TILE_WIDTH];
    __shared__ float ds_N[TILE_WIDTH][TILE_WIDTH];
    
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    int Row = by * blockDim.y + ty;
    int Col = bx * blockDim.x + tx;
    
    float Pvalue = 0;
    
    for (int p = 0; p < Width / TILE_WIDTH; ++p) {
        ds_M[ty][tx] = M[Row * Width + p * TILE_WIDTH + tx];
        ds_N[ty][tx] = N[(p * TILE_WIDTH + ty) * Width + Col];
        
        __syncthreads();
        
        for (int i = 0; i < TILE_WIDTH; ++i) {
            Pvalue += ds_M[ty][i] * ds_N[i][tx];
        }
        
        __syncthreads();
    }
    
    P[Row * Width + Col] = Pvalue;
}

int main() {
    int MWidth, NWidth, PWidth;
    printf("Enter the dimensions of matrices M, N, P (space-separated): ");
    scanf("%d %d %d", &MWidth, &NWidth, &PWidth);
    
    float *h_M, *h_N, *h_P; 
    float *d_M, *d_N, *d_P; 
    
    h_M = (float*)malloc(MWidth * NWidth * sizeof(float));
    h_N = (float*)malloc(NWidth * PWidth * sizeof(float));
    h_P = (float*)malloc(MWidth * PWidth * sizeof(float));
    
    cudaMalloc((void**)&d_M, MWidth * NWidth * sizeof(float));
    cudaMalloc((void**)&d_N, NWidth * PWidth * sizeof(float));
    cudaMalloc((void**)&d_P, MWidth * PWidth * sizeof(float));
    
    for (int i = 0; i < MWidth * NWidth; ++i) {
        h_M[i] = 1.0f; 
    }
    for (int i = 0; i < NWidth * PWidth; ++i) {
        h_N[i] = 2.0f;
    }
    
    cudaMemcpy(d_M, h_M, MWidth * NWidth * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_N, h_N, NWidth * PWidth * sizeof(float), cudaMemcpyHostToDevice);
    
    dim3 threadsPerBlock(TILE_WIDTH, TILE_WIDTH);
    dim3 blocksPerGrid((PWidth + threadsPerBlock.x - 1) / threadsPerBlock.x, (MWidth + threadsPerBlock.y - 1) / threadsPerBlock.y);
    
    MatrixMulKernel<<<blocksPerGrid, threadsPerBlock>>>(d_M, d_N, d_P, MWidth);
    
    cudaMemcpy(h_P, d_P, MWidth * PWidth * sizeof(float), cudaMemcpyDeviceToHost);
    
    cudaFree(d_M);
    cudaFree(d_N);
    cudaFree(d_P);
    
    free(h_M);
    free(h_N);
    free(h_P);
    
    return 0;
}
