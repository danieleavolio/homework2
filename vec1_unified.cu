#include <stdio.h>
#include <iostream>
#include <cuda.h>
using namespace std;

__global__ void vectorAddMono(float *d_A, float *d_B, float *d_C, int n)
{
    // Monolitic version
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    // The thread does only his job
    if (index < n)
        d_C[index] = d_A[index] + d_B[index];
}

__global__ void vectorAddGridStride(float *d_A, float *d_B, float *d_C, int n)
{
    // Monolitic version
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = index; i < n; i += stride)
        d_C[index] = d_A[index] + d_B[index];
}

__global__ void vectorInit(float *x, int n, float value)
{
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    if (index < n)
        x[index] = value;
}

void vecAdd(float *A, float *B, float *C, int n)
{

    int size = n * sizeof(float);
    int block_size = 32;
    int number_of_blocks = ceil(n / block_size);

    vectorInit<<<number_of_blocks, block_size>>>(A, n, 1.1f);
    vectorInit<<<number_of_blocks, block_size>>>(B, n, 2.0f);
    vectorInit<<<number_of_blocks, block_size>>>(C, n, 0.0f);
    // wait for things to be done
    vectorAddMono<<<number_of_blocks, block_size>>>(A, B, C, n);
    // wait for sync
    cudaDeviceSynchronize();
    
    
    cudaFree(A);
    cudaFree(B);
    cudaFree(C);
}

int main()
{

    int N = 1 << 24;
    float *A;
    float *B;
    float *C;

    cudaMallocManaged(&A, N * sizeof(float));
    cudaMallocManaged(&B, N * sizeof(float));
    cudaMallocManaged(&C, N * sizeof(float));

    vecAdd(A, B, C, N);
}