#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

__global__ void matrixMultiplyGPU(float *A,float *B,float *C,int N) {
    int row =blockIdx.y *blockDim.y+threadIdx.y;
    int col =blockIdx.x*blockDim.x+threadIdx.x;
    if (row<N&& col<N) {
        float sum=0.0f;
        for (int k=0;k < N;k++) {
            sum += A[row*N+k]*B[k*N+col];
        }
        C[row*N+col]=sum;
    }
}
int main(int argc, char **argv) {
    int N=(argc>1) ? atoi(argv[1]):512;
    size_t size=N*N*sizeof(float);
    float *h_A=(float*)malloc(size);
    float *h_B  =(float*)malloc(size);
    float *h_C= (float*)malloc(size);
    for (int i=0;i<N*N;i++) {
        h_A[i]=1.0f;
        h_B[i]=1.0f;
    }
    float *d_A,*d_B,*d_C;
    cudaMalloc((void**)&d_A,size);
    cudaMalloc((void**)&d_B,size);
    cudaMalloc((void**)&d_C,size);
    cudaMemcpy(d_A,h_A,size,cudaMemcpyHostToDevice);
    cudaMemcpy(d_B,h_B,size,cudaMemcpyHostToDevice);
    dim3 blockDim(16,16);
    dim3 gridDim(
        (N +blockDim.x - 1)/blockDim.x,
        (N+  blockDim.y - 1)/blockDim.y
    );
    matrixMultiplyGPU<<<gridDim, blockDim>>>(d_A,d_B,d_C,N);
    cudaDeviceSynchronize(); 
    cudaEvent_t start,stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    matrixMultiplyGPU<<<gridDim, blockDim>>>(d_A,d_B,d_C,N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float elapsed_ms;
    cudaEventElapsedTime(&elapsed_ms,start,stop);
    cudaMemcpy(h_C,d_C,size,cudaMemcpyDeviceToHost);
    printf("Naive CUDA execution time (N=%d): %f ms\n", N, elapsed_ms);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);
    return 0;
}
