#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char **argv) {
    int N=(argc>1) ? atoi(argv[1]):512;
    size_t size=N*N*sizeof(float);
    float *h_A=(float*)malloc(size);
    float *h_B=(float*)malloc(size);
    float *h_C=(float*)malloc(size);
    for (int i= 0;i< N * N;i++) {
        h_A[i]=1.0f;
        h_B[i]=1.0f;
    }
    float *d_A,*d_B,*d_C;
    cudaMalloc((void**)&d_A,size);
    cudaMalloc((void**)&d_B,size);
    cudaMalloc((void**)&d_C,size);
    cudaMemcpy(d_A,h_A,size,cudaMemcpyHostToDevice);
    cudaMemcpy(d_B,h_B,size,cudaMemcpyHostToDevice);
    cublasHandle_t handle;
    cublasCreate(&handle);
    const float alpha=1.0f;
    const float beta=0.0f;
    cudaEvent_t start,stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cublasSgemm(handle,CUBLAS_OP_N,CUBLAS_OP_N,N,N,N,&alpha,d_B,N,d_A,N,&beta,d_C,N);
    cudaDeviceSynchronize();
    cudaEventRecord(start);
    cublasSgemm(handle,CUBLAS_OP_N,CUBLAS_OP_N,N,N,N,&alpha,d_B,N,d_A,N,&beta,d_C,N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float elapsed_ms;
    cudaEventElapsedTime(&elapsed_ms,start,stop);
    cudaMemcpy(h_C,d_C,size,cudaMemcpyDeviceToHost);
    printf("cuBLAS SGEMM execution time (N=%d): %f ms\n",N,elapsed_ms);
    cublasDestroy(handle);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);
    return 0;
}
