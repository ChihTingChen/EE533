#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#define M 512
#define N 7
__global__ void conv2d_cuda(unsigned int *image,int *kernel,unsigned int *output);
extern "C" __declspec(dllexport)
void conv2d_cuda_lib(unsigned int *h_img,unsigned int *h_out){
    int out_size=M-N+1;
    size_t img_bytes =M*M*sizeof(unsigned int);
    size_t out_bytes =out_size*out_size*sizeof(unsigned int);
    size_t ker_bytes =N*N*sizeof(int);
    int h_kernel[49]={-1,-2,-3,0,3,2,1,-2,-4,-6,0,6,4,2,-3,-6,-9,0,9,6,3,-4,-8,-12,0,12,8,4,-3,-6,-9,0,9,6,3,-2,-4,-6,0,6,4,2,-1,-2,-3,0,3,2,1};
    unsigned int *d_img,*d_out;
    int *d_kernel;
    cudaMalloc((void**)&d_img,img_bytes);
    cudaMalloc((void**)&d_out,out_bytes);
    cudaMalloc((void**)&d_kernel,ker_bytes);
    cudaMemcpy(d_img,h_img,img_bytes,cudaMemcpyHostToDevice);
    cudaMemcpy(d_kernel,h_kernel,ker_bytes,cudaMemcpyHostToDevice);
    dim3 block(16,16);
    dim3 grid((out_size+15)/16,(out_size+15)/16);
    conv2d_cuda<<<grid,block>>>(d_img,d_kernel,d_out);
    cudaDeviceSynchronize();
    cudaMemcpy(h_out,d_out,out_bytes,cudaMemcpyDeviceToHost);
    cudaFree(d_img);
    cudaFree(d_out);
    cudaFree(d_kernel);
}
__global__ void conv2d_cuda(unsigned int *image,int *kernel,unsigned int *output){
    int x=blockIdx.x*blockDim.x +threadIdx.x;
    int y=blockIdx.y*blockDim.y +threadIdx.y;
    int out_size=M-N+1;
    if(x<out_size&&y<out_size){
        int sum=0;
        for(int ki=0;ki<N;ki++){
            for(int kj=0;kj<N;kj++){
                sum += image[(y+ki)*M + (x+kj)] * kernel[ki*N + kj];
            }
        }
        if(sum<0) sum=-sum;
        output[y*out_size+x]=(unsigned int)sum;
    }
}
