#include <stdio.h>
#include <stdlib.h>
#define M 512
#define N 3
void conv2d_cpu(unsigned int *image,int *kernel,unsigned int *output){
    int out_size=M-N+1;
    for(int i=0;i<out_size;i++){
        for(int j=0;j<out_size;j++){
            int sum=0;
            for(int ki=0;ki<N;ki++){
                for(int kj=0;kj<N;kj++){
                    sum+=image[(i+ki)*M+(j+kj)]*kernel[ki*N+kj];
                }
            }
            if(sum<0) sum=-sum;
            output[i*out_size+j]=(unsigned int)sum;
        }
    }
}
int main(){
    int out_size=M-N+1;
    unsigned int*image=malloc(M*M*sizeof(unsigned int));
    unsigned int *output=malloc(out_size*out_size*sizeof(unsigned int));
    int kernel[9]={-1,0,1,-2,0,2,-1,0,1};
    //int kernel[25]={-1,-2,0,2,1,-4,-8,0,8,4,-6,-12,0,12,6,-4,-8,0,8,4,-1,-2,0,2,1};
    //int kernel[49]={-1,-2,-3,0,3,2,1,-2,-4,-6,0,6,4,2,-3,-6,-9,0,9,6,3,-4,-8,-12,0,12,8,4,-3,-6,-9,0,9,6,3,-2,-4,-6,0,6,4,2,-1,-2,-3,0,3,2,1};
    FILE *fi=fopen("C:\\chihtinghw\\EE533\\Lab_file\\lab_CUDA\\cuda_lab\\512tunyuan.raw","rb");
    fread(image,sizeof(unsigned int),M*M,fi);
    fclose(fi);
    conv2d_cpu(image,kernel,output);
    FILE *fo=fopen("C:\\chihtinghw\\EE533\\Lab_file\\lab_CUDA\\cuda_lab\\512tunyuan3_conv.raw","wb");
    fwrite(output,sizeof(unsigned int),out_size*out_size,fo);
    fclose(fo);
    free(image);
    free(output);
    return 0;
}
