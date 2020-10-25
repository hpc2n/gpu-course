#include <stdlib.h>
#include <stdio.h>

#define CHECK_CUDA_ERROR(exp) {                     \
    cudaError_t ret = (exp);                        \
    if (ret != cudaSuccess) {                       \
        fprintf(stderr, "[error] %s:%d: %s (%s)\n", \
            __FILE__, __LINE__,                     \
            cudaGetErrorName(ret),                  \
            cudaGetErrorString(ret));               \
        exit(EXIT_FAILURE);                         \
    }                                               \
}

// a kernel that prints the contents of an array
__global__ void print_array(int n, int *array)
{
    int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (thread_id < n)
        printf("array[%d] = %d\n", thread_id, array[thread_id]);
}

int main(int argc, char const **argv)
{
    cudaError_t ret;
    
    // array size
    int n = 32;
    
    // initialize host memory
    int *host = (int*) malloc(n*sizeof(int));
    for (int i = 0; i < n; i++)
        host[i] = i;

    // allocate device memory
    int *device;
    ret = cudaMalloc(&device, n*sizeof(int));
    if (ret != cudaSuccess) {
        printf("Function cudaMalloc failed\n");
        printf("Error code: %s\n", cudaGetErrorName(ret));
        printf("Error message: %s\n", cudaGetErrorString(ret));
        exit(EXIT_FAILURE);
    }
    
    // move data from **host to device**
    //ret = cudaMemcpy(device, host, n*sizeof(int), cudaMemcpyDeviceToHost);
    ret = cudaMemcpy(device, host, n*sizeof(int), cudaMemcpyHostToDevice);
    if (ret != cudaSuccess) {
        printf("Function cudaMemcpy failed\n");
        printf("Error code: %s\n", cudaGetErrorName(ret));
        printf("Error message: %s\n", cudaGetErrorString(ret));
        exit(EXIT_FAILURE);
    }

    // call the kernel
    dim3 threads = 32;
    dim3 blocks = (n+threads.x-1)/threads.x; 
    print_array<<<blocks, threads>>>(n, device);
    
    ret = cudaGetLastError();
    if (ret != cudaSuccess) {
        printf("Kernel launch failed\n");
        printf("Error code: %s\n", cudaGetErrorName(ret));
        printf("Error message: %s\n", cudaGetErrorString(ret));
        exit(EXIT_FAILURE);
    }

    // wait until the GPU has executed the kernel
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    
    return EXIT_SUCCESS;
}
