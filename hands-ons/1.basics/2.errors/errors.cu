#include <stdlib.h>
#include <stdio.h>

// a kernel that prints the contents of an array
__global__ void print_array(int n, int *array)
{
    int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (thread_id < n)
        printf("array[%d] = %d\n", thread_id, array[thread_id]);
}

int main(int argc, char const **argv)
{
    // array size
    int n = 32;
    
    // initialize host memory
    int *host = (int*) malloc(n*sizeof(int));
    for (int i = 0; i < n; i++)
        host[i] = i;

    // allocate device memory
    int *device;
    cudaMalloc(&device, n*sizeof(int));
    
    // move data from host to device
    cudaMemcpy(device, host, n*sizeof(int), cudaMemcpyDeviceToHost);

    // call the kernel
    dim3 threads = 32;
    dim3 blocks = (n+threads.x-1)/threads.x; 
    print_array<<<blocks, threads>>>(n, device);

    // wait until the GPU has executed the kernel
    cudaDeviceSynchronize();
    
    return EXIT_SUCCESS;
}
