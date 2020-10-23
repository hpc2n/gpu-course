#include <stdlib.h>
#include <stdio.h>

// CUDA kernel
__global__ void say_hello()
{
    // a CUDA core executes this line
    printf("GPU says, Hello world!\n");
}

// CUDA kernel for step 2
__global__ void say_hello2()
{
    // a CUDA core executes this line
    printf("Thread %d says, Hello world!\n", threadIdx.x);
}

// CUDA kernel for step 3
__global__ void say_hello3()
{
    // a CUDA core executes this line
    printf("Thread %d in block %d says, Hello world!\n",
        threadIdx.x, blockIdx.x);
}

// CUDA kernel for step 4
__global__ void say_hello4()
{
    // a CUDA core executes this line
    if (blockIdx.x == 0)
        printf("Thread %d in block %d says, Hello world!\n",
            threadIdx.x, blockIdx.x);
    else
        printf("Thread %d in block %d says, Hello Umea!\n",
            threadIdx.x, blockIdx.x);
}

// CUDA kernel for step 5
__global__ void say_hello5()
{
    // a CUDA core executes this line
    int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    printf("Thread %d says, Hello world!\n", thread_id);
}

// CUDA kernel for step 6
__global__ void say_hello6()
{
    // a CUDA core executes this line
    printf("Thread (%d,%d) in block (%d,%d) says, Hello world!\n",
        threadIdx.x, threadIdx.y, blockIdx.x, blockIdx.y);
}

int main()
{
    // a CPU core executes these lines

    printf("====== Step 1 ======\n");
    
    printf("Host says, Hello world!\n");
    
    // call the kernel
    say_hello<<<1,1>>>();
    
    // wait until the GPU has executed the kernel
    cudaDeviceSynchronize();
    
    printf("====== Step 2 ======\n");
    
    printf("Host says, Hello world!\n");
    say_hello2<<<1,16>>>();
    cudaDeviceSynchronize();
    
    printf("====== Step 3 ======\n");
    
    printf("Host says, Hello world!\n");
    say_hello3<<<2,16>>>();
    cudaDeviceSynchronize();
    
    printf("====== Step 4 ======\n");
    
    printf("Host says, Hello world!\n");
    say_hello4<<<2,16>>>();
    cudaDeviceSynchronize();

    printf("====== Step 5 ======\n");
    
    printf("Host says, Hello world!\n");
    say_hello5<<<2,16>>>();
    cudaDeviceSynchronize();
    
    printf("====== Step 6 ======\n");
    
    printf("Host says, Hello world!\n");
    dim3 threads(4, 2);
    dim3 blocks(2, 3);
    say_hello6<<<blocks,threads>>>();
    cudaDeviceSynchronize();
    
    printf("====== Step 7 ======\n");
    
    printf("Host says, Hello world!\n");
    //dim3 threads(4, 2);
    //dim3 blocks(2, 3);
    say_hello6<<<blocks,threads>>>();
    
    return EXIT_SUCCESS;
}
