#include <stdlib.h>
#include <stdio.h>

// CUDA kernel
__global__ void say_hello()
{
    // a CUDA core executes this line
    printf("GPU says, Hello world!\n");
}

int main()
{
    // a CPU core executes these lines

    printf("Host says, Hello world!\n");

    // call the kernel
    say_hello<<<1,1>>>();

    // wait until the GPU has executed the kernel
    cudaDeviceSynchronize();

    return EXIT_SUCCESS;
}
