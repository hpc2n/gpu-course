//
// A simple CUDA program that multiplies a vector with a scalar.
//
// Author: Mirko Myllykoski, Umeå University, 2019
//

#include <stdlib.h>
#include <stdio.h>

// CUDA kernel
__global__ void ax_kernel(int n, double alpha, double *x)
{
    // Query the global thread index. Each thread block contains blockDim.x
    // threads, the index number of the current thread block is blockIdx.x and
    // the index number of the current thread inside the current thread block
    // is threadIdx.x.
    int thread_id = blockIdx.x * blockDim.x + threadIdx.x;

    // each thread updates one row
    if (thread_id < n)
        x[thread_id] = alpha * x[thread_id];
}

int main(int argc, char const **argv)
{
    const double alpha = 2.0;

    // read and validate the command line arguments

    if (argc < 2) {
        fprintf(stderr, "[error] No vector lenght was supplied.\n");
        return EXIT_FAILURE;
    }

    size_t n = atof(argv[1]);
    if (n < 1) {
        fprintf(stderr, "[error] The vector lenght was invalid.\n");
        return EXIT_FAILURE;
    }

    // allocate host memory

    double *x;
    if ((x = (double *) malloc(n*sizeof(double))) == NULL) {
        fprintf(stderr,
            "[error] Failed to allocate host memory for vector x.\n");
        return EXIT_FAILURE;
    }

    // initialize memory

    for (int i = 0; i < n; i++)
        x[i] = i;

    // allocate device memory

    double *d_x;
    if (cudaMalloc(&d_x, n*sizeof(double)) != cudaSuccess) {
        fprintf(stderr,
            "[error] Failed to allocate device memory for vector x.\n");
        return EXIT_FAILURE;
    }

    // copy the vector from the host memory to the device memory

    if (cudaMemcpy(d_x, x, n*sizeof(double), cudaMemcpyHostToDevice) != cudaSuccess) {
        fprintf(stderr,
            "[error] Failed to copy x to device memory.\n");
        return EXIT_FAILURE;
    }

    // launch the kernel

    dim3 threads = 256;
    dim3 blocks = (n+threads.x)/threads.x;
    ax_kernel<<<blocks, threads>>>(n, alpha, d_x);

    if (cudaGetLastError()  != cudaSuccess) {
        fprintf(stderr,
            "[error] Failed to launch the kernel.\n");
        return EXIT_FAILURE;
    }

    // copy the vector from the device memory to the host memory

    if (cudaMemcpy(x, d_x, n*sizeof(double), cudaMemcpyDeviceToHost) != cudaSuccess) {
        fprintf(stderr,
            "[error] Failed to copy x from from memory.\n");
        return EXIT_FAILURE;
    }

    // wait until the GPU has finished computing and transfering the data

    if (cudaDeviceSynchronize() != cudaSuccess) {
        fprintf(stderr, "[error] cudaDeviceSynchronize() failed.\n");
        return EXIT_FAILURE;
    }

    // validate the result

    for (int i = 0; i < n; i++) {
        if (x[i] != 2.0 * i) {
            fprintf(stderr, "[error] The computed result was incorrect.\n");
            return EXIT_FAILURE;
        }
    }

    printf("The result was correct.\n");

    // free the allocated memory

    free(x);
    cudaFree(d_x);
}
