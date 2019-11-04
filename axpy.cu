//
// A simple CUDA program that adds two vectors together.
//
// Author: Mirko Myllykoski, Umeå University, 2019
//

#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <cublas_v2.h>

int main(int argc, char const **argv)
{
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

    // allocate memory

    double *x;
    if (cudaMallocManaged(&x, n*sizeof(double)) != cudaSuccess) {
        fprintf(stderr, "[error] Failed to allocate memory for vector x.\n");
        return EXIT_FAILURE;
    }

    double *y;
    if (cudaMallocManaged(&y, n*sizeof(double)) != cudaSuccess) {
        fprintf(stderr, "[error] Failed to allocate memory for vector y.\n");
        return EXIT_FAILURE;
    }

    // initialize memory

    for (int i = 0; i < n; i++) {
        x[i] = 2.0 * rand()/RAND_MAX - 1.0;
        y[i] = 2.0 * rand()/RAND_MAX - 1.0;
    }

    // prefetch data to GPU memory

    int device = -1;
    if (cudaGetDevice(&device) != cudaSuccess) {
        fprintf(stderr, "[error] cudaGetDevice() failed.\n");
        return EXIT_FAILURE;
    }

    if (cudaMemPrefetchAsync(x, n*sizeof(double), device, NULL) != cudaSuccess) {
        fprintf(stderr, "[error] cudaMemPrefetchAsync() failed.\n");
        return EXIT_FAILURE;
    }

    if (cudaMemPrefetchAsync(y, n*sizeof(double), device, NULL) != cudaSuccess) {
        fprintf(stderr, "[error] cudaMemPrefetchAsync() failed.\n");
        return EXIT_FAILURE;
    }

    if (cudaDeviceSynchronize() != cudaSuccess) {
        fprintf(stderr, "[error] cudaDeviceSynchronize() failed.\n");
        return EXIT_FAILURE;
    }

    // initialize cuBLAS

    cublasHandle_t handle;
    if (cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "[error] Failed to initialize cuBLAS.\n");
        return EXIT_FAILURE;
    }

    //
    // start timer
    //

    struct timespec start;
    clock_gettime(CLOCK_REALTIME, &start);

    //
    // compute y <- 2 * x + y
    //

    double alpha = 2.0;
    if (cublasDaxpy(handle, n, &alpha, x, 1, y, 1) != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr,"[error] cublasDaxpy() failed.\n");
        return EXIT_FAILURE;
    }

    //
    // wait until the cublasDaxpy has finished
    //

    if (cudaDeviceSynchronize() != cudaSuccess) {
        fprintf(stderr, "[error] cudaDeviceSynchronize() failed.\n");
        return EXIT_FAILURE;
    }

    //
    // stop timer and report
    //

    struct timespec stop;
    clock_gettime(CLOCK_REALTIME, &stop);

    double time =
        (stop.tv_sec - start.tv_sec) + (stop.tv_nsec - start.tv_nsec)*1E-9;

    printf("Runtime was %.3f s.\n", time);
    printf("Floprate was %.0f GFlops.\n", (2*n/time)*1E-9);
    printf("Memory throughput %.0f GB/s.\n", (3*n*sizeof(double)/time)*1E-9);

    // de-initialize cuBLAS

    if (cublasDestroy(handle) != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "[error] cublasDestroy() failed.\n");
        return EXIT_FAILURE;
    }

    // free the allocated memory

    if (cudaFree(x) != cudaSuccess) {
        fprintf(stderr, "[error] cudaFree() failed.\n");
        return EXIT_FAILURE;
    }

    if (cudaFree(y) != cudaSuccess) {
        fprintf(stderr, "[error] cudaFree() failed.\n");
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
