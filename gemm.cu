//
// A simple CUDA program that multiplies two matrices together.
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
        fprintf(stderr, "[error] No matrix size was supplied.\n");
        return EXIT_FAILURE;
    }

    int n = atoi(argv[1]);
    if (n < 1) {
        fprintf(stderr, "[error] The matrix size was invalid.\n");
        return EXIT_FAILURE;
    }

    // allocate memory

    double *A; int ldA = (n/8+1)*8;
    if (cudaMallocManaged(&A, n*ldA*sizeof(double)) != cudaSuccess) {
        fprintf(stderr, "[error] Failed to allocate memory for matrix A.\n");
        return EXIT_FAILURE;
    }

    double *B; int ldB = (n/8+1)*8;
    if (cudaMallocManaged(&B, n*ldB*sizeof(double)) != cudaSuccess) {
        fprintf(stderr, "[error] Failed to allocate memory for matrix B.\n");
        return EXIT_FAILURE;
    }

    double *C; int ldC = (n/8+1)*8;
    if (cudaMallocManaged(&C, n*ldC*sizeof(double)) != cudaSuccess) {
        fprintf(stderr, "[error] Failed to allocate memory for matrix C.\n");
        return EXIT_FAILURE;
    }

    // initialize memory

    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            A[i*ldA+j] = 2.0 * rand()/RAND_MAX - 1.0;

    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            B[i*ldB+j] = 2.0 * rand()/RAND_MAX - 1.0;

    // prefetch data to GPU memory

    int device = -1;
    if (cudaGetDevice(&device) != cudaSuccess) {
        fprintf(stderr, "[error] cudaGetDevice() failed.\n");
        return EXIT_FAILURE;
    }

    if (cudaMemPrefetchAsync(A, n*ldA*sizeof(double), device, NULL) != cudaSuccess) {
        fprintf(stderr, "[error] cudaMemPrefetchAsync() failed.\n");
        return EXIT_FAILURE;
    }

    if (cudaMemPrefetchAsync(B, n*ldB*sizeof(double), device, NULL) != cudaSuccess) {
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
    // compute C <- A * B
    //

    double alpha = 1.0, beta = 0.0;
    if (cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n,
        &alpha, A, ldA, B, ldB, &beta, C, ldC) != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr,"[error] cublasDgemm() failed.\n");
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
    printf("Floprate was %.0f GFlops.\n", ((1.0*n*n)*(2*n-1)/time)*1E-9);
    printf("Memory throughput %.0f GB/s.\n", ((1.0*n*n)*(2*n+1)*sizeof(double)/time)*1E-9);

    // de-initialize cuBLAS

    if (cublasDestroy(handle) != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "[error] cublasDestroy() failed.\n");
        return EXIT_FAILURE;
    }

    // free the allocated memory

    if (cudaFree(A) != cudaSuccess) {
        fprintf(stderr, "[error] cudaFree() failed.\n");
        return EXIT_FAILURE;
    }

    if (cudaFree(B) != cudaSuccess) {
        fprintf(stderr, "[error] cudaFree() failed.\n");
        return EXIT_FAILURE;
    }

    if (cudaFree(C) != cudaSuccess) {
        fprintf(stderr, "[error] cudaFree() failed.\n");
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
