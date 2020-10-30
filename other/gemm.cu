#include <stdlib.h>
#include <stdio.h>
#include <cublas_v2.h>

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

#define CHECK_CUBLAS_ERROR(exp) {                   \
    cublasStatus_t ret = (exp);                     \
    if (ret != CUBLAS_STATUS_SUCCESS) {             \
        fprintf(stderr,                             \
            "[error] %s:%d: cuBLAS error\n",        \
            __FILE__, __LINE__);                    \
        exit(EXIT_FAILURE);                         \
    }                                               \
}

///
/// Returns the ceil of a/b.
///
/// @param[in] a    denominator
/// @param[in] b    numerator
///
/// @returns ceil of a/b
///
static inline int DIVCEIL(int a, int b)
{
    return (a+b-1)/b;
}

int main(int argc, char const **argv)
{
    // read and validate the command line arguments

    if (argc < 2) {
        fprintf(stderr, "[error] No matrix size was supplied.\n");
        return EXIT_FAILURE;
    }

    int n = atof(argv[1]);
    if (n < 1) {
        fprintf(stderr, "[error] The matrix size was invalid.\n");
        return EXIT_FAILURE;
    }
    
    srand(time(NULL));
    
    cublasHandle_t handle;
    CHECK_CUBLAS_ERROR(cublasCreate(&handle));

    // allocate host memory

    int ldA, ldB, ldC;
    ldA = ldB = ldC = DIVCEIL(n, 32)*32; // align to 256 bytes
    double *A = (double *) aligned_alloc(32, n*ldB*sizeof(double));
    double *B = (double *) aligned_alloc(32, n*ldB*sizeof(double));
    double *C = (double *) aligned_alloc(32, n*ldC*sizeof(double));
    
    if (A == NULL || B == NULL || C == NULL) {
        fprintf(stderr, "[error] Failed to allocate memory.\n");
        return EXIT_FAILURE;
    }

    // initialize host memory

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            A[i*ldA+j] = 2.0*rand()/RAND_MAX - 1.0;
            B[i*ldB+j] = 2.0*rand()/RAND_MAX - 1.0;
            C[i*ldC+j] = 2.0*rand()/RAND_MAX - 1.0;
        }
    }

    // allocate device memory

    double *_A, *_B, *_C;
    CHECK_CUDA_ERROR(cudaMalloc(&_A, n*ldA*sizeof(double)));
    CHECK_CUDA_ERROR(cudaMalloc(&_B, n*ldB*sizeof(double)));
    CHECK_CUDA_ERROR(cudaMalloc(&_C, n*ldC*sizeof(double)));

    // copy to the device memory

    CHECK_CUDA_ERROR(
        cudaMemcpy(_A, A, n*ldA*sizeof(double), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(
        cudaMemcpy(_B, B, n*ldB*sizeof(double), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(
        cudaMemcpy(_C, C, n*ldC*sizeof(double), cudaMemcpyHostToDevice));

    // start timer
    struct timespec ts_start;
    clock_gettime(CLOCK_MONOTONIC, &ts_start);

    // launch the kernel

    double one = 1.0, zero = 0.0;
    CHECK_CUBLAS_ERROR(cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
        n, n, n, &one, _A, ldA, _B, ldB, &zero, _C, ldC));

    // wait until the device is ready and stop the timer
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    struct timespec ts_stop;
    clock_gettime(CLOCK_MONOTONIC, &ts_stop);

    // calculate metrics
    double time = ts_stop.tv_sec - ts_start.tv_sec +
        1.0e-9*(ts_stop.tv_nsec - ts_start.tv_nsec); 
    printf("Runtime was %.3f s.\n", time);
    printf("Floprate was %.0f GFlops.\n", ((1.0*n*n)*(2*n-1)/time)*1E-9);
    printf("Memory throughput (naive) %.0f GB/s.\n", 
        ((1.0*n*n)*(2*n+1)*sizeof(double)/time)*1E-9);

    // free the allocated memory

    free(A); free(B); free(C);
    CHECK_CUBLAS_ERROR(cublasDestroy(handle));
    CHECK_CUDA_ERROR(cudaFree(_A));
    CHECK_CUDA_ERROR(cudaFree(_B));
    CHECK_CUDA_ERROR(cudaFree(_C));
}
