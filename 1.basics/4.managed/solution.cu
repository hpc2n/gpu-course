#include <stdlib.h>
#include <stdio.h>
#include <time.h>
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


int main(int argc, char const **argv)
{
    // read and validate the command line arguments

    if (argc < 2) {
        fprintf(stderr, "[error] No vector lenght was supplied.\n");
        return EXIT_FAILURE;
    }

    int n = atof(argv[1]);
    if (n < 1) {
        fprintf(stderr, "[error] The vector lenght was invalid.\n");
        return EXIT_FAILURE;
    }
    
    srand(time(NULL));

    // allocate memory

    double *x, *y, *_y;
    CHECK_CUDA_ERROR(cudaMallocManaged(&x, n*sizeof(double)));
    CHECK_CUDA_ERROR(cudaMallocManaged(&y, n*sizeof(double)));
    if ((_y = (double *) malloc(n*sizeof(double))) == NULL) {
        fprintf(stderr,
            "[error] Failed to allocate host memory for vector _y.\n");
        return EXIT_FAILURE;
    }

    // initialize memory

    for (int i = 0; i < n; i++) {
        x[i] = 2.0 * rand()/RAND_MAX - 1.0;
        y[i] = _y[i] = 2.0 * rand()/RAND_MAX - 1.0;
    }

    cublasHandle_t handle;
    CHECK_CUBLAS_ERROR(cublasCreate(&handle));
    
    // compute y <- 2 * x + y

    double alpha = 2.0;
    CHECK_CUBLAS_ERROR(cublasDaxpy(handle, n, &alpha, x, 1, y, 1));
    
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    // validate the result

    double res = 0.0;
    for (int i = 0; i < n; i++)
        res +=
            (y[i] - (alpha * x[i] + _y[i])) * (y[i] - (alpha * x[i] + _y[i]));
    printf("Residual = %e\n", sqrt(res));

    // free the allocated memory

    CHECK_CUBLAS_ERROR(cublasDestroy(handle));
    
    CHECK_CUDA_ERROR(cudaFree(x));
    CHECK_CUDA_ERROR(cudaFree(y));
    free(_y);

    return EXIT_SUCCESS;
}
