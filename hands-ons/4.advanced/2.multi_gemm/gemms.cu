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

int DIVCEIL(int a, int b)
{
    return (a+b-1)/b;
}

int main(int argc, char const **argv)
{
    //
    // read and validate the command line arguments
    //
    
    if (argc < 2) {
        fprintf(stderr, "[error] No matrix size was supplied.\n");
        return EXIT_FAILURE;
    }
    
    if (argc < 2) {
        fprintf(stderr, "[error] No matrix count was supplied.\n");
        return EXIT_FAILURE;
    }
    
    int n = atoi(argv[1]);
    if (n < 1) {
        fprintf(stderr, "[error] The matrix size was invalid.\n");
        return EXIT_FAILURE;
    }
    
    int count = atoi(argv[2]);
    if (count < 1) {
        fprintf(stderr, "[error] The matric count was invalid.\n");
        return EXIT_FAILURE;
    }
    
    //
    // initialization
    //
    
    // initialize the random number generator
    
    srand(time(NULL));
    
    // initialize cuBLAS
    
    cublasHandle_t handle;
    CHECK_CUBLAS_ERROR(cublasCreate(&handle));
    
    //
    // allocate memory for several (A, B, C) matrix triplets
    //
    
    double **A = (double **) malloc(count*sizeof(double *));
    double **B = (double **) malloc(count*sizeof(double *));
    double **C = (double **) malloc(count*sizeof(double *));
    
    double **_A = (double **) malloc(count*sizeof(double *));
    double **_B = (double **) malloc(count*sizeof(double *));
    double **_C = (double **) malloc(count*sizeof(double *));
    
    int **p = (int **) malloc(count*sizeof(int *));
    
    int ldA, ldB, ldC;
    ldA = ldB = ldC = DIVCEIL(n, 32)*32; // align to 256 bytes
    
    for (int i = 0; i < count; i++) {
        
        // allocate host memory
        
        A[i] = (double *) aligned_alloc(32, n*ldA*sizeof(double));
        B[i] = (double *) aligned_alloc(32, n*ldB*sizeof(double));
        C[i] = (double *) aligned_alloc(32, n*ldC*sizeof(double));
    
        if (A[i] == NULL || B[i] == NULL || C[i] == NULL) {
            fprintf(stderr, "[error] Failed to allocate memory.\n");
            return EXIT_FAILURE;
        }
        
        // allocate device memory

        CHECK_CUDA_ERROR(cudaMalloc(&_A[i], n*ldA*sizeof(double)));
        CHECK_CUDA_ERROR(cudaMalloc(&_B[i], n*ldB*sizeof(double)));
        CHECK_CUDA_ERROR(cudaMalloc(&_C[i], n*ldC*sizeof(double)));
        
        //
        // initialize the matrices A and B
        //
        
        // p <- random permutation vector p
            
        p[i] = (int *) malloc(n*sizeof(int));
        for (int j = 0; j < n; j++)
            p[i][j] = j;
        
        for (int j = 0; j < n; j++) {
            int s1 = rand() % n;
            int s2 = rand() % n;
            int t = p[i][s1];
            p[i][s1] = p[i][s2];
            p[i][s2] = t;
        }

        // A <- permutation matrix induced by the permutation vector p
        
        for (int j = 0; j < n; j++)
            for (int k = 0; k < n; k++)
                A[i][j*ldA+k] = 0.0;
            
        for (int j = 0; j < n; j++)
            A[i][p[i][j]*ldA+j] = 1.0;
        
        // B <- random matrix
        
        for (int j = 0; j < n; j++)
            for (int k = 0; k < n; k++)
                B[i][j*ldB+k] = 2.0*rand()/RAND_MAX - 1.0;
    }

    //
    // start timer
    //

    struct timespec ts_start;
    clock_gettime(CLOCK_MONOTONIC, &ts_start);
        
    //
    // C[i] <- A[i] * B[i] (permute the rows of the matrix B):
    //
    
    for (int i = 0; i < count; i++) {
        
        // copy the matrices A and B to the global memory
    
        CHECK_CUDA_ERROR(
            cudaMemcpy(
                _A[i], A[i], n*ldA*sizeof(double), cudaMemcpyHostToDevice));
        CHECK_CUDA_ERROR(
            cudaMemcpy(
                _B[i], B[i], n*ldB*sizeof(double), cudaMemcpyHostToDevice));
        
        // call GEMM
        
        double one = 1.0, zero = 0.0;
        CHECK_CUBLAS_ERROR(cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
            n, n, n, &one, _A[i], ldA, _B[i], ldB, &zero, _C[i], ldC));
        
        // copy the matrix C from the global memory
    
        CHECK_CUDA_ERROR(
            cudaMemcpy(
                C[i], _C[i], n*ldC*sizeof(double), cudaMemcpyDeviceToHost));
    }
    
    //
    // stop timer
    //
    
    struct timespec ts_stop;
    clock_gettime(CLOCK_MONOTONIC, &ts_stop);
    
    double time = ts_stop.tv_sec - ts_start.tv_sec +
        1.0e-9*(ts_stop.tv_nsec - ts_start.tv_nsec); 
    printf("Runtime was %.3f s.\n", time);

    //
    // validate the result and free resources
    //
    
    double max_error = 0.0;
    
    for (int i = 0; i < count; i++) {
        
        for (int j = 0; j < n; j++) {
            for (int k = 0; k < n; k++) {
                max_error = 
                    max(max_error, fabs(C[i][j*ldC+k] - B[i][j*ldB+p[i][k]]));
            }
        }
    
        free(A[i]); free(B[i]); free(C[i]); free(p[i]);
        CHECK_CUDA_ERROR(cudaFree(_A[i]));
        CHECK_CUDA_ERROR(cudaFree(_B[i]));
        CHECK_CUDA_ERROR(cudaFree(_C[i]));
    }
    
    printf("Max error = %e.\n", max_error);
    
    free(A); free(B); free(C); free(p);
    free(_A); free(_B); free(_C);

    // shutdown cuBLAS
    CHECK_CUBLAS_ERROR(cublasDestroy(handle));

    return EXIT_SUCCESS;
}
