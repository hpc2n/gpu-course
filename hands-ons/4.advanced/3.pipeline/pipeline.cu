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
        fprintf(stderr, "[error] No matrix height was supplied.\n");
        return EXIT_FAILURE;
    }
    
    if (argc < 3) {
        fprintf(stderr, "[error] No width width was supplied.\n");
        return EXIT_FAILURE;
    }
    
    if (argc < 4) {
        fprintf(stderr, "[error] No splice size was supplied.\n");
        return EXIT_FAILURE;
    }

    int m = atoi(argv[1]);
    if (m < 1) {
        fprintf(stderr, "[error] The matrix height was invalid.\n");
        return EXIT_FAILURE;
    }
    
    int n = atoi(argv[2]);
    if (n < 1) {
        fprintf(stderr, "[error] The matrix width was invalid.\n");
        return EXIT_FAILURE;
    }
    
    int splice = atoi(argv[3]);
    if (splice < 1) {
        fprintf(stderr, "[error] The splice size was invalid.\n");
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
    
    // create stream
    
    cudaStream_t stream;
    CHECK_CUDA_ERROR(cudaStreamCreate(&stream));
    
    //
    // allocate memory for a matrix-matrix multiplication
    //
    //            C               A            B
    //   +-----------------+    +---+ +-----------------+
    // m |  :  :  :  :  :  | <- |   | |  :  :  :  :  :  |
    //   +-----------------+    +---+ +-----------------+ 
    //   <------- n ------->
    //
    
    int ldA, ldB, ldC;
    ldA = ldB = ldC = DIVCEIL(m, 32)*32; // align to 256 bytes
    double *A = (double *) aligned_alloc(32, m*ldA*sizeof(double));
    double *B = (double *) aligned_alloc(32, n*ldB*sizeof(double));
    double *C = (double *) aligned_alloc(32, n*ldC*sizeof(double));
    
    if (A == NULL || B == NULL || C == NULL) {
        fprintf(stderr, "[error] Failed to allocate memory.\n");
        return EXIT_FAILURE;
    }
    
    // pin matrices A, B and C to the host memory
    
    CHECK_CUDA_ERROR(
        cudaHostRegister(A, m*ldA*sizeof(double), cudaHostRegisterDefault));
    CHECK_CUDA_ERROR(
        cudaHostRegister(B, n*ldB*sizeof(double), cudaHostRegisterDefault));
    CHECK_CUDA_ERROR(
        cudaHostRegister(C, n*ldC*sizeof(double), cudaHostRegisterDefault));
    
    // allocate device memory

    double *_A, *_B, *_C;
    CHECK_CUDA_ERROR(cudaMalloc(&_A, n*ldA*sizeof(double)));
    CHECK_CUDA_ERROR(cudaMalloc(&_B, splice*ldB*sizeof(double)));
    CHECK_CUDA_ERROR(cudaMalloc(&_C, splice*ldC*sizeof(double)));
    
    //
    // initialize the matrices A and B
    //
    
    // p <- random permutation vector p
        
    int *p = (int *) malloc(m*sizeof(int));
    for (int i = 0; i < m; i++)
        p[i] = i;
    
    for (int i = 0; i < m; i++) {
        int s1 = rand() % m;
        int s2 = rand() % m;
        int t = p[s1];
        p[s1] = p[s2];
        p[s2] = t;
    }

    // A <- permutation matrix induced by the permutation vector p
    
    for (int i = 0; i < m; i++)
        for (int j = 0; j < m; j++)
            A[i*ldA+j] = 0.0;
        
    for (int i = 0; i < m; i++)
        A[p[i]*ldA+i] = 1.0;
    
    // B <- random matrix
    
    for (int i = 0; i < n; i++)
        for (int j = 0; j < m; j++)
            B[i*ldB+j] = 2.0*rand()/RAND_MAX - 1.0;

    //
    // start timer
    //

    struct timespec ts_start;
    clock_gettime(CLOCK_MONOTONIC, &ts_start);
        
    //
    // C <- A * B (permute the rows of the matrix B):
    //
    //            C               A            B
    //   +-----------------+    +---+ +-----------------+
    // m |  :  :  :  :  :  | <- |   | |  :  :  :  :  :  |
    //   +-----------------+    +---+ +-----------------+ 
    //   <------- n ------->
    //
    
    // copy the matrix A to the global memory
    
    CHECK_CUDA_ERROR(
        cudaMemcpy(_A, A, m*ldA*sizeof(double), cudaMemcpyHostToDevice));

    // splice the matrices B and C horizontally into m-by-splice sub-matrices
    for (int i = 0; i < n; i+= splice) {
        
        // the width of the spliced sub-matrix pair
        
        int splice_width = min(splice, n-i);
        
        // _B <- corresponding slice from the matrix B
        
        CHECK_CUDA_ERROR(
            cudaMemcpyAsync(_B, B+i*ldB, splice_width*ldB*sizeof(double),
                cudaMemcpyHostToDevice, stream));
        
        // set CUDA stream for the cuBLAS call
        
        CHECK_CUBLAS_ERROR(cublasSetStream(handle, stream));
        
        // multiply each pair of sub-matrices
        //          _C                A          _B
        //   +-----------------+    +---+ +-----------------+
        // m |  :  :##:  :  :  | <- |###| |  :  :##:  :  :  |
        //   +-----------------+    +---+ +-----------------+ 
        //         ^-- i                        ^-- i
        
        double one = 1.0, zero = 0.0;
        CHECK_CUBLAS_ERROR(cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
            m, splice_width, m, &one, _A, ldA, _B, ldB, &zero, _C, ldC));
        
        // copy _C to the corresponding slice of the matrix C
        
        CHECK_CUDA_ERROR(
            cudaMemcpyAsync(C+i*ldC, _C, splice_width*ldC*sizeof(double),
                cudaMemcpyDeviceToHost, stream));
    }
    
    //
    // wait until the device is ready and stop the timer
    //
 
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    
    struct timespec ts_stop;
    clock_gettime(CLOCK_MONOTONIC, &ts_stop);
    
    double time = ts_stop.tv_sec - ts_start.tv_sec +
        1.0e-9*(ts_stop.tv_nsec - ts_start.tv_nsec); 
    printf("Runtime was %.3f s.\n", time);

    //
    // validate the result
    //
    
    double max_error = 0.0;
    for (int i = 0; i < n; i++)
        for (int j = 0; j < m; j++)
            max_error = max(max_error, fabs(C[i*ldC+j] - B[i*ldB+p[j]]));
    printf("Max error = %e.\n", max_error);

    // shutdown cuBLAS
    CHECK_CUBLAS_ERROR(cublasDestroy(handle));
    
    // destroy the streams
    
    CHECK_CUDA_ERROR(cudaStreamDestroy(stream));
    
    // un-pin the matrices A, B and C
    
    CHECK_CUDA_ERROR(cudaHostUnregister(A));
    CHECK_CUDA_ERROR(cudaHostUnregister(B));
    CHECK_CUDA_ERROR(cudaHostUnregister(C));
    
    // free device memory
    CHECK_CUDA_ERROR(cudaFree(_A));
    CHECK_CUDA_ERROR(cudaFree(_B));
    CHECK_CUDA_ERROR(cudaFree(_C));
    
    // free the allocated memory
    
    free(A); free(B); free(C); free(p);

    return EXIT_SUCCESS;
}
