#include <stdlib.h>
#include <stdio.h>

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

//
// A macro that accesses the element on the i'th row and the j'th column of a
// matrix A.
//
// The matrix A (m rows, n columns) is stored in column-major format, i.e., the
// columns are stored continuously in the memory. The leading dimension (ldA)
// defines how many words (double-precision floating point numbers in this case)
// are allocated for each column. That is, A[j*ldA+i] is the element on the i'th
// row and the j'th column of the matrix.
//
#define _A(i, j) (A[(size_t)(j)*ldA+(i)])

// fix thread block dimensions so that blockDim.x = blockDim.y = warp size
#define THREAD_BLOCK_SIZE 32

// a kernel that perform a matrix-vector multiplication y = A * x, where the
// matrix A has m rows and n columns
__global__ void gemv_kernel(
    int m, int n, int ldA, double const *A, double const *x, double *y)
{
    // dynamically allocated shared memory array
    __shared__ double tmp[THREAD_BLOCK_SIZE][THREAD_BLOCK_SIZE];
    
    // we are assuming that each row of the vector y gets it's own thread in
    // the y dimension
    int thread_id = blockIdx.x * THREAD_BLOCK_SIZE + threadIdx.x;
    
    double v = 0.0;
    if (thread_id < m) {

        //
        // loop over the corresponding row of the matrix A and the vector x
        //
        // |y_0|   |A_00 A_01 A_02 .... |   |x_0|
        // |y_1|   |A_10 A_11 A_12 .... |   |x_1|
        // |y_2| = |A_20 A_21 A_22 .... | * |x_2|
        // |...|   |.... .... .... .... |   |...|
        // |...|   |.... .... .... .... |   |...|
        //
        // y_k = A_k0 * x_0 + A_k1 * x_1 + A_k2 * x_2 ...
        //
        for (int i = threadIdx.y; i < n; i += THREAD_BLOCK_SIZE)
            v += _A(thread_id, i) * x[i];
    }
    
    // each thread stores it's partial sum to the shared memory array
    tmp[threadIdx.x][threadIdx.y] = v;
    
    // wait until all threads are ready
    __syncthreads();

    // sum together the partial sums (note the swapped x-y dimensions)
    int active = THREAD_BLOCK_SIZE/2;
    while (0 < active) {
        if (threadIdx.x < active)
            tmp[threadIdx.y][threadIdx.x] += 
                tmp[threadIdx.y][threadIdx.x + active];
        active /= 2;
        
        __syncthreads();
        
        // The above __syncthreads() call could be replaced with a __syncwarp()
        // call. The __syncwarp() function synchronizes the **warp**. This could
        // potentially improve the performance as the warps do not have to wait
        // each other.
        // __syncwarp();    
    }
    
    // __syncthreads(); // needed with __syncwarp

    if (thread_id < m && threadIdx.y == 0)
        y[thread_id] = tmp[threadIdx.x][0];
}

int main(int argc, char **argv)
{
    // read and validate the command line arguments

    if (argc < 2) {
        fprintf(stderr, "[error] No matrix height was supplied.\n");
        return EXIT_FAILURE;
    }
    if (argc < 3) {
        fprintf(stderr, "[error] No matrix width was supplied.\n");
        return EXIT_FAILURE;
    }

    int m = atof(argv[1]);
    if (m < 1) {
        fprintf(stderr, "[error] The matrix height was invalid.\n");
        return EXIT_FAILURE;
    }
    int n = atof(argv[2]);
    if (n < 1) {
        fprintf(stderr, "[error] The matrix width was invalid.\n");
        return EXIT_FAILURE;
    }
        
    srand(time(NULL));
    
    // allocate host memory for the matrix A and the vectors y and x
    
    double *A; int ldA = m; // only for illustrational purposes only
    if ((A = (double *) malloc(n*ldA*sizeof(double))) == NULL) {
        fprintf(stderr,
            "[error] Failed to allocate host memory for matrix A.\n");
        return EXIT_FAILURE;
    }
    double *y, *x;
    if ((y = (double *) malloc(m*sizeof(double))) == NULL) {
        fprintf(stderr,
            "[error] Failed to allocate host memory for vector y.\n");
        return EXIT_FAILURE;
    }
    if ((x = (double *) malloc(n*sizeof(double))) == NULL) {
        fprintf(stderr,
            "[error] Failed to allocate host memory for vector x.\n");
        return EXIT_FAILURE;
    }
    
    // initialize host memory

    for (int i = 0; i < n; i++) {
        x[i] = 2.0*rand()/RAND_MAX - 1.0;
        for (int j = 0; j < m; j++)
            _A(j, i) = 2.0*rand()/RAND_MAX - 1.0;
    }

    // allocate device memory

    double *d_A, *d_y, *d_x; int ld_dA;
    {
        size_t pitch;
        CHECK_CUDA_ERROR(cudaMallocPitch(&d_A, &pitch, m*sizeof(double), n));
        ld_dA = pitch/sizeof(double);
    }
    CHECK_CUDA_ERROR(cudaMalloc(&d_y, m*sizeof(double)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_x, n*sizeof(double)));
    
    // copy the matrix A and the vector x from the host memory to the device
    // memory

    CHECK_CUDA_ERROR(
        cudaMemcpy2D(d_A, ld_dA*sizeof(double), A, ldA*sizeof(double), 
            m*sizeof(double), n, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(
        cudaMemcpy(d_x, x, n*sizeof(double), cudaMemcpyHostToDevice));

    // launch the kernel
    
    dim3 threads(THREAD_BLOCK_SIZE, THREAD_BLOCK_SIZE);
    dim3 blocks((m+THREAD_BLOCK_SIZE-1)/THREAD_BLOCK_SIZE, 1);
    gemv_kernel<<<blocks, threads>>>(m, n, ld_dA, d_A, d_x, d_y);
    
    // copy the vector y from the device memory to the host memory
    
    CHECK_CUDA_ERROR(
        cudaMemcpy(y, d_y, m*sizeof(double), cudaMemcpyDeviceToHost));

    // validate the result by computing sqrt((A*x - y)^2)
    
    double res = 0.0, nor = 0.0;
    for (int i = 0; i < m; i++) {
        double v = 0.0;
        for (int j = 0; j < n; j++)
            v += _A(i, j) * x[j];
        res += (v - y[i]) * (v - y[i]);
        nor += v*v;
    }
    printf("Residual = %e\n", sqrt(res)/sqrt(nor));
    
    // free the allocated memory
    
    free(A); free(y); free(x);
    CHECK_CUDA_ERROR(cudaFree(d_A));
    CHECK_CUDA_ERROR(cudaFree(d_y));
    CHECK_CUDA_ERROR(cudaFree(d_x));

    return EXIT_SUCCESS;
}
