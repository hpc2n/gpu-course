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

// a kernel that multiplies a vector y with a scalar alpha
__global__ void ax_kernel(int n, double alpha, double *y)
{
    //
    // Each thread is going to begin from the array element that matches it's
    // global index number. For blockDim.x = 4, gridDim.x 2, we have:
    // threadIdx.x : 0 1 2 3 0 1 2 3
    // blockIdx.x  : 0 0 0 0 1 1 1 1
    // blockDim.x  : 4 4 4 4 4 4 4 4
    // thread_id   : 0 1 2 3,4 5 6 7
    //
    int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    int thread_count = gridDim.x * blockDim.x;

    //
    // Each thread is going to jump over <grid dimension> * <block dimension>
    // array elements. For blockDim.x = 4, gridDim.x 2, we have:
    // 0 1 2 3,4 5 6 7|0 1 2 3,4 5 6 7|0 1 2 3,4 5 6 7|0 1 2 3,4 5 6 7|0 ...
    //
    for (int i = thread_id; i < n; i += thread_count)
        y[i] = alpha * y[i];
}

int main(int argc, char const **argv)
{
    double alpha = 2.0;

    // read and validate the command line arguments

    if (argc < 2) {
        fprintf(stderr, "[error] No vector length was supplied.\n");
        return EXIT_FAILURE;
    }

    int n = atof(argv[1]);
    if (n < 1) {
        fprintf(stderr, "[error] The vector length was invalid.\n");
        return EXIT_FAILURE;
    }
    
    srand(time(NULL));

    // allocate host memory for the vector and it's duplicate

    double *y, *_y;
    if ((y = (double *) malloc(n*sizeof(double))) == NULL) {
        fprintf(stderr,
            "[error] Failed to allocate host memory for vector y.\n");
        return EXIT_FAILURE;
    }
    if ((_y = (double *) malloc(n*sizeof(double))) == NULL) {
        fprintf(stderr,
            "[error] Failed to allocate host memory for vector _y.\n");
        return EXIT_FAILURE;
    }

    // initialize host memory and store a copy for a later validation

    for (int i = 0; i < n; i++)
        y[i] = _y[i] = 1.0*rand()/RAND_MAX;

    // allocate device memory

    double *d_y;
    CHECK_CUDA_ERROR(cudaMalloc(&d_y, n*sizeof(double)));

    // copy the vector from the host memory to the device memory

    CHECK_CUDA_ERROR(
        cudaMemcpy(d_y, y, n*sizeof(double), cudaMemcpyHostToDevice));

    // launch the kernel

    dim3 threads = 128;
    dim3 blocks = min(64, (int) ceil(n/threads.x));
    ax_kernel<<<blocks, threads>>>(n, alpha, d_y);

    CHECK_CUDA_ERROR(cudaGetLastError());

    // copy the vector from the device memory to the host memory

    CHECK_CUDA_ERROR(
        cudaMemcpy(y, d_y, n*sizeof(double), cudaMemcpyDeviceToHost));

    // validate the result by computing sqrt((x-alpha*_x)^2)

    double res = 0.0;
    
    for (int i = 0; i < n; i++)
        res += (y[i]-alpha*_y[i]) * (y[i]-alpha*_y[i]);
    
    // A little assistance: comment out the above for loop and use this one
    // to validate the output of the AXPY operation.
    // for (int i = 0; i < n; i++)
    //    res +=
    //        (y[i] - (alpha * x[i] + _y[i])) * (y[i] - (alpha * x[i] + _y[i]));
    
    printf("Residual = %e\n", sqrt(res));

    // free the allocated memory

    free(y); free(_y);
    CHECK_CUDA_ERROR(cudaFree(d_y));
}
