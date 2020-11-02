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

// in order to keep things simple, the thread block size is fixed
#define THREAD_BLOCK_SIZE 128

// a function that returns the ceil of a/b. That is,
//     DIVCEIL(5, 2) = ceil(5/2) = ceil(2.5) = 3.
static int DIVCEIL(int a, int b)
{
    return (a+b-1)/b;
}

//
// A kernel that partially sums together the elements of a vector x. The partial
// sums are stored to a vector y such that
//     y[i] = x[i] + x[i+thread_count] + x[i+2*thread_count] + ...
//
__global__ void partial_sum_kernel(int n, double const *x, double *y)
{
    int thread_id = blockIdx.x * THREAD_BLOCK_SIZE + threadIdx.x;
    int thread_count = gridDim.x * THREAD_BLOCK_SIZE;

    double v = 0.0;
    for (int i = thread_id; i < n; i += thread_count)
        v += x[i];
    
    y[thread_id] = v;
}

// a function that sums together the elements of a vector x
double final_sum(int n, double *x)
{
    double v = 0;
    for (int i = 0; i < n; i++)
        v += x[i];
    return v;
}

int main(int argc, char **argv)
{
    // read and validate the command line arguments

    if (argc < 2) {
        fprintf(stderr, "[error] No vector length was supplied.\n");
        return EXIT_FAILURE;
    }
    
    if (argc < 3) {
        fprintf(stderr, 
            "[error] No intermediate vector length was supplied.\n");
        return EXIT_FAILURE;
    }

    int n = atof(argv[1]);
    if (n < 1) {
        fprintf(stderr, "[error] The vector length was invalid.\n");
        return EXIT_FAILURE;
    }
    int m = atof(argv[2]);
    if (m < 1) {
        fprintf(stderr, 
            "[error] The intermediate vector length was invalid.\n");
        return EXIT_FAILURE;
    }
    
    // in order to keep things simple, m is converted to a multiple of
    // THREAD_BLOCK_SIZE
    m = DIVCEIL(m, THREAD_BLOCK_SIZE)*THREAD_BLOCK_SIZE;
        
    srand(time(NULL));
    
    // allocate host memory for the vectors y and x
    
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

    for (int i = 0; i < n; i++)
        x[i] = 2.0*rand()/RAND_MAX - 1.0;

    // allocate device memory

    double *d_y, *d_x;
    CHECK_CUDA_ERROR(cudaMalloc(&d_y, m*sizeof(double)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_x, n*sizeof(double)));
    
    // copy the vector x from the host memory to the device memory

    CHECK_CUDA_ERROR(
        cudaMemcpy(d_x, x, n*sizeof(double), cudaMemcpyHostToDevice));

    // launch the kernel, note that m is a multiple of THREAD_BLOCK_SIZE
    
    partial_sum_kernel<<<m/THREAD_BLOCK_SIZE, THREAD_BLOCK_SIZE>>>(n, d_x, d_y);
    
    // copy the vector y from the device memory to the host memory
    
    CHECK_CUDA_ERROR(
        cudaMemcpy(y, d_y, m*sizeof(double), cudaMemcpyDeviceToHost));
    
    // compute the final sum
    
    double sum = final_sum(m, y);

    // validate the result (Kahan)
    
    double sum2 = 0.0, c = 0.0;
    for (int i = 0; i < n; i++) {
        double z = x[i] - c;
        double t = sum2 + z;
        c = (t - sum2) - z;
        sum2 = t;
    }
    printf("Residual = %e\n", fabs(sum2 - sum)/fabs(sum2));
    
    // free the allocated memory
    
    free(y); free(x);
    CHECK_CUDA_ERROR(cudaFree(d_y));
    CHECK_CUDA_ERROR(cudaFree(d_x));

    return EXIT_SUCCESS;
}
