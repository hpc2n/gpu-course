# Computing a matrix-vector multiplication

## Objectives

 - Learn how to use shared memory.
 - Learn how to coordinate thread execution.
 - Learn how to manage matrices.

## Instructions

 1. Carefully read through the `gemv.cu` file. Make sure that you have an idea
    of what each line of code does.

 2. The program requires two argument. Compile and run the program:
 
    ```
    $ nvcc -o gemv gemv.cu
    $ srun ... ./gemv 800 900
    Residual = 3.865318e-16
    ```
    
    The program does the following:
     
     - A random vector `A` and a random vector `x` are generated and moved to
       the global memory.
       
     - A kernel `gemv_kernel` computes `y <- A * x` as follows:
     
       ```
       y_k = A_k0 * x_0 + A_k1 * x_1 + A_k2 * x_2 + ...
       ```
       
     - The vector `y` is copied to the host memory and validated.
     
    The first program argument defines the height of the matrix `A` and the
    second program argument defines the width of the matrix `A`.
    
    Note that the global indexes are are computed as follows:
    
    ```
    int thread_id = blockIdx.y * blockDim.y + threadIdx.y;
    ```

    Furthermore, note that the thread block size is 1 x 128 x 1 and the grid
    size is 1 x Gy x 1, where Gy = (m+threads.y-1)/threads.y.
    
 3. Modify the program such that global memory buffer `d_A` is allocated using
    the `cudaMallocPitch` function and transferred using the `cudaMemcpy2D`
    function:
    
    ```
    cudaError_t cudaMallocPitch (
        void ** devPtr,
        size_t * pitch,
        size_t width,
        size_t height	 
    )
    cudaError_t cudaMemcpy2D (
        void * dst,
        size_t dpitch,
        const void * src,
        size_t spitch,
        size_t width,
        size_t height,
        enum cudaMemcpyKind kind	 
    )	
    ```
    
    Remember, since the matrix is stored in the column-major format, `width` is
    the **height** of the matrix in **bytes** and `height` is the width of the
    matrix. Pitch is the leading dimension of the matrix in **bytes**.
    
    Compile and test your modified program.

 4. Modify the `gemv_kernel` kernel such that it uses two-dimensional thread
    blocks. For now, use the `y` dimension for computations. Simply make sure
    that all threads that have `threadIdx.x != 0` skip the `if` block. Set the
    thread block size to `32 x 32`. 
    
    Compile and test your modified program.

 5. Modify the `gemv_kernel` kernel such that the thread block's `x` dimension
    is used to loop over the columns of the matrix. That is, parallelize the
    `for` loop. Use shared memory to communicate the partial sums.
    
    Remember that all threads must encounter the `__syncthreads()` barrier.
    Therefore, the barrier **cannot** be inside the `if` block!

    Can you tell why are we using the the thread block indices in this manner?
    Pay attention to how the memory is accessed.
    
    Compile and test your modified program.

    Hint: Allocate `threads.y * threads.x * sizeof(double)` bytes of shared
    memory:
 
    ```
    __global__ void gemv_kernel(
    int m, int n, int ldA, double const *A, double const *x, double *y)
    {
        extern __shared__ double tmp[];
        
        ....
    }
    
    ....
    
    size_t shared_size = threads.y*threads.x*sizeof(double);
    gemv_kernel<<<blocks, threads, shared_size>>>(....);
    ```
    
    Each treads should store it's partial sum `v` to `tmp` as follows:
    
    ```
    tmp[threadIdx.y*blockDim.x + threadIdx.x] = v;
    ```
    
    The final result is computed by summing together elements 
    `tmp[threadIdx.y*blockDim.x + 0]`, `tmp[threadIdx.y*blockDim.x + 1]`, `...`, 
    and `tmp[threadIdx.y*blockDim.x + blockDim.x-1]`.

    Remember, threads that belong to the warp access the memory together.
    
 6. (challenge) Modify the program so that it uses managed memory. Make sure
    that the leading dimension is a multiple of the L2 cache line width (128) in
    bytes. The driver aligns memory to 256 bytes.
    
