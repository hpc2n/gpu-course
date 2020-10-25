# Threads, thread blocks and grids

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

 3. Modify the `gemv_kernel` kernel such that it uses two-dimensional thread
    blocks. For now, use the `y` dimension for computations. Simply make sure
    that all threads that have `threadIdx.x != 0` skip the `if` block. Set the
    thread block size to `32 x 32`. 
    
    Compile and test your modified program.

 4. Modify the `gemv_kernel` kernel such that the thread block's `x` dimension
    is used to loop over the columns of the matrix. That is, parallelize the
    `for` loop. Use shared memory to communicate the partial sums.
    
    Remember that all threads must encounter the `__syncthreads()` barrier.
    Therefore, the barrier **cannot** be inside the `if` block!
    
    Compile and test your modified program.

    Can you tell why are we using the the thread block indices in this manner?
    Pay attention to how the memory is accessed.
    
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
    
    Each treads should store it's partial sum to
    
    ```
    tmp[threadIdx.y*blockDim.x + threadIdx.x] = v;
    ```

    Remember, threads that belong to the warp access the memory together.
    
    
    
    
