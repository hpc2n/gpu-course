# Summing a vector

## Objectives

 - Learn how to use shared memory.
 - Learn how to coordinate thread execution.
 - Learn how to launch multiple CUDA kernels.

## Remark

The purpose of this hands-on is not to learn the optimal way of summing together
the elements of a vector. It is simply meant illustrate how shared memory and
multiple CUDA kernels are used.
 
## Instructions

 1. Carefully read through the `sum.cu` file. Make sure that you have an idea
    of what each line of code does.

 2. The program requires two argument. Compile and run the program:
 
    ```
    $ nvcc -o sum sum.cu
    $ srun ... ./sum 10000 1024
    Residual = 2.025358e-15
    ```
    
    The program does the following:
     
     - A random vector `x` is generated and moved to the global memory.
     
     - CUDA kernel `partial_sum_kernel` partially sums together the elements
       of the vector `x` and stores the partial sums to a vector `y` such that
       
       ```
       y[i] = x[i] + x[i+thread_count] + x[i+2*thread_count] + ...
       ```
       
     - The vector `y` is copied to the host memory and it's elements are summed
       together.
    
     - The final result is validated using the Kahan summation algorithm.
     
    The first program argument defines the length of the vector `x`, the second
    program argument defines the length of the vector `y`.
    
    Note that in order to keep things simple, the thread block size is fixed
    to `THREAD_BLOCK_SIZE` and the length of the vector `y` (given by the
    variable `m`) is converted to a multiple of `THREAD_BLOCK_SIZE`.

 3. Create a second kernel (`final_sum_kernel`) that sums together the elements
    of the vector `y`. At this point, it is sufficient that the kernel is
    **single-threaded**, i.e.:
    
    ```
    final_sum_kernel<<<1, 1>>>(m, d_y);
    ```
    
    The kernel should store the final sum to the first element of the vector
    `y`. Transfer the **first** element back to the host memory and validate 
    the result. Remove the now obsolite `final_sum` function.
    
    Compile and test your modified program.

 4. Modify the `final_sum_kernel` kernel such that it uses multiple threads,
    i.e.:
 
    ```
    final_sum_kernel<<<1, THREAD_BLOCK_SIZE>>>(m, d_y);
    ```
    
    Start by replacing the body of the kernel with the following:
    
    ```
    __global__ void final_sum_kernel(int n, double *x)
    {
        // allocate an array of shared memory, one element per thread
        __shared__ double tmp[THREAD_BLOCK_SIZE];
    
        double v = 0;
        
        // Each thread computes a partial sum as done in the partial_sum_kernel
        // kernel and stores the result to the variable v. Remember, in this
        // case, we have just one thread block, i.e., use threadIdx.x and
        // blockDim.x.
        for (....
    
        // store the partial sums to the shared memory array
        tmp[threadIdx.x] = v;
        
        // wait until all threads in the same thread block are ready
        __syncthreads();
        
        // the first thread of the thread block computes the final sum
        if (threadIdx.x == 0) {
            double vv = 0;
            for (int i = 0; i < THREAD_BLOCK_SIZE; i++)
                vv += tmp[i];
            x[0] = vv;
        }
    }
    ```
    
    Implement the missing `for` loop. Compile and test your modified program.
    
 5. Replace the second half of the `final_sum_kernel` kernel (everything after
    `__syncthreads` function call) with the following:
    
    ```
    int active = THREAD_BLOCK_SIZE/2;
    while (0 < active) {
        if (threadIdx.x < active)
            tmp[threadIdx.x] += tmp[active+threadIdx.x];
        active /= 2;
        __syncthreads();
    }

    if (threadIdx.x == 0)
        x[0] = tmp[0];
    ```
    
    As explained during the lecture, only a subset of the thread are *active*
    during an iteration. If the `i`th thread is active, then it sums together
    the elements `tmp[i]` and `tmp[i+active]`, stores the result back to
    `tmp[i]`, and waits until all other threads have done the same. The number
    of *active* threads is halved after each iteration. Imagine the following
    example:
    
    ```
    3 3 5 1|4 5 2 4  THREAD_BLOCK_SIZE = 8
    
    3+4 3+5 5+2 1+4  active = 4
      7   8   7   5
    ---------------  __syncthreads();
    7 8|7 5
    
    7+8 7+5          active = 2
     15  12
    -------          __syncthreads();
    15|12
    
    15+12            active = 1
       27
    -----            __syncthreads();
    27
    ```
    
    Compile and test your modified program.

 6. Modify the `partial_sum_kernel` such that each thread block computes only
    a single partial sum. In the end, the `i`'th thread block should store it's
    partial sum to `y[i]`. Take a look at the `final_sum_kernel` kernel and see
    what could be reused.
    
    Remember to modify the `main` function such that the length of the vector
    `y` (given by the variable `m`) is the same as the number of thread block in
    first kernel call.
    
    Compile and test your modified program.
    
    Hint: If you are clever, you can combine both kernel into a single kernel
    that is called twice: Once to compute `m` partial sums and once to sum the
    `m` partial sums together.
    
