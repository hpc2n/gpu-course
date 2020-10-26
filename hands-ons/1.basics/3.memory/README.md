# Memory transfers

## Objectives

 - Learn how to modify CUDA kernels.
 - Learn how to allocate memory.
 - Learn how to move data from the host memory to the global memory and back.
 - Learn how to record runtime.

## Instructions

 1. Carefully read through the `ax.cu` file. Make sure that you have an idea
    of what each line of code does.

 2. The program requires a single argument. Compile and run the program:
 
    ```
    $ nvcc -o ax ax.cu
    $ srun ... ./ax 10000
    Residual = 0.000000e+00
    ```
    
    The program does the following:
     - A random vector `y` and it's duplicate `_y` are generated.
       The program argument `n` defines the length of the vector `y`.
     - The vector `y` is copied to a global memory buffer `d_y`.
     - A CUDA kernel multiplies the vector `d_y` with a supplied scalar `alpha`.
     - The vector `d_y` is copied back to the host memory buffer `y`.
     - The result is validated by computing
     
       `sqrt((y - (alpha * _y))^2)`.

 3. Modify the program such that instead of computing the operation
    
    `y[i] <- alpha * y[i], i = 0, ..., n-1`, 
    
    the program computes the so-called AXPY operation
    
    `y[i] <- alpha * x[i] + y[i], i = 0, ..., n-1`. 
    
    Validate the result.

    Necessary steps:
     - Generate a random vector `x`.
     - Allocate a global memory buffer `d_x` and copy the vector `x` to it.
     - Pass the vector `d_x` to the kernel and modify the `for` loop.
     - Launch the modified kernel.
     - Compute

       `sqrt((y - (alpha * x + _y))^2)`.
       
     - Free the vector `x`.

 4. Time how long it takes to compute the AXPY operation with different
    vector lengths. Use n = 100, n = 10000 and n = 1000000. You should record
    the current time i) just before launching the kernel and ii) just after the
    `cudaMemcpy` function call. Write down your results.
    
    The execution time can be measured in many different ways. For now, we will
    use the `clock_gettime` function:
     
       ```
       struct timespec {
           time_t   tv_sec;        /* seconds */
           long     tv_nsec;       /* nanoseconds */
       };
       int clock_gettime ( clockid_t clk_id, struct timespec *tp )
       ```
       
       For example,
       
       ```
       struct timespec start, stop;
       clock_gettime(CLOCK_REALTIME, &start);
       
       ....
       
       clock_gettime(CLOCK_REALTIME, &stop);

       double time =
           (stop.tv_sec - start.tv_sec) + (stop.tv_nsec - start.tv_nsec)*1E-9;

       printf("Runtime was %f seconds.\n", time);
       ```

 5. Try end the timing just after the kernel launch. What happens? Why?   
    
    
