# Memory transfers

## Objectives

 - Learn how to modify CUDA kernels.
 - Learn how to allocate memory.
 - Learn how to move data from the host memory to the global memory and back.

## Instructions

 1. Carefully read through the `ax.cu` file. Make sure that you have an idea
    of what each line of code does.

 2. Compile and run the program. The program requires a single argument:
 
    ```
    $ srun ... ./ax 10000
    Residual = 0.000000e+00
    ```
    
    The program does the following:
     - A random vector `y` and it's duplicated `_y` are generated.
       The program argument `n` defines the length of the vector `y`.
     - The vector `y` is copied to a global memory buffer `d_y`.
     - A CUDA kernel multiplies the vector `d_y` with a supplied scalar `alpha`.
     - The vector `d_y` is copied back to the host memory buffer `y`.
     - The result is validated by computing
     
       `sqrt((y - (alpha * _y))^2)`.

 3. Modify the program such that instead of computing the operation
    
    `y[i] <- alpha * y[i], i = 0, ..., n-1`, 
    
    the program computes so-called AXPY operation
    
    `y[i] <- alpha * x[i] + y[i], i = 0, ..., n-1`. 
    
    Validate the result.

    Necessary steps:
     - Allocate host memory for the vector `x`.
     - Allocate a global memory buffer `d_x` and copy the vector `x` to it.
     - Pass the vector `d_x` to the kernel and modify the `for` loop.
     - Launch the modified kernel.
     - Compute

       `sqrt((y - (alpha * x + _y))^2)`.
