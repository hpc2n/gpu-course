# Memory transfers

## Objectives

 - Learn how to allocate memory.
 - Learn how to move data from the host memory to the global memory.

## Instructions

 1. Carefully read through the `ax.cu` file. Make sure that you have an idea
    of what each line of code does.

 2. Compile and run the program. The program requires a single argument:
 
    ```
    $ srun ... ./ax 10000
    $ Residual = 0.000000e+00
    ```
    
    The program does the following:
     - A random vector `x` and it's duplicated `_x` are generated.
       The program argument defines the length of the vector `x`.
     - The vector `x` is copied to the global memory buffer `d_x`.
     - A CUDA kernel multiplies the vector `d_x` with a supplied scalar `alpha`.
     - The vector `d_x` is copied back to the host memory buffer `x`.
     - The result is validated by computing
     
       `sqrt((x - (alpha * _x))^2)`.

 3. Modify the program such that instead of computing the operation
    
    `x[i] <- alpha * x[i]`, 
    
    the program computes the AXPY operation
    
    `y[i] <- alpha * x[i] + y[i]`. 
    
    Validate the result.

    Necessary steps:
     - Pass the vector `y` to the kernel and modify the `for` loop.
     - Allocate host memory for the vector `y` and it's duplicate `_y`.
     - Allocate global memory for the vector `y`.
     - Copy the vector `y` to the global memory.
     - Launch the modified kernel.
     - Copy the vector `y` back to the host memory. Note, you do not have to
       copy the vector `x` since it is not modified.
     - Compute

       `sqrt((y - (alpha * x + _y))^2)`.
