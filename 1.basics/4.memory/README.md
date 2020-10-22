# Error handling

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
     - A random vector (array) is generated and duplicated for later validation.
       The program argument defines the length of the vector.
     - The vector is copied to the global memory.
     - A CUDA kernel multiplies the vector with a supplied scalar.
     - The vector is copied back to the host memory.
     - The result is validated using the duplicated copy of the vector.

 3. Modify the program such that instead of computing the operation
    
    `x[i] <- alpha * x[i]`, 
    
    the program computes the AXPY operation
    
    `y[i] <- alpha * x[i] + y[i]`. 
    
    Validate the result.
