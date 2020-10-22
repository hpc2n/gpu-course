# Managed memory

## Objectives

 - Learn how to use managed memory.
 - Learn how to use cuBLAS

## Instructions

 1. Carefully read through the `axpy.cu` file. Make sure that you have an idea
    of what each line of code does.

 2. Compile and run the program (note the variable `${LIBBLAS}`):
 
    ```
    $ nvcc -o axpy axpy.cu ${LIBBLAS}
    $ srun ... ./axpy 10000
    Residual = 0.000000e+00
    ```
    
    The program implements the same functionality as the hands-on "4.memory" but
    relies on the BLAS library to perform the AXPY operation.

 3. Modify the program such that the vectors `x` and `y` are allocated using
    managed memory. Use the following functions:
    
    ```
    cudaError_t cudaMallocManaged ( void** devPtr, size_t size, unsigned int flags = cudaMemAttachGlobal )
    cudaError_t cudaFree ( void* devPtr )
    ```
    
    Compile and test the program.

 4. Modify the program such that is uses the cuBLAS library to perform the AXPY
    operation.
    
    Steps:
    
     - Include the `cublas_v2.h` header file.
     - Initialize cuBLAS with the function:
       
       ```
       cublasStatus_t cublasCreate(cublasHandle_t *handle)
       ```
     
     - Replace the `cblas_daxpy` function call with the corresponding cuBLAS
       function:
       
       ```
       cublasStatus_t cublasDaxpy(cublasHandle_t handle, int n,
                                  const double *alpha,
                                  const double *x, int incx,
                                  double       *y, int incy)
       ```
       
     - Shutdown the cuBLAS library with the function:
     
       ```
       cublasStatus_t cublasDestroy(cublasHandle_t handle)
       ```
    
    Compile and test the program. The program should compile and run without
    any problems but the residual is going to be very large. Why? Can you
    fix the problem?
    
    Hint: Is some familiar command missing? Compare to previous programs.
