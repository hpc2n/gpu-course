# LU factorization

## Objectives

 - Learn how to distribute computations to both the host and the device.

## Instructions

 1. Carefully read through the `blocked.cu` file. Make sure that you have an
    idea of what each line of code does.

 2. The program requires two arguments. Compile and run the program:
 
    ```
    $ nvcc -o blocked blocked.cu ${LIBBLAS}
    $ srun ... ./blocked 10000 128
    ./blocked 10000 128
    Time = 5.014233 s
    Residual = 5.804721E-16
    ```
    
    Write down your runtime.
    
    The program does the following:
     
     - A random matrix `A` is generated.
     
     - A function `blocked_lu` computes a LU factorization of the matrix `A`:
       
       ```
       A = L * U,
       ```
       
       where `L` is a lower triangular matrix and `U` an upper triangular
       matrix. The computations are done in a blocked manner.
    
     - The final result is validated.
     
    The first program argument defines size of the matrix `A` and the second
    program argument defines the block size.

 3. Copy the `blocked.cu` to a new file called `managed.cu`. Modify the
    `managed.cu`program such that the matrix `A` is allocated using managed
    memory. Align the leading dimension (`ldA`) to 256 bytes (32 doubles).
    
    Compile and test you modified program.
    
 4. Modify the `managed.cu` program such that is uses the cuBLAS library to
    perform the TRSM and GEMM operations. You can use the CHECK_CUBLAS_ERROR
    macro for error checking (see `common.h`). Also, see the earlier hands-on
    [1.basics/4.managed](../../1.basics/4.managed).
    
    More information:
     - https://docs.nvidia.com/cuda/cublas/index.html#cublas-lt-t-gt-trsm
     - https://docs.nvidia.com/cuda/cublas/index.html#cublas-lt-t-gt-gemm
    
    Remember to call `cudaDeviceSynchronize()` before the host accesses any
    data. You must do this in **two** different locations on the code.
    
    Compile and test you modified program. Write down your runtime.

 5. Copy the `managed.cu` to a new file called `manual1.cu`. Modify the
    `manual1.cu`program such that the matrix is allocated and transferred
    **without** using managed memory. Turn the `simple_lu` function into a
    single-threaded kernel.
    
    Compile and test you modified program. Write down your runtime.

 6. Copy the `manual1.cu` to a new file called `manual2.cu`. Modify the
    `manual2.cu` program such that
    
     - the diagonal block is copied to a page locked memory buffer,
     
       ```
       __host__​cudaError_t cudaMallocHost ( void** ptr, size_t size )
        ```
     
     - call the `simple_lu` function (not the kernel) for the diagonal block,
       and
       
     - copy the diagonal block back to global memory.
    
    Compile and test you modified program. Write down your runtime.
