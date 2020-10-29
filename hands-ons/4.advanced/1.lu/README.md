# LU factorization

## Objectives

 - Learn how to distribute computations to both the host and the device.

## Instructions

 1. Carefully read through the `blocked.cu` file. Make sure that you have an
    idea of what each line of code does.

 2. Compile the program:
 
    ```
    $ nvcc -o blocked blocked.cu ${LIBLAPACK} ${LIBBLAS}
    ```
    
    The program requires two arguments. Run the program both sequentially and
    in parallel:
    
    ```
    $ OMP_NUM_THREADS=1 srun ... ./blocked 10000 128
    Time = ....
    Residual = 5.888637E-16
    $ OMP_NUM_THREADS=14 srun ... ./blocked 10000 128
    Time = ....
    Residual = 5.815569E-16
    ```
    
    Write down your runtimes.
    
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

 3. Copy `blocked.cu` to a new file called `managed.cu`. Modify the `managed.cu`
    program such that the matrix `A` is allocated using managed memory. Align
    the leading dimension (`ldA`) to 256 bytes (32 doubles).
    
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
    
    Compile and test you modified program. Write down your runtime:
    
    ```
    $ nvcc -o managed managed.cu -lcublas ${LIBLAPACK} ${LIBBLAS}
    $ OMP_NUM_THREADS=14 srun ....
    ```

 5. Copy `managed.cu` to a new file called `manual1.cu`. Modify the `manual1.cu`
    program such that the matrix is allocated and transferred **without** using
    managed memory. For now, turn the `simple_lu` function into a
    single-threaded kernel. 
    
    Compile and test you modified program. Write down your runtime.

 6. Copy `manual1.cu` to a new file called `manual2.cu`. Modify the `manual2.cu`
    program such that
    
     - each diagonal block is copied to a page-locked host memory buffer `T`,
     
       ```
       __host__ cudaError_t cudaMallocHost ( void** ptr, size_t size )
       __host__ cudaError_t cudaFreeHost ( void* ptr )
       __host__ cudaError_t cudaMemcpy2D (
           void* dst, size_t dpitch, const void* src, size_t spitch, 
           size_t width, size_t height, cudaMemcpyKind kind )
        ```
     
     - the `simple_lu` function (not the kernel) is called for the buffer `T`,
       and
       
     - the buffer `T` is copied back to the diagonal block.
    
    Compile and test you modified program. Write down your runtime.
