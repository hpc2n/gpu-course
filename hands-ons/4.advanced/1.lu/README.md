# LU factorization

## Objectives

 - Learn what type of computations are suitable for GPUs and which are not.

## Instructions

 1. Read through the `blocked.cu` file. Make sure that you have an overall idea
    of what the code does. There is **no need** to understand the algorithmic
    details.

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
    the leading dimension (`ldA`) to 256 bytes (32 doubles). You can use the
    `CHECK_CUDA_ERROR` macro for error checking (see `common.h`).
    
    Compile and test you modified program.
    
 4. Modify the `managed.cu` program such that is uses the cuBLAS library to
    perform the TRSM and GEMM operations. You can use the `CHECK_CUBLAS_ERROR`
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

 5. Copy `managed.cu` to a new file called `gpu_only.cu`. Modify the
    `gpu_only.cu` program such that the `simple_lu` function is executed as a
    single-threaded kernel on the device.
    
    Compile and test you modified program. Write down your runtime.

 6. Modify both `managed.cu` and `gpu_only.cu` such that profiling is started
    just before the `blocked_lu` function call and stopped after the function
    call and a synchronisation.
    
    Include the `cuda_profiler_api.h` header and use the following functions:
    
    ```
    void cudaProfilerStart ( void )
    void cudaProfilerStop ( void )
    ```

    Use the `nvprof` profiler to analyse both programs:
    
    ```
    $ srun ... nvprof --unified-memory-profiling off ./managed 10000 128
    $ srun ... nvprof --unified-memory-profiling off ./gpu_only 10000 128
    ```
    
    Note that the unified memory profiling needs to be disabled for now.
    
    Compare the "Profiling result" sections. What can you conclude?
