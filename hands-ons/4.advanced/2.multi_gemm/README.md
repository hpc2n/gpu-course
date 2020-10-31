# Multiple CUDA streams

## Objectives

 - Learn how to manage multiple streams.

## Remark

In this hands-on we will permute the rows of a `m`-by-`n` matrix `B` by
multiplying it from the left with a permutation matrix `A`. The overall goal is
to compute the matrix-matrix multiplication `C <- A * B`. The matrix `A` is
initialized as a permutation matrix for space and time saving reasons.
 
## Instructions

 1. Carefully read through the `gemms.cu` file. Make sure that you have an
    idea of what each line of code does.

 2. The program requires two arguments. Compile and run the program:
 
    ```
    $ nvcc -o gemms gemms.cu -lcublas
    $ srun ... ./gemms 1000 50
    Runtime was 0.644 s.
    Max error = 0.000000e+00
    ```
    
    The program does the following:
     
     - A set of matrix triplets `{(A, B, C)}}` is generated. In each triplet,
       `A` is a random `n`-by-`n` matrix and `B` is a random `n`-by-`n`
       permutation matrix.
       
     - For each triplet `(A, B, C)}`, we compute:
       
       ```
       C <- A * B
       ```
    
    The program arguments are `n` and the size of the set `{(A, B, C)}}`.

 3. Modify the program as follows:
 
     - Create a stream for each triplet in the set `{(A, B, C)}}`. Remember
       to destroy all stream.
     
     - Pin the buffers `A[i]`, `B[i]`, `C[i]`, `i = 0, ..., count-1`, to the
       host memory. Remember to unpin the memory.
       
     - Modify the program such that all data transfers are done asynchronously.
       Each matrix triplet should use a different stream.
     
     - Place each GEMM call to a different stream by calling the following
       function **before** each GEMM call:
       
       ```
       cublasStatus_t cublasSetStream(cublasHandle_t handle, cudaStream_t streamId)
       ```
    
    Since the streams operate independently from each other, computations and
    transfers from one stream can overlap with the computations and transfers
    from the other stream.
    
    Compile and test your modified program

 4. (challenge) Modify the program such that all matrix-matrix multiplications are
    performed using batched BLAS routines:
    
    ```
    cublasStatus_t cublasDgemmBatched(
        cublasHandle_t handle, 
        cublasOperation_t transa, cublasOperation_t transb,
        int m, int n, int k,
        const double *alpha,
        const double *Aarray[], int lda,
        const double *Barray[], int ldb,
        const double *beta,
        double       *Carray[], int ldc,
        int batchCount)
    ```
    
    More information: https://docs.nvidia.com/cuda/cublas/index.html#cublas-lt-t-gt-gemmbatched
