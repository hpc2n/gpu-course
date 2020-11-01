# Streams, asynchronous data transfers and events

## Objectives

 - Learn how to use streams
 - Learn how to use asynchronous data transfers
 - Learn how to use events

## Instructions

 1. Carefully read through the `axpy.cu` file. Make sure that you have an idea
    of what each line of code does. Note that this is the solution to the
    hands-on [1.basics/3.memory](../../1.basics/3.memory/). You can use your
    own solution if you want.

 2. Create a CUDA stream using the following functions:
 
    ```
    __host__ cudaError_t cudaStreamCreate ( cudaStream_t* pStream )
    __host__ __device__ cudaError_t cudaStreamDestroy ( cudaStream_t stream )
    ```

 3. Pin the vectors `x` and `y` to the host memory:
 
    ```
    __host__ cudaError_t cudaHostRegister (
        void* ptr, size_t size, unsigned int flags )
    __host__ cudaError_t cudaHostUnregister ( void* ptr )
    ```

    Use the `cudaHostRegisterDefault` flag.

 4. Modify the program such that all data transfers are done asynchronously:
 
    ```
    __host__ __device__ cudaError_t cudaMemcpyAsync (
        void* dst, const void* src, size_t count, cudaMemcpyKind kind, 
        cudaStream_t stream = 0 )
    ```

 5. Modify the kernel launch such that the kernel is placed to the stream
    you created:
    
    ```
    axpy_kernel<<<blocks, threads, 0, stream>>>(n, alpha, d_x, d_y);
    ```
    
 6. Wait until the stream is empty:
 
    ```
    __host__ cudaError_t cudaStreamSynchronize ( cudaStream_t stream )
    ```
    
    Compile and run your modified program.

 7. Measure the run time using events:

    ```
    __host__ cudaError_t cudaEventCreate ( cudaEvent_t* event )
    __host__ __device__ cudaError_t cudaEventDestroy ( cudaEvent_t event )
    __host__ __device__ cudaError_t cudaEventRecord (
        cudaEvent_t event, cudaStream_t stream = 0 )
    __host__ cudaError_t cudaEventElapsedTime (
        float* ms, cudaEvent_t start, cudaEvent_t end )
    ```

    Compile and run your modified program.
