# Error handling

## Objectives

 - Learn how to handle errors.

## Remark

In this hands-on, we are using very small thread block sizes for illustration
purposes only. In practice, the threads blocks should be significantly larger.
Furthermore, using the `printf` function inside kernels is very expensive and
should be avoided.

## Instructions

 1. Carefully read through the `errors.cu` file. Make sure that you have an idea
    of what each line of code does. In particular, pay attention to how the
    array is initialized:
    
    ```
    for (int i = 0; i < n; i++)
        host[i] = i;
    ```

    And what the kernel does:
    
    ```
    // a kernel that prints the contents of an array
    __global__ void print_array(int n, int *array)
    {
        int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
        if (thread_id < n)
            printf("array[%d] = %d\n", thread_id, array[thread_id]);
    }
    ```
    
 2. Compile and run the program. You should see that instead of printing the
    contents of the array, the program prints the following:
    
    ```
    array[0] = 0
    array[1] = 0
    array[2] = 0
    ...
    ```
    
    Even trough the program compiles and seems to run without any errors, the
    output is clearly incorrect.

 3. Use the `cudaGetLastError` function to investigate whether any errors
    occurred when the kernel was launched:
 
    ```
    cudaError_t cudaGetLastError ( void )
    ```

    Query the error code just after the kernel launch. If the error code is not
    `cudaSuccess`, print the error code and the corresponding error message
    using the following functions:
    
    ```
    const char* cudaGetErrorName ( cudaError_t error )
    const char* cudaGetErrorString ( cudaError_t error )
    ```

    Compile and run the program. Was there an error? Do you see anything wrong
    with the kernel? It did print something...
    
 4. Most CUDA API calls return an error code. This includes the `cudaMalloc`,
    `cudaMemcpy` and `cudaDeviceSynchronize` functions:
    
    ```
    ​cudaError_t cudaMalloc ( void** devPtr, size_t size )
    cudaError_t cudaMemcpy ( void* dst, const void* src, size_t count, cudaMemcpyKind kind )
    cudaError_t cudaDeviceSynchronize ( void )
    ```
    
    Check the error codes returned by all these three function calls. Also, 
    print the name of the function so that you can more easily see which
    function returned the error code. 
    
    Compile, run the program and locate the function that caused the problem.
    
    Can you see anything wrong with it? Can you fix it? Why did the kernel
    launch return an error?
    
    You can use the following macro to make error checks easier:
    
    ```
    #define CHECK_CUDA_ERROR(exp) {                 \
        cudaError_t ret = (exp);                    \
        fprintf(stderr, "[error] %s:%d: %s (%s)\n", \
            __FILE__, __LINE__,                     \
            cudaGetErrorName(ret),                  \
            cudaGetErrorString(ret));               \
        exit(EXIT_FAILURE);                         \
    }
    ```
