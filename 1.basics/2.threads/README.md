# Hello world!

## Objectives

 - Learn what CUDA kernels.
 - Learn what threads, thread blocks and grids are.

## Remark

In this hands-on, we are using very small thread block sizes for illustration
purposes only. In practice, the threads blocks should be significantly larger.
Furthermore, using the `printf` function inside kernels is very expensive and
should be avoided.

## Instructions

 1. Carefully read through the `hello.cu` file. Make sure that you have an idea
    of what each line of code does.

 2. Modify the kernel such that it prints the thread index:
 
    ```
    printf("Thread %d says, Hello world!\n", threadIdx.x);
    ```
    
    Change the thread block size to 16 (the second argument between the
    `<<< . , . >>>` brackets). Run the modified program. How many rows are
    printed? The correct number should be 17. Why?

 3. Modify the kernel such that it prints the thread block index:
 
    ```
    printf("Thread %d in block %d says, Hello world!\n",
        threadIdx.x, blockIdx.x);
    ```
 
    Change the thread grid size to 2 (the first argument between the
    `<<< . , . >>>` brackets). Run the modified program. How many rows are
    printed? The correct number should be 33. Why?

 4. Modify the kernel such that the threads that belong to the thread block `0`
    print "Hello world!" and the threads that belong to the thread block `1`
    print "Hello Umea!". Run the modified program.

 5. Modify the the kernel such that each thread prints an unique global index
    number. Remember,
     - `threadIdx.x` is thread's index number inside a thread block.
     - `blockDim.x` is the thread block size, i.e., how many threads are in a
       thread block.
     - `blockIdx.x` is the thread block's index number.
     - `gridDim.x` is the thread grid size, i.e., how many thread blocks are in
       the grid.
    
    One way of computing an unique index number is to multiply the thread block
    index with the thread block size and add thread's index number. Run the
    modified program. Check that the indexes `0-31` are printed.

 6. Modify the program such that the thread block dimensions are `4 x 2` and 
    the thread grid dimensions are `2 x 3`:
 
    ```
    dim3 threads(4, 2);
    dim3 blocks(2, 3);
    say_hello<<<blocks,threads>>>();
    ```
    
    Modify the kernel such that it prints the thread and thread block indexes
    in both `x` and `y` dimensions. Run the modified program. Check that the
    thread indexes `(0,0) - (3,1)` and the thread block indexes `(0,0) - (1,2)`
    are printed.
    
 7. Comment out the `cudaDeviceSynchronize()` call. What happens? Why?
 
