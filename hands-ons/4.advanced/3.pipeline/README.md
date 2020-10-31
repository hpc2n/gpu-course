# Pipelines

## Objectives

 - Learn how to overlap computations and transfers using pipelining.

## Remark

In this hands-on we will permute the rows of a `m`-by-`n` matrix `B` by
multiplying it from the left with a permutation matrix `A`. The overall goal is
to compute the matrix-matrix multiplication `C <- A * B` in a pipelined manner.
The matrix `A` is initialized as a permutation matrix for space and time saving
reasons.
 
## Instructions

 1. Carefully read through the `pipeline.cu` file. Make sure that you have an
    idea of what each line of code does.

 2. The program requires three arguments. Compile and run the program:
 
    ```
    $ nvcc -o pipeline pipeline.cu -lcublas
    $ srun ... ./pipeline 3000 30000 3000
    Runtime was 0.118 s.
    Max error = 0.000000e+00.
    ```
    
    The program does the following:
     
     - A random matrix `B` with `m` rows and `n` columns, and a random 
       permutation matrix `A` with `m` rows and `m` columns are generated.
       
     - The rows of the matrix `B` are permuted by computing a permuted matrix
       `C`:
       
       ```
                  C               A            B
         +-----------------+    +---+ +-----------------+
       m |                 | <- |   | |                 |
         +-----------------+    +---+ +-----------------+ 
         <------- n ------->
       ```
       
     - The matrices `B` and `C` are spliced horizontally into sub-matrices:
     
       ```
                  C               A            B
         +-----------------+    +---+ +-----------------+
         |  :  :##:  :  :  | <- |###| |  :  :##:  :  :  |
         +-----------------+    +---+ +-----------------+ 
                ^-- C*                       ^-- B*
       ```
       
       Each sub-matrix pair (`B*`, `C*`) is processed independently:
       
       ```
       C* <- A * B*
       ```
    
    The program arguments are `m`, `n`, and the splice width, respectively.

 3. Modify the program as follows:
 
     - Rename the stream `stream` as `stream1` and create a second stream called
       `stream2`. Remember to destroy both streams.
    
     - Instead of allocating buffers `_B` and `_C` for the device-side
       computations, allocate two pairs of buffers: (`_B1`, `_C1`) and
       (`_B2`, `_C2`). Each stream gets its own buffer pair. Remember to free
       all buffers.
    
     - Modify the main `for` loop thus that during the first iteration the
       variable `stream` points to the stream `stream1`. Also, set the pointers
       `_B` and `_C` to point to `_B1` and `_C1`, respectively.
    
     - Swap `stream`, `_B` and `_C` after each iteration. That is (`stream`,
       `_B`, `_C`) should alternate between (`stream1`, `_B1`, `_C1`) and
       (`stream2`, `_B2`, `_C2`).
    
    Since the two streams operate independently from each other, computations and
    transfers from one stream can overlap with the computation and transfers from
    the other stream.
    
    Compile and test your modified program

 4. (challenge) Modify the program such that it uses an arbitrary number of
    streams.
