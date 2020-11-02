# Floprate and memory throughput

## Objectives

 - Learn about floprates and memory throughputs.
 - Learn how to use the `nv-nsight-cu-cli` profiler.

## Instructions

 1. Compile all four codes:
 
    ```
    $ gcc -o ax.cpu ax.c -fopenmp -O3 -march=native
    $ nvcc -o ax.cuda ax.cu
    $ gcc -o gemm.cpu gemm.c ${LIBBLAS}
    $ nvcc -o gemm.cuda gemm.cu -lcublas
    ```
    
 2. Run all four programs. Write down the run times, floprates and memory
    throughputs. 
 
    ```
    $ OMP_NUM_THREADS=1 srun .... ./ax.cpu 500E6
    $ OMP_NUM_THREADS=14 srun .... ./ax.cpu 500E6
    ```
    
    Does the core count effect the performance? How well does the code scale?
    Is it 14 times faster with `OMP_NUM_THREADS=14`? Why? Is the reported
    floprate and memory throughput reasonable? Why?
    
    ```
    $ srun .... ./ax.cuda 500E6
    ```
    
    How much faster is the GPU? Is the reported floprate and memory throughput
    reasonable? Why?
    
    ```
    $ OMP_NUM_THREADS=1 srun .... ./gemm.cpu 5000
    $ OMP_NUM_THREADS=14 srun .... ./gemm.cpu 5000
    ```
    
    Does the core count effect the performance? How well does the code scale?
    Is it 14 times faster with `OMP_NUM_THREADS=14`? Why? Is the reported
    floprate and memory throughput reasonable? Why?
    
    ```
    $ srun .... ./gemm.cuda 5000
    ```
    
    How much faster is the GPU? Is the reported floprate and memory throughput
    reasonable? Why?

 3. Try different problem sizes:
 
     - ax: 1000 1000000
     - gemm: 500 1000 2000
     
    How do the codes compare against each other.
    
 4. Profile the codes with the `nv-nsight-cu-cli` profiler:
 
    ```
    $ srun .... nv-nsight-cu-cli ax.cuda 500E6
    $ srun .... nv-nsight-cu-cli gemm.cuda 5000
    ```
    
    Investigate the output. Use the "GPU Speed Of Light" (SOL) analysis to
    explain the performance figures.

 5. Modify the CUDA codes such that the timing is started before the data
    transfers. How does this effect the performance? Why?
