# Profiling

## Objectives

 - Learn how to use the `nv-nsight-cu-cli` profiler.
 - Learn how much you improved the codes yesterday.

## Instructions

 1. Copy your solutions to `2.intermediate/1.sum` and `2.intermediate/2.gemv` to
    this directory. This directory already contains the starting points
    (`sum_start.cu` and `gemv_start.cu`) and the model solutions (`sum_model.cu`
    and `gemv_model.cu`).
    
    Compile all codes.
    
 2. Profile all codes with the `nv-nsight-cu-cli` profiler:
 
    ```
    $ srun .... nv-nsight-cu-cli ./sum_start 500E6 10000
    $ srun .... nv-nsight-cu-cli ./sum_start 500E6 100000
    $ srun .... nv-nsight-cu-cli ./sum_yours 500E6 512
    $ srun .... nv-nsight-cu-cli ./sum_model 500E6 512
    ```
    
    ```
    $ srun .... nv-nsight-cu-cli ./gemv_start 5000 5000
    $ srun .... nv-nsight-cu-cli ./gemv_yours 5000 5000
    $ srun .... nv-nsight-cu-cli ./gemv_model 5000 5000
    ```
    
    Investigate the outputs. Use the "GPU Speed Of Light" (SOL) analysis to
    explain the performance figures.

 3. Calculate arithmetical intensity for both codes:
 
    ```
                                       total number of flops
    arithmetical intensity (AI) = --------------------------------- 
                                  total number of bytes transferred
    ```
    
    Use the following formula to estimate the expected Floprate:
    
    ```
    Floprate = min { Peak floprate, AI * Bandwidth },
    ```
    
    where `Peak floprate = 7000 Gflops` and `Bandwidth = 900 GB/s`. Does your
    results agree with `SM [%]`?
