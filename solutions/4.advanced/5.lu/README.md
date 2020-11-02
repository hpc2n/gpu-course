# Solution

## Step 6

### managed.cu

```
$ run_gpu nvprof --unified-memory-profiling off --profile-from-start off ./managed 10000 128
srun: job 10432169 queued and waiting for resources
srun: job 10432169 has been allocated resources
==941854== NVPROF is profiling process 941854, command: ./managed 10000 128
==941854== Profiling application: ./managed 10000 128
Time = 0.946326 s
Residual = 5.845104E-16
==941854== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   44.54%  170.38ms        38  4.4837ms  1.7520ms  97.955ms  void kernel_trsm_l_mul32<double, int=8, bool=0, bool=0, bool=0, bool=1>(int, int, double const *, double const *, int, double*, int, double, int)
                   28.46%  108.88ms        48  2.2683ms  697.21us  4.5168ms  volta_dgemm_128x64_nn
                   22.34%  85.458ms        78  1.0956ms  822.68us  1.4914ms  void trsm_left_kernel<double, int=256, int=4, bool=1, bool=0, bool=0, bool=0, bool=1>(cublasTrsmParams<double>, double, double const *, int)
                    2.48%  9.4871ms        80  118.59us  20.768us  702.33us  volta_dgemm_64x64_nn
                    1.69%  6.4746ms       156  41.504us  19.040us  76.127us  void trsm_right_kernel<double, int=256, int=4, bool=1, bool=0, bool=0, bool=1, bool=0>(cublasTrsmParams<double>, double, double const *, int)
                    0.30%  1.1481ms         1  1.1481ms  1.1481ms  1.1481ms  void trsm_ln_kernel<double, unsigned int=32, unsigned int=32, unsigned int=4, bool=1>(int, int, double const *, int, double*, int, double, double const *, int, int*)
                    0.19%  710.75us        66  10.768us  5.2800us  18.816us  void gemm_kernel2x2_tile_multiple_core<double, bool=1, bool=0, bool=0, bool=0, bool=0>(double*, double const *, double const *, int, int, int, int, int, int, double*, double*, double, double, int)
                    0.00%  7.0080us         1  7.0080us  7.0080us  7.0080us  void gemmSN_NN_kernel<double, int=128, int=2, int=4, int=8, int=4, int=4, cublasGemvTensorStridedBatched<double const >, cublasGemvTensorStridedBatched<double>>(cublasGemmSmallNParams<double const , cublasGemvTensorStridedBatched<double const >, double>)
                    0.00%  1.9840us         1  1.9840us  1.9840us  1.9840us  dtrsv_init(int*)
      API calls:   66.69%  380.36ms        80  4.7544ms  2.3590us  102.54ms  cudaDeviceSynchronize
                   33.29%  189.87ms       469  404.85us  4.6590us  186.59ms  cudaLaunchKernel
                    0.02%  108.07us       702     153ns      90ns  4.9130us  cudaGetLastError
                    0.00%  4.9390us         1  4.9390us  4.9390us  4.9390us  cudaEventQuery
                    0.00%  2.7540us         1  2.7540us  2.7540us  2.7540us  cudaEventRecord
```

### gpu_only.cu

```
$ run_gpu nvprof --unified-memory-profiling off --profile-from-start off ./gpu_only 10000 128
srun: job 10432159 queued and waiting for resources
srun: job 10432159 has been allocated resources
==941678== NVPROF is profiling process 941678, command: ./gpu_only 10000 128
==941678== Profiling application: ./gpu_only 10000 128
Time = 3.434807 s
Residual = 5.818241E-16
==941678== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   93.06%  3.01804s        79  38.203ms  87.071us  43.778ms  simple_lu(int, int, double*)
                    3.33%  107.86ms        48  2.2471ms  689.47us  4.5173ms  volta_dgemm_128x64_nn
                    3.05%  98.954ms        38  2.6041ms  86.815us  94.595ms  void kernel_trsm_l_mul32<double, int=8, bool=0, bool=0, bool=0, bool=1>(int, int, double const *, double const *, int, double*, int, double, int)
                    0.29%  9.4531ms        80  118.16us  21.504us  694.30us  volta_dgemm_64x64_nn
                    0.19%  6.3145ms       156  40.477us  18.784us  75.871us  void trsm_right_kernel<double, int=256, int=4, bool=1, bool=0, bool=0, bool=1, bool=0>(cublasTrsmParams<double>, double, double const *, int)
                    0.05%  1.5810ms        78  20.269us  11.616us  30.784us  void trsm_left_kernel<double, int=256, int=4, bool=1, bool=0, bool=0, bool=0, bool=1>(cublasTrsmParams<double>, double, double const *, int)
                    0.02%  676.09us        66  10.243us  4.1600us  18.336us  void gemm_kernel2x2_tile_multiple_core<double, bool=1, bool=0, bool=0, bool=0, bool=0>(double*, double const *, double const *, int, int, int, int, int, int, double*, double*, double, double, int)
                    0.00%  60.735us         1  60.735us  60.735us  60.735us  void trsm_ln_kernel<double, unsigned int=32, unsigned int=32, unsigned int=4, bool=1>(int, int, double const *, int, double*, int, double, double const *, int, int*)
                    0.00%  6.6560us         1  6.6560us  6.6560us  6.6560us  void gemmSN_NN_kernel<double, int=128, int=2, int=4, int=8, int=4, int=4, cublasGemvTensorStridedBatched<double const >, cublasGemvTensorStridedBatched<double>>(cublasGemmSmallNParams<double const , cublasGemvTensorStridedBatched<double const >, double>)
                    0.00%  1.3760us         1  1.3760us  1.3760us  1.3760us  dtrsv_init(int*)
      API calls:   94.47%  3.23996s         1  3.23996s  3.23996s  3.23996s  cudaDeviceSynchronize
                    5.52%  189.48ms       548  345.76us  4.6360us  186.57ms  cudaLaunchKernel
                    0.00%  92.027us       702     131ns      89ns  1.5510us  cudaGetLastError
                    0.00%  5.3560us         1  5.3560us  5.3560us  5.3560us  cudaEventQuery
                    0.00%  2.1880us         1  2.1880us  2.1880us  2.1880us  cudaEventRecord
```  
                  
