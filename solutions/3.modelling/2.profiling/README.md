# Solution

## Step 2

### sum

```
$ run_gpu nv-nsight-cu-cli ./sum_start 500E6 10000
  partial_sum_kernel, 2020-Nov-02 19:24:33, Context 1, Stream 7
    Section: GPU Speed Of Light
    ---------------------------------------------------------------------- --------------- ------------------------------
    Memory Frequency                                                         cycle/usecond                         866,86
    SOL FB                                                                               %                          23,27
    Elapsed Cycles                                                                   cycle                     23 982 976
    SM Frequency                                                             cycle/nsecond                           1,24
    Memory [%]                                                                           %                          23,27
    Duration                                                                       msecond                          19,38
    SOL L2                                                                               %                           8,58
    SM Active Cycles                                                                 cycle                  23 111 929,56
    SM [%]                                                                               %                           1,63
    SOL TEX                                                                              %                           6,76
    ---------------------------------------------------------------------- --------------- ------------------------------
```

We can see that `SOL FB` and `Memory [%]` are both around 23% which is low. 
`SM [%]` is very low and indicates that the code is memory bound.

```
$ run_gpu nv-nsight-cu-cli ./sum_start 500E6 100000 
  partial_sum_kernel, 2020-Nov-02 19:18:14, Context 1, Stream 7
    Section: GPU Speed Of Light
    ---------------------------------------------------------------------- --------------- ------------------------------
    Memory Frequency                                                         cycle/usecond                         879,44
    SOL FB                                                                               %                          93,00
    Elapsed Cycles                                                                   cycle                      6 002 517
    SM Frequency                                                             cycle/nsecond                           1,25
    Memory [%]                                                                           %                          93,00
    Duration                                                                       msecond                           4,78
    SOL L2                                                                               %                          34,63
    SM Active Cycles                                                                 cycle                   5 881 348,64
    SM [%]                                                                               %                           6,52
    SOL TEX                                                                              %                          26,57
    ---------------------------------------------------------------------- --------------- ------------------------------
```

We can see that `SOL FB` and `Memory [%]` are both around 93% which is high. `SM [%]` is
very low and indicates that the code is memory bound.

```
$ run_gpu nv-nsight-cu-cli ./sum_model 500E6 512
  partial_sum_kernel, 2020-Nov-02 19:19:42, Context 1, Stream 7
    Section: GPU Speed Of Light
    ---------------------------------------------------------------------- --------------- ------------------------------
    Memory Frequency                                                         cycle/usecond                         875,44
    SOL FB                                                                               %                          90,02
    Elapsed Cycles                                                                   cycle                      6 200 061
    SM Frequency                                                             cycle/nsecond                           1,25
    Memory [%]                                                                           %                          90,02
    Duration                                                                       msecond                           4,96
    SOL L2                                                                               %                          33,24
    SM Active Cycles                                                                 cycle                   6 065 229,35
    SM [%]                                                                               %                           6,31
    SOL TEX                                                                              %                          25,76
    ---------------------------------------------------------------------- --------------- ------------------------------

  partial_sum_kernel, 2020-Nov-02 19:19:43, Context 1, Stream 7
    Section: GPU Speed Of Light
    ---------------------------------------------------------------------- --------------- ------------------------------
    Memory Frequency                                                         cycle/usecond                         664,74
    SOL FB                                                                               %                           0,26
    Elapsed Cycles                                                                   cycle                          5 277
    SM Frequency                                                             cycle/usecond                         953,00
    Memory [%]                                                                           %                           0,26
    Duration                                                                       usecond                           5,54
    SOL L2                                                                               %                           0,11
    SM Active Cycles                                                                 cycle                          44,52
    SM [%]                                                                               %                           0,03
    SOL TEX                                                                              %                           6,96
    ---------------------------------------------------------------------- --------------- ------------------------------
```

We can see that for the first kernel `SOL FB` and `Memory [%]` are both around
90% which is high. `SM [%]` is very low and indicates that the code is memory
bound.

However, the performance of the second kernel is much lower. When looking at the
`Duration` field, we can see that the starting point kernel was actually faster
than the two model solution kernels combined. However, note that these numbers
do not include transfer times etc.

Also remember that the goal of the `2.intermediate/1.sum` hands-on was not to
write an optimal summation code. The point was to learn to use shared memory.

### gemv

```
$ run_gpu nv-nsight-cu-cli ./gemv_start 5000 5000
  gemv_kernel, 2020-Nov-02 19:35:25, Context 1, Stream 7
    Section: GPU Speed Of Light
    ---------------------------------------------------------------------- --------------- ------------------------------
    Memory Frequency                                                         cycle/usecond                         876,11
    SOL FB                                                                               %                          33,46
    Elapsed Cycles                                                                   cycle                        848 635
    SM Frequency                                                             cycle/nsecond                           1,25
    Memory [%]                                                                           %                          33,46
    Duration                                                                       usecond                         678,56
    SOL L2                                                                               %                          12,25
    SM Active Cycles                                                                 cycle                     409 305,97
    SM [%]                                                                               %                           2,31
    SOL TEX                                                                              %                          19,24
    ---------------------------------------------------------------------- --------------- ------------------------------
```

We can see that `SOL FB` and `Memory [%]` are both around 33% which is low. 
`SM [%]` is very low and indicates that the code is memory bound.

```
$ run_gpu nv-nsight-cu-cli ./gemv_model 5000 5000     
  gemv_kernel, 2020-Nov-02 19:35:52, Context 1, Stream 7
    Section: GPU Speed Of Light
    ---------------------------------------------------------------------- --------------- ------------------------------
    Memory Frequency                                                         cycle/usecond                         867,19
    SOL FB                                                                               %                          95,91
    Elapsed Cycles                                                                   cycle                        295 731
    SM Frequency                                                             cycle/nsecond                           1,24
    Memory [%]                                                                           %                          95,91
    Duration                                                                       usecond                         239,14
    SOL L2                                                                               %                          35,45
    SM Active Cycles                                                                 cycle                     287 283,36
    SM [%]                                                                               %                           7,17
    SOL TEX                                                                              %                          27,63
    ---------------------------------------------------------------------- --------------- ------------------------------
```

We can see that `SOL FB` and `Memory [%]` are both around 95% which is high.
`SM [%]` is very low and indicates that the code is memory bound.

## Step 3

### sum

We get

```
                               n - 1              1
arithmetical intensity (AI) = ------- ---------> --- Flop / Byte.
                               8 * n   n -> inf   8
```

Therefore,

```
Floprate = min { Peak floprate, AI * Bandwidth } 
         = 900 / 8 GFlops 
         = 112.5 GFlops.
```

This is `112.5 GFlops / 7000 GFlops = 0.016 = 1.6%` of the theoretical peak.
The number is not identical with `SM [%]` but they are relatively close to each
other.

### gemm

We get

```
                                  n * (2 * n)                   1
arithmetical intensity (AI) = --------------------- ---------> --- Flop / Byte.
                               n * (2 * n + 1) * 8   n -> inf   8
```

Therefore,

```
Floprate = min { Peak floprate, AI * Bandwidth } 
         = 900 / 8 GFlops 
         = 112.5 GFlops.
```

This is `112.5 GFlops / 7000 GFlops = 0.016 = 1.6%` of the theoretical peak.
The number is not identical with `SM [%]` but they are relatively close to each
other.
