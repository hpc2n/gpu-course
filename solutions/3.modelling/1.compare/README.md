# Solution

## Step 2

```
$ OMP_NUM_THREADS=1 run_gpu ./ax.cpu 500E6
Time = 0.378740 s
Floprate = 1.3 GFlops
Memory throughput = 21 GB/s
$ OMP_NUM_THREADS=14 run_gpu ./ax.cpu 500E6
Time = 0.088828 s
Floprate = 5.6 GFlops
Memory throughput = 90 GB/s
```

The core count does effect the performance but the code does not scale past
four cores. The memory bus gets saturated. The reported floprate and memory
throughput are within the expected ranges.

```
$ run_gpu ./ax.cuda 500E6
Time = 0.010629 s
Floprate = 47.0 GFlops
Memory throughput = 753 GB/s
```

The GPU is about eight times faster. The reported floprate and memory throughput
are within the expected ranges.

```
$ OMP_NUM_THREADS=1 run_gpu ./gemm.cpu 5000
Runtime was 4.968 s.
Floprate was 50 GFlops.
Memory throughput (naive) 3221 GB/s.
$ OMP_NUM_THREADS=14 run_gpu ./gemm.cpu 5000
Runtime was 0.525 s.
Floprate was 476 GFlops.
Memory throughput (naive) 30453 GB/s.
```

The core count does effect the performance and the code scales past 10 cores.
The reported floprate is within the expected range. The reported memory
throughput is incorrect because modern GEMM implementations cache data to the
L2 and L1 caches.

```
$ run_gpu ./gemm.cuda 5000
Runtime was 0.041 s.
Floprate was 6078 GFlops.
Memory throughput (naive) 389032 GB/s.
```

The GPU is about 13 times faster. The reported floprate is within the expected
range. The reported memory throughput is incorrect because modern GEMM
implementations cache data to the shared memory.

## Step 3

### AX

```
$ OMP_NUM_THREADS=14 run_gpu ./ax.cpu 1000
Time = 0.026744 s
Floprate = 0.0 GFlops
Memory throughput = 0 GB/s
$ run_gpu ./ax.cuda 1000
Time = 0.000029 s
Floprate = 0.0 GFlops
Memory throughput = 1 GB/s
```

The measurements are meaningless for 1000. 

```
$ OMP_NUM_THREADS=14 run_gpu ./ax.cpu 1000000
Time = 0.000632 s
Floprate = 1.6 GFlops
Memory throughput = 25 GB/s
$ run_gpu ./ax.cuda 1000000
Time = 0.000087 s
Floprate = 11.5 GFlops
Memory throughput = 185 GB/s
```

The performance has dropped significantly. The problem size should clearly be
larger. 

### GEMM

```
$ OMP_NUM_THREADS=14 run_gpu ./gemm.cpu 500
Runtime was 0.018 s.
Floprate was 14 GFlops.
Memory throughput (naive) 911 GB/s.
$ run_gpu ./gemm.cuda 500
Runtime was 0.000 s.
Floprate was 1330 GFlops.
Memory throughput (naive) 85189 GB/s.
```

The performance has dropped significantly. The problem size should clearly be
larger. 

```
$ OMP_NUM_THREADS=14 run_gpu ./gemm.cpu 1000
Runtime was 0.039 s.
Floprate was 51 GFlops.
Memory throughput (naive) 3256 GB/s.
$ run_gpu ./gemm.cuda 1000
Runtime was 0.001 s.
Floprate was 3862 GFlops.
Memory throughput (naive) 247302 GB/s.
```

The CPU performance has dropped significantly. The problem size should clearly
be larger. The GPU is performing reasonably well. 

```
$ OMP_NUM_THREADS=14 run_gpu ./gemm.cpu 2000
Runtime was 0.060 s.
Floprate was 267 GFlops.
Memory throughput (naive) 17101 GB/s.
$ run_gpu ./gemm.cuda 2000
Runtime was 0.003 s.
Floprate was 4721 GFlops.
Memory throughput (naive) 302243 GB/s.
```

The CPU performance has dropped significantly. The problem size should clearly
be larger. The GPU is performing reasonably well. 

## Step 4

```
$ run_gpu nv-nsight-cu-cli ./ax.cuda 500E6
  ax_kernel, 2020-Oct-30 17:40:42, Context 1, Stream 7
    Section: GPU Speed Of Light
    ---------------------------------------------------------------------- --------------- ------------------------------
    Memory Frequency                                                         cycle/usecond                         875,84
    SOL FB                                                                               %                          84,58
    Elapsed Cycles                                                                   cycle                     13 181 137
    SM Frequency                                                             cycle/nsecond                           1,25
    Memory [%]                                                                           %                          84,58
    Duration                                                                       msecond                          10,55
    SOL L2                                                                               %                          31,21
    SM Active Cycles                                                                 cycle                  12 837 852,30
    SM [%]                                                                               %                           3,34
    SOL TEX                                                                              %                          15,21
    ---------------------------------------------------------------------- --------------- ------------------------------
```

The field `SM [%]` indicates that CUDA cores are mostly inactive. The `SOL FB`
and `Memory [%]` indicate than the memory bus is busy. The code is memory bound.

```
$ run_gpu nv-nsight-cu-cli ./gemm.cuda 5000
    Section: GPU Speed Of Light
    ---------------------------------------------------------------------- --------------- ------------------------------
    Memory Frequency                                                         cycle/usecond                         878,08
    SOL FB                                                                               %                          36,45
    Elapsed Cycles                                                                   cycle                     51 458 926
    SM Frequency                                                             cycle/nsecond                           1,25
    Memory [%]                                                                           %                          37,18
    Duration                                                                       msecond                          41,06
    SOL L2                                                                               %                          23,69
    SM Active Cycles                                                                 cycle                  50 755 341,94
    SM [%]                                                                               %                          98,31
    SOL TEX                                                                              %                          37,68
    ---------------------------------------------------------------------- --------------- ------------------------------
```

The field `SM [%]` indicates that CUDA cores are busy. The `SOL FB` and `Memory [%]` 
indicate than the memory bus reasonably busy. The code is memory bound.

## Step 5

### AX

```
diff --git a/hands-ons/3.modelling/1.compare/ax.cu b/hands-ons/3.modelling/1.compare/ax.cu
index b5f8bef..d5699ba 100644
--- a/hands-ons/3.modelling/1.compare/ax.cu
+++ b/hands-ons/3.modelling/1.compare/ax.cu
@@ -78,15 +78,15 @@ int main(int argc, char const **argv)
     double *d_y;
     CHECK_CUDA_ERROR(cudaMalloc(&d_y, n*sizeof(double)));
 
+    // start timer
+    struct timespec ts_start;
+    clock_gettime(CLOCK_MONOTONIC, &ts_start);
+
     // copy the vector from the host memory to the device memory
 
     CHECK_CUDA_ERROR(
         cudaMemcpy(d_y, y, n*sizeof(double), cudaMemcpyHostToDevice));
 
-    // start timer
-    struct timespec ts_start;
-    clock_gettime(CLOCK_MONOTONIC, &ts_start);
-
     // launch the kernel
 
     dim3 threads = 256;
```

```
$ run_gpu ./a.out 500E6
Time = 0.835948 s
Floprate = 0.6 GFlops
Memory throughput = 10 GB/s
```

The performance has dropped to one 79th of what it was earlier. The PCI-E bus
(16GB/s to both directions) forms a bottleneck 

### GEMM

```
diff --git a/hands-ons/3.modelling/1.compare/gemm.cu b/hands-ons/3.modelling/1.compare/gemm.cu
index b9304f3..7eca6a4 100644
--- a/hands-ons/3.modelling/1.compare/gemm.cu
+++ b/hands-ons/3.modelling/1.compare/gemm.cu
@@ -77,6 +77,10 @@ int main(int argc, char const **argv)
     CHECK_CUDA_ERROR(cudaMalloc(&_B, n*ldB*sizeof(double)));
     CHECK_CUDA_ERROR(cudaMalloc(&_C, n*ldC*sizeof(double)));
 
+    // start timer
+    struct timespec ts_start;
+    clock_gettime(CLOCK_MONOTONIC, &ts_start);
+
     // copy to the device memory
 
     CHECK_CUDA_ERROR(
@@ -84,10 +88,6 @@ int main(int argc, char const **argv)
     CHECK_CUDA_ERROR(
         cudaMemcpy(_B, B, n*ldB*sizeof(double), cudaMemcpyHostToDevice));
 
-    // start timer
-    struct timespec ts_start;
-    clock_gettime(CLOCK_MONOTONIC, &ts_start);
-
     // launch the kernel
 
     double one = 1.0, zero = 0.0;
```

```
$ run_gpu ./a.out 5000
Runtime was 0.144 s.
Floprate was 1732 GFlops.
Memory throughput (naive) 110881 GB/s.
```

The performance has dropped to one fourth of what it was earlier. The PCI-E bus
(16GB/s to both directions) forms a bottleneck but the effect is less drastic
because data is being cached to the shared memory.
