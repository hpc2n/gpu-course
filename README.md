# gpu_course

Introduction to GPU programming: When and how to use GPU-acceleration?

```
$ ml intelcuda/2019a buildenv
$ nvcc -o hello.cuda hello.cu
$ srun -A SNIC2019-5-41 --gres=gpu:v100:1,gpuexcl --time=00:05:00 --ntasks=1 ./hello.cuda
Host says, Hello world!
GPU says, Hello world!
```
