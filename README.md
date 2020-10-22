# Introduction to GPU programming: When and how to use GPU-acceleration? (fall 2019)

## Course description

GPU-acceleration has been shown to provide significant performance benefits in
many different applications. However, for a novice, or even for a moderately
experienced scientist or programmer, it is not always clear which applications
could potentially benefit from GPU-acceleration and which do not. For example, a
Nvidia V100 GPU can perform artificial intelligence (AI) related computations in
a fraction of the time it takes a regular CPU to perform the same computations
but ill-informed OpenACC compiler pragma can actually make a code run slower.
Why is this? When should one invest time in GPU-acceleration? How much speedup
can be expected with a given application?

**Purpose**: The main goal of this one day course is to start answering these
questions. The course also covers the basics of GPU programming and aims to
provide the necessary information for avoiding the most common pitfalls.

**Requirements**: The course does not require any existing GPU programming
knowledge but basic understanding of the C language is required for the
hands-ons.

**Date**: 2019-11-05

**Time**: 9:00-17:00

**Location**: MC313

**Instructors**: Mirko Myllykoski (HPC2N)

**Helpers**: Birgitte Brydsö (HPC2N)

## Materials

 - [Introduction to the course](https://www.hpc2n.umu.se/sites/default/files/conferences-courses/2019/GPU-intro/intro.pdf)
 - [Introduction to HPC2N and Kebnekaise](https://www.hpc2n.umu.se/sites/default/files/conferences-courses/2019/GPU-intro/intro-kebnekaise.pdf)
 - [GPU hardware and CUDA basics](https://www.hpc2n.umu.se/sites/default/files/conferences-courses/2019/GPU-intro/basics.pdf)
 - [Where is my performance?](https://www.hpc2n.umu.se/sites/default/files/conferences-courses/2019/GPU-intro/performance.pdf)

## Howto

```
$ ml intelcuda/2019a buildenv
$ nvcc -o hello.cuda hello.cu
$ srun -A SNIC2019-5-142 --gres=gpu:v100:1,gpuexcl --time=00:05:00 --ntasks=1 ./hello.cuda
Host says, Hello world!
GPU says, Hello world!
```
