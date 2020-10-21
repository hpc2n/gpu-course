# Compiling and batch jobs

## Objective

 - Learn how to load the necessary modules on Kebnekaise.
 - Learn how to compile CUDA code.
 - Learn how to place jobs to job queue.
 - Learn how to use the **course reservations**.

## Instructions

 1. Load the necessary modules (only in Kebnekaise):
 
    ```
    $ ml purge
    $ ml fosscuda/2019b buildenv
    ```
    
    The `purge` command unload existing modules. Note that the `purge` command
    will warn that two modules where not unloaded. This is normal and you should
    **NOT** force unload them. The `fosscuda` module loads the GNU compiler and
    the CUDA SDK. The `buildenv` module sets certain environment variables
    correctly for code compiling.

 2. Compile the `hello.cu` source file with `nvcc` compiler:
 
    ```
    $ nvcc -o hello hello.cu
    ```
    
    In some situations, it is beneficial to pass additional arguments to the
    host compiler (`g++` in this case):
    
    ```
    $ nvcc -o hello hello.cu -Xcompiler="-Wall"
    ```
    
    This passed the `-Wall` flag to `g++`. The flag causes the compiler to print
    extra warnings.

 3. Run the program:
 
    ```
    $ srun -A SNIC2020-9-161 --gres=gpu:v100:1,gpuexcl --time=00:05:00 --ntasks=1 ./hello
    Host says, Hello world!
    GPU says, Hello world!
    ```
    
    This can take a few minutes if several people are trying to use the GPUs
    simultaneously. 
    
    The `srun` command places the program into a queue, `-A SNIC2020-9-161` sets
    the project, `--gres=gpu:v100:1,gpuexcl` requests exclusive access to a
    single Nvidia Tesla V100 GPU (and 14 CPU cores), `--time=00:05:00` sets the
    maximum run time to five minutes, `--ntasks=1` sets the number of tasks to
    one and the last argument the is the program itself.
    
    During the course, you can also use the course reservations (6 GPUs) to get
    faster access:
    
    ```
    $ srun ... --reservation=<reservation name> ...
    ```
    
    The reservation `snic2020-9-161-day1` is valid during Wednesday and the
    reservation `snic2020-9-161-day2` is valid during Thursday.

 4. Create a file called `batch.sh` with the following contents:
 
    ```
    #!/bin/bash
    #SBATCH -A SNIC2020-9-161
    #SBATCH --gres=gpu:v100:1,gpuexcl
    #SBATCH --time=00:05:00
    #SBATCH --ntasks=1

    ml purge
    ml fosscuda/2019b buildenv

    ./hello
    ```
    
    Submit the batch file:
    
    ```
    $ sbatch batch.sh 
    Submitted batch job ....
    ```
    
    You can investigate the job queue with the following command: 
    
    ```
    $ squeue -u $USER
    ```
    
    If you want an estimate for when the job will start running, you can
    give the `squeue` command the argument `--start`. By default, the output of
    the batch file goes to `slurm-<job id>.out`.
