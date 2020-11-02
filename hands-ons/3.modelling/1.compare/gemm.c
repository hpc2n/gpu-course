#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <cblas.h>

int DIVCEIL(int a, int b)
{
    return (a+b-1)/b;
}

int main(int argc, char const **argv)
{
    // read and validate the command line arguments

    if (argc < 2) {
        fprintf(stderr, "[error] No matrix size was supplied.\n");
        return EXIT_FAILURE;
    }

    int n = atof(argv[1]);
    if (n < 1) {
        fprintf(stderr, "[error] The matrix size was invalid.\n");
        return EXIT_FAILURE;
    }
    
    srand(time(NULL));
    
    // allocate memory

    int ldA, ldB, ldC;
    ldA = ldB = ldC = DIVCEIL(n, 8)*8; // align to 64 bytes
    double *A = (double *) aligned_alloc(8, n*ldB*sizeof(double));
    double *B = (double *) aligned_alloc(8, n*ldB*sizeof(double));
    double *C = (double *) aligned_alloc(8, n*ldC*sizeof(double));
    
    if (A == NULL || B == NULL || C == NULL) {
        fprintf(stderr, "[error] Failed to allocate memory.\n");
        return EXIT_FAILURE;
    }

    // initialize memory

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            A[i*ldA+j] = 2.0*rand()/RAND_MAX - 1.0;
            B[i*ldB+j] = 2.0*rand()/RAND_MAX - 1.0;
        }
    }

    // start timer
    struct timespec ts_start;
    clock_gettime(CLOCK_MONOTONIC, &ts_start);

    // call the function

    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
        n, n, n, 1.0, A, ldA, B, ldB, 0.0, C, ldC);

    // stop the timer
    
    struct timespec ts_stop;
    clock_gettime(CLOCK_MONOTONIC, &ts_stop);

    // calculate metrics
    
    double time = ts_stop.tv_sec - ts_start.tv_sec +
        1.0e-9*(ts_stop.tv_nsec - ts_start.tv_nsec); 
    printf("Runtime was %.3f s.\n", time);
    printf("Floprate was %.0f GFlops.\n", (2.0*n*n*n/time)*1E-9);
    printf("Memory throughput (naive) %.0f GB/s.\n", 
        ((8.0*n*n)*(2*n+1)*sizeof(double)/time)*1E-9);

    // free the allocated memory

    free(A); free(B); free(C);
}
