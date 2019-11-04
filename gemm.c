//
// A simple C program that multiplies two matrices together.
//
// Author: Mirko Myllykoski, Umeå University, 2019
//

#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <cblas.h>

int main(int argc, char const **argv)
{
    // read and validate the command line arguments

    if (argc < 2) {
        fprintf(stderr, "[error] No matrix size was supplied.\n");
        return EXIT_FAILURE;
    }

    int n = atoi(argv[1]);
    if (n < 1) {
        fprintf(stderr, "[error] The matrix size was invalid.\n");
        return EXIT_FAILURE;
    }

    // allocate memory

    double *A; int ldA = (n/8+1)*8;
    if ((A = malloc(n*ldA*sizeof(double))) == NULL) {
        fprintf(stderr, "[error] Failed to allocate memory for matrix A.\n");
        return EXIT_FAILURE;
    }

    double *B; int ldB = (n/8+1)*8;
    if ((B = malloc(n*ldB*sizeof(double))) == NULL) {
        fprintf(stderr, "[error] Failed to allocate memory for matrix B.\n");
        return EXIT_FAILURE;
    }

    double *C; int ldC = (n/8+1)*8;
    if ((C = malloc(n*ldC*sizeof(double))) == NULL) {
        fprintf(stderr, "[error] Failed to allocate memory for matrix C.\n");
        return EXIT_FAILURE;
    }

    // initialize memory

    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            A[i*ldA+j] = 2.0 * rand()/RAND_MAX - 1.0;

    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            B[i*ldB+j] = 2.0 * rand()/RAND_MAX - 1.0;

    //
    // start timer
    //

    struct timespec start;
    clock_gettime(CLOCK_REALTIME, &start);

    //
    // compute C <- A * B
    //

    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, n, n, n,
        1.0, A, ldA, B, ldB, 0.0, C, ldC);

    //
    // stop timer and report
    //

    struct timespec stop;
    clock_gettime(CLOCK_REALTIME, &stop);

    double time =
        (stop.tv_sec - start.tv_sec) + (stop.tv_nsec - start.tv_nsec)*1E-9;

    printf("Runtime was %.3f s.\n", time);
    printf("Floprate was %.0f GFlops.\n", ((1.0*n*n)*(2*n-1)/time)*1E-9);
    printf("Memory throughput %.0f GB/s.\n", ((1.0*n*n)*(2*n+1)*sizeof(double)/time)*1E-9);

    // free the allocated memory

    free(A);
    free(B);
    free(C);

    return EXIT_SUCCESS;
}
