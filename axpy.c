//
// A simple C program that adds two vectors together.
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
        fprintf(stderr, "[error] No vector lenght was supplied.\n");
        return EXIT_FAILURE;
    }

    size_t n = atof(argv[1]);
    if (n < 1) {
        fprintf(stderr, "[error] The vector lenght was invalid.\n");
        return EXIT_FAILURE;
    }

    // allocate memory

    double *x;
    if ((x = malloc(n*sizeof(double))) == NULL) {
        fprintf(stderr, "[error] Failed to allocate memory for vector x.\n");
        return EXIT_FAILURE;
    }

    double *y;
    if ((y = malloc(n*sizeof(double))) == NULL) {
        fprintf(stderr, "[error] Failed to allocate memory for vector y.\n");
        return EXIT_FAILURE;
    }

    // initialize memory

    for (int i = 0; i < n; i++) {
        x[i] = 2.0 * rand()/RAND_MAX - 1.0;
        y[i] = 2.0 * rand()/RAND_MAX - 1.0;
    }

    //
    // start timer
    //

    struct timespec start;
    clock_gettime(CLOCK_REALTIME, &start);

    //
    // compute y <- 2 * x + y
    //

    cblas_daxpy(n, 2.0, x, 1, y, 1);

    //
    // stop timer and report
    //

    struct timespec stop;
    clock_gettime(CLOCK_REALTIME, &stop);

    double time =
        (stop.tv_sec - start.tv_sec) + (stop.tv_nsec - start.tv_nsec)*1E-9;

    printf("Runtime was %.3f s.\n", time);
    printf("Floprate was %.0f GFlops.\n", (2*n/time)*1E-9);
    printf("Memory throughput %.0f GB/s.\n", (3*n*sizeof(double)/time)*1E-9);

    // free the allocated memory

    free(x);
    free(y);

    return EXIT_SUCCESS;
}
