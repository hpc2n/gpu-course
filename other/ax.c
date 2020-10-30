#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <math.h>

// a function that multiplies a vector y with a scalar alpha
void ax(int n, double alpha, double *y)
{
    #pragma omp parallel for
    for (int i = 0; i < n; i++)
        y[i] = alpha * y[i];
}

int main(int argc, char const **argv)
{
    double alpha = 2.0;

    // read and validate the command line arguments

    if (argc < 2) {
        fprintf(stderr, "[error] No vector length was supplied.\n");
        return EXIT_FAILURE;
    }

    int n = atof(argv[1]);
    if (n < 1) {
        fprintf(stderr, "[error] The vector length was invalid.\n");
        return EXIT_FAILURE;
    }
    
    srand(time(NULL));

    // allocate memory for the vector and it's duplicate

    double *y, *_y;
    if ((y = (double *) malloc(n*sizeof(double))) == NULL) {
        fprintf(stderr,
            "[error] Failed to allocate host memory for vector y.\n");
        return EXIT_FAILURE;
    }
    if ((_y = (double *) malloc(n*sizeof(double))) == NULL) {
        fprintf(stderr,
            "[error] Failed to allocate host memory for vector _y.\n");
        return EXIT_FAILURE;
    }

    // initialize memory and store a copy for a later validation

    for (int i = 0; i < n; i++)
        y[i] = _y[i] = 1.0*rand()/RAND_MAX;

    // start timer
    struct timespec ts_start;
    clock_gettime(CLOCK_MONOTONIC, &ts_start);

    // call the function

    ax(n, alpha, y);

    // stop the timer
    struct timespec ts_stop;
    clock_gettime(CLOCK_MONOTONIC, &ts_stop);

    // calculate time
    double time = ts_stop.tv_sec - ts_start.tv_sec +
        1.0e-9*(ts_stop.tv_nsec - ts_start.tv_nsec); 
    printf("Time = %f s\n", time);

    // calculate flop rate and memory througput
    printf("Floprate = %.0f GFlops\n", 1.0E-9 * n / time);
    printf("Memory throughput = %.0f GB/s\n", 
        1.0E-9 * 2 * n * sizeof(double) / time);

    // validate the result by computing sqrt((x-alpha*_x)^2)

    double res = 0.0;
    for (int i = 0; i < n; i++)
        res += (y[i]-alpha*_y[i]) * (y[i]-alpha*_y[i]);
    printf("Residual = %e\n", sqrt(res));

    // free the allocated memory

    free(y); free(_y);
}
