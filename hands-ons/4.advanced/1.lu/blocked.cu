#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "common.h"

const double one = 1.0;
const double minus_one = -1.0;

///
/// @brief Forms a LU decomposition in a scalar manner.
///
/// @param[in]          matrix size
/// @param[in]    ldA   leading dimension
/// @param[inout]   A   in: matrix, out: LU decomposition
///
void simple_lu(int n, int ldA, double *A)
{
    for (int i = 0; i < n; i++) {
        for (int j = i+1; j < n; j++) {
            A[i*ldA+j] /= A[i*ldA+i];

            for (int k = i+1; k < n; k++)
                 A[k*ldA+j] -= A[i*ldA+j] * A[k*ldA+i];
        }
    }
}

///
/// @brief Forms a LU decomposition in a blocked manner.
///
/// @param[in] block_size   block size
/// @param[in]          n   matrix dimension
/// @param[in]        ldA   leading dimension
/// @param[inout]       A   in: matrix, out: LU decomposition
///
void blocked_lu(int block_size, int n, int ldA, double *A)
{
    int block_count = DIVCEIL(n, block_size);

    // allocate and fill an array that stores the block pointers
    double ***blocks = (double ***) malloc(block_count*sizeof(double**));
    for (int i = 0; i < block_count; i++) {
        blocks[i] = (double **) malloc(block_count*sizeof(double*));

        for (int j = 0; j < block_count; j++)
            blocks[i][j] = A+(j*ldA+i)*block_size;
    }

    //
    // iterate through the diagonal blocks
    //
    // +--+--+--+--+
    // | 0|  |  |  |
    // +--+--+--+--+
    // |  | 1|  |  |
    // +--+--+--+--+
    // |  |  | 2|  |
    // +--+--+--+--+
    // |  |  |  | 3|
    // +--+--+--+--+
    //
    for (int i = 0; i < block_count; i++) {

        // calculate diagonal block size
        int dsize = min(block_size, n-i*block_size);

        // calculate trailing matrix size
        int tsize = n-(i+1)*block_size;

        //
        // compute the LU decomposition of the diagonal block
        //
        // +--+--+--+--+
        // |  |  |  |  |
        // +--+--+--+--+   ## - process (read-write)
        // |  |##|  |  |
        // +--+--+--+--+
        // |  |  |  |  |
        // +--+--+--+--+
        // |  |  |  |  |
        // +--+--+--+--+
        //
        simple_lu(dsize, ldA, blocks[i][i]);

        if (0 < tsize) {

            //
            // blocks[i][i+1:] <- L1(blocks[i][i]) \ blocks[i][i+1:]
            //
            // +--+--+--+--+
            // |  |  |  |  |
            // +--+--+--+--+
            // |  |rr|##|##|   ## - process (read-write)
            // +--+--+--+--+   rr - read
            // |  |  |  |  |
            // +--+--+--+--+
            // |  |  |  |  |
            // +--+--+--+--+
            //
            // cublasDtrsm(handle, CUBLAS_SIDE_LEFT, 
            //     CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, CUBLAS_DIAG_UNIT,
            //     ....
            dtrsm_("Left", "Lower", "No transpose", "Unit triangular",
                &dsize, &tsize, &one, blocks[i][i], &ldA, blocks[i][i+1], &ldA);

            //
            // blocks[i+1:][i] <- U(blocks[i][i]) / blocks[i+1:][i]
            //
            // +--+--+--+--+
            // |  |  |  |  |
            // +--+--+--+--+
            // |  |rr|  |  |   ## - process (read-write)
            // +--+--+--+--+   rr - read
            // |  |##|  |  |
            // +--+--+--+--+
            // |  |##|  |  |
            // +--+--+--+--+
            //
            // cublasDtrsm(handle, CUBLAS_SIDE_RIGHT,
            //     CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT,
            //     ....
            dtrsm_("Right", "Upper", "No Transpose", "Not unit triangular",
                &tsize, &dsize, &one, blocks[i][i], &ldA, blocks[i+1][i], &ldA);

            //
            // blocks[i+1:][i+1:] <- blocks[i+1:][i+1:] -
            //                          blocks[i+1:][i] * blocks[i][i+1:]
            //
            // +--+--+--+--+
            // |  |  |  |  |
            // +--+--+--+--+
            // |  |  |rr|rr|   ## - process (read-write)
            // +--+--+--+--+   rr - read
            // |  |rr|##|##|
            // +--+--+--+--+
            // |  |rr|##|##|
            // +--+--+--+--+
            //
            // cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
            //     ....
            dgemm_("No Transpose", "No Transpose",
                &tsize, &tsize, &dsize, &minus_one, blocks[i+1][i],
                &ldA, blocks[i][i+1], &ldA, &one, blocks[i+1][i+1], &ldA);
        }
    }

    // free allocated resources
    for (int i = 0; i < block_count; i++)
        free(blocks[i]);
    free(blocks);
}

int main(int argc, char **argv)
{    
    //
    // check arguments
    //
    
    if (argc != 3) {
        fprintf(stderr, 
            "[error] Incorrect arguments. Use %s (n) (block size)\n", argv[0]);
        return EXIT_FAILURE;
    }

    int n = atoi(argv[1]);
    if (n < 1)  {
        fprintf(stderr, "[error] Invalid matrix dimension.\n");
        return EXIT_FAILURE;
    }

    int block_size = atoi(argv[2]);
    if (block_size < 2)  {
        fprintf(stderr, "[error] Invalid block size.\n");
        return EXIT_FAILURE;
    }

    //
    // Initialize matrix A and store a duplicate to matrix B. Matrix C is for
    // validation.
    //
    
    srand(time(NULL));

    int ldA, ldB, ldC;
    ldA = ldB = ldC = DIVCEIL(n, 8)*8; // align to 64 bytes
    double *A = (double *) aligned_alloc(8, n*ldA*sizeof(double));
    double *B = (double *) aligned_alloc(8, n*ldB*sizeof(double));
    double *C = (double *) aligned_alloc(8, n*ldC*sizeof(double));
    
    if (A == NULL || B == NULL || C == NULL) {
        fprintf(stderr, "[error] Failed to allocate memory.\n");
        return EXIT_FAILURE;
    }

    // A <- random diagonally dominant matrix
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++)
            A[i*ldA+j] = B[i*ldB+j] = 2.0*rand()/RAND_MAX - 1.0;
        A[i*ldA+i] = B[i*ldB+i] = 1.0*rand()/RAND_MAX + n;
    }

    //
    // compute
    //
    
    struct timespec ts_start;
    clock_gettime(CLOCK_MONOTONIC, &ts_start);

    // A <- (L,U)
    blocked_lu(block_size, n, ldA, A);

    struct timespec ts_stop;
    clock_gettime(CLOCK_MONOTONIC, &ts_stop);

    printf("Time = %f s\n",
        ts_stop.tv_sec - ts_start.tv_sec +
        1.0E-9*(ts_stop.tv_nsec - ts_start.tv_nsec));

    // C <- L * U
    mul_lu(n, ldA, ldC, A, C);

    //
    // validate
    //
    
    // C <- L * U - B
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            C[i*ldC+j] -= B[i*ldB+j];

    // compute || C ||_F / || B ||_F = || L * U - B ||_F  / || B ||_F
    double residual = dlange_("Frobenius", &n, &n, C, &ldC, NULL) /
        dlange_("Frobenius", &n, &n, B, &ldB, NULL);
        
    printf("Residual = %E\n", residual);
    
    int ret = EXIT_SUCCESS;
    if (1.0E-12 < residual) {
        fprintf(stderr, "The residual is too large.\n");
        ret = EXIT_FAILURE;
    }
    
    //
    // cleanup
    //

    free(A);
    free(B);
    free(C);

    return ret;
}
