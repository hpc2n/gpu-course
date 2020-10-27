#ifndef COMMON_H_
#define COMMON_H_

//
// BLAS and LAPACK subroutines
//

extern "C" {

extern double dnrm2_(int const *, double const *, int const *);

extern void dtrmm_(char const *, char const *, char const *, char const *,
    int const *, int const *, double const *, double const *, int const *,
    double *, int const *);

extern void dlacpy_(char const *, int const *, int const *, double const *,
    int const *, double *, int const *);

extern double dlange_(char const *, int const *, int const *, double const *,
    int const *, double *);

extern void dtrsm_(char const *, char const *, char const *, char const *,
    int const *, int const *, double const *, double const *, int const *,
    double *, int const *);

extern void dgemm_(char const *, char const *, int const *, int const *,
    int const *, double const *, double const *, int const *, double const *,
    int const *, double const *, double *, int const *);

}

#define CHECK_CUDA_ERROR(exp) {                     \
    cudaError_t ret = (exp);                        \
    if (ret != cudaSuccess) {                       \
        fprintf(stderr, "[error] %s:%d: %s (%s)\n", \
            __FILE__, __LINE__,                     \
            cudaGetErrorName(ret),                  \
            cudaGetErrorString(ret));               \
        exit(EXIT_FAILURE);                         \
    }                                               \
}

#define CHECK_CUBLAS_ERROR(exp) {                   \
    cublasStatus_t ret = (exp);                     \
    if (ret != CUBLAS_STATUS_SUCCESS) {             \
        fprintf(stderr,                             \
            "[error] %s:%d: cuBLAS error\n",        \
            __FILE__, __LINE__);                    \
        exit(EXIT_FAILURE);                         \
    }                                               \
}

///
/// Returns the ceil of a/b.
///
/// @param[in] a    denominator
/// @param[in] b    numerator
///
/// @returns ceil of a/b
///
static inline int DIVCEIL(int a, int b)
{
    return (a+b-1)/b;
}

///
/// @brief Multiplies the components of a LU decomposition (B = L1(A) * U(A)).
///
/// @param[in] n        matrix dimension
/// @param[in] lda      leading dimension of the input matrix
/// @param[in] ldb      leading dimension of the output matrix
/// @param[in] A        input matrix
/// @param[out] B       output matrix
///
static inline void mul_lu(int n, int lda, int ldb, double const *A, double *B)
{
    const double one = 1.0;

    // B <- U(A) = U
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < i+1; j++)
            B[i*ldb+j] = A[i*lda+j];
        for (int j = i+1; j < n; j++)
            B[i*ldb+j] = 0.0;
    }

    // B <- L1(A) * B = L * U
    dtrmm_("Left", "Lower", "No Transpose", "Unit triangular",
        &n, &n, &one, A, &lda, B, &ldb);
}

#endif
