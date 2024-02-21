# cython: profile=False
from libc.math cimport sqrt
cimport numpy as np
import numpy as np
cimport scipy.linalg.cython_blas as blas
cimport scipy.linalg.cython_lapack as lapack
cimport mkl
from .c_kernels cimport Kernel, get_kernel, kernel_cleanup
from .c_kernels cimport covariance_vector, variance_vector

cdef int __argmax(double[::1] x):
    """ Get the index corresponding to the largest (positive) value of x. """
    cdef:
        int i, k
        double v, best

    k, best = -1, -1
    for i in range(x.shape[0]):
        v = x[i]
        if v > best:
            k, best = i, v
    return k

### selection methods

cdef void __chol_update(double[::1, :] L, int i, int k):
    """ Updates the ith column of the Cholesky factor L with column k. """
    cdef:
        char *trans
        int M, N, lda, incx, incy, j
        double alpha, beta
        double *A
        double *x
        double *y

    # update Cholesky factor
    trans = 'n'
    M = L.shape[0]
    N = i
    alpha = -1
    A = &L[0, 0]
    lda = L.shape[0]
    x = &L[k, 0]
    incx = lda
    beta = 1
    y = &L[0, i]
    incy = 1
    blas.dgemv(trans, &M, &N, &alpha, A, &lda, x, &incx, &beta, y, &incy)
    alpha = 1/sqrt(L[k, i])
    blas.dscal(&M, &alpha, y, &incy)

cdef long[::1] __entropy_chol(double[:, ::1] x, Kernel *kernel, int s):
    """ Returns a list of the most entropic points in x greedily. """
    cdef:
        int n, i, j, k
        double v
        long[::1] indexes
        double[::1, :] L
        double[::1] cond_var

    n = x.shape[0]
    s = min(s, n)
    # initialization
    indexes = np.zeros(s, dtype=np.int64)
    L = np.zeros((n, s), order="F")
    cond_var = np.zeros(n)
    variance_vector(kernel, x, &cond_var[0])

    for i in range(s):
        # pick best entry
        k = __argmax(cond_var)
        indexes[i] = k
        # update Cholesky factor
        covariance_vector(kernel, x, x[k], &L[0, i])
        __chol_update(L, i, k)
        # update conditional variance
        for j in range(n):
            v = L[j, i]
            cond_var[j] -= v*v
        # clear out selected index
        cond_var[k] = -1

    return indexes

### wrapper functions

def entropy_chol(double[:, ::1] x, kernel_object, int s) -> np.ndarray:
    """ Returns a list of the most entropic points in x greedily. """
    cdef Kernel *kernel = get_kernel(kernel_object)
    selected = __entropy_chol(x, kernel, s)
    kernel_cleanup(kernel)
    return np.asarray(selected)

