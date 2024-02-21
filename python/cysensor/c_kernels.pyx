# cython: profile=False
from cpython.mem cimport PyMem_Free, PyMem_Malloc
from cpython.ref cimport PyObject
from libc.math cimport exp, sqrt

import numpy as np
import sklearn.gaussian_process.kernels as kernels

cimport scipy.linalg.cython_blas as blas

from . cimport mkl


cdef (Kernel *) get_kernel(kernel_object: kernels.Kernel):
    """Turn a Python scikit-learn kernel object into a C kernel struct."""
    cdef Kernel *kernel
    kernel = <Kernel *> PyMem_Malloc(sizeof(Kernel))
    # default to generic Python implementation
    kernel.params = <void *> kernel_object
    kernel.kernel_function = &__python_covariance
    kernel.diag = &__python_variance
    # don't free a Python object (not malloc'd)
    kernel.cleanup = False

    # specialize to optimized C if possible
    if (
        isinstance(kernel_object, kernels.Matern)
        and not isinstance(kernel_object.length_scale, list)
        and not isinstance(kernel_object.length_scale, np.ndarray)
    ):
        kernel.params = __matern_params(kernel_object)
        kernel.kernel_function = &__matern_covariance
        kernel.diag = &__matern_variance
        kernel.cleanup = True

    return kernel


cdef void covariance_vector(
    Kernel *kernel,
    double[:, ::1] points,
    double[::1] point,
    double *vector,
):
    """Covariance between each point in points and given point."""
    kernel.kernel_function(kernel.params, points, point, vector)


cdef void variance_vector(
    Kernel *kernel,
    double[:, ::1] points,
    double *vector,
):
    """Variance for each point in points."""
    kernel.diag(kernel.params, points, vector)


cdef void kernel_cleanup(Kernel *kernel):
    """Free dynamically allocated memory."""
    if kernel.cleanup:
        PyMem_Free(kernel.params)
    PyMem_Free(kernel)


### generic Python scikit-learn kernel


cdef void __python_covariance(
    void *params,
    double[:, ::1] points,
    double[::1] point,
    double *vector,
):
    """Wrapper over a scikit-learn kernel object's __call__ method."""
    cdef:
        object kernel
        double[:, :] cov
        int i

    kernel = <object> (<PyObject *> params)
    cov = kernel(points, [point])
    for i in range(points.shape[0]):
        vector[i] = cov[i, 0]


cdef void __python_variance(
    void *params,
    double[:, ::1] points,
    double *vector,
):
    """Wrapper over a scikit-learn kernel object's diag method."""
    cdef:
        object kernel
        double[:] var
        int i

    kernel = <object> (<PyObject *> params)
    var = kernel.diag(points)
    for i in range(points.shape[0]):
        vector[i] = var[i]


### matern covariance

cdef double SQRT3 = sqrt(3)
cdef double SQRT5 = sqrt(5)


cdef struct MaternParams:
    double nu
    double length_scale


cdef (void *) __matern_params(kernel: kernels.Kernel):
    """Intialize a MaternParams struct based on the given kernel."""
    cdef MaternParams *params_ptr
    params_ptr = <MaternParams *> PyMem_Malloc(sizeof(MaternParams))
    params = kernel.get_params()
    params_ptr.nu = params["nu"]
    params_ptr.length_scale = params["length_scale"]
    return <void *>params_ptr


cdef void __distance_vector(
    double[:, ::1] points,
    double[::1] point,
    double *vector,
):
    """ Euclidean distance between each point in points and given point. """
    cdef:
        int n, i, j
        double dist, d
        double *start
        double *p

    n = points.shape[1]
    start = &points[0, 0]
    p = &point[0]
    for i in range(points.shape[0]):
        dist = 0
        for j in range(n):
            d = (start + i*n)[j] - p[j]
            dist += d*d
        vector[i] = dist

    mkl.vdSqrt(points.shape[0], vector, vector)


cdef void __matern_covariance(
    void *params,
    double[:, ::1] points,
    double[::1] point,
    double *vector,
):
    """ Matern covariance between each point in points and given point. """
    cdef:
        MaternParams *matern_params
        int n, incx, i
        double nu, length_scale, alpha, x
        double *u

    matern_params = <MaternParams *> params
    nu = matern_params.nu
    length_scale = matern_params.length_scale

    __distance_vector(points, point, vector)

    n = points.shape[0]
    if nu == 0.5:
        alpha = 1
    elif nu == 1.5:
        alpha = SQRT3
    else:
        alpha = SQRT5
    alpha /= -length_scale
    incx = 1
    blas.dscal(&n, &alpha, vector, &incx)
    u = <double *> PyMem_Malloc(n * sizeof(double))
    mkl.vdExp(n, vector, u)

    if nu == 0.5:
        for i in range(n):
            vector[i] = u[i]
    elif nu == 1.5:
        for i in range(n):
            x = vector[i]
            vector[i] = (1 - x)*u[i]
    else:
        for i in range(n):
            x = vector[i]
            vector[i] = (1 - x + x*x/3)*u[i]

    PyMem_Free(u)


cdef void __matern_variance(
    void *params,
    double[:, ::1] points,
    double *vector,
):
    """ Matern variance for each point in points. """
    cdef int i
    for i in range(points.shape[0]):
        # Matern kernels have covariance one between a point and itself
        vector[i] = 1
