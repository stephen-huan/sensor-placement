import numpy as np
import scipy
from sklearn.gaussian_process.kernels import Kernel


def inv(m: np.ndarray) -> np.ndarray:
    """Inverts a symmetric positive definite matrix m."""
    return np.linalg.inv(m)
    # below only starts to get faster for large matrices (~10^4)
    # return solve(m, np.identity(m.shape[0]))


def solve(A: np.ndarray, b: np.ndarray, **kwargs) -> np.ndarray:
    """Solve the system Ax = b for symmetric positive definite A."""
    return scipy.linalg.solve(A, b, assume_a="pos", **kwargs)


def solve_triangular(
    L: np.ndarray, b: np.ndarray, lower=True, **kwargs
) -> np.ndarray:
    """Solve the system Lx = b for triangular L."""
    return scipy.linalg.solve_triangular(L, b, lower=lower, **kwargs)


def dot(theta: np.ndarray, x: np.ndarray, factor: bool = True):
    """Compute the quadratic form of the precision x^T theta^{-1} x."""
    # use Cholesky factor x^T (L L^T)^{-1} x = (L^{-1} x)^T (L^{-1} x)
    if x.shape[0] == 0:
        return 0
    L = np.linalg.cholesky(theta) if factor else theta
    temp = solve_triangular(L, x)
    return np.sum(temp * temp, axis=0)


def chol_downdate(
    L: np.ndarray, u: np.ndarray, j: int | None = None
) -> np.ndarray:
    """Computes the rank-one downdate in-place, L -> chol(L L^T - u u^T)."""
    if j is None:
        j = L.shape[0]
    for i in range(j):
        c1, c2 = L[i, i], u[i]
        dp = np.sqrt(c1 * c1 - c2 * c2)
        c1, c2 = c1 / dp, c2 / dp
        L[:, i] *= c1
        L[:, i] -= c2 * u
        u *= 1 / c1
        u -= c2 / c1 * L[:, i]
    return L


### Gaussian process sensor placement

# see: "Near-Optimal Sensor Placements in Gaussian Processes: Theory,
# Efficient Algorithms and Empirical Studies" by Krause et al., 2008


def entropy_naive(X: np.ndarray, kernel: Kernel, s: int) -> np.ndarray:
    """Returns a list of the most entropic points in X greedily."""
    # O(s*(s^3 + n*s^2)) = O(n s^3 + s^4)
    n = len(X)
    s = min(s, n)
    indexes = np.zeros(s, dtype=np.int64)
    theta = np.zeros((s, s))
    cov = np.zeros((s, n))
    var: np.ndarray = kernel.diag(X)  # type: ignore

    for i in range(s):
        # compute conditional variance
        cond_var = var - dot(theta[:i, :i], cov[:i, :])  # type: ignore
        # pick best entry
        k = np.argmax(cond_var)
        indexes[i] = k
        var[k] = -1
        # store kernel function evaluations
        cov[i] = kernel(X, X[k : k + 1]).flatten()  # type: ignore
        theta[: i + 1, i] = cov[i, indexes[: i + 1]].flatten()
        theta[i, : i + 1] = cov[i, indexes[: i + 1]].flatten()

    return indexes


def entropy_prec(X: np.ndarray, kernel: Kernel, s: int) -> np.ndarray:
    """Returns a list of the most entropic points in X greedily."""
    # O(s*(n*s + s^2)) = O(n s^2)
    n = len(X)
    s = min(s, n)
    # initialization
    indexes = np.zeros(s, dtype=np.int64)
    prec = np.zeros((s, s))
    cov = np.zeros((n, s))
    cond_var: np.ndarray = kernel.diag(X)  # type: ignore

    for i in range(s):
        # pick best entry
        k = np.argmax(cond_var)
        indexes[i] = k
        cov[:, i] = kernel(X, X[k : k + 1]).flatten()  # type: ignore
        # update precision of selected entries
        var = 1 / cond_var[k]
        v = prec[:i, :i] @ cov[indexes[:i], i]
        prec[:i, :i] += var * np.outer(v, v)
        prec[:i, i] = -var * v
        prec[i, :i] = -var * v
        prec[i, i] = var
        # compute column k of conditional covariance
        cond_cov_k = cov[:, i].copy()
        cond_cov_k -= cov[:, :i] @ v
        cond_cov_k /= np.sqrt(cond_var[k])
        # update conditional variance
        cond_var -= cond_cov_k**2
        cond_var[k] = -1

    return indexes


def entropy_prechol(X: np.ndarray, kernel: Kernel, s: int) -> np.ndarray:
    """Returns a list of the most entropic points in X greedily."""
    # O(s*(n*s + s^2)) = O(n s^2)
    n = len(X)
    s = min(s, n)
    # initialization
    indexes = np.zeros(s, dtype=np.int64)
    L = np.zeros((s, s))
    cov = np.zeros((n, s))
    cond_var: np.ndarray = kernel.diag(X)  # type: ignore

    for i in range(s):
        # pick best entry
        k = np.argmax(cond_var)
        indexes[i] = k
        cov[:, i] = kernel(X, X[k : k + 1]).flatten()  # type: ignore
        # update Cholesky factor by up looking
        L[i, :i] = cov[indexes[:i], i]
        Lp, row = L[:i, :i], L[i, :i]
        if i > 0:
            solve_triangular(Lp, row, overwrite_b=True)
        L[i, i] = np.sqrt(cov[k, i] - np.dot(row, row))
        # compute column k of conditional covariance
        cond_cov_k = cov[:, i].copy()
        if i > 0:
            cond_cov_k -= cov[:, :i] @ solve_triangular(Lp, row, trans="T")
        cond_cov_k /= np.sqrt(cond_var[k])
        # update conditional variance
        cond_var -= cond_cov_k**2
        cond_var[k] = -1

    return indexes


def entropy_chol(X: np.ndarray, kernel: Kernel, s: int) -> np.ndarray:
    """Returns a list of the most entropic points in X greedily."""
    # O(s*(n*s + s^2)) = O(n s^2)
    n = len(X)
    s = min(s, n)
    # initialization
    indexes = np.zeros(s, dtype=np.int64)
    L = np.zeros((n, s))
    cond_var: np.ndarray = kernel.diag(X)  # type: ignore

    for i in range(s):
        # pick best entry
        k = np.argmax(cond_var)
        indexes[i] = k
        # update Cholesky factor by left looking
        L[:, i] = kernel(X, X[k : k + 1]).flatten()  # type: ignore
        L[:, i] -= L[:, :i] @ L[k, :i]
        L[:, i] /= np.sqrt(L[k, i])
        # update conditional variance
        cond_var -= L[:, i] ** 2
        cond_var[k] = -1

    return indexes


# simple improvement to Algorithm 1 in the Krause paper


def mi_naive(X: np.ndarray, kernel: Kernel, s: int) -> list:
    """Max mutual information between selected and non-selected points."""
    # O(s*(s^3 + n*n^3)) = O(n^4 s)
    n = len(X)
    s = min(s, n)
    indexes, candidates = [], list(range(n))
    cov: np.ndarray = kernel(X)  # type: ignore
    var = np.diagonal(cov)

    for _ in range(s):
        theta1 = cov[np.ix_(indexes, indexes)]
        cond_var1 = var - dot(theta1, cov[indexes, :])
        theta2 = cov[np.ix_(candidates, candidates)]
        cond_var2 = -np.ones(n)
        # cond_var2[candidates] = 1/np.diagonal(inv(theta2))
        cond_var2[candidates] = dot(theta2, np.identity(len(candidates)))
        k = np.argmax(cond_var1 * cond_var2)
        indexes.append(k)
        candidates.remove(k)  # type: ignore

    return indexes


def mi_prec(X: np.ndarray, kernel: Kernel, s: int) -> np.ndarray:
    """Max mutual information between selected and non-selected points."""
    # O(n^3 + s*(n^2)) = O(n^3)
    n = len(X)
    # initialization
    indexes, candidates = np.zeros(s, dtype=np.int64), np.arange(n)
    L = np.zeros((n, s))
    prec: np.ndarray = inv(kernel(X))  # type: ignore
    cond_var1: np.ndarray = kernel.diag(X)  # type: ignore
    # the full conditional of i corresponds to the ith diagonal in precision
    cond_var2 = np.copy(np.diagonal(prec))

    for i in range(s):
        # pick best entry
        k = np.argmax(cond_var1 * cond_var2)
        indexes[i] = k
        j = np.argwhere(candidates == k).item()
        candidates = np.delete(candidates, j)
        # update Cholesky factor
        L[:, i] = kernel(X, X[k : k + 1]).flatten()  # type: ignore
        L[:, i] -= L[:, :i] @ L[k, :i]
        L[:, i] /= np.sqrt(L[k, i])
        # update conditional variance
        cond_var1 -= L[:, i] ** 2
        cond_var1[k] = -1
        # update precision of candidates
        # marginalization in covariance is conditioning in precision
        prec -= np.outer(prec[j], prec[j]) / prec[j, j]
        prec = np.delete(np.delete(prec, j, 0), j, 1)
        # update conditional variance of candidates
        cond_var2[candidates] = np.diagonal(prec)

    return indexes


def mi_chol(X: np.ndarray, kernel: Kernel, s: int) -> np.ndarray:
    """Max mutual information between selected and non-selected points."""
    # O(n^3 + s*(n^2)) = O(n^3)
    n = len(X)
    # initialization
    indexes, candidates = np.zeros(s, dtype=np.int64), np.arange(n)
    L1 = np.zeros((n, s))
    L2: np.ndarray = np.flip(
        solve_triangular(
            np.linalg.cholesky(kernel(X[::-1])),  # type: ignore
            np.identity(n),
        )
    ).T
    cond_var1: np.ndarray = kernel.diag(X)  # type: ignore
    # the full conditional of i corresponds to the ith diagonal in precision
    cond_var2 = np.sum(L2 * L2, axis=1)

    for i in range(s):
        # pick best entry
        k = np.argmax(cond_var1 * cond_var2)
        indexes[i] = k
        j = np.argwhere(candidates == k).item()
        candidates = np.delete(candidates, j)
        # update Cholesky factor
        L1[:, i] = kernel(X, X[k : k + 1]).flatten()  # type: ignore
        L1[:, i] -= L1[:, :i] @ L1[k, :i]
        L1[:, i] /= np.sqrt(L1[k, i])
        # update conditional variance
        cond_var1 -= L1[:, i] ** 2
        cond_var1[k] = -1
        # update Cholesky factor of precision of candidates
        # marginalization in covariance is conditioning in precision
        u = L2 @ L2[j]
        u /= np.sqrt(u[j])
        L2 = chol_downdate(L2, u, j)
        L2 = np.delete(np.delete(L2, j, 0), j, 1)
        # update conditional variance of candidates
        cond_var2[candidates] = np.sum(L2 * L2, axis=1)

    return indexes
