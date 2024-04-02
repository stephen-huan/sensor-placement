from bisect import bisect_left
from functools import partial, wraps
from typing import TypeAlias

import jax
import jax.numpy as jnp
from gpjax.kernels import AbstractKernel, DenseKernelComputation
from jax import Array, lax
from jax.numpy.linalg import inv

dense = DenseKernelComputation()
cross_covariance = dense.cross_covariance
diagonal = dense.diagonal
gram = dense.gram


def jit(f, *args, **kwargs):
    """Workaround LSP showing JitWrapped on hover."""
    return wraps(f)(jax.jit(f, *args, **kwargs))


def index_dtype(x: Array, unsigned: bool = True):
    """Return the smallest integer datatype that can represent indices in x."""
    max_value = lambda dtype: jnp.iinfo(dtype).max  # noqa: E731
    dtypes = sorted(
        (
            [jnp.uint8, jnp.uint16, jnp.uint32, jnp.uint64]
            if unsigned
            else [jnp.int8, jnp.int16, jnp.int32, jnp.int64]
        ),
        key=max_value,
    )
    sizes = list(map(max_value, dtypes))
    return dtypes[bisect_left(sizes, x.shape[0] - 1)]


@jit
def argmax_masked(x: Array, mask: Array) -> Array:
    """Argmax of x restricted to the indices in mask, including nans."""
    return jnp.nanargmax(jnp.nan_to_num(x) + jnp.where(mask, 0.0, jnp.nan))


@jit
def __chol_update(
    cond_var: Array, factor: Array, i: int, k: int | Array
) -> tuple[Array, Array]:
    """Condition the i-th column of the Cholesky factor by the k-th point."""
    n = cond_var.shape[0]
    # update Cholesky factor by left looking
    # -factor[:, :i] @ factor[k, :i] is more efficient but size must be static
    row = factor[k]
    row = row.at[i].set(0.0)
    factor = factor.at[:, i].add(-factor @ row)
    # https://github.com/google/jax/issues/19162
    factor = factor.at[:, i].multiply(jnp.reciprocal(jnp.sqrt(factor[k, i])))
    # update conditional variance
    cond_var = cond_var.at[:].add(-jnp.square(factor[:n, i]))
    return cond_var, factor


@partial(jit, static_argnums=2)
def entropy(x: Array, kernel: AbstractKernel, s: int) -> Array:
    """Greedily select the s most entropic points from x."""
    n = x.shape[0]
    s = min(s, n)
    # initialization
    int_dtype = index_dtype(x)
    indices = jnp.zeros(s, dtype=int_dtype)
    candidates = jnp.ones(n, dtype=jnp.bool_)
    cond_var = diagonal(kernel, x).diag
    factor = jnp.zeros((n, s), dtype=cond_var.dtype)
    State: TypeAlias = tuple[Array, Array, Array, Array]  # type: ignore
    state = (indices, candidates, cond_var, factor)

    def body_fun(i: int, state: State) -> State:
        """Select the best index on the i-th iteration."""
        indices, candidates, cond_var, factor = state
        # pick best entry
        k = argmax_masked(cond_var, candidates)
        # update data structures
        cov_k = cross_covariance(kernel, x, x[k, jnp.newaxis])
        factor = factor.at[:n, i].set(cov_k.flatten())
        return (
            indices.at[i].set(int_dtype(k)),
            candidates.at[k].set(False),
            *__chol_update(cond_var, factor, i, k),
        )

    indices, *_ = lax.fori_loop(0, s, body_fun, state)
    return indices


@partial(jit, static_argnums=2)
def mi(x: Array, kernel: AbstractKernel, s: int) -> Array:
    """Greedily select the s most informative points from x."""
    n = x.shape[0]
    s = min(s, n)
    # initialization
    int_dtype = index_dtype(x)
    indices = jnp.zeros(s, dtype=int_dtype)
    candidates = jnp.ones(n, dtype=jnp.bool_)
    cond_var = diagonal(kernel, x).diag
    factor = jnp.zeros((n, s), dtype=cond_var.dtype)
    prec = inv(gram(kernel, x).to_dense())
    State: TypeAlias = tuple[Array, Array, Array, Array, Array]  # type: ignore
    state = (indices, candidates, cond_var, factor, prec)

    def body_fun(i: int, state: State) -> State:
        """Select the best index on the i-th iteration."""
        indices, candidates, cond_var, factor, prec = state
        # pick best entry
        k = argmax_masked(cond_var * jnp.diagonal(prec), candidates)
        # update data structures
        cov_k = cross_covariance(kernel, x, x[k, jnp.newaxis])
        factor = factor.at[:n, i].set(cov_k.flatten())
        return (
            indices.at[i].set(int_dtype(k)),
            candidates.at[k].set(False),
            *__chol_update(cond_var, factor, i, k),
            prec.at[:].add(-jnp.outer(prec[k], prec[k]) / prec[k, k]),
        )

    indices, *_ = lax.fori_loop(0, s, body_fun, state)
    return indices
