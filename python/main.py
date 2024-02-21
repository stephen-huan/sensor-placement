import time

import matplotlib.pyplot as plt
import numpy as np
import sklearn.gaussian_process.kernels as kernels

import cysensor
import pysensor as sensor

# fmt: off
lightblue     = "#a1b4c7"
orange        = "#ea8810"
silver        = "#b0aba8"
rust          = "#b8420f"
seagreen      = "#23553c"

lightsilver   = "#e7e6e5"
darkorange    = "#c7740e"
darksilver    = "#96918f"
darklightblue = "#8999a9"
darkrust      = "#9c380d"
darkseagreen  = "#1e4833"
# fmt: on

POINT_SIZE = 20  # point sizes
BIG_POINT = 40  # large point


def grid(n: int, a: float = 0, b: float = 1, d: int = 2) -> np.ndarray:
    """Generate n points evenly spaced in a [a, b]^d hypercube."""
    spaced = np.linspace(a, b, round(n ** (1 / d)))
    cube = (spaced,) * d
    return np.stack(np.meshgrid(*cube), axis=-1).reshape(-1, d)


def perturbed_grid(
    rng: np.random.Generator,
    n: int,
    a: float = 0,
    b: float = 1,
    d: int = 2,
    delta: float | None = None,
) -> np.ndarray:
    """Generate n points roughly evenly spaced in a [a, b]^d hypercube."""
    points = grid(n, a, b, d)
    # compute level of perturbation as half width
    if delta is None:
        # one point, width ill-defined
        width = (b - a) / (n ** (1 / d) - 1) if n ** (1 / d) > 1 else 0
        delta = 1 / 2 * width
    return points + rng.uniform(-delta, delta, points.shape)  # type: ignore


if __name__ == "__main__":
    np.set_printoptions(precision=3, suppress=True)
    rng = np.random.default_rng(1)

    kernel = kernels.Matern(length_scale=1, nu=5 / 2)
    X = rng.random((100, 2))
    s = 75

    # correctness and export data to Julia

    ans = sensor.entropy_naive(X, kernel, s)
    indexes = sensor.entropy_prec(X, kernel, s)
    assert np.allclose(ans, indexes), "python prec    entropy wrong"
    indexes = sensor.entropy_prechol(X, kernel, s)
    assert np.allclose(ans, indexes), "python prechol entropy wrong"
    indexes = sensor.entropy_chol(X, kernel, s)
    assert np.allclose(ans, indexes), "python chol    entropy wrong"
    indexes = cysensor.entropy_chol(X, kernel, s)
    assert np.allclose(ans, indexes), "cython chol    entropy wrong"

    np.save("data/entropy_X.npy", X)
    np.save("data/entropy_indexes.npy", indexes)

    ans = sensor.mi_naive(X, kernel, s)
    indexes = sensor.mi_prec(X, kernel, s)
    assert np.allclose(ans, indexes), "python mi prec wrong"
    indexes = sensor.mi_chol(X, kernel, s)
    assert np.allclose(ans, indexes), "python mi chol wrong"

    np.save("data/mi_X.npy", X)
    np.save("data/mi_indexes.npy", indexes)

    # graphing

    kernel = kernels.Matern(length_scale=1, nu=5 / 2)

    n = 10
    s = 10
    X = grid(n * n, 0, 1)
    entropy = sensor.entropy_naive(X, kernel, s)
    mi = sensor.mi_naive(X, kernel, s)

    # plt.style.available for available themes
    plt.style.use("seaborn-v0_8-paper")

    plt.scatter(
        X[:, 0],
        X[:, 1],
        label="candidates",
        zorder=1,
        s=POINT_SIZE,
        color=silver,
    )
    plt.scatter(
        X[entropy, 0],
        X[entropy, 1],
        label="entropy",
        zorder=2,
        s=BIG_POINT,
        marker="D",
        color=lightblue,
    )
    plt.scatter(
        X[mi, 0],
        X[mi, 1],
        label="mutual info",
        zorder=3,
        s=BIG_POINT,
        marker="s",
        color=rust,
    )
    plt.gca().set_aspect("equal")
    plt.legend()
    plt.savefig("data/sensor.png", bbox_inches="tight")
    plt.clf()

    # sample from underlying Gaussian process

    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")

    # n = 10
    # s = 10
    # X = grid(n*n, 0, 1)
    mu = np.zeros(n * n)
    # rng = np.random.default_rng(1)
    y = rng.multivariate_normal(mu, kernel(X))

    entropy = sensor.entropy_naive(X, kernel, s)
    mi = sensor.mi_naive(X, kernel, s)

    surf = ax.plot_surface(  # type: ignore
        X[:, 0].reshape(n, n),
        X[:, 1].reshape(n, n),
        y.reshape(n, n),
        label="measurements",
        zorder=0,
        color=silver,
        alpha=0.5,
    )
    # https://stackoverflow.com/questions/54994600/65554278#65554278
    surf._facecolors2d = surf._facecolor3d
    surf._edgecolors2d = surf._edgecolor3d
    ax.scatter(
        X[:, 0],
        X[:, 1],
        y,
        label="candidates",
        zorder=1,
        s=POINT_SIZE,  # pyright: ignore
        color=silver,
    )
    ax.scatter(
        X[entropy, 0],
        X[entropy, 1],
        y[entropy],
        label="entropy",
        zorder=2,
        s=BIG_POINT,  # pyright: ignore
        marker="D",
        color=lightblue,
    )
    ax.scatter(
        X[mi, 0],
        X[mi, 1],
        y[mi],
        label="mutual info",
        zorder=3,
        s=BIG_POINT,  # pyright: ignore
        marker="s",
        color=rust,
    )
    plt.legend()
    plt.savefig("data/gp.png", bbox_inches="tight")
    ax.clear()

    # exit()

    # benchmarking

    X = rng.random((10000, 3))
    s = 200

    # entropy

    start = time.time()
    indexes = sensor.entropy_naive(X, kernel, s)
    t1 = t2 = time.time() - start
    print(f"entropy   naive: {t1:9.3e} ({t1/t1:7.3f})")

    start = time.time()
    indexes = sensor.entropy_prec(X, kernel, s)
    # t1 = t2 = time.time() - start
    t2 = time.time() - start
    print(f"python     prec: {t2:9.3e} ({t1/t2:7.3f})")

    start = time.time()
    indexes = sensor.entropy_prechol(X, kernel, s)
    t2 = time.time() - start
    print(f"python  prechol: {t2:9.3e} ({t1/t2:7.3f})")

    start = time.time()
    indexes = sensor.entropy_chol(X, kernel, s)
    t2 = time.time() - start
    print(f"python     chol: {t2:9.3e} ({t1/t2:7.3f})")

    start = time.time()
    indexes = cysensor.entropy_chol(X, kernel, s)
    t2 = time.time() - start
    print(f"cython     chol: {t2:9.3e} ({t1/t2:7.3f})")

    # mutual information

    X = rng.random((1000, 3))
    s = 100

    start = time.time()
    indexes = sensor.mi_naive(X, kernel, s)
    t1 = t2 = time.time() - start
    print(f"mi        naive: {t1:9.3e} ({t1/t1:7.3f})")

    start = time.time()
    indexes = sensor.mi_prec(X, kernel, s)
    t2 = time.time() - start
    print(f"python     prec: {t2:9.3e} ({t1/t2:7.3f})")

    start = time.time()
    indexes = sensor.mi_chol(X, kernel, s)
    t2 = time.time() - start
    print(f"python     chol: {t2:9.3e} ({t1/t2:7.3f})")
