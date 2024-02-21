# sensor

Information-theoretic sensor placement [1]
implemented in Python, Cython, JAX, and Julia.

[1] A. Krause, A. Singh, and C. Guestrin, "Near-Optimal Sensor Placements
in Gaussian Processes: Theory, Efficient Algorithms and Empirical
Studies," J. Mach. Learn. Res., vol. 9, pp. 235–284, Jun. 2008.

## Installation

The Cython implementation relies on Intel's oneMKL library which is unfree.

Uncomment `pkgs.mkl` in `python/flake.nix` and run

```bash
NIXPKGS_ALLOW_UNFREE=1 nix develop --impure
python setup.py build_ext --inplace
```

to compile the Cython extensions. MKL can be
disabled once the compilation is complete.

## Benchmarks

All experiments are done with a Matérn kernel with smoothness `5 / 2`
and length scale `1`. Points are uniformly drawn from the unit hypercube
with spacial dimension `d` = 3. `n` denotes the number of points and `s`
the number of points selected. `entropy` refers to greedily selecting
the points with maximum conditional entropy and `mi` to maximizing the
conditional mutual information with unselected points.

### entropy (n = 1000, s = 100)

| method         | time (seconds)                                  |
| -------------- | ----------------------------------------------- |
| entropy naive  | 2.884e-02 ( 1.000)                              |
| python prec    | 9.967e-03 ( 2.893)                              |
| python prechol | 1.234e-02 ( 2.337)                              |
| python chol    | 7.763e-03 ( 3.714)                              |
| cython chol    | 1.481e-03 (19.473)                              |
| julia          | 1.419e-03 (20.324) (5 allocations: 798.047 KiB) |

### mi (n = 100, s = 10)

| method      | time (seconds)                                    |
| ----------- | ------------------------------------------------- |
| mi naive    | 2.970e-03 (1.000)                                 |
| python prec | 1.420e-03 (2.091)                                 |
| python chol | 3.988e-03 (0.745)                                 |
| julia       | 7.320e-04 (4.057) (1.22 k allocations: 1.416 MiB) |

### entropy (n = 10_000, s = 200)

| method         | time (seconds)                                 |
| -------------- | ---------------------------------------------- |
| entropy naive  | 2.075e+00 ( 1.000)                             |
| python prec    | 2.203e-01 ( 9.416)                             |
| python prechol | 2.219e-01 ( 9.352)                             |
| python chol    | 1.925e-01 (10.779)                             |
| cython chol    | 7.852e-02 (26.423)                             |
| julia          | 8.293e-02 (25.021) (7 allocations: 15.413 MiB) |

### mi (n = 1000, s = 100)

| method      | time (seconds)                                                      |
| ----------- | ------------------------------------------------------------------- |
| mi naive    | 2.929e+00 (1.000)                                                   |
| python prec | 8.127e-01 (3.604)                                                   |
| python chol | 1.329e+00 (2.203)                                                   |
| julia       | 5.660e-01 (5.175) (92.78 k allocations: 729.614 MiB, 2.48% gc time) |
