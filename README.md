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

| method        | time (seconds)                              |
| ------------- | ------------------------------------------- |
| entropy naive | 0.030 ( 1.000)                              |
| python prec   | 0.011 ( 2.801)                              |
| python prchol | 0.014 ( 2.213)                              |
| python chol   | 0.008 ( 3.613)                              |
| cython chol   | 0.002 (15.417)                              |
| julia         | 0.001 (20.718) (5 allocations: 798.047 KiB) |

### mi (n = 100, s = 10)

| method      | time (seconds)                                 |
| ----------- | ---------------------------------------------- |
| mi naive    | 0.003 ( 1.000)                                 |
| python prec | 0.001 ( 2.334)                                 |
| python chol | 0.005 ( 0.664)                                 |
| julia       | 0.001 ( 3.597) (1.22 k allocations: 1.416 MiB) |

### entropy (n = 10_000, s = 200)

| method        | time (seconds)                             |
| ------------- | ------------------------------------------ |
| entropy naive | 1.812 ( 1.000)                             |
| python prec   | 0.316 ( 5.738)                             |
| python prchol | 0.325 ( 5.575)                             |
| python chol   | 0.301 ( 6.020)                             |
| cython chol   | 0.107 (16.870)                             |
| julia         | 0.075 (24.319) (7 allocations: 15.413 MiB) |

### mi (n = 1000, s = 100)

| method      | time (seconds)                                                   |
| ----------- | ---------------------------------------------------------------- |
| mi naive    | 2.599 ( 1.000)                                                   |
| python prec | 0.769 ( 3.378)                                                   |
| python chol | 1.331 ( 1.953)                                                   |
| julia       | 0.552 ( 4.707) (92.78 k allocations: 729.614 MiB, 3.28% gc time) |
