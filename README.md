# sensor-placement

Information-theoretic sensor placement [1]
implemented in Python, Cython, JAX, and Julia.

[1] A. Krause, A. Singh, and C. Guestrin, "Near-Optimal Sensor Placements
in Gaussian Processes: Theory, Efficient Algorithms and Empirical
Studies," J. Mach. Learn. Res., vol. 9, pp. 235–284, Jun. 2008.

## Installation

### Python

The Python dependencies are installed with Nix.

Navigate to the `python` directory and run

```bash
nix develop
```

### Cython

The Cython implementation relies on Intel's oneMKL library which is unfree.

Navigate to the `python` directory and uncomment `pkgs.mkl` in `flake.nix`. Run

```bash
NIXPKGS_ALLOW_UNFREE=1 nix develop --impure
python setup.py build_ext --inplace
```

to compile the Cython extensions.

### JAX

JAX on CPU is automatically installed with the Python dependencies.

For GPU support, create a Conda environment with
[micromamba](https://mamba.readthedocs.io/en/latest/installation/micromamba-installation.html)
from the explicitly pinned dependencies with

```bash
micromamba create -y --prefix ./.venv -f linux-64-spec-list.txt
eval "$(micromamba shell hook)"
micromamba activate ./.venv
pip install --requirement requirements.txt
```

from the `python` directory. Note that this requires a x86-64 Linux system.

Alternatively, create from the environment specification with

```bash
micromamba create --prefix ./.venv -f environment.yml
eval "$(micromamba shell hook)"
micromamba activate ./.venv
```

This is up-to-date and works on more platforms but
the resolved dependencies may not work as expected.

### Julia

Installation is done using the standard `Pkg` interface.

Navigate to the `Sensors.jl/examples` directory and run

```bash
julia --project="@."
```

```julia
(examples) pkg> instantiate
```

## Running

### Python, Cython, and JAX

Navigate to the `python` directory and run

```bash
python main.py
```

### Julia

Navigate to the `Sensors.jl/examples` directory and run

```bash
julia --project="@." main.jl
```

## Benchmarks

All experiments are done with a Matérn kernel with smoothness `5 / 2`
and length scale `1`. Points are uniformly drawn from the unit hypercube
with spacial dimension `d` = 3. `n` denotes the number of points and `s`
the number of points selected. `entropy` refers to greedily selecting
the points with maximum conditional entropy and `mi` to maximizing the
conditional mutual information with unselected points.

CPU experiments were run on a laptop. The
GPU used was an A100 with 12 GB of memory.

### entropy (n = 1000, s = 100)

| method         | time (seconds)                                  |
| -------------- | ----------------------------------------------- |
| entropy naive  | 2.884e-02 (1.000)                               |
| python prec    | 9.967e-03 (2.893)                               |
| python prechol | 1.234e-02 (2.337)                               |
| python chol    | 7.763e-03 (3.714)                               |
| cython chol    | 1.481e-03 (19.473)                              |
| julia          | 1.419e-03 (20.324) (5 allocations: 798.047 KiB) |
| jax (cpu)      | 1.875e-02 (1.538)                               |
| jax (gpu)      | 4.372e-03 ( 6.596)                              |

### mi (n = 100, s = 10)

| method      | time (seconds)                                    |
| ----------- | ------------------------------------------------- |
| mi naive    | 2.970e-03 (1.000)                                 |
| python prec | 1.420e-03 (2.091)                                 |
| python chol | 3.988e-03 (0.745)                                 |
| julia       | 7.320e-04 (4.057) (1.22 k allocations: 1.416 MiB) |
| jax (cpu)   | 6.456e-04 (4.600)                                 |
| jax (gpu)   | 1.675e-03 (4.400)                                 |

### entropy (n = 10_000, s = 200)

| method         | time (seconds)                                 |
| -------------- | ---------------------------------------------- |
| entropy naive  | 2.075e+00 (1.000)                              |
| python prec    | 2.203e-01 (9.416)                              |
| python prechol | 2.219e-01 (9.352)                              |
| python chol    | 1.925e-01 (10.779)                             |
| cython chol    | 7.852e-02 (26.423)                             |
| julia          | 8.293e-02 (25.021) (7 allocations: 15.413 MiB) |
| jax (cpu)      | 5.322e-01 (3.899)                              |
| jax (gpu)      | 1.309e-02 (158.517)                            |

### mi (n = 1000, s = 100)

| method      | time (seconds)                                                      |
| ----------- | ------------------------------------------------------------------- |
| mi naive    | 2.929e+00 (1.000)                                                   |
| python prec | 8.127e-01 (3.604)                                                   |
| python chol | 1.329e+00 (2.203)                                                   |
| julia       | 5.660e-01 (5.175) (92.78 k allocations: 729.614 MiB, 2.48% gc time) |
| jax (cpu)   | 2.934e-01 (9.983)                                                   |
| jax (gpu)   | 2.364e-02 (123.900)                                                 |
