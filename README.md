# Compact Matrix Square Roots

Python and C++ implementation of algorithms for computing matrix square
roots with small entries, designed to generate compact sigma points for
nonlinear Gaussian estimation.

For a symmetric positive-definite matrix `Sigma`, the package computes a
matrix `L` satisfying

``` python
L @ L.T == Sigma
```

while reducing its weighted L4 loss. The performance-critical algorithms
are implemented in dependency-free C++ with a thin pybind11 interface
for Python.

## Main functions

### `compact_matrix_sqrt_D`

``` python
compact_matrix_sqrt_D(Sigma, row_weights=None)
```

Computes a square root using a DCT transformation followed by
conditional expectation optimization of a sign diagonal.

With the natural row weights (`row_weights=None`), the weighted L4 loss
is guaranteed to be at most 4.5.

### `compact_matrix_sqrt_Q`

``` python
compact_matrix_sqrt_Q(Sigma, row_weights=None)
```

Computes a square root using conditional expectation optimization of an
orthogonal transformation.

With the natural row weights (`row_weights=None`), the weighted L4 loss
is guaranteed to be at most

    3 - 2 / (n + 2).

Both functions return a matrix `L` satisfying `L @ L.T = Sigma`.

## Usage

``` python
import numpy as np
from compact_matrix_sqrt import compact_matrix_sqrt_D, compact_matrix_sqrt_Q

rng = np.random.default_rng(0)
A = rng.normal(size=(10, 10))
Sigma = A @ A.T + np.eye(10)

L_D = compact_matrix_sqrt_D(Sigma)
L_Q = compact_matrix_sqrt_Q(Sigma)

np.testing.assert_allclose(L_D @ L_D.T, Sigma)
np.testing.assert_allclose(L_Q @ L_Q.T, Sigma)
```

Explicit row weights can also be supplied:

``` python
L_D = compact_matrix_sqrt_D(Sigma, row_weights)
L_Q = compact_matrix_sqrt_Q(Sigma, row_weights)
```

## Installation

For development, from the repository directory:

``` bash
python -m pip install -e .
```

The package requires a C++ compiler for building the extension.

## Research and reference implementation

`algorithms.py` contains the Python reference implementations and
additional functions used to develop, analyze, and test the algorithms
described in the paper.

These functions are not part of the main user-facing API, but can be
imported explicitly:

``` python
import compact_matrix_sqrt.algorithms as algorithms
```

The C++ and Python implementations are compared in the unit tests.

## Repository structure

-   `python/compact_matrix_sqrt/`
    -   `wrapper.py` --- main user-facing functions
    -   `algorithms.py` --- Python reference and research
        implementations
-   `cxx/`
    -   `src/` --- dependency-free C++ implementation and Python bindings
    -   `include/` --- public C++ headers
    -   `CMakeLists.txt` --- C++ and Python extension build configuration
-   `tests/` --- unit tests, including comparisons of the C++ and Python
    implementations
-   `paper/` --- experiments and figures of the paper (Python package
    `paper`, run from the repository root, e.g.
    `python -m paper.experiments.l4_loss`)
    -   `experiments/` --- L4 loss and runtime over the dimension,
        unscented transform of polar coordinates
        (`polar_matrix_sqrt.py`), filter experiment (`filtering/`)
    -   `filters/`, `systems/` --- UKF/EKF and the simulated system
    -   `scripts/` --- helper scripts (e.g. optimizing Q matrices)
    -   `visualization/` --- visualizations used to illustrate the
        algorithms
    -   `tests/` --- tests of the experiment code
    -   `data/`, `figures/` --- logs, results and generated figures

## C++ implementation

The C++ core does not depend on Python, NumPy, Eigen, or pybind11. It
operates on lightweight matrix views, making it possible to integrate
the algorithms into C++ projects using different matrix libraries.

Only the Python binding layer depends on pybind11 and NumPy.

## Testing

Install pytest if necessary and run:

``` bash
python -m pip install pytest
pytest
```

## Building a wheel

``` bash
python -m pip install build
python -m build
```

## Paper

The algorithms, theoretical guarantees, visualizations, and UKF
experiments are described in the accompanying paper.

Udo Frese, Arne Hasselbring, Compact Sigma-Points
via Low-Entry Matrix Square Roots, 
submitted to Symposium on Sensor Data Fusion, Bonn, 2026

## License

This project is licensed under the MIT License. See `LICENSE`.
