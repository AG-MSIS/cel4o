from enum import Enum

import numpy as np
import numpy.typing as npt

from compact_matrix_sqrt import (
    cel4o_D,
    cel4o_Q,
    compact_matrix_sqrt_D,
    compact_matrix_sqrt_Q,
    dct_matrix
)
from compact_matrix_sqrt.algorithms import random_Q

from ..paths import DATA_DIR


_OPTIMIZED_Q_PATH = DATA_DIR / "optimized_q_matrices.npz"
_OPTIMIZED_Q_MATRICES: dict[str, npt.NDArray[np.float64]] | None = None
_Q_CACHE: dict[tuple[int, int], npt.NDArray[np.float64]] = {}


def get_Q(size: tuple[int, int]) -> npt.NDArray[np.float64]:
    """Return the closest cached optimized matrix that can be safely cropped.

    Landscape matrices may lose rows, while portrait matrices may lose
    columns. These operations preserve row and column orthonormality,
    respectively.
    """
    n, m = size
    if n < 1 or m < 1:
        raise ValueError("matrix dimensions must be positive")

    cached = _Q_CACHE.get(size)
    if cached is not None:
        return cached

    global _OPTIMIZED_Q_MATRICES
    if _OPTIMIZED_Q_MATRICES is None:
        if not _OPTIMIZED_Q_PATH.is_file():
            raise FileNotFoundError(
                f"Optimized Q archive not found: {_OPTIMIZED_Q_PATH.resolve()}"
            )
        with np.load(_OPTIMIZED_Q_PATH) as archive:
            _OPTIMIZED_Q_MATRICES = {
                key: np.asarray(archive[key], dtype=np.float64)
                for key in archive.files
                if key.startswith("Q_")
            }

    candidates: list[tuple[int, str, npt.NDArray[np.float64]]] = []
    for name, matrix in _OPTIMIZED_Q_MATRICES.items():
        rows, cols = matrix.shape
        if n <= m and rows <= cols and cols == m and rows >= n:
            candidates.append((rows - n, name, matrix[:n, :]))
        elif n > m and rows >= cols and rows == n and cols >= m:
            candidates.append((cols - m, name, matrix[:, :m]))

    if not candidates:
        available = sorted(matrix.shape for matrix in _OPTIMIZED_Q_MATRICES.values())
        raise KeyError(
            f"No optimized Q can be cropped to {size}; available shapes: {available}"
        )

    _, _, result = min(candidates, key=lambda candidate: (candidate[0], candidate[1]))
    result.setflags(write=False)
    _Q_CACHE[size] = result
    return result

class SigmaPointMethod(Enum):
    CHOLESKY = 0
    SQRTM = 1
    CEL4O_D_DCT = 2
    CEL4O_D = 2  # Backward-compatible alias.
    CEL4O_D_RANDOM = 3
    CEL4O_D_N_RANDOM_Q = 3  # Backward-compatible alias.
    CEL4O_Q = 4
    CEL4O_D_2N_OPTIMIZED = 5
    CEL4O_D_2N_DCT = 6
    CEL4O_Q_2N = 7
    CEL4O_D_2N_RANDOM_Q = 8
    CEL4O_D_OPTIMIZED = 9


def sigma_point_direction_count(n: int, method: SigmaPointMethod) -> int:
    """Number of positive/negative direction pairs used by ``method``."""
    if method in (SigmaPointMethod.CEL4O_D_2N_OPTIMIZED, SigmaPointMethod.CEL4O_D_2N_DCT,
        SigmaPointMethod.CEL4O_Q_2N, SigmaPointMethod.CEL4O_D_2N_RANDOM_Q):
        return 2 * n
    return n


def _sqrtm(matrix: npt.NDArray[np.float64], tolerance: float = 1e-6) -> npt.NDArray[np.float64]:
    """Calculates a "safe" matrix square root of a symmetric positive semi-definite matrix. Negative
    eigenvalues are clipped as long as they aren't too negative.

    :param matrix: The matrix.
        (n, n)
    :param tolerance: Fraction of the largest eigenvalue by which the smallest one may be negative.
    :return: Its square root.
        (n, n)
    """
    eva, eve = np.linalg.eigh(matrix)
    # Smallest eigenvalue is more negative than a fraction of the largest:
    if eva[0] < -tolerance * np.abs(eva[-1]):
        raise ValueError(f"Matrix too far from positive semidefinite {eva[0]} {eva[-1]}")
    return (eve * np.sqrt(np.maximum(eva, 0))) @ eve.T


def generate_sigma_points(
    mean,
    covariance: npt.NDArray[np.float64],
    method: SigmaPointMethod,
    *,
    kappa: float,
    offset: npt.NDArray[np.float64] | None = None,
) -> list:
    """Generates sigma points around a mean state and a covariance matrix.

    :param mean: An element of the state manifold that will be the mean of the sigma points.
    :param covariance: A matrix that will be the covariance of the sigma points.
        (tangent_dim, tangent_dim)
    :param method: The method to use to draw sigma points.
    :param kappa: Additional spread of sigma points.
    :param offset: Additional offset (in tangent space) for each sigma point. This is needed for
        moving the reference on non-Euclidean boxplus manifolds.
        (tangent_dim,)
    :return: List of ``2 * m + 1`` sigma points, where ``m`` is the
        method-specific number of directions (``n`` or ``2 * n``).
    """
    assert len(covariance.shape) == 2
    assert covariance.shape[0] == covariance.shape[1]
    assert kappa >= 0
    n = covariance.shape[0]
    m = sigma_point_direction_count(n, method)
    scaled_covariance = (m + kappa) * covariance
    match method:
        case SigmaPointMethod.CHOLESKY:
            directions = np.linalg.cholesky(scaled_covariance)
        case SigmaPointMethod.SQRTM:
            directions = _sqrtm(scaled_covariance)
        case SigmaPointMethod.CEL4O_D_DCT:
            directions = compact_matrix_sqrt_D(scaled_covariance)
        case SigmaPointMethod.CEL4O_D_OPTIMIZED:
            cholesky = np.linalg.cholesky(scaled_covariance)
            Q = get_Q((n, n))
            directions = cel4o_D(cholesky, Q)
        case SigmaPointMethod.CEL4O_Q:
            directions = compact_matrix_sqrt_Q(scaled_covariance)
        case SigmaPointMethod.CEL4O_D_2N_OPTIMIZED:
            cholesky = np.linalg.cholesky(scaled_covariance)
            Q = get_Q((n, m))
            directions = cel4o_D(cholesky, Q)
        case SigmaPointMethod.CEL4O_D_2N_DCT:
            cholesky = np.linalg.cholesky(scaled_covariance)
            Q = dct_matrix ((m,n), make_orthonormal=True).T
            directions = cel4o_D(cholesky, Q)
        case SigmaPointMethod.CEL4O_D_2N_RANDOM_Q:
            cholesky = np.linalg.cholesky(scaled_covariance)
            Q = random_Q(np.random.default_rng(90), (n, m))
            directions = cel4o_D(cholesky, Q)
        case SigmaPointMethod.CEL4O_D_RANDOM:
            cholesky = np.linalg.cholesky(scaled_covariance)
            Q = random_Q(np.random.default_rng(90), (n, m))
            directions = cel4o_D(cholesky, Q)
        case SigmaPointMethod.CEL4O_Q_2N:
            cholesky = np.linalg.cholesky(scaled_covariance)
            rectangular = np.pad(cholesky, ((0, 0), (0, m - n)))
            directions = cel4o_Q(rectangular)
    delta = np.concatenate([np.zeros((n, 1), dtype=np.float64), directions, -directions], axis=-1)
    if offset is not None:
        delta = delta + offset[:, np.newaxis]
    return [mean + delta[:, i] for i in range(delta.shape[-1])]


def generate_sigma_weights(
    n: int, *, kappa: float, num_directions: int | None = None
) -> npt.NDArray[np.float64]:
    """Generates weight vector to go with sigma points from generate_sigma_points.

    :param n: The dimension of the tangent space.
    :param kappa: Additional spread of sigma points.
    :param num_directions: Number of positive/negative direction pairs.
        Defaults to ``n``.
    :return: Array of weights, corresponding to the result of generate_sigma_points.
        (2 * num_directions + 1,)
    """
    assert kappa >= 0
    m = n if num_directions is None else num_directions
    assert m >= n
    return np.array(
        [kappa / (m + kappa)] + [1 / (2 * (m + kappa))] * 2 * m,
        dtype=np.float64,
    )


def mean_of_sigma_points(elems: list, *, weights: npt.NDArray[np.float64]):
    """Calculates the mean of a list of sigma points, potentially on a [+]-manifold.

    :param elems: The list of elements.
    :param weights: Array of weights per element.
        (num_of_points,)
    :return: The mean.
    """
    assert len(elems) > 0
    assert len(weights.shape) == 1
    assert weights.shape[0] == len(elems)
    if isinstance(elems[0], np.ndarray):
        assert len(elems[0].shape) == 1
        return np.sum(np.stack(elems, axis=-1) * weights, axis=-1)
    elif hasattr(elems[0].__class__, "mean"):
        return elems[0].__class__.mean(elems, weights=weights)
    else:
        # Fixed point iteration. The 0th element is a good start since it is often the "mean" sigma
        # point.
        mean = elems[0]
        for _ in range(20):
            delta = np.sum(np.stack([_ - mean for _ in elems], axis=-1) * weights, axis=-1)
            if np.linalg.norm(delta) < 1e-7:
                return mean
            mean = mean + delta
        raise RuntimeError(f"[+] mean iteration did not converge {mean} {elems}.")


def covariance_of_sigma_points(
    elems1: list,
    mean1,
    elems2: list | None = None,
    mean2=None,
    *,
    weights: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Calculates the covariance matrix over a list of sigma points, or the cross-covariance matrix
    with another list (of the same length, but not necessarily same dimension).

    :param elems1: The list of elements.
    :param mean1: The mean of the elements.
    :param elems2: Optional second list of elements.
    :param mean2: Optional mean of the second list of elements.
    :param weights: Array of weights per element.
        (num_of_points,)
    :return: The (cross-)covariance matrix.
        (tangent_dim1, tangent_dim2)
    """
    assert len(elems1) > 0
    if mean2 is None or elems2 is None:
        mean2 = mean1
        elems2 = elems1
    assert len(weights.shape) == 1
    assert weights.shape[0] == len(elems1)
    return np.sum(
        np.stack(
            [np.outer(_[0] - mean1, _[1] - mean2) for _ in zip(elems1, elems2, strict=True)],
            axis=-1,
        )
        * weights,
        axis=-1,
    )
