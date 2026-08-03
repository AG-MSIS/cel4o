import math
from itertools import product
from typing import overload, Literal
import numpy as np
import numpy.typing as npt
import numpy.random
import pytest
import hashlib
from scipy.stats import t

from compact_matrix_sqrt import *
from compact_matrix_sqrt.algorithms import *

def stable_seed (name: str) -> int:
    """Maps a string deterministically to a 64-bit seed."""
    return int.from_bytes(
        hashlib.blake2b(name.encode(), digest_size=8).digest(),
        "little",
    )

@pytest.fixture
def rng(request):
    """Random number generator seeded based on test name
    This is reproducible but different for different test
    cases"""
    seed = stable_seed(request.node.nodeid)
    return np.random.default_rng(seed)

def assert_mean_plausible(samples: npt.ArrayLike, expected: float,
    relation: Literal["below", "above", "equal"], *, alpha: float = 1e-2,
    atol=1E-10, rtol=1E-10
) -> None:
    """
    Test whether the sample mean is statistically compatible with being
    below, above, or equal to a specified expected value.

    Parameters
    ----------
    samples:
        Independent Monte-Carlo samples.

    expected:
        The theoretical value or bound to compare against.

    relation:
        ``"below"``:
            Tests compatibility with E[X] <= expected.
            Fails if the sample mean is significantly above expected.

        ``"above"``:
            Tests compatibility with E[X] >= expected.
            Fails if the sample mean is significantly below expected.

        ``"equal"``:
            Tests compatibility with E[X] == expected.
            Fails if the sample mean differs significantly in either
            direction.

    alpha:
        Probability of rejecting the null hypothesis although it is true.
        For ``"equal"``, alpha is divided between the two tails.

    atol, rtol:
        Additional absolute and relative tolerance added to the std.
        deviation. This handles numerical errors in the case of
        identical samples (zero variance).
    """
    samples_a = np.asarray(samples, dtype=np.float64)

    if samples_a.ndim != 1:
        raise ValueError("samples must be one-dimensional")

    if samples_a.size < 2:
        raise ValueError("at least two samples are required")

    if not np.all(np.isfinite(samples_a)):
        raise ValueError("samples must contain only finite values")

    if not np.isfinite(expected):
        raise ValueError("expected must be finite")

    if relation not in ("below", "above", "equal"):
        raise ValueError(
            "relation must be 'below', 'above', or 'equal'"
        )

    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must satisfy 0 < alpha < 1")

    n = samples_a.size
    sample_mean = float(np.mean(samples_a))
    sample_std = float(np.std(samples_a, ddof=1))
    standard_error = sample_std / np.sqrt(n) + atol + rtol*np.abs(sample_mean)
    difference = sample_mean - expected

    # Degenerate case: all samples are identical.
    if standard_error == 0.0:
        if relation == "below":
            plausible = sample_mean <= expected
        elif relation == "above":
            plausible = sample_mean >= expected
        else:
            plausible = sample_mean == expected

        assert plausible, (
            "Sample mean is incompatible with the expected relation:\n"
            f"  relation        = {relation}\n"
            f"  sample mean     = {sample_mean:.12g}\n"
            f"  expected        = {expected:.12g}\n"
            "  standard error  = 0"
        )
        return

    t_statistic = difference / standard_error
    degrees_of_freedom = n - 1

    if relation == "below":
        # H0: E[X] <= expected
        # Reject for a large positive t statistic.
        critical_value = float(
            t.ppf(1.0 - alpha, degrees_of_freedom)
        )
        plausible = t_statistic <= critical_value
        acceptance_description = (
            f"t <= {critical_value:.6g}"
        )

    elif relation == "above":
        # H0: E[X] >= expected
        # Reject for a large negative t statistic.
        critical_value = float(
            t.ppf(1.0 - alpha, degrees_of_freedom)
        )
        plausible = t_statistic >= -critical_value
        acceptance_description = (
            f"t >= {-critical_value:.6g}"
        )

    else:
        # H0: E[X] == expected
        # Two-sided test, alpha / 2 in each tail.
        critical_value = float(
            t.ppf(1.0 - alpha / 2.0, degrees_of_freedom)
        )
        plausible = abs(t_statistic) <= critical_value
        acceptance_description = (
            f"|t| <= {critical_value:.6g}"
        )

    assert plausible, (
        "Monte-Carlo mean is statistically incompatible with the "
        "expected relation:\n"
        f"  relation        = {relation}\n"
        f"  sample mean     = {sample_mean:.12g}\n"
        f"  expected        = {expected:.12g}\n"
        f"  difference      = {difference:.12g}\n"
        f"  sample std      = {sample_std:.12g}\n"
        f"  standard error  = {standard_error:.12g}\n"
        f"  t statistic     = {t_statistic:.6g}\n"
        f"  acceptance      = {acceptance_description}\n"
        f"  alpha           = {alpha:.6g}"
    )


@pytest.mark.parametrize("n", [1, 2, 3, 10, 100])
def test_random_Q (n:int, rng:numpy.random.Generator)-> None:
    """Checks random orthonormal matrix for orthonormality
    and that two calls return different matrices"""
    Q = random_Q (rng, n)
    np.testing.assert_allclose (Q@Q.T, np.eye(n), atol=1E-10, rtol=0.0)
    Q2 = random_Q (rng, n)
    assert n<=2 or float(np.linalg.norm(Q-Q2))>0.1

def test_random_Q2 (rng:numpy.random.Generator)->None:
    n_tests = 100
    for ctr in range(n_tests):
        size = (int(rng.integers(1,8)), int(rng.integers(1,8)))
        Q = random_Q (rng, size)
        assert Q.shape == size
        if size[0]>=size[1]: np.testing.assert_allclose (Q.T@Q, np.eye(size[1]), atol=1E-10)
        if size[1]>=size[0]: np.testing.assert_allclose (Q@Q.T, np.eye(size[0]), atol=1E-10)

@pytest.mark.parametrize("shape", [(1, 2, 2), (3, 4, 7), (8, 3, 5)])
def test_random_LQ(
    shape:tuple[int, int, int], rng:np.random.Generator
) -> None:
    n, p, m = shape
    L, Q = random_LQ(rng, n, p, m)

    assert L.shape == (n, p)
    assert Q.shape == (p, m)
    np.testing.assert_allclose(Q @ Q.T, np.eye(p), atol=1e-12, rtol=0.0)
    np.testing.assert_allclose((L @ Q) @ (L @ Q).T, L @ L.T, atol=1e-12, rtol=0.0)

    with pytest.raises(ValueError, match="m must be >= p"):
        random_LQ(rng, n, p, p - 1)


@pytest.mark.parametrize("n", [10, 30, 100])
@pytest.mark.slow
def test_random_L (n:int, rng:numpy.random.Generator)-> None:
    """
    Checks the statistics of random_L regarding singular values
    """
    n_samples = 200
    samples = np.empty((2,n_samples))
    for i in range(n_samples):
        L = random_L (rng, n)
        sigmas = np.linalg.svdvals (L)
        samples[0,i] = np.min(sigmas)
        samples[1,i] = np.max(sigmas)

    # Approximately formulars for the smallest and largest
    # singular value of a matrix with normal entries
    # Doesn't work for really small n
    assert_mean_plausible (samples[0], 0.65568/n, 'equal', atol=1/n)
    assert_mean_plausible (samples[1], 2 - 0.760*n**(-2/3), 'equal', atol=1/n)

def test_make_row_weights (rng:numpy.random.Generator) -> None:
    """Tests that computed weights normalize the
    4th moment to one."""
    n = 4
    A = rng.normal (size=(n,n))
    w = make_row_weights (A, row_weights=None)
    row_sum = w*np.sum(A**2, axis=1)**2
    np.testing.assert_allclose (row_sum, np.ones(n), atol=1E-10, rtol=0.0)

@pytest.mark.parametrize("n", [1, 2, 3, 10, 100])
def test_weighted_l4_loss_identity(n:int) -> None:
    """Test with known formula L4(I_n)=n."""
    result = weighted_l4_loss_python (np.eye(n))

    assert result == pytest.approx(n)

@pytest.mark.parametrize("n", [1, 2, 3, 10, 100])
def test_weighted_l4_loss_identity(n:int) -> None:
    """Test with known formula L4(I_n)=n."""
    result = weighted_l4_loss_python (np.eye(n))

    assert result == pytest.approx(n)

@pytest.mark.parametrize("shape", [(3, 3), (3, 7), (7, 3)])
def test_weighted_l4_loss_cpp(rng, shape: tuple[int, int]) -> None:
    """Compares to the python implementation."""
    for _ in range(20):
        L = rng.normal(size=shape)
        l4loss_true = weighted_l4_loss_python (L)
        l4loss = weighted_l4_loss (L)
        assert l4loss == pytest.approx(l4loss_true, abs=1E-9)


def test_weighted_l4_loss_rectangular_normalization() -> None:
    L = np.ones((2, 6), dtype=np.float64)
    row_weights = np.ones(2, dtype=np.float64)

    # Raw weighted sum is 12; m/n = 3.
    assert weighted_l4_loss_python(L, row_weights) == pytest.approx(36.0)
    assert weighted_l4_loss(L, row_weights) == pytest.approx(36.0)


def hadamard (log2n:int, orthonormal=False):
    """Returns a hadamard matrix of dimension 2^n.
    If orthonormal is False, it is the usual with entries +/-1.
    If True, the entries are +/-2^(log2n/2), so the matrix
    becomes orthonormal"""
    if log2n==1: return np.eye(1)
    H = hadamard (log2n-1)/np.sqrt(2)
    return np.block ([[H, H], [H, -H]])

@pytest.mark.parametrize("log2n", range(1,6))
def test_weighted_l4_loss_hadamard(log2n:int) -> None:
    """Test with a hadamard matrix where L4(H)=1"""
    Q = hadamard (log2n)
    print(Q)
    np.testing.assert_allclose (Q@Q.T, np.eye(Q.shape[0]), atol=1E-10, rtol=0.0)
    result = weighted_l4_loss_python (Q)

    assert result == pytest.approx(1)

@pytest.mark.parametrize("n", [1, 2, 3, 10, 100])
def test_dct_matrix(n:int) -> None:
    Q = dct_matrix (n, True)
    np.testing.assert_allclose (Q@Q.T, np.eye(n), atol=1E-10, rtol=0.0)
    assert np.all(np.abs(Q)<=np.sqrt(2/n)+1E-10)

def test_dct_matrix2(rng:numpy.random.Generator) -> None:
    """Tests rectangular (portrait) DCT matrices by checking the
    affirmed properties."""
    n_tests = 100
    for ctr in range(n_tests):
        m = int(rng.integers (1,10))
        n = int(rng.integers (1, m+1)) # <=m
        Q = dct_matrix ((m,n), True)
        np.testing.assert_allclose (Q.T@Q, np.eye(n), atol=1E-10, rtol=0.0)
        l4 = np.sum(Q**4, axis=0)
        assert np.all(l4<=3/(2*m)+1E-6)

@pytest.mark.parametrize("n", [1, 2, 3, 10, 100])
def test_symmetric_matrix_sqrt (n:int, rng:numpy.random.Generator) -> None:
    Sigma = random_SPSD (rng, n)
    L = symmetric_matrix_sqrt (Sigma)
    np.testing.assert_allclose (L@L, Sigma, atol=1E-10, rtol=0.0)

@pytest.mark.parametrize("n", [3, 10, 100])
@pytest.mark.slow
def test_adversarial (n:int, rng:np.random.Generator) -> None:
    losses = []
    for ctr in range(20):
        Q = random_Q (rng, n)
        L = adversarial (Q)
        l4_loss = weighted_l4_loss_python (L@Q)
        losses.append (l4_loss)
        assert l4_loss >= n/3+5/6
    print (f"Mean adversarial l4-loss n={n} n {np.mean(losses)}")

@pytest.mark.parametrize("size", [(1, 2), (3, 5), (4, 10)])
def test_adversarial_landscape(
    size:tuple[int, int], rng:np.random.Generator
) -> None:
    n, m = size
    Q = random_Q(rng, size)
    L = adversarial(Q)

    assert L.shape == (n, n)
    assert (L @ Q).shape == size
    np.testing.assert_allclose(L, np.tril(L), atol=0.0, rtol=0.0)
    np.testing.assert_allclose(np.sum(L**2, axis=1), np.ones(n), atol=1e-12)
    l4_loss = weighted_l4_loss_python(L @ Q)
    assert l4_loss >= n**2/(3*m)

def test_adversarial_transposed_portrait_dct() -> None:
    m, n = 10, 4
    Q = dct_matrix((m, n), make_orthonormal=True).T
    L = adversarial(Q)

    assert L.shape == (n, n)
    assert (L @ Q).shape == (n, m)
    assert weighted_l4_loss_python(L @ Q) >= n**2/(3*m)

def test_adversarial_rejects_portrait_Q(rng:np.random.Generator) -> None:
    with pytest.raises(ValueError, match="portrait"):
        adversarial(random_Q(rng, (5, 3)))

@pytest.mark.parametrize("n", [3, 10, 100])
def test_jacobi_rotate_two_columns_l4 (n:int, rng:np.random.Generator) -> None:
    """Performs optimal rotations on random columns and checks..."""
    A = rng.normal(size=(n,n))
    oldpair = None
    for ctr in range(25):
        k = rng.integers (1, n)
        l = rng.integers (0, k)
        if (k,l)==oldpair: continue
        oldpair = (k,l)
        old_A = A.copy()
        _, gain = jacobi_rotate_two_columns_l4 (A, k, l, row_weights=None, inplace=True)
        assert gain>0 # may not become worse, very unlikely that exactly at optimum
        diff = np.abs(A-old_A)
        diff[:,k] = 0
        diff[:,l] = 0
        np.testing.assert_equal (diff, np.zeros((n,n))) # only column k and l may change
        np.testing.assert_allclose (A@A.T, old_A@old_A.T, atol=1E-10, rtol=0.0) # same covariance expressed
        l4_old = weighted_l4_loss_python (old_A)
        l4_new = weighted_l4_loss_python (A)
        assert math.isclose (l4_old-l4_new, gain, abs_tol=1E-10, rel_tol=0.0) # gain equals difference in l4 loss values

def test_jacobi_rotate_two_columns_l4_2 (rng:np.random.Generator) -> None:
    """Checks that after a rotation and additional rotation gives angle 0."""
    n = 13
    A = rng.normal(size=(n,n))
    oldpair = None
    for ctr in range(25):
        k = rng.integers (1, n)
        l = rng.integers (0, k)
        if (k,l)==oldpair: continue
        oldpair = (k,l)
        _, gain = jacobi_rotate_two_columns_l4 (A, k, l, row_weights=None, inplace=True)
        assert gain>0 # may not become worse, very unlikely that exactly at optimum
        alpha, gain = jacobi_rotate_two_columns_l4 (A, k, l, row_weights=None, inplace=True)
        assert alpha == pytest.approx(0, abs=1E-5)
        assert gain<1E-5


@pytest.mark.parametrize("n", [3, 10, 43])
def test_cyclic_distance_sweep_l4 (n:int, rng:np.random.Generator) -> None:
    """Runs cyclic_distance_sweep on a random matrix and checks, that
    the represented covariance is maintained and the l4_loss reduced."""
    A = rng.normal(size=(n,n))
    old_A = A.copy()
    distance = 7
    if distance>=n: distance = n-1
    gain = cyclic_distance_sweep_l4 (A, distance, row_weights=None, inplace=True)
    assert isinstance (gain, float)
    assert gain>0
    np.testing.assert_allclose (A@A.T, old_A@old_A.T, atol=1E-10, rtol=0.0) # same covariance expressed
    l4_old = weighted_l4_loss_python (old_A)
    l4_new = weighted_l4_loss_python (A)
    assert l4_old-gain == pytest.approx(l4_new, abs=1E-10, rel=0.0) # gain equals difference in l4 loss values

@pytest.mark.parametrize("n", [3, 10, 15])
def test_optimize_l4 (n:int, rng:np.random.Generator) -> None:
    """Runs optimize_l4 with a random SPSD Sigma and checks, that
    the represented covariance is maintained and the l4_loss reduced over cholesky."""
    Sigma = random_SPSD (rng, n)
    new_l4, gain, U = optimize_l4 (Sigma, row_weights=None, inplace=False, from_sigma=True, iterations=3)
    assert gain>0
    np.testing.assert_allclose (U@U.T, Sigma, atol=1E-10, rtol=0.0) # Right covariance expressed
    L = np.linalg.cholesky (Sigma)
    old_l4 = weighted_l4_loss_python (L)
    assert math.isclose (new_l4, weighted_l4_loss_python(U), abs_tol=1E-10, rel_tol=0)
    assert math.isclose (new_l4, old_l4-gain)

@pytest.mark.parametrize("size", [(3, 7), (7, 3)])
def test_l4_column_optimization_rectangular(
    size:tuple[int, int], rng:np.random.Generator
) -> None:
    """All column-rotation optimization stages support rectangular U."""
    U = rng.normal(size=size)
    covariance = U @ U.T
    initial_loss = weighted_l4_loss_python(U)

    alpha, rotation_gain, rotated = jacobi_rotate_two_columns_l4(
        U, 0, 1, inplace=False
    )
    assert isinstance(alpha, float)
    assert rotation_gain >= 0.0
    assert rotated.shape == size
    np.testing.assert_allclose(U @ U.T, covariance, atol=1e-12, rtol=0.0)
    np.testing.assert_allclose(rotated @ rotated.T, covariance, atol=1e-12, rtol=0.0)
    assert initial_loss - rotation_gain == pytest.approx(
        weighted_l4_loss_python(rotated), abs=1e-10, rel=0.0
    )

    sweep_gain, swept = cyclic_distance_sweep_l4(
        U, 1, row_weights=None, inplace=False
    )
    assert sweep_gain >= 0.0
    assert swept.shape == size
    np.testing.assert_allclose(swept @ swept.T, covariance, atol=1e-12, rtol=0.0)
    assert initial_loss - sweep_gain == pytest.approx(
        weighted_l4_loss_python(swept), abs=1e-10, rel=0.0
    )

    final_loss, total_gain, optimized = optimize_l4(
        U, inplace=False, iterations=2, verbose=False
    )
    assert optimized.shape == size
    assert total_gain >= 0.0
    np.testing.assert_allclose(optimized @ optimized.T, covariance, atol=1e-11, rtol=0.0)
    assert final_loss == pytest.approx(
        weighted_l4_loss_python(optimized), abs=1e-10, rel=0.0
    )
    assert initial_loss - total_gain == pytest.approx(final_loss, abs=1e-10, rel=0.0)


@pytest.mark.parametrize("n", [3, 10, 15])
def test_rademacher_l4_bound (n:int, rng:np.random.Generator) -> None:
    """Checks that the computed rademacher bound is actuall <=
    the Monte Carlo derived expected value for random Ls and Qs."""
    L, Q = random_LQ (rng, n, n, n)
    n_samples = 1000
    mc_samples = [
        weighted_l4_loss_python (L@np.diag(rng.choice([-1.0, 1.0], size=n))@Q)
        for ctr in range(n_samples)
    ]
    r = rademacher_l4_bound (L, Q)
    print (np.mean(mc_samples), r)
    assert_mean_plausible (mc_samples, r, 'below')


def slow_conditional_expected_value_signs (L: Matrix, Q: Matrix, signs: Vector,
    row_weights: None | npt.ArrayLike = None) -> float:
    """
    Compute the conditional expected weighted L4 loss of L @ D @ Q by
    enumerating all currently unfixed diagonal signs.

    ``signs[k]`` is:
        -1 or +1  for a fixed sign,
         0        for an independent random Rademacher sign.
    """
    L = np.asarray(L, dtype=np.float64)
    Q = np.asarray(Q, dtype=np.float64)
    signs = np.asarray(signs, dtype=np.float64)

    if L.ndim != 2:
        raise ValueError("L must be two-dimensional")

    if Q.ndim != 2:
        raise ValueError("Q must be two-dimensional")

    m, n = L.shape

    if Q.shape[0] != n:
        raise ValueError(
            f"Q must have {n} rows, got {Q.shape[0]}"
        )

    if signs.shape != (n,):
        raise ValueError(
            f"signs must have shape ({n},), got {signs.shape}"
        )

    if not np.all(np.isin(signs, (-1.0, 0.0, 1.0))):
        raise ValueError("signs must contain only -1, 0, and +1")

    row_weights = make_row_weights (L, row_weights)
    random_indices = np.flatnonzero(signs == 0.0)
    loss_sum = 0.0
    number_of_combinations = 1 << random_indices.size

    for random_values in product((-1.0, 1.0), repeat=random_indices.size):
        complete_signs = signs.copy()
        complete_signs[random_indices] = random_values

        LDQ = (L * complete_signs) @ Q
        row_losses = np.sum(LDQ**4, axis=1)

        loss_sum += float(row_weights @ row_losses)

    output_rows = L.shape[0]
    output_columns = Q.shape[1]
    return output_columns / output_rows * loss_sum / number_of_combinations


def slow_conditional_expected_value_signs_change (L: Matrix, Q: Matrix, signs: Vector, 
    j:int, row_weights: None | npt.ArrayLike = None) -> tuple[float, float]:
    "Retuns slow_conditional_expected_value_signs with sign[k] to +1 and -1 as a tuple"
    row_weights = make_row_weights (L, row_weights)
    signstest = signs.copy()
    signstest[j] = 1
    eplus = slow_conditional_expected_value_signs (L, Q, signstest, row_weights)
    signstest[j] = -1
    eminus = slow_conditional_expected_value_signs (L, Q, signstest, row_weights)
    return eplus, eminus


def test_slow_conditional_expected_value_signs (rng:np.random.Generator):
    """Test slow_conditional_expected_value_signs on a random LQ,
    by comparing the values for individual signs fixed -1 and +1
    to the one where they are random."""
    n = 9
    L, Q = random_LQ (rng, n, n, n)
    signs = np.zeros(n, dtype=float)
    signs[0]=1 # without loss of generality
    eall = slow_conditional_expected_value_signs (L, Q, signs)
    for j in range(1, n):
        eplus, eminus = slow_conditional_expected_value_signs_change (L, Q, signs, j)
        print(eall, eplus, eminus)
        assert eall == pytest.approx ((eplus+eminus)/2)

@overload
def slow_cel4o_D(
    L: Matrix, Q: Matrix, row_weights:Weights = None, *, return_history: Literal[False]=False
    )->  tuple[Matrix, float, Vector]: ... 

@overload
def slow_cel4o_D(
    L: Matrix, Q: Matrix, row_weights:Weights = None, *, return_history: Literal[True]
    )->  tuple[Matrix, float, Vector, Vector]: ... 


def slow_cel4o_D(
    L: Matrix, Q: Matrix, row_weights:Weights =None, *, return_history: bool = False
    )->  tuple[Matrix, float, Vector] | tuple[Matrix, float, Vector, Vector]:

    assert L.ndim == 2
    assert Q.ndim == 2
    _, n = L.shape
    assert Q.shape[0] == n

    row_weights = make_row_weights (L, row_weights)
    signs = np.zeros(n, dtype=float)
    signs[0] = 1 # the first decision might be not equal due to numerics, so fix it here
    if return_history:
        history = np.empty(n, dtype=float)
        history[0] = slow_conditional_expected_value_signs (L, Q, signs)
    else: history = None

    for j in range(1, n):
        eplus, eminus = slow_conditional_expected_value_signs_change (L, Q, signs, j, row_weights)
        print (f'{j}: {eplus}, {eminus}')
        signs[j] = 1 if eplus<=eminus else -1
        if return_history:  
            assert history is not None
            history[j] = min(eplus, eminus)
        
    A = (L*signs)@Q
    if return_history:
        assert history is not None
        return A, weighted_l4_loss_python (A, row_weights), signs, history

    return A, weighted_l4_loss_python (A, row_weights), signs

@pytest.mark.parametrize("n", [3, 9, 12])
def test_cel4o_d_with_history (n:int, rng:np.random.Generator):
    "Compares it to slow_cel4o_D on a random problem"
    L, Q = random_LQ (rng, n, n, n)
    A, l4_loss, signs, history = cel4o_D_with_history(L, Q)
    A_true, l4_loss_true, signs_true, history_true = slow_cel4o_D (L, Q, return_history=True)
    sign_history = np.array([
        entry["econd"] for entry in history if entry["step_type"] == "sign"
    ])

    assert l4_loss == pytest.approx(l4_loss_true)
    np.testing.assert_allclose(A@A.T, L@L.T, atol=1E-10)
    np.testing.assert_allclose(A_true@A_true.T, L@L.T, atol=1E-10, rtol=0.0)
    np.testing.assert_equal (signs, signs_true)
    np.testing.assert_allclose (sign_history, history_true, atol=1E-10, rtol=0.0)
    np.testing.assert_allclose (A, A_true, atol=1E-10, rtol=0.0)
    assert history[0]["step_type"] == "init"
    assert history[-1]["step_type"] == "final"
    assert history[-1]["econd"] == pytest.approx(weighted_l4_loss_python(A))

@pytest.mark.parametrize("n", [3, 9, 12])
def test_cel4o_d_python (n:int, rng:np.random.Generator):
    "Compares it to slow_cel4o_D on a random problem"
    L, Q = random_LQ (rng, n, n, n)
    A = cel4o_D_python (L, Q)
    A_true, _, _ = slow_cel4o_D (L, Q)
    np.testing.assert_allclose(A@A.T, L@L.T, atol=1E-10)
    np.testing.assert_allclose(A_true@A_true.T, L@L.T, atol=1E-10, rtol=0.0)
    np.testing.assert_allclose (A, A_true, atol=1E-10, rtol=0.0)

@pytest.mark.parametrize("shape", [(2, 3, 4), (5, 2, 7), (4, 6, 1)])
def test_cel4o_d_rectangular(
    shape:tuple[int, int, int], rng:np.random.Generator
) -> None:
    n, p, m = shape
    L = rng.normal(size=(n, p))
    Q = rng.normal(size=(p, m))
    row_weights = rng.uniform(0.1, 2.0, size=n)

    A = cel4o_D_python(L, Q, row_weights)
    A_cpp = cel4o_D(L, Q, row_weights)
    A_true, l4_loss_true, signs, history = slow_cel4o_D(
        L, Q, row_weights, return_history=True
    )

    assert A.shape == (n, m)
    np.testing.assert_allclose(A, (L * signs) @ Q, atol=1e-12, rtol=0.0)
    np.testing.assert_allclose(A, A_true, atol=1e-12, rtol=0.0)
    np.testing.assert_allclose(A_cpp, A_true, atol=1e-12, rtol=0.0)
    assert weighted_l4_loss_python(A, row_weights) == pytest.approx(l4_loss_true)
    assert history[-1] == pytest.approx(l4_loss_true)

def test_cel4o_d_python_rejects_incompatible_shapes() -> None:
    with pytest.raises(ValueError, match="Incompatible shapes"):
        cel4o_D_python(np.zeros((2, 3)), np.zeros((4, 5)))
    with pytest.raises(ValueError, match="incompatible shapes"):
        cel4o_D(np.zeros((2, 3)), np.zeros((4, 5)))

@pytest.mark.parametrize("n", [3, 9, 12])
def test_cel4o_d_cpp (n:int, rng:np.random.Generator):
    "Compares it to slow_cel4o_D on a random problem"
    L, Q = random_LQ (rng, n, n, n)
    A = cel4o_D (L, Q)
    A_true, _, _ = slow_cel4o_D (L, Q)
    np.testing.assert_allclose(A@A.T, L@L.T, atol=1E-10)
    np.testing.assert_allclose(A_true@A_true.T, L@L.T, atol=1E-10, rtol=0.0)
    np.testing.assert_allclose (A, A_true, atol=1E-10, rtol=0.0)


def test_compact_matrix_sqrt_D (rng:np.random.Generator):
    """Tests that compact_matrix_sqrt fulfills the documentation.
    This is actually pretty redundant to test_cel4o_D"""
    n_tests = 100
    for ctr in range(n_tests):
        n = rng.integers (3, 20)
        Sigma = random_SPSD (rng, n)
        L = compact_matrix_sqrt_D (Sigma)
        assert weighted_l4_loss_python (L) <= 9/2
        np.testing.assert_allclose (L@L.T, Sigma, atol=1E-8)

def test_compact_matrix_sqrt_D_python(rng:np.random.Generator) -> None:
    n = 7
    Sigma = random_SPSD(rng, n)
    QDct = dct_matrix(n, make_orthonormal=True)
    expected = cel4o_D_python(np.linalg.cholesky(Sigma), QDct.T)

    actual = compact_matrix_sqrt_D_python(Sigma, QDct=QDct)
    actual_default = compact_matrix_sqrt_D_python(Sigma)

    np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=0.0)
    np.testing.assert_allclose(actual_default, expected, atol=1e-12, rtol=0.0)
    np.testing.assert_allclose(actual @ actual.T, Sigma, atol=1e-10, rtol=0.0)

@pytest.mark.parametrize("n", [3, 9, 12])
@pytest.mark.slow
def test_ceol4_d_bound1 (n:int, rng:np.random.Generator)->None:
    "Test for random L, Q that the expected value over signs is below the entioned bound."
    for ctr in range(5):
        L, Q = random_LQ (rng, n, n, n)
        bound = ceol4_d_bound1 (Q)
        el4all = slow_conditional_expected_value_signs (L, Q, np.zeros(n))
        assert el4all <= bound


def test_ceol4_d_bound1_rectangular(
    rng: np.random.Generator,
) -> None:
    n, m = 3, 7
    for _ in range(100):
        L, Q = random_LQ(rng, n, n, m)

        bound = ceol4_d_bound1(Q)
        expected_loss = slow_conditional_expected_value_signs(L, Q, np.zeros(n))

        assert expected_loss <= bound


@pytest.mark.parametrize("n", [2, 3, 5, 8])
def test_ceol4_d_bound2_random_Q(n: int, rng: np.random.Generator) -> None:
    """The improved bound contains the exact expectation for random Q and L."""
    for _ in range(10):
        L, Q = random_LQ(rng, n, n, n)

        expected_loss = slow_conditional_expected_value_signs(
            L, Q, np.zeros(n)
        )
        bound1 = ceol4_d_bound1(Q)
        bound2 = ceol4_d_bound2(Q)

        assert expected_loss <= bound2 + 1e-12
        assert bound2 <= bound1 + 1e-12


def test_ceol4_d_bound_metrics_identity() -> None:
    r_max, delta_min = ceol4_d_bound_metrics(np.eye(3))

    assert r_max == pytest.approx(1.0)
    assert delta_min == pytest.approx(2.0)
    assert not ceol4_d_bound2_second_branch_active(np.eye(3))


def test_ceol4_d_soft_delta_min_gradient_matches_finite_difference(
    rng: np.random.Generator,
) -> None:
    Q = rng.normal(size=(4, 6)) * 0.4
    direction = rng.normal(size=Q.shape)
    direction /= np.linalg.norm(direction)
    temperature = 0.2

    gradient, _ = ceol4_d_soft_delta_min_gradient(
        Q, temperature=temperature, return_value=True
    )
    step = 1e-6
    _, value_plus = ceol4_d_soft_delta_min_gradient(
        Q + step * direction, temperature=temperature, return_value=True
    )
    _, value_minus = ceol4_d_soft_delta_min_gradient(
        Q - step * direction, temperature=temperature, return_value=True
    )

    finite_difference = (value_plus - value_minus) / (2.0 * step)
    np.testing.assert_allclose(
        np.sum(gradient * direction), finite_difference, atol=1e-9, rtol=1e-8
    )


def test_ceol4_d_bound2_gradient_matches_finite_difference(
    rng: np.random.Generator,
) -> None:
    Q = rng.normal(size=(4, 6)) * 0.4
    direction = rng.normal(size=Q.shape)
    direction /= np.linalg.norm(direction)
    kwargs = dict(
        r_temperature=0.2,
        delta_temperature=0.2,
        barrier_temperature=0.2,
        barrier_weight=3.0,
        return_value=True,
    )

    gradient, _ = ceol4_d_bound2_gradient(Q, **kwargs)
    step = 1e-6
    _, value_plus = ceol4_d_bound2_gradient(Q + step * direction, **kwargs)
    _, value_minus = ceol4_d_bound2_gradient(Q - step * direction, **kwargs)

    finite_difference = (value_plus - value_minus) / (2.0 * step)
    np.testing.assert_allclose(
        np.sum(gradient * direction), finite_difference, atol=1e-8, rtol=1e-8
    )


@pytest.mark.parametrize("shape", [(4, 4), (4, 7)])
def test_optimize_ceol4_d_bound2_python_stays_on_stiefel_and_descends(
    shape: tuple[int, int], rng: np.random.Generator,
) -> None:
    Q0 = random_Q(rng, shape)
    _, initial_smooth_bound = ceol4_d_bound2_gradient(
        Q0, return_value=True
    )

    Q, history = optimize_ceol4_d_bound2_python(
        Q0, iterations=30, learning_rate=0.05, return_history=True
    )
    _, final_smooth_bound = ceol4_d_bound2_gradient(Q, return_value=True)

    assert Q.shape == shape
    assert history
    assert all(info["accepted"] for info in history)
    assert all(
        info["smooth_bound"] <= info["smooth_bound_before"] + 1e-14
        for info in history
    )
    assert final_smooth_bound <= initial_smooth_bound
    np.testing.assert_allclose(
        Q @ Q.T, np.eye(shape[0]), atol=1e-12, rtol=0.0
    )
    assert history[-1]["orthogonality_error"] < 1e-12


@pytest.mark.parametrize("shape", [(3, 3), (4, 7)])
def test_optimize_ceol4_d_bound2_python_enforces_branch_constraint(
    shape: tuple[int, int], rng: np.random.Generator,
) -> None:
    Q = optimize_ceol4_d_bound2_python(
        random_Q(rng, shape), iterations=500, learning_rate=0.05
    )
    r_max, delta_min = ceol4_d_bound_metrics(Q)

    assert delta_min <= (4.0 / 3.0) * r_max + 1e-10
    assert ceol4_d_bound2_second_branch_active(Q)


def test_optimize_ceol4_d_bound2_python_stops_at_max_bound(
    rng: np.random.Generator,
) -> None:
    Q0 = random_Q(rng, (4, 7))
    _, full_history = optimize_ceol4_d_bound2_python(
        Q0, iterations=30, learning_rate=0.05, return_history=True
    )
    assert len(full_history) >= 2
    target = full_history[1]["bound"]

    Q, history = optimize_ceol4_d_bound2_python(
        Q0,
        iterations=30,
        learning_rate=0.05,
        max_bound=target,
        return_history=True,
    )

    assert len(history) <= 2
    assert ceol4_d_bound2(Q) <= target


@pytest.mark.parametrize("shape", [(2, 3), (2, 20), (3, 7), (5, 8)])
def test_ceol4_d_bound2_random_rectangular_Q(
    shape: tuple[int, int], rng: np.random.Generator
) -> None:
    """The improved bound also holds for rectangular row-orthonormal Q."""
    n, m = shape
    for _ in range(10):
        L, Q = random_LQ(rng, n, n, m)

        expected_loss = slow_conditional_expected_value_signs(
            L, Q, np.zeros(n)
        )
        bound1 = ceol4_d_bound1(Q)
        bound2 = ceol4_d_bound2(Q)

        assert expected_loss <= bound2 + 1e-12
        assert bound2 <= bound1 + 1e-12


def test_minimize_quartic_on_interval_python()->None:
    # p(x) = (x - 0.3)^4 + 2
    x_min, p_min = minimize_quartic_on_interval(
        [1.0, -1.2, 0.54, -0.108, 2.0081]
    )
    assert x_min == pytest.approx(0.3, abs=1E-5)
    assert p_min == pytest.approx(2.0, abs=1E-5)


    # p(x) = -x^4, Minimum at one of the borders
    x_min, p_min = minimize_quartic_on_interval(
        [-1.0, 0.0, 0.0, 0.0, 0.0], limits=(-2,2)
    )
    assert np.abs(x_min) == pytest.approx(2.0, abs=1E-5)
    assert p_min == pytest.approx(-16.0, abs=1E-5)


    # Auch eine quadratische Funktion ist erlaubt:
    # p(x) = (x + 0.4)^2
    x_min, p_min = minimize_quartic_on_interval_python(
        [1.0, 0.8, 0.16]
    )
    assert x_min == pytest.approx(-0.4, abs=1E-5)
    assert p_min == pytest.approx(0.0, abs=1E-5)
    # ungefähr: -0.4, 0.0

def minimize_quartic_on_interval_grid(coefficients, limits=(-1.0, 1.0), num=100001, *, only_limits=False):
    coefficients = np.asarray(coefficients, dtype=float)
    if coefficients.size == 0:
        raise ValueError("coefficients must not be empty.")
    if coefficients.size > 5:
        raise ValueError("Polynomial degree must not exceed four.")
    if not np.all(np.isfinite(coefficients)):
        raise ValueError("coefficients must contain only finite values.")

    lo, hi = limits
    if lo > hi: lo, hi = hi, lo

    if only_limits: x = np.array([lo, hi])
    else:x = np.linspace(lo, hi, num)

    values = np.polyval(coefficients, x)
    i = np.argmin(values)

    return x[i], values[i]

def test_minimize_quartic_on_interval_cpp(rng)->None:
    """Test the cpp implementation by comparing to the python
    implementation on a somewhat random polynomial."""
    for ctr in range(5000):
        a = rng.normal (size=(5,))
        n = rng.integers(0,6)
        a[:n] = 0 # systematically test cases where the order of the polynomial reduces
        limits = (-1.2, 1.2)

        #It's very slow but you can use it for debugging
        #x_min_strue, p_min_strue = minimize_quartic_on_interval_grid (a, limits=limits)
        x_min_true, p_min_true = minimize_quartic_on_interval_python (a, limits=limits)
        x_min, p_min = minimize_quartic_on_interval (a, limits=limits)

        assert limits[0]<=x_min_true<=limits[1]
        assert limits[0]<=x_min<=limits[1]
        assert p_min_true == pytest.approx(np.polyval (a, x_min_true), abs=1E-3)
        assert p_min == pytest.approx(np.polyval (a, x_min), abs=1E-3)
        if (ctr==11):
            print(f"a={a}")
            #print(f"GIRD: {x_min_strue}, {p_min_strue}")
            print(f"PYTHON: {x_min_true}, {p_min_true}")
            print(f"CPP: {x_min}, {p_min}")
        #assert x_min == pytest.approx (x_min_true, abs=1E-3) #If mins are ambiguous
        assert p_min == pytest.approx (p_min_true, abs=1E-3)


@pytest.mark.parametrize("shape", [(2, 3), (7, 3), (3, 7), (10, 10)])
def test_integrate_trailing_column_householder_python (
    shape:tuple[int, int], rng:np.random.Generator
)->None:
    """Integrates a random column q with random index (but 0 in leading entries) 
    into a random L and checks that L_i=q and L@L.T did not change."""
    n, m = shape
    for ctr in range(10):
        L = rng.normal(size=(n, m))
        j = rng.integers (m)

        q = rng.normal(size=m-j)
        q /= np.linalg.norm(q)

        L_new = integrate_trailing_column_householder_python (L, q, j)

        np.testing.assert_allclose (L_new[:, j], L @ np.concat([np.zeros(j), q]), atol=1e-10, rtol=0.0)
        np.testing.assert_allclose (L@L.T, L_new@L_new.T, atol=1e-10, rtol=0.0)

@pytest.mark.parametrize("shape", [(2, 5), (7, 3), (10, 10)])
def test_integrate_trailing_column_householder_cpp (
    shape:tuple[int, int], rng:np.random.Generator
)->None:
    """The native Householder update supports arbitrary L shapes."""
    n, m = shape
    for _ in range(10):
        L = rng.normal(size=shape)
        j = rng.integers(m)

        q = rng.normal(size=m-j)
        q /= np.linalg.norm(q)

        L_new = integrate_trailing_column_householder (L, q, j)

        np.testing.assert_allclose (L_new[:, j], L @ np.concat([np.zeros(j), q]), atol=1e-10, rtol=0.0)
        np.testing.assert_allclose (L@L.T, L_new@L_new.T, atol=1e-10, rtol=0.0)

def sample_conditioned_current_column(
    Q: Matrix, i: int,  j: int,
    rng: np.random.Generator, *,
    atol: float = 1e-10,
) -> Vector:
    """
    Sample column j from its conditional Haar distribution.
    This means the random sampledcolumn has unit length and
    is orthogonal to previous columns.

    Columns 0,...,j-1 and entries 0,...,i of column j are fixed.
    """
    n = Q.shape[0]
    if Q.shape != (n, n):
        raise ValueError("Q must be square")

    if not -1 <= i < n:
        raise IndexError("i must satisfy -1 <= i < n")

    if not 0 <= j < n:
        raise IndexError("j must satisfy 0 <= j < n")

    prefix_length = i + 1

    # Constraints C q = b:
    #
    # 1. q is orthogonal to columns 0,...,j-1
    # 2. q[:prefix_length] equals the fixed prefix
    C = np.zeros((j + prefix_length, n), dtype=np.float64)
    b = np.zeros(j + prefix_length, dtype=np.float64)

    if j > 0:
        C[:j] = Q[:, :j].T

    if prefix_length > 0:
        C[j:, :prefix_length] = np.eye(prefix_length)
        b[j:] = Q[:prefix_length, j]

    if C.shape[0] == 0:
        q0 = np.zeros(n)
        null_basis = np.eye(n)

    else:
        U, singular_values, Vh = np.linalg.svd(
            C,
            full_matrices=True,
        )

        tolerance = atol * max(C.shape) * singular_values[0]
        rank = int(np.sum(singular_values > tolerance))

        U_rank = U[:, :rank]
        V_rank = Vh[:rank].T

        # Minimum-norm solution of C q = b.
        q0 = V_rank @ ((U_rank.T @ b) / singular_values[:rank])
        null_basis = Vh[rank:].T

        if not np.allclose(C @ q0, b, atol=1e-10, rtol=1e-10):
            raise ValueError("The fixed constraints are inconsistent")

    radius_squared = 1.0 - float(q0 @ q0)

    if radius_squared < -atol:
        raise ValueError("The fixed entries cannot belong to a unit vector")

    radius_squared = max(radius_squared, 0.0)
    free_dimension = null_basis.shape[1]

    if radius_squared <= atol: return q0

    if free_dimension == 0:
        raise ValueError("Positive residual radius but no free direction")

    # Uniform point on S^(free_dimension-1).
    z = rng.normal(size=free_dimension)
    z /= np.linalg.norm(z)

    q = q0 + np.sqrt(radius_squared) * (null_basis @ z)

    # Only correct tiny roundoff errors.
    q /= np.linalg.norm(q)
    return q

@pytest.mark.slow
def test_sample_conditioned_current_column (rng:np.random.Generator)->None:
    """Tests that a conditionally sampled column has unit length
     and is orthogonal to the previous columns and the monte carlo
     statistics of its coordinate moments matches that of
     the originally drawn column of Q (over random Q)."""
    n = 17
    i = 4
    j = 7
    n_samples = 2000
    n_tests = 4
    samples = []
    for ctr in range(n_samples):
        Q = random_Q (rng, n)
        q = sample_conditioned_current_column (Q, i, j, rng)
        samples.append ((Q, q))
        assert np.linalg.norm(q) == pytest.approx (1, abs=1E-10)
        qq = (Q.T@q)[:j]
        np.testing.assert_allclose (qq, 0, atol=1E-10, rtol=0.0)
        np.testing.assert_allclose (Q[:i+1,j], q[:i+1], atol=1E-10, rtol=0.0)

    # Test moments 1-4 along random directions a
    # and assert that they are the same in the original random Q column
    # and the conditionally sampled q
    for m in range(1,5): # moment 1-4
        for ctr in range(n_tests):
            a = rng.normal(size=n) # projection directions
            a/= np.linalg.norm(a)
            vals = np.asarray([((a@Q[:,j])**m, (a@q)**m) for (Q,q) in samples], dtype=np.float64)
            print(f'order {m} {np.mean(vals, axis=0)}')
            print(a)
            assert_mean_plausible (vals[:,1]-vals[:,0], 0, 'equal')
        

def sample_conditioned_orthogonal_matrix(
    Q: Matrix, i: int, j: int,
    rng: np.random.Generator,
) -> Matrix:
    """
    Draw an orthogonal matrix from Haar measure conditioned on

        columns < j being fixed, and
        entries <= i of column j being fixed.

    Q supplies the fixed values only.
    """
    n = Q.shape[0]
    if Q.shape != (n, n):
        raise ValueError("Q must be square")

    result = np.empty_like(Q)

    if j > 0:
        result[:, :j] = Q[:, :j]

    current = sample_conditioned_current_column(
        Q, i, j, rng,
    )
    result[:, j] = current
    fixed = result[:, :j + 1]
    future_dimension = n - j - 1

    if future_dimension == 0: return result

    # Gaussian columns, projected onto the orthogonal complement.
    G = rng.normal(size=(n, future_dimension))
    G -= fixed @ (fixed.T @ G)
    future, R = np.linalg.qr(G, mode="reduced")

    # Optional sign convention correction. It removes the deterministic
    # sign convention of QR and gives the usual Gaussian-QR Haar sample.
    signs = np.sign(np.diag(R))
    signs[signs == 0.0] = 1.0
    future *= signs

    result[:, j + 1:] = future
    return result

@pytest.mark.slow
def test_sample_conditioned_orthogonal_matrix(rng: np.random.Generator) -> None:
    """
    Test that a conditionally sampled orthogonal matrix

    - is orthogonal,
    - preserves the conditioned entries,
    - has projected moments matching those of the original random
      orthogonal matrix, averaged over random original matrices.
    """
    n = 17
    i = 4
    j = 7

    n_samples = 2_000
    n_tests = 20

    original_matrices = np.empty((n_samples, n, n))
    sampled_matrices = np.empty((n_samples, n, n))

    identity = np.eye(n)

    for sample in range(n_samples):
        Q = random_Q(rng, n)
        Q_new = sample_conditioned_orthogonal_matrix(Q, i, j, rng)

        # Orthogonality
        np.testing.assert_allclose(Q_new.T @ Q_new, identity, atol=1e-10, rtol=0.0)
        # Previous columns remain fixed.
        np.testing.assert_allclose(Q_new[:, :j], Q[:, :j],  atol=1e-10, rtol=0.0)
        # Fixed prefix of current column remains fixed.
        np.testing.assert_allclose(Q_new[:i + 1, j], Q[:i + 1, j], atol=1e-10, rtol=0.0)

        original_matrices[sample] = Q
        sampled_matrices[sample] = Q_new

    # Compare moments of random bilinear projections a.T @ Q @ b.
    for test in range(n_tests):
        a = rng.normal(size=n)
        a /= np.linalg.norm(a)

        b = rng.normal(size=n)
        b /= np.linalg.norm(b)

        original_values = np.einsum("i,sij,j->s", a, original_matrices, b)
        sampled_values = np.einsum("i,sij,j->s", a, sampled_matrices, b)

        for moment in range(1, 5):
            differences = sampled_values**moment - original_values**moment
            assert_mean_plausible(differences, 0.0, "equal", alpha=1e-4)

def sample_conditional_l4_loss_Q (rng:np.random.Generator, L:Matrix, Q:Matrix, 
                                  i:int, j:int, row_weights:Weights=None):
    QNew = sample_conditioned_orthogonal_matrix (Q, i, j, rng)
    return weighted_l4_loss_python (L@QNew, row_weights)

def test_conditional_expected_loss_Q_1 (rng:np.random.Generator) -> None:
    """Checks that for j=n as the whole matrix is fixed, the result is
    just the weighted loss."""
    n_tests = 12
    for ctr in range(n_tests):
        m = rng.integers(6,15)
        n = rng.integers(1, 20)
        L = rng.normal(size=(n, m))
        Q = random_Q(rng, m)
        loss = conditional_expected_loss_Q (L, Q, 0, m)
        loss_true = weighted_l4_loss_python (L@Q)
        assert loss == pytest.approx (loss_true, abs=1E-10)

@pytest.mark.slow
def test_conditional_expected_loss_Q_2 (rng:np.random.Generator) -> None:
    """Takes random L, Q and current element (i,j) of Q and 
    checks that the conditional expected value computed is
    compatible with the samples obtained from 
    sample_conditioned_orthogonal_matrix."""
    n_samples = 2_000
    n_tests = 4
    for ctr in range(n_tests):
        m = i = j = 0
        try:
            m = rng.integers(6,15)
            i = rng.integers(-1, m)
            j = rng.integers(0, m)

            n = rng.integers(1, 20)
            L = rng.normal(size=(n, m))
            Q = random_Q(rng, m)
            #print(n, i, j)
            econd = conditional_expected_loss_Q (L, Q, i, j)
            samples = [sample_conditional_l4_loss_Q (rng, L, Q, i, j) 
                    for ctr in range(n_samples)]
            assert_mean_plausible (samples, econd, 'equal', alpha=1e-3)
        except AssertionError as error:
            raise AssertionError(
                f"Conditional expectation test failed for m = {m} i = {i}\n j = {j}\n"
            ) from error

def test_conditional_quartic_coefficients_Q_python (rng:np.random.Generator) -> None:
    """Tests conditional_quartic_coefficients_Q by obtaining the 4th order
    polynomial, applying it to the actual coefficient at i,j and comparing
    the result to conditional_expected_loss_Q."""
    n_tests = 100
    for ctr in range(n_tests):
        m = rng.integers(6,15)
        j = rng.integers(0, m-2)
        i = rng.integers(j, m-2)

        n = rng.integers(1, 20)
        L = rng.normal(size=(n, m))
        row_weights = make_row_weights (L)
        # A random Q with leading j columns/rows unity
        Q = np.eye(m)
        Q[j:,j:] = random_Q (rng, m-j)

        # Compute the rather technical parameters for conditional_quartic...
        assert i>=j

        # Current column:
        # q = [x, y, z]
        # x is fixed, y = Q[i, j] is variable, and z is random.
        x = Q[j:i, j]

        # Matching split of every active row of L:
        #
        # ell_r = [a_r, b_r, c_r].
        a = L[:, j:i]
        alpha = a @ x
        beta = L[:, i]
        c = L[:, i + 1:]

        # gamma_r = ||c_r||^2.
        gamma = np.sum(c * c, axis=1)

        # sigma_r = ||L[r, j:]||^2
        #
        # Computing it directly avoids separately summing
        # ||a_r||^2 + beta_r^2 + gamma_r.
        sigma = np.sum(L[:, j:] ** 2, axis=1)

        # rho^2(y) = upsilon - y^2.
        upsilon = float(1.0 - np.dot(x, x))

        # Dimension of the random tail z = Q[i+1:, j].
        tail_dimension = m - i - 1

        # Since Q is identity on rows/columns before j, the already fixed
        # columns of LQ equal L[:, :j].
        phi = np.sum(L[:, :j] ** 4, axis=1)

        econd_poly = conditional_quartic_coefficients_Q_python (
            phi, alpha, beta, gamma, sigma, upsilon, i, j, L.shape[1], row_weights
        )
        assert abs(Q[i,j]) <= math.sqrt(upsilon)
        econd = np.polyval (econd_poly, Q[i,j])

        econd_true = conditional_expected_loss_Q (L, Q, i, j)
        assert econd == pytest.approx (econd_true, abs = 1E-6)

def test_conditional_quartic_coefficients_Q_cpp(rng):
    """Tests the cpp implementation by comparing to the
    python implementation."""
    n_tests = 100
    for ctr in range(n_tests):
        m = rng.integers(6,15)
        n = rng.integers(1, 20)
        j = rng.integers(0, m-2)
        i = rng.integers(j, m-2)

        L = rng.normal(size=(n, m))
        row_weights = make_row_weights (L)
        # A random Q with leading j columns/rows unity
        Q = np.eye(m)
        Q[j:,j:] = random_Q (rng, m-j)

        # Compute the rather technical parameters for conditional_quartic...
        assert i>=j

        # Current column:
        # q = [x, y, z]
        # x is fixed, y = Q[i, j] is variable, and z is random.
        x = Q[j:i, j]

        # Matching split of every active row of L:
        #
        # ell_r = [a_r, b_r, c_r].
        a = L[:, j:i]
        alpha = a @ x
        beta = L[:, i]
        c = L[:, i + 1:]

        # gamma_r = ||c_r||^2.
        gamma = np.sum(c * c, axis=1)

        # sigma_r = ||L[r, j:]||^2
        #
        # Computing it directly avoids separately summing
        # ||a_r||^2 + beta_r^2 + gamma_r.
        sigma = np.sum(L[:, j:] ** 2, axis=1)

        # rho^2(y) = upsilon - y^2.
        upsilon = float(1.0 - np.dot(x, x))

        # Dimension of the random tail z = Q[i+1:, j].
        tail_dimension = m - i - 1

        # Since Q is identity on rows/columns before j, the already fixed
        # columns of LQ equal L[:, :j].
        phi = np.sum(L[:, :j] ** 4, axis=1)

        k_python = conditional_quartic_coefficients_Q_python (
            phi, alpha, beta, gamma, sigma,
            upsilon, i, j, L.shape[1], row_weights
        )

        k_cpp = conditional_quartic_coefficients_Q(
            phi, alpha, beta, gamma, sigma,
            upsilon, i, j, L.shape[1], row_weights
        )

        np.testing.assert_allclose(k_cpp, k_python, rtol=1e-12, atol=1e-12)

def test_optimize_conditional_Q_column_python (rng:np.random.Generator)->None:
    """Tests that for random L, random j and Q identity matrix,
    optimize_conditional_Q_column provides a unit length column
    with 0s before row j that does not increase the expected
    L4 loss. Q is an indentity matrix here, because at the beginning
    of optimizing a column, the previous housholder transforms
    have made Q identity in the relevant parts. Since it is
    constant it is not actually passed."""
    n_tests = 200
    for ctr in range(n_tests):
        m = rng.integers (3, 20)
        if ctr>5:
            n = rng.integers(1, 25)
            L = rng.normal(size=(n, m))
        else:
            L = np.diag(rng.uniform(0.5, 2.0, size=m))
        row_weights = make_row_weights (L)
        j = rng.integers(0, m)
        Q = np.eye(m)

        #lossprior = weighted_l4_loss_python (L, row_weights)
        econdprior = conditional_expected_loss_Q (L, Q, -1, j, row_weights)
        if j==0:
            n = L.shape[0]
            assert econdprior <= 3*m/(m+2) + 1e-12

        q, _, history = optimize_conditional_Q_column_python (L, j, row_weights, return_history=True)
        history = np.concatenate ([[econdprior], history], dtype = np.float64)
        print(q)
        print(history)

        assert np.linalg.norm (q) == pytest.approx(1, abs = 1E-6), "q is not unit vector"
        assert np.all(history[:-1]+1E-10 >= history[1:]), "conditional expectation never becomes worse after a decision"

        Q[j:,j] = q
        econdposterior = conditional_expected_loss_Q (L, Q, -1, j+1, row_weights)
        assert econdposterior == pytest.approx (history[-1], abs=1E-8), "reported final conditional expectation wrong"

def test_optimize_conditional_Q_column_cpp (rng:np.random.Generator)->None:
    """Tests the cpp implementation by comparing that it is as good as the python implementation.
    Unfortunately we cannot compare it's equal because multiple equal minima."""
    n_tests = 200
    for ctr in range(n_tests):
        m = rng.integers (3, 20)
        if ctr>5:
            n = rng.integers(1, 25)
            L = rng.normal(size=(n, m))
        else:
            L = np.diag(rng.uniform(0.5, 2.0, size=m))
        row_weights = make_row_weights (L)
        j = rng.integers(0, m)
        Q = np.eye(m)

        q_true, _ = optimize_conditional_Q_column_python (L, j, row_weights)
        q, _ = optimize_conditional_Q_column (L, j, row_weights)

        Q[j:,j] = q_true
        econdposterior_true = conditional_expected_loss_Q (L, Q, -1, j+1, row_weights)

        Q[j:,j] = q
        econdposterior = conditional_expected_loss_Q (L, Q, -1, j+1, row_weights)

        assert econdposterior == pytest.approx(econdposterior_true, abs=1E-6)

@pytest.mark.slow
def test_cel4o_Q_python (rng:np.random.Generator):
    """Tests that cel4o_Q_python returns a matrix L that has
    normalized l4-loss <= 3*m/(m+2) and represents the right covariance."""
    n_tests = 150
    for ctr in range(n_tests):
        m = rng.integers (3, 20)
        if ctr>5:
            n = rng.integers(1, 25)
            L = rng.normal(size=(n, m))
        else:
            L = np.diag(rng.uniform(0.5, 2.0, size=m))
        Lnew  = cel4o_Q_python (L)
        loss4 = weighted_l4_loss_python (Lnew)
        assert loss4 == pytest.approx(weighted_l4_loss(Lnew))
        n = L.shape[0]
        assert loss4 <= 3*m/(m+2) + 1e-10
        np.testing.assert_allclose (Lnew@Lnew.T, L@L.T, atol=1E-8)

@pytest.mark.parametrize("shape", [(1, 4), (3, 7), (9, 4)])
def test_cel4o_Q_with_history_rectangular(
    shape:tuple[int, int], rng:np.random.Generator
) -> None:
    n, m = shape
    L = rng.normal(size=shape)
    row_weights = make_row_weights(L)

    Lnew, econd, history = cel4o_Q_with_history(L, row_weights)
    expected = cel4o_Q_python(L, row_weights)

    assert Lnew.shape == shape
    np.testing.assert_allclose(Lnew, expected, atol=1e-12, rtol=0.0)
    np.testing.assert_allclose(Lnew @ Lnew.T, L @ L.T, atol=1e-10, rtol=0.0)
    assert weighted_l4_loss_python(Lnew, row_weights) <= 3*m/(m+2) + 1e-10
    assert econd == pytest.approx(weighted_l4_loss_python(Lnew, row_weights))
    assert history[0]["Q"].shape == (m, m)
    assert history[-1]["L"].shape == shape
    assert history[-1]["l4loss"] == pytest.approx(econd)

@pytest.mark.slow
def test_cel4o_Q_cpp (rng:np.random.Generator):
    """Tests that cel4o_Q returns a matrix L that has
    normalized l4-loss <= 3*m/(m+2) and represents the right covariance."""
    n_tests = 150
    for ctr in range(n_tests):
        m = rng.integers (3, 20)
        if ctr>5:
            n = rng.integers(1, 25)
            L = rng.normal(size=(n, m))
        else:
            L = np.diag(rng.uniform(0.5, 2.0, size=m))
        Lnew = cel4o_Q (L)
        loss4 = weighted_l4_loss_python (Lnew)
        assert loss4 == pytest.approx(weighted_l4_loss(Lnew))
        n = L.shape[0]
        assert loss4 <= 3*m/(m+2) + 1e-10
        np.testing.assert_allclose (Lnew@Lnew.T, L@L.T, atol=1E-8)

@pytest.mark.parametrize("shape", [(1, 4), (3, 7), (9, 4)])
def test_cel4o_Q_cpp_rectangular_reference(
    shape:tuple[int, int], rng:np.random.Generator
) -> None:
    """The native Q variant agrees with Python for rectangular L."""
    n, m = shape
    L = rng.normal(size=shape)
    row_weights = make_row_weights(L)

    actual = cel4o_Q(L, row_weights)
    expected = cel4o_Q_python(L, row_weights)

    assert actual.shape == shape
    np.testing.assert_allclose(actual @ actual.T, L @ L.T, atol=1e-9, rtol=0.0)
    assert weighted_l4_loss(actual, row_weights) == pytest.approx(
        weighted_l4_loss(expected, row_weights), abs=1e-8
    )
    assert weighted_l4_loss(actual, row_weights) <= 3*m/(m+2) + 1e-10

def test_compact_matrix_sqrt_Q (rng:np.random.Generator):
    """Tests that compact_matrix_sqrt fulfills the documentation.
    This is actually pretty redundant to test_optimize_conditional_Q"""
    n_tests = 100
    for ctr in range(n_tests):
        n = rng.integers (3, 20)
        Sigma = random_SPSD (rng, n)
        L = compact_matrix_sqrt_Q (Sigma)
        l4_loss = weighted_l4_loss (L)
        assert l4_loss == pytest.approx(weighted_l4_loss_python (L))
        assert l4_loss <= 3 - 2/(n+2)
        np.testing.assert_allclose (L@L.T, Sigma, atol=1E-8)

def test_dct_matrix_cpp():
    for n in [1, 2, 5, 20]:
        expected = dct_matrix_python(n)
        actual = dct_matrix(n)

        np.testing.assert_allclose(actual, expected, rtol=1e-14, atol=1e-14)

    for size in [(1, 0), (2, 1), (5, 3), (20, 7)]:
        expected = dct_matrix_python(size)
        actual = dct_matrix(size)

        assert actual.shape == size
        np.testing.assert_allclose(actual, expected, rtol=1e-14, atol=1e-14)

    with pytest.raises(ValueError, match="landscape"):
        dct_matrix((3, 4))

def test_dct_matrix_orthogonal():
    m, n = 20, 7
    Q = dct_matrix((m, n), make_orthonormal=True)

    np.testing.assert_allclose(Q.T @ Q, np.eye(n), rtol=1e-13, atol=1e-13)
    assert np.all(np.sum(Q**4, axis=0) <= 3/(2*m) + 1e-14)
    assert np.all(np.abs(Q)<=np.sqrt(2/m)+1E-10)
