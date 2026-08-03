from __future__ import annotations

from typing import overload, Literal, List, Dict
import numpy as np

from .wrapper import *


## Coded in parts by ChatGPT 5.5

## Auxiliary functions
def random_SPSD (rng:np.random.Generator, n:int)->Matrix:
    """
    Returns a random n*n symmetric positive semidefinite
    matrix with eigenvalues on the order of 1.
    """
    A = rng.normal (size=(n,n))/np.sqrt(n)
    return A@A.T

def random_L (rng:np.random.Generator, n:int)->Matrix:
    """
    Returns a random lower-triangular matrix with
    singularvalues on average [0.65568/n...2].
    """
    A = rng.normal (size=(n,n))/np.sqrt(n)
    _, R = np.linalg.qr (A)
    return R.T

def random_Q (rng:np.random.Generator, size:int|tuple[int,int])->Matrix:
    """Returns a random size orthogonal matrix. If size=n a square
    (n,n) matrix is returned."""
    if not isinstance(size, tuple):
        n = int(size)
        size = (n, n)
    if size[0]>=size[1]:
        A = rng.normal(size=size)
        Q, R = np.linalg.qr(A, mode="reduced")
        # Haar-Verteilung korrigieren
        Q *= np.sign(np.diag(R))
        return Q
    else:
        A = rng.normal(size=(size[1], size[0]))
        Q, R = np.linalg.qr(A, mode="reduced")
        # Haar-Verteilung korrigieren
        Q *= np.sign(np.diag(R))
        return Q.T


def random_LQ (
    rng:np.random.Generator, n:int, p:int, m:int
) -> tuple[Matrix, Matrix]:
    """Return compatible random matrices L and Q.

    L has shape (n, p). Q has shape (p, m) and orthonormal rows, so
    Q@Q.T=I and (L*signs)@Q preserves L@L.T. Consequently m must be at
    least p.
    """
    if n <= 0 or p <= 0 or m <= 0:
        raise ValueError("n, p and m must be positive")
    if m < p:
        raise ValueError("m must be >= p")

    L = rng.normal(size=(n, p)) / np.sqrt(p)
    Q = random_Q(rng, (p, m))
    return L, Q

## General code

def make_row_weights (L:Matrix, row_weights:Weights = None)->Matrix:
    """Computes row_weights from L if row_weights is None and
    otherwise returns row_weights."""
    rw:Vector
    if row_weights is None:
        rw = (np.sum(L**2, axis=-1))**2
        rw = np.divide(1.0, rw, out=np.zeros_like(rw), where=rw!=0)
    elif isinstance (row_weights, float):
        rw = np.ones(L.shape[0:], dtype=np.float64)
    else:
        rw = np.asarray(row_weights, dtype=np.float64)
        if rw.shape != (L.shape[0],):
            raise ValueError("row_weights must have shape (L.shape[0],)")
    return rw

def weighted_l4_loss_python(L:Matrix, row_weights:Weights=None)->float:
    """
    Computes m/n * sum_i row_weights[i] * sum_j L[i,j]**4 for an
    (n, m) matrix.
    """
    row_weights = make_row_weights (L, row_weights)

    return L.shape[1] / L.shape[0] * np.sum(row_weights[:, None] * L**4)

def dct_matrix_python (size:int|tuple[int,int], make_orthonormal=False, dtype=np.float64) -> Matrix:
    '''Returns a discrete cosine transform (dct) matrix of dimension ``size``.
    The rows of the DCT matrix correspond to sample-points, the cols to frequency.
    '''
    if isinstance(size, tuple):
        m, n = size
        N = max(m, n)
    else:
        N = n = m = int(size)
    I = np.arange(m, dtype=dtype)
    J = np.arange(n, dtype=dtype)
    I, J = np.meshgrid(I, J, indexing='ij')
    dct = np.cos(np.pi/N*(I+0.5)*J)
    dct[:,1:] *= np.sqrt(2)
    if make_orthonormal:
        dct /= np.sqrt(N)
    return dct

def symmetric_matrix_sqrt(A:Matrix)->Matrix:
    # eigh leverages symmetry and returns eigenvalues (w) and eigenvectors (v)
    w, v = np.linalg.eigh(A)
    
    # Ensure eigenvalues are non-negative to avoid complex results
    w = np.clip(w, 0, None)
    
    # Reconstruct: V * diag(sqrt(w)) * V.T
    return (v * np.sqrt(w)) @ v.T

def adversarial (Q:Matrix)->Matrix:
    """Returns a lower triangular matrix L that
    has a large weighted_l4_loss when multiplied with Q.
    The technique is to go row by row (i), choose
    the column of Q that has most energy in its first
    i+1 entries, normalize that to unit length and
    take it as the i-th row of the adversarial. Based on
    this it perform 5 additional iterations.

    Q may be an n*m landscape matrix (n<=m), for example a transposed
    portrait DCT matrix. The result is always an n*n matrix, so L@Q has
    the same shape as Q. For square Q, this adversarial is guaranteed to
    have n/3+5/6 as L4-loss. For landscape Q, the general lower bound is
    n**3/(3*m**2)."""
    if Q.ndim != 2:
        raise ValueError ("Q must be a matrix")
    n, m = Q.shape
    if n>m:
        raise ValueError ("portrait Q matrix is not allowed")

    L = np.zeros((n, n), dtype=Q.dtype)
    column_norm_sq = np.zeros(m, dtype=Q.dtype)
    for i in range(n):
        column_norm_sq += Q[i] ** 2
        j = np.argmax(column_norm_sq)
        Qi = Q[:i+1, :]
        v = Qi[:, j] / np.sqrt(column_norm_sq[j])
        for _ in range(8):
            g = Qi @ ((Qi.T @ v)**3)
            v = g / np.linalg.norm(g)
        L[i, :i+1] = v
    return L 


## Code for Jacobi rotation based optimization

def jacobi_rotate_two_columns_l4(
        U:Matrix, 
        k:int, l:int, 
        row_weights:Weights=None, 
        inplace:bool=True)->tuple:
    """
    Optimally Givens-rotates columns k,l of U to minimize the weighted 4th moment
        sum_i row_weights[i] * (U[i,k]**4 + U[i,l]**4)
    for this column pair.

    Rotation convention:
        a_new =  a*cos(alpha) + b*sin(alpha)
        b_new = -a*sin(alpha) + b*cos(alpha)

    Returns:
        alpha, gain
    """
    if k == l:
        return 0.0, 0.0

    M = U if inplace else U.copy()

    a = M[:, k].copy()
    b = M[:, l].copy()

    w = make_row_weights (M, row_weights)

    a2 = a * a
    b2 = b * b

    # F(alpha) = C + A*cos(4 alpha) + B*sin(4 alpha)
    A = np.sum(w * (a2 * a2 + b2 * b2 - 6.0 * a2 * b2)) / 4.0
    B = np.sum(w * a * b * (a2 - b2))

    R = np.hypot(A, B)
    if R != 0.0:

        # Minimum of A cos(4a) + B sin(4a)
        alpha = (np.arctan2(B, A) + np.pi) / 4.0

        # Same solution modulo pi/2; choose small representative
        alpha = (alpha + np.pi / 4.0) % (np.pi / 2.0) - np.pi / 4.0

        c = np.cos(alpha)
        s = np.sin(alpha)

        M[:, k] = a * c + b * s
        M[:, l] = -a * s + b * c

        # F(0) = C + A, F_min = C - R
        gain = M.shape[1] / M.shape[0] * (A + R)
    else:
        alpha = gain = 0.0

    return (alpha, gain) if inplace else (alpha, gain, M)

def cyclic_distance_sweep_l4 (
        U:Matrix, distance:int, row_weights:Weights, 
        inplace:bool=True)->float | tuple[float, Matrix]:
    """
    One sweep over column pairs (k, (k + distance) % m).
    Applies the optimal weighted L4 Jacobi rotation for each pair.

    Returns:
        total_gain, updated U if not inplace
    """
    M = U if inplace else U.copy()
    m = M.shape[1]

    row_weights = make_row_weights (M, row_weights)

    distance %= m
    if distance == 0:
        return 0.0 if inplace else (0.0, M)

    total_gain = 0.0
    num_rotations = 0

    for k in range(m):
        l = (k + distance) % m

        alpha, gain = jacobi_rotate_two_columns_l4(
            M, k, l, row_weights=row_weights, inplace=True
        )

        total_gain += gain

    return total_gain if inplace else (total_gain, M)

@overload
def optimize_l4(
    U: Matrix, row_weights: Weights = None,
    *,
    inplace: Literal[True] = True, from_sigma: bool = False, iterations: int = 1,  verbose: bool = True,
) -> tuple[float, float]:
    ...


@overload
def optimize_l4(
    U: Matrix, row_weights: Weights = None,
    *,
    inplace: Literal[False], from_sigma: bool = False, iterations: int = 1, verbose: bool = True,
) -> tuple[float, float, Matrix]:
    ...

def optimize_l4 (
        U:Matrix, row_weights:Weights =None, 
        inplace:bool=True, from_sigma:bool=False, iterations:int=1, verbose:bool=True
        )-> tuple[float, float] | tuple[float, float, Matrix]:
    """
    Reduces the variance-weighted L4 loss of U while keeping
    U*U^T fixed. For that it sweeps over columns of U and performs
    a L4 minimizing Jacobi-rotation between the column and the
    column in a given distance. This distance doubles with every
    sweep. This policy is similar to a butterfly pattern, more 
    expensive but works with non-power-of-two dimensions.

    If row_weights is None is is computed as (sum_j U[i,j]**2)**2 
    corresponding to the squared diagonal of U*U^T.

    If from_sigma is True, the incoming U ist used as Sigma and
    Cholesky decomposed for an initial U which is then optimized.
    from_sigma and inplace cannot be used together.

    If iterations>1 this process is done iterations times.
    
    Returns:
        final_loss, total_gain, final U if not inplace
    """
    m = U.shape[1]
    if from_sigma:
        if inplace: 
            raise ValueError("from_sigma and inplace cannot be used together")
        M = np.linalg.cholesky(U)
    elif inplace:
        M = U
    else:
        M = U.copy()

    row_weights = make_row_weights (M, row_weights)
    
    total_gain = 0.0
    final_loss = weighted_l4_loss_python (M, row_weights)
    if verbose: print(final_loss)
    for _ in range(iterations):
        distance = 1
        while distance < m:
            gain = cyclic_distance_sweep_l4(M, distance, row_weights)
            assert isinstance(gain, float)
            total_gain += gain
            final_loss = weighted_l4_loss_python (M, row_weights)
            if verbose: print(distance, gain, final_loss)

            distance *= 2
    if inplace:
        return final_loss, total_gain
    else:
        return final_loss, total_gain, M


def row_orthonormalize_polar_python(Q: Matrix) -> Matrix:
    """Return the closest row-orthonormal matrix to Q in Frobenius norm."""
    U, _, Vh = np.linalg.svd(Q, full_matrices=False)
    return U @ Vh

def row_orthonormal_tangent_projection_python(Q: Matrix, G: Matrix) -> Matrix:
    """Project Euclidean gradient G to the tangent space of Q@Q.T=I."""
    GQt = G @ Q.T
    return G - 0.5 * (GQt + GQt.T) @ Q

## Code for the conditional_expectation driven sign-D optimization

def rademacher_l4_bound(L:Matrix, Q:Matrix, row_weights:Weights=None)->float:
    """
    Computes

        3m/n * sum_{i,j} w_i * (sum_k L[i,k]^2 * Q[k,j]^2)^2

    where w_i = 1 if row_weights is None.

    This value is an upper bound for the expected l4-loss of
    LDQ for a random sign diagonal D. 
    It is actually comparatively sharp but depends on L and Q.
    """
    L2 = L**2
    Q2 = Q**2

    M = L2 @ Q2  # M[i,j] = sum_k L[i,k]^2 Q[k,j]^2

    w = make_row_weights (L, row_weights)

    if w.shape != (L.shape[0],):
        raise ValueError("row_weights must have shape (L.shape[0],)")

    n = L.shape[0]
    m = Q.shape[1]
    return float(3.0 * m / n * np.sum(w[:, None] * M**2))


def cel4o_D_python (L: Matrix, Q:Matrix, row_weights:Weights=None):
    """Implementation from the conditional expectation l4 optimization D
    algorithm from the paper without any additional analysis information.

    L may have shape (n, p) and Q shape (p, m). The returned matrix then
    has shape (n, m).
    """
    if L.ndim != 2 or Q.ndim != 2:
        raise ValueError("L and Q must both be two-dimensional arrays.")
    n, p = L.shape
    p_q, m = Q.shape
    if p_q != p:
        raise ValueError(
            f"Incompatible shapes: L has {p} columns, "
            f"but Q has {p_q} rows."
        )

    w = make_row_weights (L, row_weights)
    alpha = np.zeros((n, m), dtype=np.result_type(L, Q))
    gamma = np.square(L)@np.square(Q)
    for k in range(p):
        beta   = np.outer(L[:,k], Q[k,:])
        alpha2 = np.square (alpha)
        beta2  = np.square(beta)
        gamma -= beta2
        delta_beta_E = 8*np.sum(w[:,None]*alpha*beta*(3*gamma + alpha2 + beta2))
        if delta_beta_E>0: alpha -= beta 
        else: alpha += beta
    return alpha


def cel4o_D_with_history (
    L: Matrix, Q: Matrix,
    row_weights:Weights =None,
) ->  tuple[Matrix, float, Vector, list[dict]]:
    """
    Version of cel4o-D that returns a full history
    for visualization.
    Parameters
    ----------
    L:
        Array with shape (n, p).
    Q:
        Array with shape (p, m), orthonormal.
    row_weights:
        a 1d vector or None. If None it is computed from L.

    Returns
    -------
    L*D*Q:
        The result as least as good as the expected value
    l4_loss:
        The obtained L4-value
    signs:
        Array of shape (n,) containing values -1 or +1.
    history:
    history is a list. Each entry is the status after a step and
    formally a dict:
         step:  consecutive counter
         step_type: 'init', 'sign', 'final'
         k: sign currently considered
         L: value of the L matrix
         Q: value of the Q matrix
         signs: vector of signs with +1/-1 and 0 for the not chosen
                ones
         LDQ: Product with all the signs already set
         econd: conditional expectation
         l4loss: l4loss of LQ
    The first entry has step_type 'init' and reflects the prior
    state. For each sign there is a 'sign' step.
    In the final end, there is a 'final step.

    This information can be used for diagnosis or visualization.
    """
    if L.ndim != 2 or Q.ndim != 2:
        raise ValueError("L and Q must both be two-dimensional arrays.")

    n, p = L.shape
    p_q, m = Q.shape

    if p_q != p:
        raise ValueError(
            f"Incompatible shapes: L has {p} columns, "
            f"but Q has {p_q} rows."
        )

    w = make_row_weights (L, row_weights)
    A = np.zeros((n, m), dtype=float)
    V = np.square(L) @ np.square(Q)
    W = np.power(L, 4) @ np.power(Q, 4)

    signs = np.zeros(p, dtype=float)
    history = []
    normalization = m / n
    econd = normalization * np.sum(w[:,None]*
                (np.power(A, 4) + 6.0 * np.square(A) * V
                + 3.0 * np.square(V) - 2.0 * W))
    history.append({
        'step':len(history), 'step_type':'init', 'k':None,
        'L':L.copy(), 'Q':Q.copy(), 'signs':signs.copy(),
        'LDQ':A.copy(), 'w':w.copy(), 'econd':econd,
        'l4loss':weighted_l4_loss_python (L@Q, w)
    })

    for k in range(p):
        # B[i,j] = L[i,k] Q[k,j]
        B = np.multiply.outer(L[:, k], Q[k, :])
        # The random remainder must contain only ell > k.
        B2 = np.square(B)
        B4 = np.square(B2)

        V -= B2
        W -= B4

        # Delta = E(+1) - E(-1)
        delta = 8.0 * np.sum(w[:,None]*
            A * B * (np.square(A) + B2 + 3.0 * V)
        )

        sign = 1 if delta <= 0.0 else -1
        signs[k] = sign
        A += sign * B

        econd = normalization * np.sum(w[:,None]*
            (np.power(A, 4) + 6.0 * np.square(A) * V
            + 3.0 * np.square(V) - 2.0 * W))

        history.append({
            'step':len(history), 'step_type':'sign', 'k':k,
            'L':L.copy(), 'Q':Q.copy(), 'signs':signs.copy(), 
            'LDQ':A.copy(), 'w':w.copy(), 'econd':econd,
            'l4loss':weighted_l4_loss_python (L@Q, w)
        })


    #At the end, A is the final result L @ diag(s) @ Q
    #U = (L*signs)@Q
    l4_loss = normalization * np.sum(w[:,None]*np.power(A, 4))
    history.append({
        'step':len(history), 'step_type':'final', 'k':p,
        'L':L.copy(), 'Q':Q.copy(), 'signs':signs.copy(), 
        'LDQ':A.copy(), 'w':w.copy(), 'econd':l4_loss,
        'l4loss':weighted_l4_loss_python (L@Q, w)
    })

    return A, l4_loss, signs, history


def compact_matrix_sqrt_D_python (
    Sigma:Matrix, row_weights:Weights=None, QDct:Matrix|None=None
) -> Matrix:
    """Python variant of compact_matrix_sqrt_D."""
    L = np.linalg.cholesky (Sigma)
    if QDct is None: QDct = dct_matrix (Sigma.shape[0], True)
    A = cel4o_D_python (L, QDct.T, row_weights)
    return A


def ceol4_d_bound1 (Q:Matrix)->float:
    """Computes the bound achieved by CEL4O-D
    for the given Q in R^{n*m} and any L. It is simply:
        
         3m max_k sum_j Q^4_kj

    This is the simple, not so sharp bound from the 
    MFI-conference paper.
    """
    m = Q.shape[1]
    return 3*m*np.max(np.sum(Q**4,axis=1))

def ceol4_d_bound_metrics(Q: Matrix) -> tuple[float, float]:
    """Return ``(r_max, delta_min)`` used by the CEL4O-D bounds.

    Here ``r_k = sum_j Q[k,j]**4`` and ``delta_min`` is the smallest
    squared distance between two distinct squared rows of Q.
    """
    Q2 = Q**2
    r = np.sum(Q2**2, axis=1)
    r_max = float(np.max(r))

    distances2 = r[:, None] + r[None, :] - 2 * (Q2 @ Q2.T)
    np.fill_diagonal(distances2, np.inf)
    delta_min = float(np.min(distances2))
    return r_max, delta_min


def ceol4_d_bound2_second_branch_active(Q: Matrix) -> bool:
    """Return whether the second branch of the bound2 maximum is active.

    The second branch is greater than or equal to the first exactly when
    ``delta_min <= 4/3 * r_max``.
    """
    r_max, delta_min = ceol4_d_bound_metrics(Q)
    return bool(delta_min <= (4.0 / 3.0) * r_max)


def ceol4_d_soft_r_max_gradient(
    Q: Matrix, *, temperature: float = 0.01, return_value: bool = False
) -> Matrix | tuple[Matrix, float]:
    """Gradient of a smooth maximum of ``r_k = sum_j Q[k,j]**4``."""
    Q = np.asarray(Q)
    if Q.ndim != 2:
        raise ValueError("Q must be a matrix")
    if temperature <= 0.0:
        raise ValueError("temperature must be positive")

    r = np.sum(Q**4, axis=1)
    scaled = r / temperature
    scaled_max = np.max(scaled)
    exponentials = np.exp(scaled - scaled_max)
    weights = exponentials / np.sum(exponentials)
    value = temperature * (scaled_max + np.log(np.sum(exponentials)))
    gradient = weights[:, None] * 4.0 * Q**3
    return (gradient, float(value)) if return_value else gradient


def ceol4_d_soft_delta_min_gradient(
    Q: Matrix, *, temperature: float = 0.01, return_value: bool = False
) -> Matrix | tuple[Matrix, float]:
    """Gradient of a smooth minimum of the squared-row distances.

    The smoothed quantity is ``-temperature * logsumexp(-delta_kl /
    temperature)`` over all unordered row pairs, where
    ``delta_kl = sum_j (Q[k,j]**2 - Q[l,j]**2)**2``.
    """
    Q = np.asarray(Q)
    if Q.ndim != 2:
        raise ValueError("Q must be a matrix")
    if Q.shape[0] < 2:
        raise ValueError("Q must have at least two rows")
    if temperature <= 0.0:
        raise ValueError("temperature must be positive")

    pair_k, pair_l = np.triu_indices(Q.shape[0], k=1)
    squared_difference = Q[pair_k]**2 - Q[pair_l]**2
    distances2 = np.sum(squared_difference**2, axis=1)
    scaled = -distances2 / temperature
    scaled_max = np.max(scaled)
    exponentials = np.exp(scaled - scaled_max)
    weights = exponentials / np.sum(exponentials)
    value = -temperature * (scaled_max + np.log(np.sum(exponentials)))

    gradient = np.zeros_like(Q)
    pair_gradient_k = 4.0 * Q[pair_k] * squared_difference
    pair_gradient_l = -4.0 * Q[pair_l] * squared_difference
    np.add.at(gradient, pair_k, weights[:, None] * pair_gradient_k)
    np.add.at(gradient, pair_l, weights[:, None] * pair_gradient_l)
    return (gradient, float(value)) if return_value else gradient


def ceol4_d_bound2_gradient(
    Q: Matrix, *, r_temperature: float = 0.01,
    delta_temperature: float = 0.01, barrier_temperature: float = 0.01,
    barrier_weight: float = 100.0,
    return_value: bool = False
) -> Matrix | tuple[Matrix, float]:
    """Gradient of the constrained, smooth CEL4O-D objective.

    The objective minimizes the smooth second branch
    ``(3 - 2/n)*r_max - 3/2*(1 - 1/n)*delta_min`` and adds a squared
    softplus barrier for violations of ``delta_min <= 4/3*r_max``.  The
    constraint fixes the branch selection, so the smooth maximum over the
    two branches is unnecessary.
    """
    Q = np.asarray(Q)
    if Q.ndim != 2:
        raise ValueError("Q must be a matrix")
    if barrier_temperature <= 0.0:
        raise ValueError("barrier_temperature must be positive")
    if barrier_weight <= 0.0:
        raise ValueError("barrier_weight must be positive")
    n, m = Q.shape

    r_gradient, r_max = ceol4_d_soft_r_max_gradient(
        Q, temperature=r_temperature, return_value=True
    )
    delta_gradient, delta_min = ceol4_d_soft_delta_min_gradient(
        Q, temperature=delta_temperature, return_value=True
    )
    violation = delta_min - (4.0 / 3.0) * r_max
    scaled_violation = violation / barrier_temperature
    barrier = barrier_temperature * np.logaddexp(0.0, scaled_violation)
    if scaled_violation >= 0.0:
        sigmoid = 1.0 / (1.0 + np.exp(-scaled_violation))
    else:
        exp_violation = np.exp(scaled_violation)
        sigmoid = exp_violation / (1.0 + exp_violation)

    violation_gradient = delta_gradient - (4.0 / 3.0) * r_gradient
    barrier_gradient_factor = (
        2.0 * barrier_weight * barrier * sigmoid / barrier_temperature
    )
    r_factor = 3.0 - 2.0 / n
    delta_factor = 1.5 * (1.0 - 1.0 / n)
    branch_value = r_factor * r_max - delta_factor * delta_min
    branch_gradient = (
        r_factor * r_gradient - delta_factor * delta_gradient
    )
    value = m * (
        branch_value + barrier_weight * barrier**2 / barrier_temperature
    )
    gradient = m * (
        branch_gradient + barrier_gradient_factor * violation_gradient
    )
    return (gradient, float(value)) if return_value else gradient


def optimize_ceol4_d_bound2_python(
    Q: Matrix, *, iterations: int = 1000, learning_rate: float = 0.05,
    r_temperature: float = 0.01, delta_temperature: float = 0.01,
    barrier_temperature: float = 0.01, barrier_weight: float = 100.0,
    max_backtracking: int = 20,
    gradient_tolerance: float = 1e-10, constraint_tolerance: float = 1e-12,
    max_bound: float | None = None,
    return_history: bool = False,
) -> Matrix | tuple[Matrix, list[dict[str, float]]]:
    """Minimize the constrained smooth CEL4O-D bound on the row Stiefel manifold.

    ``Q`` may be square or landscape-shaped, with ``Q.shape[0] <=
    Q.shape[1]``.  Every iteration projects the Euclidean gradient onto the
    tangent space of ``Q @ Q.T = I`` and retracts the trial step back to the
    manifold using the polar/SVD retraction. Backtracking ensures that the
    smooth objective does not increase. If ``max_bound`` is given, optimization
    stops as soon as ``ceol4_d_bound2(Q) <= max_bound`` and the branch constraint
    is satisfied; ``iterations`` is then only the maximum number of iterations.
    """
    Q = np.asarray(Q, dtype=np.float64)
    if Q.ndim != 2:
        raise ValueError("Q must be a matrix")
    if not np.all(np.isfinite(Q)):
        raise ValueError("Q must contain only finite values")
    n, m = Q.shape
    if n < 2:
        raise ValueError("Q must have at least two rows")
    if n > m:
        raise ValueError("Q must be landscape, i.e. Q.shape[0] <= Q.shape[1]")
    if iterations < 0:
        raise ValueError("iterations must be nonnegative")
    if learning_rate <= 0.0:
        raise ValueError("learning_rate must be positive")
    if max_backtracking < 0:
        raise ValueError("max_backtracking must be nonnegative")
    if gradient_tolerance < 0.0:
        raise ValueError("gradient_tolerance must be nonnegative")
    if constraint_tolerance < 0.0:
        raise ValueError("constraint_tolerance must be nonnegative")
    if max_bound is not None and (not np.isfinite(max_bound) or max_bound <= 0.0):
        raise ValueError("max_bound must be finite and positive")

    Q = row_orthonormalize_polar_python(Q)
    history: list[dict[str, float]] = []

    if (
        max_bound is not None
        and ceol4_d_bound2(Q) <= max_bound
        and ceol4_d_bound2_second_branch_active(Q)
    ):
        return (Q, history) if return_history else Q

    for iteration in range(iterations):
        current_r_max, current_delta_min = ceol4_d_bound_metrics(Q)
        current_violation = current_delta_min - (4.0 / 3.0) * current_r_max
        euclidean_gradient, smooth_bound = ceol4_d_bound2_gradient(
            Q,
            r_temperature=r_temperature,
            delta_temperature=delta_temperature,
            barrier_temperature=barrier_temperature,
            barrier_weight=barrier_weight,
            return_value=True,
        )
        gradient = row_orthonormal_tangent_projection_python(
            Q, euclidean_gradient
        )
        gradient_norm = float(np.linalg.norm(gradient))
        if gradient_norm <= gradient_tolerance:
            break

        step_size = learning_rate
        accepted = False
        candidate = Q
        candidate_bound = smooth_bound
        backtracking = 0
        for backtracking in range(max_backtracking + 1):
            trial = row_orthonormalize_polar_python(Q - step_size * gradient)
            _, trial_bound = ceol4_d_bound2_gradient(
                trial,
                r_temperature=r_temperature,
                delta_temperature=delta_temperature,
                barrier_temperature=barrier_temperature,
                barrier_weight=barrier_weight,
                return_value=True,
            )
            trial_r_max, trial_delta_min = ceol4_d_bound_metrics(trial)
            trial_violation = trial_delta_min - (4.0 / 3.0) * trial_r_max
            if current_violation <= constraint_tolerance:
                constraint_improves = trial_violation <= constraint_tolerance
            else:
                constraint_improves = trial_violation < current_violation
            if trial_bound <= smooth_bound and constraint_improves:
                candidate = trial
                candidate_bound = trial_bound
                accepted = True
                break
            step_size *= 0.5

        r_max, delta_min = ceol4_d_bound_metrics(candidate)
        history.append({
            "iteration": float(iteration + 1),
            "smooth_bound_before": float(smooth_bound),
            "smooth_bound": float(candidate_bound),
            "bound1": float(ceol4_d_bound1(candidate)),
            "bound": float(ceol4_d_bound2(candidate)),
            "r_max": float(r_max),
            "delta_min": float(delta_min),
            "second_branch_active": float(
                ceol4_d_bound2_second_branch_active(candidate)
            ),
            "gradient_norm": gradient_norm,
            "step_size": float(step_size if accepted else 0.0),
            "backtracking": float(backtracking),
            "orthogonality_error": float(
                np.linalg.norm(candidate @ candidate.T - np.eye(n))
            ),
            "accepted": float(accepted),
        })
        if not accepted:
            break
        Q = candidate
        if (
            max_bound is not None
            and ceol4_d_bound2(Q) <= max_bound
            and ceol4_d_bound2_second_branch_active(Q)
        ):
            break

    return (Q, history) if return_history else Q


def ceol4_d_bound2 (Q: Matrix) -> float:
    """Improved expected L4-loss bound for CEL4O-D.

        q_k   = sum_j Q[k,j]^4
        q_max = max_k r_k
        delta_min = min_{k != l} sum_j (Q[k,j]^2 - Q[l,j]^2)^2

        bound = m * max(q_max, q_max (3-2/n) - delta_min 3/2(1-1/n))    

    Q has shape (n, m) and orthonormal rows: Q @ Q.T = I_n.
    """
    n, m = Q.shape
    r_max, delta_min = ceol4_d_bound_metrics(Q)

    bound_spiky = r_max
    bound_flat = r_max * (3 - 2/n) - 1.5 * delta_min * (1 - 1/n)

    return float(m * max(bound_spiky, bound_flat))

##

## Code for the conditional expectation driven Q optimization

def minimize_quartic_on_interval_python (coefficients: ArrayLike, limits:tuple[float, float] = (-1,1), *,
    imag_tol: float = 1e-10, only_limits: bool = False
) -> tuple[float, float]:
    """
    Minimize a polynomial of degree at most four on the interval limits.

    Parameters
    ----------
    coefficients:
        Polynomial coefficients in descending order, as used by np.polyval:

            [a4, a3, a2, a1, a0]

        Leading zero coefficients are allowed, so lower-degree polynomials
        are handled as well.

    limits:
        interval in which values are considered

    limits_only:
        if true the minimum of the limits is returned and no interior
        points considered

    imag_tol:
        A root of the derivative is considered real if its imaginary part is
        no larger than

            imag_tol * max(1, abs(real_part)).

    Returns
    -------
    x_min:
        A global minimizer in limits.

    value_min:
        The corresponding minimum polynomial value.

    Raises
    ------
    ValueError:
        If the coefficient array is empty, contains non-finite values, or
        represents a polynomial of degree greater than four.
    """
    coeff = np.asarray(coefficients, dtype=np.float64)

    if coeff.ndim != 1 or coeff.size == 0:
        raise ValueError("coefficients must be a non-empty one-dimensional array")

    if coeff.size > 5:
        raise ValueError("the polynomial degree must not exceed four")

    if not np.all(np.isfinite(coeff)):
        raise ValueError("all coefficients must be finite")

    if limits[1]<limits[0]: raise ValueError (f"Interval is empty {limits}.")

    # Permit abbreviated input such as [a2, a1, a0].
    # Remove leading zeros to avoid spurious roots from np.roots.
    coeff = np.trim_zeros(coeff, trim="f")

    # The zero polynomial is constant.
    if coeff.size == 0:
        return 0.0, 0.0

    # Endpoints are always candidates.
    candidates = list(limits)

    if not only_limits:
        # Interior extrema satisfy p'(x) = 0.
        derivative = np.polyder(coeff)
        if derivative.size > 1:
            for root in np.roots(derivative):
                real_part = float(root.real)
                imaginary_part = float(root.imag)
                if abs(imaginary_part) <= imag_tol * max(1.0, abs(real_part)):
                    # Accept tiny numerical excursions beyond the interval.
                    if limits[0] - imag_tol <= real_part <= limits[1] + imag_tol:
                        candidates.append(float(np.clip(real_part, limits[0], limits[1])))

    x = np.asarray(candidates, dtype=np.float64)
    values = np.polyval(coeff, x)

    index = int(np.argmin(values))
    return float(x[index]), float(values[index])


def integrate_trailing_column_householder_python (
    L: Matrix, q: Vector, j: int, *, inplace: bool = False, atol: float = 1e-12,
) -> Matrix:
    """
    Integrate q as column j of an implicit orthogonal transformation.

    The first j columns of L remain unchanged. The vector q is expressed in
    trailing coordinates and must therefore have shape (n-j,).

    A numerically stable Householder construction is used. It chooses between
    +e_0 and -e_0 to avoid cancellation in the Householder vector.

    The returned matrix satisfies

        L_new[:, j] = L[:, j:] @ q.

    Parameters
    ----------
    L:
        Matrix of shape (n, m).

    q:
        Unit vector of shape (m-j,), expressed in trailing coordinates.

    j:
        Zero-based index of the column to integrate.

    inplace:
        If True, overwrite L. Otherwise, operate on a copy.

    atol:
        Tolerance used to check whether q has unit norm.

    Returns
    -------
    L_new:
        Updated matrix with q integrated as column j.
    """
    if L.ndim != 2:
        raise ValueError("L must be a two-dimensional matrix")

    _, m = L.shape

    if not 0 <= j < m:
        raise IndexError(f"j must satisfy 0 <= j < {m}")

    if not np.issubdtype(L.dtype, np.floating):
        raise TypeError("L must have a floating-point dtype")

    trailing_dimension = m - j
    if q.shape != (trailing_dimension,):
        raise ValueError(
            f"q must have shape ({trailing_dimension},), got {q.shape}"
        )

    if not np.all(np.isfinite(q)):
        raise ValueError("q must contain only finite values")

    q_norm = float(np.linalg.norm(q))
    if not np.isclose(q_norm, 1.0, atol=atol, rtol=atol):
        raise ValueError(f"q must be a unit vector, got norm {q_norm}")

    result = L if inplace else L.copy()
    trailing = result[:, j:]

    # Choose the sign so that q[0] and s have the same sign.
    # Thus q[0] + s has magnitude at least 1 and cannot suffer cancellation.
    s = 1.0 if q[0] >= 0.0 else -1.0

    # v = q + s e_0
    v = q.copy()
    v[0] += s

    # Since ||q|| = 1:
    #
    #     v^T v = 2 (1 + s q[0]).
    #
    # Computing it explicitly also works, but this expression avoids an
    # additional dot product.
    v_squared_norm = 2.0 * (1.0 + s * float(q[0]))

    # Apply H = I - 2 vv^T / (v^T v) from the right:
    #
    #     trailing <- trailing H
    #
    trailing -= (
        2.0 / v_squared_norm
    ) * np.outer(trailing @ v, v)

    # H e_0 = -s q. Flip the new first trailing column so that it is q.
    trailing[:, 0] *= -s

    return result

# This function is not used in the actual algorithm but implemented
# somewhat independently to verify the intermediate steps of the
# algorithm.

def conditional_expected_loss_Q (
    L:Matrix, Q:Matrix,
    i:int, j:int, row_weights:Weights=None)->float:
    """Compute the normalized conditional expected fourth-moment loss of L @ Q.

    Columns 0, ..., j-1 of Q and entries 0, ..., i of column j are
    considered fixed. All remaining entries are distributed according to
    Haar measure on O(n), conditioned on these fixed values.

    More precisely, this function computes

        m/n * E[sum_{r,k} w[r] * (L @ Q_random)[r,k]**4 | fixed entries].

    The fixed columns of Q, i.e. Q:,:j must be orthonormal. The fixed
    part of column Q must have length <=1. The remaining entries are
    ignored.
    
    Parameters
    ----------
    L:
        Matrix of shape (n, m).

    Q:
        Incomplete orthogonal matrix of shape (m, m).

    i:
        Last fixed row index in column j. The value -1 is allowed and means
        that no entry of column j is fixed.

    j:
        Current column. Columns with indices smaller than j are fully fixed.

    row_weights:
        Either an array of shape (n,) or None. For None, the
        weights are

            w[r] = 1 / ||L[r, :]||_2**4,

        corresponding to inverse squared row variances.

    Returns
    -------
    float
        Conditional expected weighted fourth-moment loss.

    """
    if L.ndim != 2:
        raise ValueError("L must be a two-dimensional matrix")
    if Q.ndim != 2:
        raise ValueError("Q must be a two-dimensional matrix")
    n, m = L.shape

    if Q.shape != (m, m):
        raise ValueError(
            f"Q must have shape ({m}, {m}), got {Q.shape}"
        )
    if not 0 <= j <= m:
        raise IndexError(f"j ({j}) must be 0 <= j <= {m}")
    if not -1 <= i < m:
        raise IndexError(f"i ({i}) must be -1 <= i < {m}")
    if not np.all(np.isfinite(L)):
        raise ValueError("L must contain only finite values")
    if not np.all(np.isfinite(Q)):
        raise ValueError("Q must contain only finite values")

    # This is a reference/comparison routine, so an explicit orthogonality
    # check is useful.
    if j>0:
        QFixed = Q[:,:j]
        if not np.allclose(QFixed.T @ QFixed, np.eye(j), atol=1e-10):
            raise ValueError("Fixed columns of Q must be orthonormal")
    if i>-1 and j<m:
        qFixed = Q[:i+1,j]
        if np.linalg.norm(qFixed)>1:
            raise ValueError("Fixed part of column j must have norm <=1.")
    row_norm2 = np.einsum("rk,rk->r", L, L)
    row_weights = make_row_weights (L, row_weights)
    if not np.all(np.isfinite(row_weights)):
        raise ValueError("row_weights must contain only finite values")
    if np.any(row_weights < 0.0):
        raise ValueError("row_weights must be nonnegative")

    # ------------------------------------------------------------------
    # Fully fixed columns 0, ..., j-1
    # ------------------------------------------------------------------

    if j > 0:
        fixed_values = L @ Q[:, :j]

        fixed_loss_per_row = np.sum(fixed_values**4, axis=1)
        if j==m:
            return m / n * (row_weights @ fixed_loss_per_row)  # nothing random left

        # Squared length of each row projected onto the orthogonal complement
        # of the already fixed columns.
        remaining_norm2 = (
            row_norm2
            - np.sum(fixed_values**2, axis=1)
        )
        remaining_norm2 = np.maximum(remaining_norm2, 0.0)

    else:
        fixed_loss_per_row = np.zeros(n, dtype=np.float64)
        remaining_norm2 = row_norm2.copy()

    # ------------------------------------------------------------------
    # Affine sphere for the current column
    #
    # The current column q must satisfy
    #
    #     Q[:, :j].T @ q = 0
    #
    # and
    #
    #     q[:i+1] = Q[:i+1, j].
    #
    # We write
    #
    #     q = q0 + rho * N u,
    #
    # where q0 is the minimum-norm solution, the columns of N span the
    # homogeneous nullspace, and u is uniform on the unit sphere.
    # ------------------------------------------------------------------

    prefix_length = i + 1
    number_of_constraints = j + prefix_length

    constraint_matrix = np.empty(
        (number_of_constraints, m),
        dtype=np.float64,
    )
    constraint_values = np.empty(
        number_of_constraints,
        dtype=np.float64,
    )

    if j > 0:
        constraint_matrix[:j, :] = Q[:, :j].T
        constraint_values[:j] = 0.0

    if prefix_length > 0:
        constraint_matrix[j:, :] = np.eye(
            m,
            dtype=np.float64,
        )[:prefix_length]
        constraint_values[j:] = Q[:prefix_length, j]

    if number_of_constraints == 0:
        q0 = np.zeros(m, dtype=np.float64)
        null_basis = np.eye(m, dtype=np.float64)

    else:
        # SVD gives both the minimum-norm solution and an orthonormal basis
        # of the homogeneous nullspace.
        U, singular_values, Vh = np.linalg.svd(
            constraint_matrix,
            full_matrices=True,
        )

        tolerance = (
            max(constraint_matrix.shape)
            * np.finfo(np.float64).eps
            * singular_values[0]
        )

        rank = int(np.sum(singular_values > tolerance))

        U_rank = U[:, :rank]
        V_rank = Vh[:rank, :].T

        q0 = V_rank @ (
            (U_rank.T @ constraint_values)
            / singular_values[:rank]
        )

        null_basis = Vh[rank:, :].T

        constraint_residual = (
            constraint_matrix @ q0 - constraint_values
        )

        if not np.allclose(
            constraint_residual,
            0.0,
            rtol=1e-9,
            atol=1e-10,
        ):
            raise ValueError(
                "The supplied fixed entries are not mutually consistent"
            )

    free_dimension = null_basis.shape[1]

    # The residual radius can be computed from the supplied feasible current
    # column. This is somewhat more stable than 1 - ||q0||² near zero.
    current_column = Q[:, j]
    residual = current_column - q0
    residual_radius2 = float(residual @ residual)

    if free_dimension == 0:
        if residual_radius2 > 1e-18:
            raise ValueError(
                "The fixed constraints leave no freedom but do not determine "
                "the supplied current column"
            )
        residual_radius2 = 0.0

    # ------------------------------------------------------------------
    # Moments of x = l_r^T q for every row l_r of L
    #
    #     x = alpha + rho * b^T u
    #
    # with
    #
    #     alpha = l_r^T q0,
    #     beta² = ||N^T l_r||².
    # ------------------------------------------------------------------

    alpha = L @ q0

    if free_dimension > 0 and residual_radius2 > 0.0:
        projected_rows = L @ null_basis
        beta2 = np.einsum(
            "rk,rk->r",
            projected_rows,
            projected_rows,
        )

        expected_x2 = (
            alpha**2 + residual_radius2 * beta2 / free_dimension
        )

        expected_x4 = (
            alpha**4
            + 6.0* alpha**2 * residual_radius2 * beta2 / free_dimension
            + 3.0* residual_radius2**2 * beta2**2/ (free_dimension * (free_dimension + 2))
        )

    else:
        expected_x2 = alpha**2
        expected_x4 = alpha**4

    # ------------------------------------------------------------------
    # Current and future columns
    #
    # Let A = ||P l_r||² be the row energy remaining after columns < j.
    # Conditional on the current column value x, the future columns contribute
    #
    #     3 / (d + 1) * (A - x²)²,
    #
    # where d = n-j is the number of columns including the current one.
    # ------------------------------------------------------------------

    remaining_columns = m - j

    if remaining_columns == 1:
        # There are no future columns.
        current_and_future = expected_x4

    else:
        future_factor = 3.0 / (remaining_columns + 1)

        current_and_future = (
            future_factor * remaining_norm2**2
            - 2.0 * future_factor * remaining_norm2 * expected_x2
            + (1.0 + future_factor) * expected_x4
        )

    expected_loss_per_row = fixed_loss_per_row + current_and_future

    return float(m / n * (row_weights @ expected_loss_per_row))


def conditional_quartic_coefficients_Q_python (phi:Vector, alpha: Vector, beta: Vector, gamma: Vector, sigma: Vector,
    upsilon: float, i: int, j: int, m:int, row_weights: Vector,
) -> Vector:
    """
    Returns coefficients [kappa_4, ..., kappa_0] of

        F(y) = kappa_0 + kappa_1*y + ... + kappa_4*y**4.

    the expected weighted L4-loss. See paper for details.
    This function encapsulates the main formula for the
    conditional expectation elementwise optimization of Q
    in L4(LQ).

    The valid range for choosing y is +/-sqrt(upsilon).
    
    Parameters
    ----------
    phi:
        phi_r = sum_{i'<i} f_{ri'}
    alpha:
        alpha_r = a_r @ x.
    beta:
        beta_r = b_r.
    gamma:
        gamma_r = ||c_r||^2.
    sigma:
        sigma_r = ||a_r||^2 + beta_r^2 + gamma_r.
    upsilon:
        1 - ||x||^2.
    i,j:
        Index of the entry to be optimized
    m:
        Number of sigma-point pairs, i.e. columns of L and rows/cols of Q.
    row_weights:
        Weights w_r.
    """
    n = row_weights.shape[0]
    arrays = (phi, alpha, beta, gamma, sigma, row_weights)
    if not all(array.shape == (n,) for array in arrays):
        raise ValueError("All row-dependent arrays must have equal shape")

    lam = 3/(m-j+1)
    Lambda = 1.0 + lam

    if m-i-1>0:
        v = gamma / (m-i-1)
        h = 3.0 * gamma**2 / ((m-i-1) * (m-i+1))
    else: v = h = 0
    alpha4 = alpha**4
    alpha2 = alpha**2
    beta2  = beta**2
    beta4  = beta**4

    k0_rows = (Lambda * (alpha4 + 6.0 * v * upsilon * alpha2 + h * upsilon**2)
        - 2.0 * lam * sigma * (alpha2 + v * upsilon)
        + lam * sigma**2 + phi
    )

    k1_rows = (4.0 * alpha * beta * (Lambda * (alpha2 + 3.0 * v * upsilon) - lam * sigma))

    k2_rows = (Lambda * (6.0 * alpha2 * beta2 
                         + 6.0 * v * (beta2 * upsilon - alpha2)
                         - 2.0 * h * upsilon)
              - 2.0 * lam * sigma * (beta2 - v)
    )

    k3_rows = (4.0 * Lambda * alpha * beta * (beta2 - 3.0 * v))

    k4_rows = (Lambda * (beta4  - 6.0 * v * beta2 + h))

    kappas = np.array([
            np.sum(row_weights * k4_rows),
            np.sum(row_weights * k3_rows),
            np.sum(row_weights * k2_rows),
            np.sum(row_weights * k1_rows),
            np.sum(row_weights * k0_rows),
        ], dtype=np.float64)

    return m / n * kappas


@overload
def optimize_conditional_Q_column_python (L: Matrix, j: int,
     row_weights: Vector, *,  return_history: Literal[False] = False,
) -> tuple[Vector, float]: ...

@overload
def optimize_conditional_Q_column_python (L: Matrix, j: int,
     row_weights: Vector, *,  return_history: Literal[True],
) -> tuple[Vector, float, Vector]: ...

def optimize_conditional_Q_column_python (L: Matrix, j: int,
     row_weights: Vector, *,  return_history: bool = False,
) -> tuple[Vector, float] | tuple[Vector, float, Vector]:
    """
    Choose column j of Q element by element by conditional-expectation
    minimization.

    Assumptions
    -----------
    Q has the Householder-normalized block structure

        Q[:j, :j] = I
        Q[:j, j:] = 0
        Q[j:, :j] = 0.

    Thus the active part of column j is Q[j:, j]. 
    The matrix Q is implicitly assumed, not passed, because the 
    non-active part is I and the active part doesn't matter,
    because it is filled by the algorithm.

    Complexity
    ----------
    O(number_of_rows(L) * (n-j)) for the incremental state updates and
    quartic coefficient calculations. For square L this is O(n^2) per
    column and O(n^3) for all columns.

    Returns
    -------
    q:
        active part of column j (Q[j:,j]) chosen by the algorithm

    expected_loss:
        Conditional expected loss after the complete column has been chosen.

    If return_history is True, also returns the conditional expected loss
    after each chosen entry.
    """
    if not isinstance(L, np.ndarray) or L.dtype!=np.float64:
        raise TypeError("L must be a NumPy float64 array.")
    if L.ndim != 2:
        raise ValueError("L must be a matrix")

    n, m = L.shape
    if not 0 <= j < m:
        raise ValueError("j must satisfy 0 <= j < m")

    # * Quantities constant throughout this column *
    # sigma_r = ||L[r, j:]||^2
    sigma = np.sum(L[:, j:] ** 2, axis=1)
    # Since previous columns of Q are coordinate vectors, the already
    # fixed columns of LQ are exactly L[:, :j].
    phi = np.sum(L[:,:j]**4, axis=1)

    # * Incrementally updated quantities. *
    # At i == j the fixed prefix x is empty.
    alpha = np.zeros(n, dtype=np.float64)
    # beta_r = L[r, i]
    beta = L[:, j]
    # gamma_r = sum_{k=i+1}^{n-1} L[r,k]^2
    gamma = np.sum(L[:, j + 1:]**2, axis=1)
    # upsilon = 1 - ||x||^2. Initially x is empty.
    upsilon = 1.0

    # * Quantities to be filled in *
    q = np.empty(m-j, dtype=np.float64)
    history = np.empty(m - j, dtype=np.float64)
    expected_loss = np.nan

    for i in range(j, m):
        # Remove tiny negative values caused by accumulated roundoff.
        # Also set tiny positive to 0
        tolerance = 1e-13 * max(1.0, m - j)
        if upsilon < 0.0:
            if upsilon < -tolerance:
                raise ArithmeticError(f"Remaining squared norm became negative: {upsilon}")
            upsilon = 0.0
        if upsilon < tolerance: upsilon = 0
        radius = np.sqrt(upsilon)
        if radius==0: y = 0 # expected_loss keeps the same
        else:
            coefficients = conditional_quartic_coefficients_Q_python (
                phi, alpha, beta, gamma, sigma, upsilon, i, j, m, row_weights
            )
            y, expected_loss = minimize_quartic_on_interval_python (
                coefficients, (-radius, radius), only_limits = (i==m-1)
            )
        q[i-j] = y
        history[i-j] = expected_loss

        # Incremental transition from i to i+1.
        alpha += beta * y
        upsilon -= y * y
        if i + 1 < m:
            beta = L[:, i + 1]
            gamma -= beta * beta
            np.maximum(gamma, 0.0, out=gamma)

    # The completed column should have unit norm.
    if not np.isclose(np.linalg.norm(q), 1.0, atol=1e-10, rtol=0):
        raise ArithmeticError(
            f"Optimized column is not normalized: ||Q[j:,j]||^2={np.linalg.norm(q)}"
        )
    if return_history:
        return q, float(expected_loss), history
    return q, float(expected_loss)

def cel4o_Q_python (L:Matrix, row_weights: Weights = None) -> Matrix:
    """
    For L with shape (n, m), finds a matrix L' with the same shape
    (not necessarily triangular) and L'@L'.T = L@L.T
    that has at most the weighted L4-loss expected from rotating with 
    a random Q and usually smaller. For natural row weights, its normalized
    L4-loss is at most 3*m/(m+2).

    The algorithm works by building an implicit m*m orthonormal rotation Q
    column by
    column (j), element by element (i) and choosing each entry Q[i,j]
    optimal regarding the expected l4-loss over all following entries
    and columns. This is called conditional expectation optimisation.
    Actually, after the algorithm has completed a column, it integrates
    that column into L' with a householder transform. The previously 
    optimized column of Q becomes identity then, so the orthogonality
    constraint for future columns reduces to operating and a column
    one element shorter.

    Returns the matrix L'. Q is implicit and 
    never explicitly formed.
    """
    if not isinstance(L, np.ndarray) or L.dtype!=np.float64:
        raise TypeError("L must be a NumPy float64 array.")
    if L.ndim != 2:
        raise ValueError("L must be a matrix")
    m = L.shape[1]
    row_weights = make_row_weights (L, row_weights)
    Ls = L.copy()
    econd = np.nan

    for j in range(m):
        q, econd = optimize_conditional_Q_column_python (Ls, j, row_weights)
        integrate_trailing_column_householder_python (Ls, q, j, inplace=True)
    return Ls

def compact_matrix_sqrt_Q_python (Sigma:Matrix, row_weights: Weights = None
    ) -> Matrix:
    """Computes a square root of Sigma with low weighted L4-loss.

    If the result is L, then L@L.T=Sigma. For natural weights (None),
    the loss is guaranteed to be <=3 - 2/(n+2). Returns the square-root
    matrix L.
    """
    L = np.linalg.cholesky (Sigma)
    return cel4o_Q_python (L, row_weights)

def cel4o_Q_with_history (L:Matrix, row_weights: Weights = None
    ) -> tuple[Matrix, float, List[Dict]]:
    """Same as optimize_conditional_Q but in addition returns 
    extensive history information for every step. 

    L may have shape (n, m); the Q matrices stored in the history have
    shape (m, m), and the returned optimized matrix has shape (n, m).
    
        Ls, econd, history

    history is a list. Each entry is the status after a step and
    formally a dict:
         step:  consecutive counter
         step_type: 'init', 'q-element', 'householder', 'final'
         i, j: element currently considered
         L: value of the L matrix
         Q: value of the Q matrix
         econd: conditional expectation
         l4ofL: current weighted L4-loss of the current L
                (changes only after completion of a column)
    The first entry has step_type 'init' and reflects the prior
    state. Then it goes columnwise. In every column the is one
    'q-element' step per element on/below the diagonal. 
    At the end of each column there is a 'householder' step.
    In the final end, there is a 'final step.

    This information can be used for diagnosis or visualization.
    """
    if not isinstance(L, np.ndarray) or L.dtype!=np.float64:
        raise TypeError("L must be a NumPy float64 array.")
    if L.ndim != 2:
        raise ValueError("L must be a matrix")
    n, m = L.shape
    row_weights = make_row_weights (L, row_weights)
    Ls = L.copy()
    step_ctr = 0
    l4loss = weighted_l4_loss_python (L, row_weights)
    econd = 3*m/(n*(m+2))*np.sum(row_weights*np.sum(L**2,axis=1)**2)
    Q = np.eye(m) # the implicit Q made explicit
    history = []
    history = [{
        'step':len(history), 'step_type':'init', 'i':None, 'j':None, 
         'L':Ls.copy(), 'Q':Q.copy(), 'w':row_weights, 
         'econd':econd, 'l4loss':l4loss
        }]

    for j in range(m):
        # optimize one column
        q, econd, econd_hist = optimize_conditional_Q_column_python (Ls, j, row_weights, return_history=True)
        # log column_hist into history
        for i in range(j, m):
            Q[i,j] = q[i-j]
            history.append ({
                'step':len(history), 'step_type':'q-element', 'i':i, 'j':j,
                'L':Ls.copy(), 'Q':Q.copy(), 'w':row_weights, 
                'econd':econd_hist[i-j], 'l4loss':l4loss
            })
        # do householder step and log it
        integrate_trailing_column_householder_python (Ls, q, j, inplace=True)
        Q = np.eye(m)
        l4loss = weighted_l4_loss_python (Ls, row_weights)
        history.append ({
            'step':len(history), 'step_type':'householder', 'i':m, 'j':j,
            'L':Ls.copy(), 'Q':Q.copy(), 'w':row_weights, 
            'econd':econd_hist[-1], 'l4loss':l4loss
        })
    history.append ({
            'step':len(history), 'step_type':'final', 'i':m, 'j':m,
            'L':Ls.copy(), 'Q':Q.copy(), 'w':row_weights, 
            'econd':l4loss, 'l4loss':l4loss
    })
    return Ls, econd, history
