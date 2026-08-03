import numpy as np
from numpy.typing import ArrayLike
from numpy.typing import NDArray
from typing import Any

Vector = NDArray[np.floating[Any]]
Matrix = NDArray[np.floating[Any]]
Weights = np.typing.ArrayLike | None

from ._cel4o import dct_matrix, cel4o_D, cel4o_Q

def compact_matrix_sqrt_D (Sigma:Matrix, row_weights:Weights=None, QDct:Matrix|None=None)-> Matrix:
    """Computes a sqrt of Sigma with low weighted l4-loss. If the result
    is L, L@L.T=Sigma using conditional l4 expectation optimization on
    a sign-diagonal. For natural weights ('auto') the loss is guaranteed
    to be <=4.5. Returns the square root matrix L.
    
    The computation is based on the DCT-Matrix QDct, which the user can provide
    to avoid repeated computation.
    """
    L = np.linalg.cholesky (Sigma)
    if QDct is None: QDct = dct_matrix (Sigma.shape[0], True)
    A = cel4o_D (L, QDct.T, row_weights) # or cel4o_d_python as a reference
    return A

def compact_matrix_sqrt_Q (Sigma:Matrix, row_weights:Weights=None)-> Matrix:
    """Computes a sqrt of Sigma with low weighted l4-loss. If the result
    is L, L@L.T=Sigma. For natural weights (None) the loss is guaranteed
    to be <=3 - 2/(n+2). Returns the square root matrix L."""
    L = np.linalg.cholesky (Sigma)
    return cel4o_Q (L, row_weights)
