"""Implementation of the cel4o-d and cel4o-q algorithm and several helper
   functions from the paper U. Frese, A. Hasselbring: 
   Compact Sigma-Points via Low-Entry Matrix Square Roots, 2026
   
   cel4o-d has a C++ implementation."""

from .wrapper import *
from ._cel4o import weighted_l4_loss, cel4o_D, dct_matrix
from ._cel4o import minimize_quartic_on_interval, integrate_trailing_column_householder
from ._cel4o import conditional_quartic_coefficients_Q, optimize_conditional_Q_column, cel4o_Q

#__all__ = ["cel4o_D"]
