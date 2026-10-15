# Copyright (C) 2026 Kent Gorday
# SPDX-License-Identifier: LGPL-3.0-only
"""Type stub for the compiled extension (csrc/gcharge.cpp).  Psi4 objects are untyped."""

from typing import Any

import numpy as np
from numpy.typing import NDArray

def gaussian_3c_packed(aux: Any, basis: Any, nthreads: int) -> NDArray[np.float64]:
    """B[k, mn] = (k|mn) for m >= n, packed row-major as m*(m+1)/2 + n."""

def gaussian_esp(aux: Any, basis: Any, D: Any, nthreads: int) -> Any:
    """psi4.core.Vector phi[k] = sum_mn D_mn (k|mn)."""

def gaussian_potential_matrix(aux: Any, basis: Any, q: Any, nthreads: int) -> Any:
    """psi4.core.Matrix V[m, n] = sum_k q_k (k|mn)."""
