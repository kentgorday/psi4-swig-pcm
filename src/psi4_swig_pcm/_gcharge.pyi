# Copyright (C) 2026 Kent Gorday
# SPDX-License-Identifier: LGPL-3.0-only
"""Type stub for the compiled extension (csrc/gcharge.cpp).  Psi4 objects are untyped."""

from typing import Any

import numpy as np
from numpy.typing import NDArray

def gaussian_3c_packed_into(
    aux: Any, basis: Any, out: NDArray[np.float64], rows: NDArray[np.intp], nthreads: int
) -> None:
    """out[rows[k], mn] = (k|mn) for m >= n, packed row-major as m*(m+1)/2 + n."""

def gaussian_esp(aux: Any, basis: Any, D: Any, nthreads: int) -> Any:
    """psi4.core.Vector phi[k] = sum_mn D_mn (k|mn), D symmetric."""

def gaussian_potential_matrix(aux: Any, basis: Any, q: Any, nthreads: int) -> Any:
    """psi4.core.Matrix V[m, n] = sum_k q_k (k|mn)."""

def gaussian_esp_deriv1(
    aux: Any, basis: Any, D: Any, w: Any | None, natom: int, nthreads: int
) -> tuple[Any, Any]:
    """(field, grad): field[k, x] = sum_mn D_mn d(k|mn)/dr_k,x (naux, 3) and
    grad[A, x] = sum_k w_k sum_mn D_mn d(k|mn)/dR_A,x over basis centres (natom, 3)."""

def gaussian_overlap_esp(aux: Any, basis: Any, D: Any, nthreads: int) -> Any:
    """psi4.core.Vector s[k] = sum_mn D_mn (k m n), three-centre overlap, D symmetric."""
