# Copyright (C) 2026 Kent Gorday
# SPDX-License-Identifier: LGPL-3.0-only
"""Type stub for the compiled extension (csrc/nonelectrostatic.cpp).  Psi4 objects are untyped."""

from typing import Any

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.intp]

def field_matrix_packed_into(
    basis: Any, points: FloatArray, normals: FloatArray, out: FloatArray, nthreads: int
) -> None:
    """out[i, mn] = n_i . grad_s int phi_m phi_n / |r - s_i| for m >= n, packed m(m+1)/2 + n."""

def normal_field(
    basis: Any, points: FloatArray, normals: FloatArray, D: FloatArray, nthreads: int
) -> FloatArray:
    """The electrons' normal field n_i . E_el(s_i) for the symmetric density matrix D."""

def normal_field_matrix(
    basis: Any, points: FloatArray, normals: FloatArray, c: FloatArray, nthreads: int
) -> FloatArray:
    """sum_i c_i S_i, S_i the matrix whose contraction with D is n_i . E_el(s_i)."""

def field_gradient(
    basis: Any, points: FloatArray, D: FloatArray, q: FloatArray, natom: int, nthreads: int
) -> tuple[FloatArray, FloatArray]:
    """(per atom, per point) gradient of sum_i q_i . E_el(s_i) at fixed D."""

def dispersion_kernel(
    points: FloatArray,
    coords: FloatArray,
    normals: FloatArray,
    areas: FloatArray,
    delta: float,
    weights: FloatArray | None = None,
    nthreads: int = 1,
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray, FloatArray]:
    """(J, dJ/dr, dE/ds, dE/dA, dE/dn) for E = sum_p weights_p J(r_p)."""

def cavity_partition(
    points: FloatArray,
    owners: IntArray,
    centres: FloatArray,
    radii: FloatArray,
    dist: FloatArray,
    inside: bool,
    width: float,
    nthreads: int,
) -> FloatArray:
    """The partition weight of each point's owner."""

def cavity_partition_gradient(
    points: FloatArray,
    owners: IntArray,
    centres: FloatArray,
    radii: FloatArray,
    dist: FloatArray,
    inside: bool,
    width: float,
    values: FloatArray,
    nthreads: int,
) -> FloatArray:
    """sum_p values_p dP_owner(p)(r_p)/dR, the points moving with their owners: (natm, 3)."""

def ssf_weights(
    points: FloatArray, owners: IntArray, centres: FloatArray, a: float, nthreads: int
) -> FloatArray:
    """p_owner(y) of the Stratmann-Scuseria-Frisch partition at each point."""

def ssf_gradient(
    points: FloatArray,
    owners: IntArray,
    centres: FloatArray,
    a: float,
    coeff: FloatArray,
    nthreads: int,
) -> tuple[FloatArray, FloatArray]:
    """For sum_p coeff_p p_owner(y_p): (d/dR at fixed points (natm, 3), d/dy (n, 3))."""

def ssf_cells_hold(
    points: FloatArray, centres: FloatArray, a: float, nthreads: int
) -> NDArray[np.bool_]:
    """(npoints, natm): whether each atom's cell holds each point."""

def ray_adjoint(
    xyz: FloatArray,
    normals: FloatArray,
    areas: FloatArray,
    centres: FloatArray,
    point: IntArray,
    atom: IntArray,
    tau: FloatArray,
    w_radial: FloatArray,
    partition: FloatArray,
    values: FloatArray,
    gy: FloatArray,
    nthreads: int,
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
    """The exterior rays' (ds, dn, da, dc) for dE/dy = gy and integrand values."""
