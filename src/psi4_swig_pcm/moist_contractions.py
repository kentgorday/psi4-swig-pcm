# Adapted from moist's PySCF host, python/moist/pyscf.py in moist 1.1.0-alpha.1
# (https://github.com/lukaswittmann/moist, by Lukas Wittmann): PySCFHost._fock_lsf,
# PySCFHost._gradient_lsf, and the nuclear terms of PySCFHost.surface_potential, _grad_phi_nuc
# and _dphi_dxi.  Modified by Kent Gorday, 2026, for Psi4's basis functions (moist_host): the
# basis values come in as arrays, a block of grid points at a time, and the Fock term is built
# from matrix products.
#
# SPDX-License-Identifier: MPL-2.0
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0. If a copy
# of the MPL was not distributed with this file, You can obtain one at
# https://mozilla.org/MPL/2.0/.
"""The parts of the moist host that follow moist's own PySCF host (MPL-2.0, as moist is).

nuclear_potential: the nuclei's potential at moist's Gaussian probes, and its derivatives.

density_fock and density_gradient: the contractions of moist's density response (the weights
an isodensity cavity hands back) with d rho/dD and d rho/dR.  Both take the basis functions'
values and derivatives on grid points as a mapping

    (a, b, c) -> (npts, nfn) array of d^a/dx^a d^b/dy^b d^c/dz^c of each function,

with the weights of moist.DensityResponse at the same points.  Both are sums over the points,
so a grid can be processed in blocks and the results added.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
from scipy.special import erf

from .cavity import FloatArray

AOValues = Mapping[tuple[int, int, int], FloatArray]


def nuclear_potential(
    points: FloatArray, xi: FloatArray, atom_xyz: FloatArray, Z: FloatArray
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """phi, dphi/dr (npts, 3) and dphi/dxi of the nuclei at Gaussian probes erf(xi r)/r."""
    delta = points[:, None, :] - atom_xyz[None, :, :]
    d = np.linalg.norm(delta, axis=2)
    x = xi[:, None] * d
    small = x < 1e-3
    safe = np.where(small, 1.0, d)
    kernel = np.where(small, 2.0 * xi[:, None] / np.sqrt(np.pi) * (1 - x**2 / 3), erf(x) / safe)
    # -(1/r) d/dr [erf(xi r)/r], with its small-x series
    radial = np.where(
        small,
        4.0 * xi[:, None] ** 3 / (3.0 * np.sqrt(np.pi)) * (1.0 - 0.6 * x**2),
        (erf(x) - 2.0 * x * np.exp(-(x**2)) / np.sqrt(np.pi)) / safe**3,
    )
    phi: FloatArray = kernel @ Z
    dphi_dr: FloatArray = -np.einsum("A,iA,iAk->ik", Z, radial, delta)
    dphi_dxi: FloatArray = 2.0 / np.sqrt(np.pi) * (np.exp(-(x**2)) @ Z)
    return phi, dphi_dr, dphi_dxi


def density_fock(
    ao: AOValues, w_rho: FloatArray, w_grad_rho: FloatArray, w_hess_rho: FloatArray
) -> FloatArray:
    """The density weights against d rho/dD_uv, summed over the points: (nfn, nfn).

    ao to second order.  With rho = sum_uv D_uv chi_u chi_v, each order of the weights pairs
    derivative blocks of the functions; all but the first term come with their transpose.
    w_hess_rho[i, b, a] is moist's index order.
    """
    a0 = ao[0, 0, 0]
    F: FloatArray = (a0 * w_rho[:, None]).T @ a0
    half = np.zeros_like(F)  # the terms that appear with their transpose
    for a in range(3):
        ea = _unit(a)
        half += (ao[ea] * w_grad_rho[:, a, None]).T @ a0
        for b in range(3):
            eb = _unit(b)
            w = w_hess_rho[:, b, a][:, None]
            half += (ao[_add(ea, eb)] * w).T @ a0 + (ao[ea] * w).T @ ao[eb]
    out: FloatArray = F + half + half.T
    return out


def density_gradient(
    ao: AOValues,
    D: FloatArray,
    w_rho: FloatArray,
    w_grad_rho: FloatArray,
    w_hess_rho: FloatArray,
) -> FloatArray:
    """The density weights against d rho/d(centre of function u) at fixed D: (3, nfn).

    ao to third order; D (nfn, nfn) symmetric.  d/dR_A of every spatial derivative of rho is -2
    times the same derivative of u(r) = sum_{c on A, d} D_cd (d_k chi_c) chi_d; Leibniz expands
    it.  An atom's gradient is the sum of the columns of its functions.
    """
    right = {k: v @ D for k, v in ao.items() if sum(k) <= 2}
    zero = (0, 0, 0)
    out = np.zeros((3, D.shape[0]))
    for k in range(3):
        ek = _unit(k)
        terms: list[tuple[FloatArray, tuple[int, int, int], tuple[int, int, int]]] = [
            (w_rho, ek, zero)
        ]
        for a in range(3):
            ea = _unit(a)
            terms += [(w_grad_rho[:, a], _add(ek, ea), zero), (w_grad_rho[:, a], ek, ea)]
            for b in range(3):
                eb, w = _unit(b), w_hess_rho[:, b, a]
                terms += [
                    (w, _add(_add(ek, ea), eb), zero),
                    (w, _add(ek, ea), eb),
                    (w, _add(ek, eb), ea),
                    (w, ek, _add(ea, eb)),
                ]
        for w, left, rgt in terms:
            out[k] += np.einsum("i,iu,iu->u", w, ao[left], right[rgt])
    return -2.0 * out


def _unit(a: int) -> tuple[int, int, int]:
    return (int(a == 0), int(a == 1), int(a == 2))


def _add(p: tuple[int, int, int], q: tuple[int, int, int]) -> tuple[int, int, int]:
    return (p[0] + q[0], p[1] + q[1], p[2] + q[2])
