# Copyright (C) 2026 Kent Gorday
# SPDX-License-Identifier: LGPL-3.0-only
"""The surface Gaussian charges as Psi4 basis sets, and their contractions with the AO density.

Each surface point carries a unit Gaussian s charge.  Psi4 holds such charges as ghost-He
s shells (the way proc.py builds "diffuse" external charges), so that its integral code,
ExternalPotential and gradients_on_gaussians can be used on them.  Two complications:

* A Psi4 Molecule rejects atoms closer than 0.1 bohr, and switching/Gaussian points can nearly
  coincide, so the points are split into several sets with no close pairs.
* Psi4 determines the full point group of every Molecule, even under fix_symmetry="c1", and
  with a fixed frame it fails ("Unrecognized point group bits") on some exactly symmetric point
  sets.  Each set therefore carries one extra Gaussian at a generic far position (the anchor),
  which breaks every symmetry, always has zero charge, and is dropped from every result.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from psi4 import core
from scipy.spatial import cKDTree

# _gcharge resolves its Psi4 symbols from psi4.core, which the import above has loaded.
from . import _gcharge
from .cavity import FloatArray, IntArray, Surface

CLOSE_POINT_TOL = 0.1  # bohr; a Psi4 Molecule rejects atoms closer than this
ANCHOR_OFFSET = np.array([17.31, -23.93, 31.71])  # bohr from a set's centroid
ANCHOR_EXPONENT = 1.0


def split_close_points(points: FloatArray, tol: float = CLOSE_POINT_TOL) -> list[IntArray]:
    """Partition point indices into subsets with no two points closer than tol."""
    neighbours: list[list[int]] = [[] for _ in points]
    for i, j in cKDTree(points).query_pairs(tol):
        neighbours[i].append(j)
        neighbours[j].append(i)
    subset = np.zeros(len(points), dtype=np.intp)
    for i in range(len(points)):  # greedy colouring; nearly every point lands in subset 0
        used = {int(subset[j]) for j in neighbours[i] if j < i}
        subset[i] = next(c for c in range(len(used) + 1) if c not in used)
    return [np.flatnonzero(subset == c) for c in range(int(subset.max()) + 1)]


def gaussian_basis(points: FloatArray, exponents: FloatArray) -> core.BasisSet:
    """Unit Gaussian s charges at points, followed by the zero-charge anchor, as a BasisSet."""
    pts = np.vstack([points, points.mean(0) + ANCHOR_OFFSET])
    exps = np.append(exponents, ANCHOR_EXPONENT)
    n = len(pts)
    mol = core.Molecule.from_arrays(
        units="Bohr",
        geom=pts,
        elem=["He"] * n,
        real=[False] * n,
        elbl=[f"@He_{i + 1}" for i in range(n)],
        fix_symmetry="c1",
        fix_com=True,
        fix_orientation=True,
    )
    spec = {
        "blend": "Diffuse Charges",
        "key": "(none)",
        "message": "",
        "name": "Diffuse Charges",
        "puream": True,
        "shell_map": [[f"HE_{i + 1}", f"{a:.2f}", [0, (a, 1.0)]] for i, a in enumerate(exps)],
    }
    bs = core.BasisSet.construct_from_pydict(mol, spec, True)
    bs.negative_gaussian_normalization_to_coefficients()
    return bs


class GaussianCharges:
    """The surface Gaussians as Psi4 bases, with phi(D) and V(q) either direct or from storage.

    In-core: B[k, mn] = (k|mn) (packed lower triangle) is computed once per geometry, after
    which each SCF iteration is two matrix-vector products.  Direct: both contractions
    recompute the integrals.  The (k|mn) carry the electron sign (see csrc/gcharge.cpp).
    """

    def __init__(
        self,
        surf: Surface,
        nthreads: int,
        basis: core.BasisSet | None = None,
        incore_mb: float = 0.0,
    ) -> None:
        self.n, self.nthreads = surf.npts, nthreads
        self.sets: list[tuple[IntArray, Any]] = [
            (idx, gaussian_basis(surf.coords[idx], surf.xi[idx] ** 2))
            for idx in split_close_points(surf.coords)
        ]
        self.B: FloatArray | None = None
        if basis is not None and incore_mb > 0:
            nbf = basis.nbf()
            if self.n * nbf * (nbf + 1) / 2 * 8 / 2**20 <= incore_mb:
                self.B = np.empty((self.n, nbf * (nbf + 1) // 2))
                for idx, aux in self.sets:
                    self.B[idx] = _gcharge.gaussian_3c_packed(aux, basis, nthreads)[:-1]
                self.tril = np.tril_indices(nbf)
                self.offdiag = self.tril[0] != self.tril[1]

    def esp(self, basis: core.BasisSet, D: core.Matrix) -> FloatArray:
        """Electronic potential at each Gaussian (negative for electrons)."""
        if self.B is not None:
            d = np.asarray(D.np)[self.tril].copy()
            d[self.offdiag] *= 2.0
            phi_incore: FloatArray = self.B @ d
            return phi_incore
        phi = np.empty(self.n)
        for idx, aux in self.sets:
            phi[idx] = np.asarray(_gcharge.gaussian_esp(aux, basis, D, self.nthreads).np)[:-1]
        return phi

    def potential_matrix(self, basis: core.BasisSet, q: FloatArray) -> core.Matrix:
        """Fock contribution sum_k q_k (k|mn) of charges q."""
        if self.B is not None:
            v = q @ self.B
            V = np.zeros((basis.nbf(), basis.nbf()))
            V[self.tril] = v
            V.T[self.tril] = v
            return core.Matrix.from_array(V)
        Vm = core.Matrix(basis.nbf(), basis.nbf())
        for idx, aux in self.sets:
            qv = core.Vector.from_array(np.append(q[idx], 0.0))
            Vm.add(_gcharge.gaussian_potential_matrix(aux, basis, qv, self.nthreads))
        return Vm

    def external_potential(self, q: FloatArray) -> core.ExternalPotential:
        """The charges q as a Psi4 ExternalPotential (for the SCF gradient)."""
        ep = core.ExternalPotential()
        for idx, aux in self.sets:
            ep.addGaussian(aux, core.Vector.from_array(np.append(q[idx], 0.0)))
        return ep

    def point_gradients(self, ep: core.ExternalPotential) -> FloatArray:
        """Per-point gradient of the interaction of ep's charges with electrons and nuclei."""
        g = np.empty((self.n, 3))
        for (idx, _), gi in zip(self.sets, ep.gradients_on_gaussians(), strict=True):
            g[idx] = np.asarray(gi.np)[:-1]
        return g
