# Copyright (C) 2026 Kent Gorday
# Portions adapted from Psi4, Copyright (c) 2007-2026 The Psi4 Developers.
# SPDX-License-Identifier: LGPL-3.0-only
#
# gaussian_basis() follows the diffuse-charge construction in Psi4's
# driver/procrouting/proc.py (_set_external_potentials_to_wavefunction): the same ghost-He
# Molecule.from_arrays call, "Diffuse Charges" shell dictionary and normalization.
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

from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np
from psi4 import core
from scipy.spatial import cKDTree

# _gcharge resolves its Psi4 symbols from psi4.core, which the import above has loaded.
from . import _gcharge
from .cavity import FloatArray, IntArray

CLOSE_POINT_TOL = 0.1  # bohr; a Psi4 Molecule rejects atoms closer than this
ANCHOR_OFFSET = np.array([17.31, -23.93, 31.71])  # bohr from a set's centroid
ANCHOR_EXPONENT = 1.0


class PointSet(Protocol):
    """What GaussianCharges needs of a surface: point positions and Gaussian widths (bohr).

    Satisfied by cavity.Surface and by GaussianPoints.
    """

    @property
    def coords(self) -> FloatArray: ...  # (npts, 3)
    @property
    def xi(self) -> FloatArray: ...  # the charge at a point is exp(-(xi r)^2), normalized
    @property
    def npts(self) -> int: ...


@dataclass(frozen=True)
class GaussianPoints:
    """A PointSet from bare arrays: positions (npts, 3) and Gaussian widths xi (bohr).

    The grid of a moist cavity (moist_host) takes this form, as do test points.
    """

    coords: FloatArray
    xi: FloatArray

    @property
    def npts(self) -> int:
        return len(self.xi)


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
    Both paths take the density to be symmetric (only m >= n is stored or computed).

    B holds npts * nbf * (nbf + 1) / 2 doubles, on top of Psi4's own memory: it is built
    only if that fits incore_mb.
    """

    def __init__(
        self,
        surf: PointSet,
        nthreads: int,
        basis: core.BasisSet | None = None,
        incore_mb: float = 0.0,
    ) -> None:
        self.n, self.nthreads = surf.npts, nthreads
        self.xi: FloatArray = np.asarray(surf.xi)
        self.sets: list[tuple[IntArray, Any]] = [
            (idx, gaussian_basis(surf.coords[idx], surf.xi[idx] ** 2))
            for idx in split_close_points(surf.coords)
        ]
        self.B: FloatArray | None = None
        if basis is not None and incore_mb > 0:
            nbf = basis.nbf()
            if self.n * nbf * (nbf + 1) / 2 * 8 / 2**20 <= incore_mb:
                self.B = np.empty((self.n, nbf * (nbf + 1) // 2))
                for idx, aux in self.sets:  # each set's rows, written in place
                    _gcharge.gaussian_3c_packed_into(aux, basis, self.B, idx, nthreads)
                self.tril = np.tril_indices(nbf)
                self.offdiag = self.tril[0] != self.tril[1]

    def esp(self, basis: core.BasisSet, D: core.Matrix) -> FloatArray:
        """Electronic potential at each Gaussian (negative for electrons); D symmetric."""
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

    def esp_deriv1(
        self, basis: core.BasisSet, D: core.Matrix, w: FloatArray | None = None
    ) -> tuple[FloatArray, FloatArray]:
        """(field, grad) of the electronic potential, D symmetric:

        field (npts, 3): d phi_el[k] / d (position of point k);
        grad (natom, 3): sum_k w_k d phi_el[k] / dR_A over the basis-function centres, points
        held fixed (zero when w is None).
        """
        field = np.empty((self.n, 3))
        grad = np.zeros((basis.molecule().natom(), 3))
        for idx, aux in self.sets:
            wv = None if w is None else core.Vector.from_array(np.append(w[idx], 0.0))
            f, g = _gcharge.gaussian_esp_deriv1(aux, basis, D, wv, len(grad), self.nthreads)
            field[idx] = np.asarray(f.np)[:-1]
            grad += np.asarray(g.np)
        return field, grad

    def esp_dxi(self, basis: core.BasisSet, D: core.Matrix) -> FloatArray:
        """d phi_el[k] / d xi_k, D symmetric.

        d/dxi erf(xi r)/r = (2/sqrt(pi)) exp(-xi^2 r^2), so this is a three-centre overlap; the
        aux functions carry -(xi^2/pi)^(3/2), which the prefactor undoes.
        """
        s = np.empty(self.n)
        for idx, aux in self.sets:
            s3 = _gcharge.gaussian_overlap_esp(aux, basis, D, self.nthreads)
            s[idx] = np.asarray(s3.np)[:-1]
        xi = self.xi
        out: FloatArray = 2.0 / np.sqrt(np.pi) * (np.pi / xi**2) ** 1.5 * s
        return out

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
