# Copyright (C) 2026 Kent Gorday
# SPDX-License-Identifier: LGPL-3.0-only
"""Surface Gaussians as Psi4 basis sets and the C++ contractions (no SCF)."""

from __future__ import annotations

import numpy as np
import psi4
from psi4 import core
from scipy.spatial.distance import pdist

from psi4_swig_pcm import cavity as cav
from psi4_swig_pcm.charges import (
    ANCHOR_OFFSET,
    CLOSE_POINT_TOL,
    GaussianCharges,
    gaussian_basis,
    split_close_points,
)
from tests.helpers import psi4_molecule
from tests.molecules import MOLECULES


def surface(name: str) -> cav.Surface:
    mol = MOLECULES[name]
    return cav.make_surface(mol.bohr, mol.Z, cav.modified_bondi_radii())


def symmetric_formate_surface() -> cav.Surface:
    """The surface Psi4 builds at MOLECULES["formate_symmetric"] (Psi4's bohr conversion)."""
    mol = MOLECULES["formate_symmetric"]
    xyz = np.array(mol.angstrom) / psi4.constants.bohr2angstroms
    return cav.make_surface(xyz, mol.Z, cav.modified_bondi_radii())


def test_split_close_points() -> None:
    surf = surface("acetone")
    sets = split_close_points(surf.coords)
    assert len(sets) > 1, "acetone should have near-coincident points"
    np.testing.assert_array_equal(np.sort(np.concatenate(sets)), np.arange(surf.npts))
    for idx in sets:
        if len(idx) > 1:
            assert pdist(surf.coords[idx]).min() >= CLOSE_POINT_TOL


def test_gaussian_basis_keeps_points_and_appends_anchor() -> None:
    surf = surface("formate")
    pts = surf.coords[:5]
    bs = gaussian_basis(pts, surf.xi[:5] ** 2)
    geom = np.array(bs.molecule().geometry().np)
    np.testing.assert_array_equal(geom[:5], pts)
    np.testing.assert_allclose(geom[5], pts.mean(0) + ANCHOR_OFFSET, rtol=0, atol=1e-12)
    assert bs.nbf() == 6


def test_symmetric_point_set_builds() -> None:
    """Regression: this surface has an exactly symmetric 4-point set (see molecules.py)."""
    surf = symmetric_formate_surface()
    charges = GaussianCharges(surf, nthreads=1)
    assert [len(idx) for idx, _ in charges.sets] == [surf.npts - 4, 4]


def test_kernels_match_dense_integrals() -> None:
    surf = surface("acetone")
    bs = core.BasisSet.build(psi4_molecule("acetone"), "ORBITAL", "def2-svp")
    charges = GaussianCharges(surf, nthreads=4)
    rng = np.random.default_rng(1)
    D = rng.standard_normal((bs.nbf(), bs.nbf()))
    D += D.T
    mints = core.MintsHelper(bs)
    sel = rng.choice(surf.npts, 40, replace=False)
    dense = [
        -np.vdot(np.array(mints.ao_potential_erf(list(surf.coords[k]), float(surf.xi[k])).np), D)
        for k in sel
    ]
    phi = charges.esp(bs, core.Matrix.from_array(D))
    np.testing.assert_allclose(phi[sel], dense, rtol=1e-12, atol=1e-12)
    q = rng.standard_normal(surf.npts) * 1e-3
    V = np.array(charges.potential_matrix(bs, q).np)
    V_ref = np.array(charges.external_potential(q).computePotentialMatrix(bs).np)
    np.testing.assert_allclose(V, V_ref, atol=1e-14)


def test_incore_matches_direct() -> None:
    surf = surface("acetone")
    bs = core.BasisSet.build(psi4_molecule("acetone"), "ORBITAL", "def2-svp")
    direct = GaussianCharges(surf, nthreads=4)
    incore = GaussianCharges(surf, nthreads=4, basis=bs, incore_mb=1e4)
    assert direct.B is None
    assert incore.B is not None
    rng = np.random.default_rng(2)
    D = rng.standard_normal((bs.nbf(), bs.nbf()))
    D = core.Matrix.from_array(D + D.T)
    np.testing.assert_allclose(incore.esp(bs, D), direct.esp(bs, D), rtol=1e-12, atol=1e-12)
    q = rng.standard_normal(surf.npts) * 1e-3
    np.testing.assert_allclose(
        np.array(incore.potential_matrix(bs, q).np),
        np.array(direct.potential_matrix(bs, q).np),
        atol=1e-14,
    )
