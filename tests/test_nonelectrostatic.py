# Copyright (C) 2026 Kent Gorday
# SPDX-License-Identifier: LGPL-3.0-only
"""The nonelectrostatic correction: integrals, potential, gradients and both SCF modes."""

from __future__ import annotations

from typing import Any

import numpy as np
import psi4
import pytest
from psi4 import core

import psi4_swig_pcm
from psi4_swig_pcm import _nonelectrostatic as _ne
from psi4_swig_pcm.cavity import BOHR, FloatArray
from psi4_swig_pcm.nonelectrostatic import (
    NonelectrostaticSettings,
    Terms,
    build_cavity_grid,
    partition_gradient,
)

# Formate off its symmetric geometry (Angstrom), so that no gradient component vanishes.
FORMATE = np.array(
    [[0.02, -0.03, 0.53], [1.15, -0.03, -0.04], [-1.11, -0.03, -0.04], [0.09, 0.01, 1.67]]
)
WATER_RHO = 0.049389617856476845  # e/bohr^3
SETTINGS = NonelectrostaticSettings(-0.004, 0.04, -0.5, 2.5, -0.2, 2.0, solvent_rho=WATER_RHO)
OPTIONS = {
    "basis": "6-31g*",
    "puream": False,
    "scf_type": "pk",
    "e_convergence": 1e-12,
    "d_convergence": 1e-10,
}
# tautomator's PySCF implementation on the same calculation (exact integrals, Cartesian d,
# the same PCM): this one agrees with it to ~1e-10 Eh.
PYSCF_TERMS = (-0.003616728100, 0.011059251053, -0.071498120677)
PYSCF_POST_SCF = -188.3549223004
PYSCF_SELF_CONSISTENT = -188.3645184599


def _molecule(xyz_angstrom: FloatArray) -> Any:
    lines = ["-1 1", "units angstrom", "symmetry c1", "no_com", "no_reorient"]
    lines += [
        f"{s} {x:.12f} {y:.12f} {z:.12f}"
        for s, (x, y, z) in zip("COOH", xyz_angstrom, strict=True)
    ]
    return psi4.geometry("\n".join(lines))


def _basis(xyz_angstrom: FloatArray, basis: str = "6-31g*") -> tuple[Any, Any]:
    mol = _molecule(xyz_angstrom)
    psi4.set_options({**OPTIONS, "basis": basis})
    return mol, core.Wavefunction.build(mol, core.get_global_option("BASIS")).basisset()


def _scf(xyz_angstrom: FloatArray, ne: NonelectrostaticSettings | None = SETTINGS) -> Any:
    psi4.core.clean()
    _molecule(xyz_angstrom)
    psi4.set_options(OPTIONS)
    with psi4_swig_pcm.enabled(psi4_swig_pcm.PCMSettings(nonelectrostatic=ne)):
        _e, wfn = psi4.energy("hf", return_wfn=True)
    return wfn


def _terms(wfn: Any) -> Terms:
    return wfn.ddx.terms  # type: ignore[no-any-return]


def _density(wfn: Any) -> FloatArray:
    return np.asarray(wfn.Da().np) + np.asarray(wfn.Db().np)


class TestFieldIntegrals:
    @pytest.fixture()
    def setup(self) -> tuple[Any, Any, FloatArray, FloatArray, FloatArray]:
        mol, basis = _basis(FORMATE, "cc-pvtz")  # s to f functions
        rng = np.random.default_rng(0)
        points = rng.normal(scale=2.0, size=(3, 3))
        points += 3.0 * np.sign(points)
        normals = rng.normal(size=(3, 3))
        normals /= np.linalg.norm(normals, axis=1)[:, None]
        D = rng.normal(size=(basis.nbf(), basis.nbf()))
        return mol, basis, points, normals, D + D.T

    def test_matrices_are_psi4s_electric_field_integrals(self, setup: Any) -> None:
        _mol, basis, points, normals, _D = setup
        mints = core.MintsHelper(basis)
        c = np.array([0.7, -1.3, 0.4])
        mine = _ne.normal_field_matrix(basis, points, normals, c, 2)
        ref = sum(
            ci * sum(n[x] * np.asarray(mints.electric_field(list(p))[x].np) for x in range(3))
            for ci, p, n in zip(c, points, normals, strict=True)
        )
        assert mine == pytest.approx(ref, abs=1e-14)

    def test_field_is_psi4s_electronic_field(self, setup: Any) -> None:
        _mol, basis, points, normals, D = setup
        mints = core.MintsHelper(basis)
        ref = np.asarray(
            mints.electric_field_value(
                core.Matrix.from_array(points), core.Matrix.from_array(D)
            ).np
        )
        F = _ne.normal_field(basis, points, normals, D, 2)
        assert pytest.approx(np.einsum("px,px->p", ref, normals), abs=1e-12) == F
        nbf = basis.nbf()
        stored = np.empty((3, nbf * (nbf + 1) // 2))
        _ne.field_matrix_packed_into(basis, points, normals, stored, 2)
        tril = np.tril_indices(nbf)
        d = D[tril].copy()
        d[tril[0] != tril[1]] *= 2.0
        assert stored @ d == pytest.approx(F, abs=1e-12)

    def test_gradient_matches_finite_differences(self, setup: Any) -> None:
        mol, basis, points, _normals, D = setup
        q = np.random.default_rng(1).normal(size=points.shape)
        atoms, at_points = _ne.field_gradient(basis, points, D, q, mol.natom(), 2)

        def energy(bas: Any, pts: FloatArray) -> float:
            total = 0.0
            for x in range(3):
                e = np.zeros_like(pts)
                e[:, x] = 1.0
                total += float(q[:, x] @ _ne.normal_field(bas, pts, e, D, 2))
            return total

        h = 1e-5
        step = np.zeros_like(points)
        step[1, 2] = h
        numeric = (energy(basis, points + step) - energy(basis, points - step)) / (2 * h)
        assert at_points[1, 2] == pytest.approx(numeric, rel=1e-7)
        for atom, axis in [(0, 0), (3, 1)]:
            plus, minus = FORMATE.copy(), FORMATE.copy()
            plus[atom, axis] += h * BOHR
            minus[atom, axis] -= h * BOHR
            numeric = (
                energy(_basis(plus, "cc-pvtz")[1], points)
                - energy(_basis(minus, "cc-pvtz")[1], points)
            ) / (2 * h)
            assert atoms[atom, axis] == pytest.approx(numeric, rel=1e-6, abs=1e-9)


class TestKernelAndGrids:
    def test_kernel_and_its_derivatives(self) -> None:
        rng = np.random.default_rng(3)
        normals = rng.normal(size=(30, 3))
        normals /= np.linalg.norm(normals, axis=1)[:, None]
        coords = 3.0 * normals + rng.normal(scale=0.3, size=(30, 3))
        areas = rng.random(30)
        points = rng.normal(scale=1.5, size=(20, 3))
        delta = 2.0
        sep = coords[None] - points[:, None]
        t = np.sum(sep**2, axis=2) ** 1.5 / delta**3
        ref = (np.einsum("pix,ix->pi", sep, normals) * np.arctan(t) / t) @ areas / (3 * delta**6)
        weights = rng.normal(size=20)
        J, dJ, dE_ds, dE_dA, dE_dn = _ne.dispersion_kernel(
            points, coords, normals, areas, delta, weights, 2
        )
        assert pytest.approx(ref, rel=1e-13) == J
        h = 1e-5
        plus, minus = points.copy(), points.copy()
        plus[4, 1] += h
        minus[4, 1] -= h
        Jp = _ne.dispersion_kernel(plus, coords, normals, areas, delta, None, 1)[0]
        Jm = _ne.dispersion_kernel(minus, coords, normals, areas, delta, None, 1)[0]
        assert dJ[4, 1] == pytest.approx((Jp[4] - Jm[4]) / (2 * h), rel=1e-7)

        def energy(c: FloatArray, a: FloatArray, n: FloatArray = normals) -> float:
            return float(weights @ _ne.dispersion_kernel(points, c, n, a, delta, None, 1)[0])

        cp, cm = coords.copy(), coords.copy()
        cp[7, 0] += h
        cm[7, 0] -= h
        assert dE_ds[7, 0] == pytest.approx(
            (energy(cp, areas) - energy(cm, areas)) / (2 * h), rel=1e-7
        )
        ap, am = areas.copy(), areas.copy()
        ap[2] += h
        am[2] -= h
        assert dE_dA[2] == pytest.approx(
            (energy(coords, ap) - energy(coords, am)) / (2 * h), rel=1e-9
        )
        npl, nmi = normals.copy(), normals.copy()  # a DROP surface's normals move
        npl[11, 2] += h
        nmi[11, 2] -= h
        assert dE_dn[11, 2] == pytest.approx(
            (energy(coords, areas, npl) - energy(coords, areas, nmi)) / (2 * h), rel=1e-7
        )

    @pytest.mark.parametrize("region", ["inside", "outside"])
    def test_partition_gradient(self, region: Any) -> None:
        centres = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 2.3], [2.4, 0.0, -1.5], [0.3, 0.2, 1.0]])
        radii = np.array([3.85, 3.45, 3.85, 2.0])
        centre = np.array([0.4, -0.3, 0.2])

        def field(p: FloatArray) -> FloatArray:
            return np.asarray(np.exp(-0.15 * np.sum((p - centre) ** 2, axis=1)))

        grid = build_cavity_grid(centres, radii, region, nthreads=2)
        analytic = partition_gradient(grid, centres, radii, field(grid.coords), 2)
        moving = (
            grid.weights[:, None] * (-0.3 * (grid.coords - centre)) * field(grid.coords)[:, None]
        )
        for a in range(len(centres)):
            analytic[a] += moving[grid.owners == a].sum(axis=0)
        h = 1e-6
        for atom, axis in [(1, 2), (3, 0)]:
            values = []
            for sign in (1.0, -1.0):
                moved = centres.copy()
                moved[atom, axis] += sign * h
                g = build_cavity_grid(moved, radii, region, nthreads=2)
                values.append(g.weights @ field(g.coords))
            assert analytic[atom, axis] == pytest.approx(
                (values[0] - values[1]) / (2 * h), rel=1e-6
            )


class TestTerms:
    def test_energy_is_the_pyscf_one(self) -> None:
        wfn = _scf(FORMATE)
        energies = _terms(wfn).energy(_density(wfn))
        assert (energies.dispersion, energies.exchange, energies.short_range) == pytest.approx(
            PYSCF_TERMS, rel=2e-9
        )

    def test_potential_and_explicit_gradient(self) -> None:
        wfn = _scf(FORMATE)
        terms, D = _terms(wfn), _density(wfn)
        _energies, V = terms.energy_and_potential(D)
        assert V is not None
        P = np.random.default_rng(1).normal(size=D.shape)
        P = 0.01 * (P + P.T)
        h = 1e-4
        numeric = (terms.energy(D + h * P).total - terms.energy(D - h * P).total) / (2 * h)
        assert np.sum(V * P) == pytest.approx(numeric, rel=1e-8)
        g = terms.gradient(D)
        assert g.sum(axis=0) == pytest.approx(0.0, abs=1e-10)
        # The cavity quadrature is C^2: its third derivative jumps where a node crosses
        # another sphere's surface, so the geometric differences need a small step.
        h = 1e-5
        values = []
        for sign in (1.0, -1.0):
            moved = FORMATE.copy()
            moved[1, 0] += sign * h * BOHR
            values.append(_terms(_scf(moved)).energy(D).total)
        assert g[1, 0] == pytest.approx((values[0] - values[1]) / (2 * h), abs=5e-8)


@pytest.mark.slow
class TestModes:
    """psi4.energy and psi4.gradient with the correction, against energy differences."""

    @pytest.mark.parametrize("self_consistent", [False, True])
    def test_gradient(self, self_consistent: bool) -> None:
        ne = NonelectrostaticSettings(**{**SETTINGS.__dict__, "self_consistent": self_consistent})
        psi4.set_options(OPTIONS)
        psi4_swig_pcm.enable(psi4_swig_pcm.PCMSettings(nonelectrostatic=ne))

        def energy(xyz: FloatArray) -> float:
            psi4.core.clean()
            _molecule(xyz)
            return float(psi4.energy("hf"))

        e = energy(FORMATE)
        reference = PYSCF_SELF_CONSISTENT if self_consistent else PYSCF_POST_SCF
        assert e == pytest.approx(reference, abs=1e-8)
        psi4.core.clean()
        _molecule(FORMATE)
        g = np.asarray(psi4.gradient("hf").np)
        assert core.variable("CURRENT ENERGY") == pytest.approx(e, abs=1e-10)
        assert g.sum(axis=0) == pytest.approx(0.0, abs=1e-9)
        # Small enough for the quadrature's third-derivative jumps, large enough that the
        # total energies' ~1e-11 of rounding stays below ~1e-7.
        h = 4e-5
        for atom, axis in [(0, 2), (3, 1)]:
            plus, minus = FORMATE.copy(), FORMATE.copy()
            plus[atom, axis] += h * BOHR
            minus[atom, axis] -= h * BOHR
            numeric = (energy(plus) - energy(minus)) / (2 * h)
            assert g[atom, axis] == pytest.approx(numeric, abs=3e-7)

    def test_post_scf_leaves_the_reference_variables(self) -> None:
        psi4.set_options(OPTIONS)
        psi4_swig_pcm.enable(psi4_swig_pcm.PCMSettings(nonelectrostatic=SETTINGS))
        _molecule(FORMATE)
        psi4.gradient("hf")
        scf = core.variable("SCF TOTAL ENERGY")
        assert core.variable("CURRENT ENERGY") == pytest.approx(
            scf + core.variable("NONELECTROSTATIC ENERGY"), abs=1e-12
        )
        assert scf == pytest.approx(PYSCF_POST_SCF - sum(PYSCF_TERMS), abs=1e-8)
