# Copyright (C) 2026 Kent Gorday
# SPDX-License-Identifier: LGPL-3.0-only
"""Analytic PCM gradients against finite differences of PCM energies (all slow)."""

from __future__ import annotations

from functools import partial

import numpy as np
import psi4
import pytest

import psi4_swig_pcm
from psi4_swig_pcm import PCMSettings, PCMSolver
from psi4_swig_pcm.cavity import Method, Switching
from tests.helpers import EPS, HF_OPTIONS, as_array, displaced, five_point, psi4_molecule
from tests.molecules import MOLECULES

DFT_OPTIONS = {
    "basis": "def2-svp",
    "scf_type": "pk",
    "e_convergence": 1e-11,
    "d_convergence": 1e-9,
    "dft_radial_points": 99,
    "dft_spherical_points": 590,
    "dft_pruning_scheme": "none",
}


@pytest.mark.slow
@pytest.mark.parametrize(
    ("method", "name", "atoms", "switching"),
    [
        ("SS(V)PE", "formate", [0, 1, 3], "SWIG"),
        ("IEF-PCM", "formate", [1, 3], "SWIG"),
        ("C-PCM", "methylammonium", [1, 2], "SWIG"),
        ("COSMO", "methylammonium", [2], "SWIG"),
        ("SS(V)PE", "glycine_zwitterion", [2, 3], "SWIG"),
        ("IEF-PCM", "glycine_zwitterion", [2, 3], "ISWIG"),
        ("IEF-PCM", "iodomethane", [1, 2], "SWIG"),  # def2-SVP ECP on iodine
        ("IEF-PCM", "formyl", [1, 2], "SWIG"),  # UHF
    ],
)
def test_gradient_matches_finite_differences(
    method: Method, name: str, atoms: list[int], switching: Switching
) -> None:
    """Hartree-Fock, so no DFT grid-weight terms blur the comparison.

    With an ECP, Psi4's gas-phase HF gradient itself differs from finite differences by
    ~6e-8 along the C-I bond, so there the PCM residual is taken relative to the gas one.
    """
    reference = "uhf" if MOLECULES[name].multiplicity > 1 else "rhf"
    psi4.set_options({**HF_OPTIONS, "reference": reference})
    xyz = MOLECULES[name].bohr
    h = 2e-3

    def energy_at(atom: int, axis: int, step: float) -> float:
        e: float = psi4.energy(
            "hf", molecule=psi4_molecule(name, displaced(xyz, atom, axis, step))
        )
        return e

    def residuals() -> dict[tuple[int, int], float]:
        g = as_array(psi4.gradient("hf", molecule=psi4_molecule(name)))
        return {
            (a, c): g[a, c] - five_point(partial(energy_at, a, c), h)
            for a in atoms
            for c in range(3)
        }

    gas = residuals() if name == "iodomethane" else {}
    psi4_swig_pcm.enable(PCMSettings(method=method, eps=EPS, switching=switching))
    for key, resid in residuals().items():
        assert abs(resid - gas.get(key, 0.0)) < 1e-8, (key, resid)


@pytest.mark.slow
def test_incore_matches_direct_scf() -> None:
    """Stored (k|mn) and the direct contractions give the same SCF and gradient, through the
    custom (dict) functional path."""
    psi4.set_options(DFT_OPTIONS)
    xc = {"name": "pbe0-dict", "xc_functionals": {"HYB_GGA_XC_PBEH": {}}}
    runs = {}
    for incore_mb in (0.0, 2000.0):
        psi4_swig_pcm.enable(PCMSettings(incore_mb=incore_mb))
        G, wfn = psi4.gradient(
            "scf", molecule=psi4_molecule("glycine_zwitterion"), dft_functional=xc, return_wfn=True
        )
        assert isinstance(wfn.ddx, PCMSolver)
        assert (wfn.ddx.charges.B is not None) == (incore_mb > 0)
        runs[incore_mb] = (wfn.energy(), as_array(G))
    assert abs(runs[0.0][0] - runs[2000.0][0]) < 1e-10
    np.testing.assert_allclose(runs[0.0][1], runs[2000.0][1], atol=1e-9)


@pytest.mark.slow
@pytest.mark.parametrize(("name", "method", "coords"), [
    ("water_ghost_water", "hf", [(0, 2), (4, 2)]),  # a real and a ghost atom
    ("water", "mp2", [(0, 2), (1, 1)]),  # finite differences of energies inside psi4.gradient
])  # fmt: skip
def test_gradient_matches_finite_differences_of_pcm_energies(
    name: str, method: str, coords: list[tuple[int, int]]
) -> None:
    # points 5: psi4.gradient("mp2") is then a 5-point finite difference of PCM energies
    psi4.set_options({**HF_OPTIONS, "basis": "sto-3g", "mp2_type": "conv", "points": 5})
    xyz = MOLECULES[name].bohr
    psi4_swig_pcm.enable()
    g = as_array(psi4.gradient(method, molecule=psi4_molecule(name)))

    def energy_at(atom: int, axis: int, step: float) -> float:
        mol = psi4_molecule(name, displaced(xyz, atom, axis, step))
        e: float = psi4.energy(method, molecule=mol)
        return e

    # The analytic MP2 gradient, PCM left out, was 2e-3 off.
    tol = 1e-8 if method == "hf" else 1e-7
    for atom, axis in coords:
        fd = five_point(partial(energy_at, atom, axis), 2e-3)
        assert abs(g[atom, axis] - fd) < tol, (atom, axis, g[atom, axis], fd)
