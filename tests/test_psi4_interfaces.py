# Copyright (C) 2026 Kent Gorday
# SPDX-License-Identifier: LGPL-3.0-only
"""One test per assumption about Psi4 internals, so that a Psi4 upgrade names what changed.

psi4-swig-pcm relies on:
  * the DDX solvation slot of scf_helper / scf_iterator (procrouting.solvent.ddx);
  * the SCF entries of driver.procedures["gradient"] and ["hessian"], and how the driver
    negotiates analytic vs finite-difference derivatives from them;
  * finite-difference Hessians running in this process (so they see the patched table);
  * Psi4 1.11 not computing DDX gradient terms itself;
  * ExternalPotential's Gaussian-charge API;
  * the C++ classes the extension calls (checked from its undefined symbols).
"""

from __future__ import annotations

import platform
import shutil
import subprocess
from typing import Any

import numpy as np
import psi4
import pytest
from psi4 import core
from psi4.driver import driver_util, procedures
from psi4.driver.procrouting import proc
from psi4.driver.procrouting import solvent as p4solvent

import psi4_swig_pcm
from psi4_swig_pcm import PCMSolver, _gcharge, driver
from psi4_swig_pcm import cavity as cav
from psi4_swig_pcm.cavity import FloatArray
from psi4_swig_pcm.charges import split_close_points
from tests.helpers import as_array, psi4_molecule
from tests.molecules import MOLECULES

STO3G = {"basis": "sto-3g", "scf_type": "pk", "e_convergence": 1e-10, "d_convergence": 1e-8}


def test_psi4_version_is_the_one_built_against() -> None:
    assert psi4.__version__.split(".")[:2] == psi4_swig_pcm.PSI4_VERSION.split(".")[:2]


def test_scf_uses_the_ddx_slot() -> None:
    """scf_helper builds DdxInterface(molecule=, options=, basisset=) and the iterator calls
    get_solvation_contributions(D, state), adding E and the returned matrix."""
    assert hasattr(p4solvent, "_have_ddx")
    psi4.set_options(STO3G)
    psi4_swig_pcm.enable()
    _, wfn = psi4.energy("hf", molecule=psi4_molecule("formate"), return_wfn=True)
    sol = wfn.ddx
    assert isinstance(sol, PCMSolver)
    assert sol is PCMSolver.last
    assert sol.calls > 2
    # Psi4 records the energy of the next-to-last call, so they agree to SCF convergence.
    assert wfn.variable("DD SOLVATION ENERGY") == pytest.approx(
        0.5 * sol.q_sym @ sol.phi, abs=1e-8
    )
    assert wfn.variable("DD SOLVATION ENERGY") < -0.05  # an anion in water


def test_scf_derivative_entries() -> None:
    for name in ("hf", "b3lyp", "wb97x-3c"):
        assert procedures["gradient"][name] is proc.select_scf_gradient
    for name in ("hf", "svwn"):
        assert procedures["hessian"][name] is proc.run_scf_hessian
    # The driver probes availability through select_* entries by name and probe=True.
    assert proc.select_scf_gradient.__name__.startswith("select_")
    psi4.set_options(STO3G)
    proc.select_scf_gradient("hf", probe=True)


def test_hessians_become_finite_differences_of_gradients() -> None:
    assert driver_util.negotiate_derivative_type("hessian", "hf", None, verbose=0) == (2, 2)
    with psi4_swig_pcm.enabled():
        assert driver_util.negotiate_derivative_type("hessian", "hf", None, verbose=0) == (2, 1)
        assert driver_util.negotiate_derivative_type("gradient", "hf", None, verbose=0) == (1, 1)
    assert driver_util.negotiate_derivative_type("hessian", "hf", None, verbose=0) == (2, 2)


def test_psi4_does_not_compute_ddx_gradient_terms(monkeypatch: pytest.MonkeyPatch) -> None:
    """Why the gradient table is patched: with only the DDX slot, psi4.gradient silently
    leaves the PCM terms out.  If this fails, Psi4 has started handling DDX gradients."""
    psi4.set_options(STO3G)
    mol = psi4_molecule("formate")
    psi4_swig_pcm.enable()
    g_pcm = as_array(psi4.gradient("hf", molecule=mol))
    monkeypatch.setitem(procedures["gradient"], "hf", proc.select_scf_gradient)
    g_plain = as_array(psi4.gradient("hf", molecule=mol))
    assert np.abs(g_plain - g_pcm).max() > 1e-3


def test_finite_difference_hessian_runs_in_process(monkeypatch: pytest.MonkeyPatch) -> None:
    """Displacements go through AtomicComputer -> qcengine with psiapi, i.e. this process."""
    calls: list[str] = []
    original = driver.gradient_from_wfn

    def counting(name: str, wfn: Any) -> FloatArray:
        calls.append(name)
        return original(name, wfn)

    monkeypatch.setattr(driver, "gradient_from_wfn", counting)
    psi4.set_options(STO3G)
    psi4_swig_pcm.enable()
    psi4.hessian("hf", molecule=psi4_molecule("formyl"), reference="uhf")
    assert len(calls) >= 2


def test_external_potential_gaussian_api() -> None:
    for attr in ("addGaussian", "gradients_on_gaussians", "computePotentialMatrix"):
        assert hasattr(core.ExternalPotential, attr), attr
    assert hasattr(core.BasisSet, "negative_gaussian_normalization_to_coefficients")


def test_symmetric_ghost_set_needs_the_anchor() -> None:
    """Psi4 cannot name the point group of this 4-point ghost set in a fixed frame.

    If this fails, Psi4 no longer trips over it: the anchor Gaussian (charges.py) may be
    unnecessary, and test_charges.test_symmetric_point_set_builds no longer exercises it.
    """
    mol = MOLECULES["formate_symmetric"]
    xyz = np.array(mol.angstrom) / psi4.constants.bohr2angstroms
    surf = cav.make_surface(xyz, mol.Z, cav.modified_bondi_radii())
    pts = surf.coords[split_close_points(surf.coords)[1]]
    with pytest.raises(RuntimeError, match="Unrecognized point group bits"):
        core.Molecule.from_arrays(
            units="Bohr",
            geom=pts,
            elem=["He"] * len(pts),
            real=[False] * len(pts),
            elbl=[f"@He_{i + 1}" for i in range(len(pts))],
            fix_symmetry="c1",
            fix_com=True,
            fix_orientation=True,
        )


# Psi4 C++ symbols the extension imports.  Anything else, or any Eigen/Boost type in its
# imports, means its ABI coupling to Psi4 has grown and the version pin needs a second look.
EXPECTED_PSI4_SYMBOLS = (
    "psi::BasisSet::shell(",
    "psi::BasisSet::zero_ao_basis_set(",
    "psi::Dimension::Dimension(",
    "psi::Dimension::sum(",
    "psi::GaussianShell::nfunction(",
    "psi::IntegralFactory::IntegralFactory(",
    "psi::Matrix::Matrix(",
    "psi::Matrix::add(",
    "psi::Matrix::zero(",
    "typeinfo for psi::Matrix",
)


@pytest.mark.skipif(shutil.which("nm") is None, reason="needs nm")
def test_extension_imports_only_expected_psi4_symbols() -> None:
    args = ["nm", "-u", "-C"] if platform.system() == "Darwin" else ["nm", "-D", "-u", "-C"]
    out = subprocess.run([*args, _gcharge.__file__], capture_output=True, text=True, check=True)
    # macOS prints bare names; GNU nm prints "U name"
    lines = [line.strip() for line in out.stdout.splitlines() if line.strip()]
    undefined = [s.removeprefix("U ") for s in lines]
    psi4_syms = [s for s in undefined if "psi::" in s]
    assert psi4_syms
    assert [s for s in psi4_syms if not s.startswith(EXPECTED_PSI4_SYMBOLS)] == []
    assert [s for s in undefined if "Eigen" in s or "boost" in s] == []
