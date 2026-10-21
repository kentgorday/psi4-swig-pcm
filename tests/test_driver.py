# Copyright (C) 2026 Kent Gorday
# SPDX-License-Identifier: LGPL-3.0-only
"""enable()/disable() and Psi4's own driver with the PCM: gradient, optimize, frequency."""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import psi4
import pytest
from psi4 import core
from psi4.driver import procedures
from psi4.driver.procrouting import solvent as p4solvent

import psi4_swig_pcm
from psi4_swig_pcm import PCMSettings, PCMSolver, driver
from tests.helpers import EPS, HF_OPTIONS, as_array, psi4_molecule
from tests.molecules import MOLECULES

STO3G = {"basis": "sto-3g", "scf_type": "pk", "e_convergence": 1e-10, "d_convergence": 1e-8}


def test_enable_and_disable_restore_psi4() -> None:
    gradient, hessian = dict(procedures["gradient"]), dict(procedures["hessian"])
    have_ddx, had_ddx = p4solvent._have_ddx, hasattr(p4solvent, "ddx")
    settings = PCMSettings(method="C-PCM")

    psi4_swig_pcm.enable(settings)
    assert psi4_swig_pcm.active_settings() is settings
    assert core.get_option("SCF", "DDX")
    assert procedures["gradient"]["hf"] is not gradient["hf"]
    assert "hf" not in procedures["hessian"]
    psi4_swig_pcm.enable()  # re-enabling replaces, it does not stack
    assert psi4_swig_pcm.active_settings() == PCMSettings()

    psi4_swig_pcm.disable()
    assert psi4_swig_pcm.active_settings() is None
    assert not core.get_option("SCF", "DDX")
    assert procedures["gradient"] == gradient
    assert procedures["hessian"] == hessian
    assert p4solvent._have_ddx == have_ddx
    assert hasattr(p4solvent, "ddx") == had_ddx
    psi4_swig_pcm.disable()  # a no-op


def test_enabled_context_manager() -> None:
    with psi4_swig_pcm.enabled(PCMSettings(eps=4.0)):
        settings = psi4_swig_pcm.active_settings()
        assert isinstance(settings, PCMSettings)
        assert settings.eps == 4.0
    assert psi4_swig_pcm.active_settings() is None


def test_psi4_gradient_is_the_pcm_gradient() -> None:
    psi4.set_options(HF_OPTIONS)
    mol = psi4_molecule("formate")
    psi4_swig_pcm.enable()
    _, wfn = psi4.energy("hf", molecule=mol, return_wfn=True)
    g_direct = driver.gradient_from_wfn("hf", wfn)
    G, wfn_g = psi4.gradient("hf", molecule=mol, return_wfn=True)
    g = as_array(G)
    np.testing.assert_allclose(g, g_direct, atol=1e-9)
    np.testing.assert_allclose(as_array(wfn_g.variable("CURRENT GRADIENT")), g, atol=0)
    assert np.abs(g.sum(0)).max() < 1e-7  # translational invariance


def test_gas_phase_is_untouched_with_ddx_off() -> None:
    psi4.set_options(STO3G)
    mol = psi4_molecule("formate")
    g_gas = as_array(psi4.gradient("hf", molecule=mol))
    psi4_swig_pcm.enable()
    psi4.set_options({"ddx": False})
    # threaded SCFs repeat to ~1e-14, not bit for bit
    np.testing.assert_allclose(as_array(psi4.gradient("hf", molecule=mol)), g_gas, atol=1e-11)


def test_settings_reach_the_solver() -> None:
    psi4.set_options(STO3G)
    mol = psi4_molecule("formate")
    energies = {}
    for eps in (EPS, float("inf"), 4.0):
        psi4_swig_pcm.enable(PCMSettings(method="C-PCM", eps=eps, lebedev_order=17))
        _, wfn = psi4.energy("hf", molecule=mol, return_wfn=True)
        sol = wfn.ddx
        assert isinstance(sol, PCMSolver)
        assert sol.settings.eps == eps
        assert sol.surf.npts < 4 * 110  # Lebedev order 17: 110 points per sphere
        energies[eps] = wfn.variable("DD SOLVATION ENERGY")
    assert energies[float("inf")] < energies[EPS] < energies[4.0] < 0


def test_restricted_open_shell_is_rejected() -> None:
    psi4.set_options({**STO3G, "reference": "rohf"})
    psi4_swig_pcm.enable()
    with pytest.raises(NotImplementedError, match="ROHF"):
        psi4.gradient("hf", molecule=psi4_molecule("formyl"))


def test_enable_refuses_other_settings() -> None:
    """A TypeError, with or without moist installed (moist_host is never imported for it)."""
    with pytest.raises(TypeError, match="PCMSettings or MoistSettings"):
        psi4_swig_pcm.enable("IEF-PCM")  # type: ignore[arg-type]
    assert psi4_swig_pcm.active_settings() is None
    assert not core.get_option("SCF", "DDX")


def test_failed_enable_leaves_psi4_untouched(monkeypatch: pytest.MonkeyPatch) -> None:
    gradient, hessian = dict(procedures["gradient"]), dict(procedures["hessian"])

    def broken(original: object) -> object:
        raise RuntimeError("broken")

    monkeypatch.setattr(driver, "_wrap_scf_gradient", broken)
    with pytest.raises(RuntimeError, match="broken"):
        psi4_swig_pcm.enable()
    assert psi4_swig_pcm.active_settings() is None
    assert not core.get_option("SCF", "DDX")
    assert procedures["gradient"] == gradient
    assert procedures["hessian"] == hessian


def test_psi4_output_reports_the_pcm(tmp_path: Path) -> None:
    out = tmp_path / "pcm.out"
    psi4.core.set_output_file(str(out), False)
    try:
        psi4.set_options(STO3G)
        psi4_swig_pcm.enable(PCMSettings(method="C-PCM"))
        psi4.energy("hf", molecule=psi4_molecule("formate"))
    finally:
        psi4.core.be_quiet()
    text = out.read_text()
    assert "==> psi4-swig-pcm: C-PCM, eps = 78.3553 <==" in text
    assert re.search(
        r"Cavity: 4 atoms, Lebedev order 29, SWIG switching, \d+ surface points", text
    )
    assert "Nuclear surface charge vs Gauss's law" in text


def test_gradient_leaves_no_external_potential_behind() -> None:
    psi4.set_options(STO3G)
    psi4_swig_pcm.enable()
    _, wfn = psi4.gradient("hf", molecule=psi4_molecule("formate"), return_wfn=True)
    assert wfn.external_pot() is None


def test_ghost_atoms_have_no_sphere() -> None:
    psi4.set_options(STO3G)
    psi4_swig_pcm.enable()
    _, alone = psi4.energy("hf", molecule=psi4_molecule("water"), return_wfn=True)
    _, ghosted = psi4.energy("hf", molecule=psi4_molecule("water_ghost_water"), return_wfn=True)
    assert ghosted.ddx.surf.npts == alone.ddx.surf.npts
    np.testing.assert_array_equal(ghosted.ddx.atoms, [0, 1, 2])


@pytest.mark.slow
def test_optimize_and_frequencies() -> None:
    """optking through psi4.optimize; then finite differences of PCM gradients vs energies.

    Formate converged this tightly passes through the geometry whose surface has a symmetric
    point set (molecules.py), which made the Gaussian-charge Molecule fail before the anchor.
    """
    psi4.set_options({**HF_OPTIONS, "g_convergence": "gau_verytight", "geom_maxiter": 50})
    mol = psi4_molecule("formate")
    psi4_swig_pcm.enable()
    psi4.optimize("hf", molecule=mol)
    assert np.abs(as_array(psi4.gradient("hf", molecule=mol))).max() < 2e-6

    _, wfn_g = psi4.frequency("hf", molecule=mol, return_wfn=True)
    _, wfn_e = psi4.frequency("hf", molecule=mol, dertype=0, return_wfn=True)
    f_g = wfn_g.frequencies().to_array()
    f_e = wfn_e.frequencies().to_array()
    assert len(f_g) == 6
    assert (f_g > 0).all()
    np.testing.assert_allclose(f_g, f_e, atol=0.05)


@pytest.mark.slow
def test_psi4_executable_input_file(tmp_path: Path) -> None:
    """The way tautomator runs Psi4: the psi4 executable on an input file."""
    mol = MOLECULES["formate"]
    atoms = "\n".join(
        f"{s} {x:.10f} {y:.10f} {z:.10f}"
        for s, (x, y, z) in zip(mol.symbols, mol.bohr, strict=True)
    )
    (tmp_path / "input.dat").write_text(
        f"""
import psi4_swig_pcm
psi4_swig_pcm.enable(psi4_swig_pcm.PCMSettings(method="IEF-PCM"))

molecule {{
{mol.charge} {mol.multiplicity}
units bohr
symmetry c1
no_com
no_reorient
{atoms}
}}
set basis def2-svp
set scf_type pk
set e_convergence 1e-10
set d_convergence 1e-8
G, wfn = gradient("hf", return_wfn=True)
e_solv = wfn.variable("DD SOLVATION ENERGY")
print_out(f"\\n=== E {{wfn.energy():.10f}} SOLV {{e_solv:.10f}} ===\\n")
"""
    )
    psi4_exe = Path(sys.prefix) / "bin" / "psi4"
    subprocess.run(
        [str(psi4_exe), "-n", "2", "input.dat", "output.dat"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        timeout=600,
    )
    # Numbers only: output.dat also echoes the input file, f-string template included.
    pattern = r"=== E (-?\d+\.\d+) SOLV (-?\d+\.\d+) ==="
    m = re.search(pattern, (tmp_path / "output.dat").read_text())
    assert m is not None

    psi4.set_options({**HF_OPTIONS, "e_convergence": 1e-10, "d_convergence": 1e-8})
    psi4_swig_pcm.enable()
    e, wfn = psi4.energy("hf", molecule=psi4_molecule("formate"), return_wfn=True)
    assert float(m.group(1)) == pytest.approx(e, abs=1e-9)
    assert float(m.group(2)) == pytest.approx(wfn.variable("DD SOLVATION ENERGY"), abs=1e-9)
