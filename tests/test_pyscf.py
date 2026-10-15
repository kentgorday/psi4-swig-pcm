# Copyright (C) 2026 Kent Gorday
# SPDX-License-Identifier: LGPL-3.0-only
"""Against PySCF 2.11's pyscf.solvent.pcm, the implementation the cavity derives from.

Runs in the crosscheck environment (pixi run -e crosscheck crosscheck).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import psi4
import pytest
from scipy.spatial import cKDTree

import psi4_swig_pcm
from psi4_swig_pcm import PCMSettings
from psi4_swig_pcm import cavity as cav
from psi4_swig_pcm.cavity import FloatArray, IntArray, Method
from tests.helpers import EPS, HF_OPTIONS, as_array, psi4_molecule
from tests.molecules import MOLECULES

pyscf = pytest.importorskip("pyscf")
from pyscf import dft, gto, scf  # noqa: E402
from pyscf.solvent import pcm  # noqa: E402
from pyscf.solvent.grad import pcm as pcm_grad  # noqa: E402

pytestmark = pytest.mark.pyscf

METHODS: list[Method] = ["C-PCM", "IEF-PCM", "SS(V)PE"]


def pyscf_mol(name: str, basis: str) -> Any:
    m = MOLECULES[name]
    atoms = list(zip(m.symbols, m.bohr.tolist(), strict=True))
    return gto.M(atom=atoms, unit="Bohr", basis=basis, charge=m.charge, verbose=0)


def pyscf_pcm(name: str, method: Method) -> tuple[Any, Any]:
    mol = pyscf_mol(name, "sto-3g")
    ws = pcm.PCM(mol)
    ws.method, ws.eps, ws.radii_table = method, EPS, cav.modified_bondi_radii()
    ws.build()
    return mol, ws


def ours(name: str, method: Method) -> tuple[cav.Surface, cav.PCMOperator]:
    m = MOLECULES[name]
    surf = cav.make_surface(m.bohr, m.Z, cav.modified_bondi_radii())
    return surf, cav.PCMOperator(surf, method, EPS)


def permutation(ws: Any, surf: cav.Surface) -> IntArray:
    """perm with ours[perm] in PySCF's point order (scipy and PySCF order Lebedev points
    differently)."""
    d, perm = cKDTree(surf.coords).query(ws.surface["grid_coords"])
    assert d.max() < 1e-10
    assert len(set(perm)) == surf.npts == len(perm)
    out: IntArray = perm
    return out


def test_default_radii_are_pyscfs() -> None:
    """modified_bondi_radii() is the radii table PySCF's PCM uses by default, every element."""
    from pyscf.data import radii

    ws = pcm.PCM(pyscf_mol("formate", "sto-3g"))
    expected = ws.vdw_scale * pcm.modified_Bondi + ws.r_probe / radii.BOHR
    np.testing.assert_allclose(cav.modified_bondi_radii(), expected, rtol=1e-15, atol=0)


@pytest.mark.parametrize("name", ["formate", "acetone", "glycine_zwitterion"])
def test_surface_matches_pyscf(name: str) -> None:
    _, ws = pyscf_pcm(name, "C-PCM")
    surf, _ = ours(name, "C-PCM")
    p = permutation(ws, surf)
    s = ws.surface
    for mine, ref in [
        (surf.switch, s["switch_fun"]),
        (surf.xi, s["charge_exp"]),
        (surf.area, s["area"]),
        (surf.normals, s["norm_vec"]),
        (surf.R_vdw, s["R_vdw"]),
    ]:
        np.testing.assert_allclose(mine[p], ref, rtol=1e-12, atol=1e-14)
    atom_ref = np.concatenate([np.full(b - a, i) for i, (a, b) in enumerate(s["gslice_by_atom"])])
    np.testing.assert_array_equal(surf.atom_of_point[p], atom_ref)


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("name", ["formate", "acetone"])
def test_operator_and_cavity_gradient_match_pyscf(name: str, method: Method) -> None:
    mol, ws = pyscf_pcm(name, method)
    surf, op = ours(name, method)
    p = permutation(ws, surf)
    inter = ws._intermediates
    K, R = inter["K"], inter["R"]
    np.testing.assert_allclose(op.K[np.ix_(p, p)], K, rtol=1e-11, atol=1e-11)
    np.testing.assert_allclose(op.R[np.ix_(p, p)], R, rtol=1e-11, atol=1e-11)

    # A potential of solute-like size and shape (screened nuclei plus noise), in our order
    rng = np.random.default_rng(0)
    phi: FloatArray = 0.1 * cav.nuclear_potential(surf, mol.atom_charges())
    phi += 0.01 * rng.standard_normal(surf.npts)
    e, q, q_sym = op.energy(phi)
    q_ref = np.linalg.solve(K, R @ phi[p])
    q_sym_ref = 0.5 * (q_ref + R.T @ np.linalg.solve(K.T, phi[p]))
    np.testing.assert_allclose(q[p], q_ref, rtol=1e-9, atol=1e-12)
    np.testing.assert_allclose(q_sym[p], q_sym_ref, rtol=1e-9, atol=1e-12)
    assert abs(e - 0.5 * q_sym_ref @ phi[p]) < 1e-12

    # Cavity gradient (blocked, tiny blocks) vs grad_solver with the same phi and q
    inter.update(v_grids=phi[p], q=q_ref, dm=np.zeros(1))
    g_ref = pcm_grad.grad_solver(ws, np.zeros(1))
    g = cav.cavity_gradient(op, phi, q, block_mb=1)
    np.testing.assert_allclose(g, g_ref, rtol=1e-8, atol=1e-11)


def pyscf_reference(name: str, method: Method | None, xc: str = "hf") -> tuple[float, FloatArray]:
    """(E_solv, gradient) from PySCF with conventional integrals; method=None for gas phase."""
    mol = pyscf_mol(name, "def2-svp")
    if xc == "hf":
        mf = scf.RHF(mol)
    else:
        mf = dft.RKS(mol, xc=xc)
        mf.grids.atom_grid, mf.grids.prune = (99, 590), None
    if method is not None:
        mf = mf.PCM()
        ws = mf.with_solvent
        ws.method, ws.eps, ws.radii_table = method, EPS, cav.modified_bondi_radii()
    mf.conv_tol, mf.conv_tol_grad = 1e-12, 1e-8
    mf.kernel()
    e_solv = float(mf.with_solvent.e) if method is not None else 0.0
    g: FloatArray = mf.nuc_grad_method().kernel()
    return e_solv, g


@pytest.mark.slow
@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("name", ["formate", "methylammonium", "glycine_zwitterion"])
def test_hf_energy_and_gradient_match_pyscf(name: str, method: Method) -> None:
    """Hartree-Fock in both codes: no DFT grids, so the PCM terms are compared directly."""
    psi4.set_options(HF_OPTIONS)
    psi4_swig_pcm.enable(PCMSettings(method=method, eps=EPS))
    G, wfn = psi4.gradient("hf", molecule=psi4_molecule(name), return_wfn=True)
    e_solv_ref, g_ref = pyscf_reference(name, method)
    assert abs(wfn.variable("DD SOLVATION ENERGY") - e_solv_ref) < 1e-9
    np.testing.assert_allclose(as_array(G), g_ref, atol=5e-8)


@pytest.mark.slow
@pytest.mark.parametrize("method", ["IEF-PCM", "SS(V)PE"])
def test_dft_solvent_gradient_matches_pyscf(method: Method) -> None:
    """PBE0: the codes' DFT grids differ (~1e-6 in gradients), so compare g(PCM) - g(gas)."""
    psi4.set_options(
        {
            **HF_OPTIONS,
            "e_convergence": 1e-11,
            "d_convergence": 1e-9,
            "dft_radial_points": 99,
            "dft_spherical_points": 590,
            "dft_pruning_scheme": "none",
        }
    )
    mol = psi4_molecule("glycine_zwitterion")
    g_gas = as_array(psi4.gradient("pbe0", molecule=mol))
    psi4_swig_pcm.enable(PCMSettings(method=method, eps=EPS))
    g_pcm = as_array(psi4.gradient("pbe0", molecule=mol))
    _, g_pcm_ref = pyscf_reference("glycine_zwitterion", method, xc="pbe0")
    _, g_gas_ref = pyscf_reference("glycine_zwitterion", None, xc="pbe0")
    np.testing.assert_allclose(g_pcm - g_gas, g_pcm_ref - g_gas_ref, atol=3e-7)
