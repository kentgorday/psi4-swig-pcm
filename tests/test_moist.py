# Copyright (C) 2026 Kent Gorday
# SPDX-License-Identifier: LGPL-3.0-only
"""moist cavities and models through moist_host (prototype): Psi4 as moist's host.

Runs in the moist environment (pixi run -e moist test-moist, test-moist-slow); the PySCF
comparisons in moist-crosscheck (pixi run -e moist-crosscheck crosscheck-moist).
"""

from __future__ import annotations

import dataclasses
import gc
import os
import weakref
from collections.abc import Callable
from functools import partial
from types import SimpleNamespace
from typing import Any

import numpy as np
import psi4
import pytest
from psi4 import core

import psi4_swig_pcm
from psi4_swig_pcm.cavity import modified_bondi_radii
from psi4_swig_pcm.charges import GaussianCharges
from tests.helpers import HF_OPTIONS, as_array, displaced, five_point, psi4_molecule
from tests.molecules import MOLECULES

moist = pytest.importorskip("moist")
from psi4_swig_pcm import moist_host  # noqa: E402
from psi4_swig_pcm.moist_host import CartesianAOs, MoistSettings, MoistSolver  # noqa: E402

pytestmark = pytest.mark.moist

EPS = 78.3553
NLEB = 194

CAVITIES: dict[str, Callable[[], Any]] = {
    "svdw": lambda: moist.DROP(lsf=moist.SvdW(), parameters=moist.DROPParameters(nleb=NLEB)),
    "isodensity": lambda: moist.DROP(
        lsf=moist.Isodensity(), parameters=moist.DROPParameters(nleb=NLEB)
    ),
    "cfc": lambda: moist.DROP(lsf=moist.CFC(), parameters=moist.DROPParameters(nleb=NLEB)),
    "iswig": lambda: moist.ISwiG(),
}


COMPONENTS: dict[str, Callable[[], Any]] = {
    "cpcm": lambda: moist.ModelComponentCPCM(EPS),
    "iefpcm": lambda: moist.ModelComponentIEFPCM(EPS),
    "ssvpe": lambda: moist.ModelComponentSSVPE(EPS),
    "lpb": lambda: moist.ModelComponentLPB(EPS, 0.1),
    "dpcm": lambda: moist.ModelComponentDPCM(EPS),
}


def settings(cavity: str, component: str = "cpcm") -> MoistSettings:
    return MoistSettings(cavity=CAVITIES[cavity](), components=(COMPONENTS[component](),))


def test_cartesian_aos_reproduce_psi4_basis_functions() -> None:
    """The monomials times T are Psi4's (pure, up to f) functions; derivatives to 3rd order."""
    bs = core.BasisSet.build(psi4_molecule("water"), "ORBITAL", "def2-tzvp")
    aos = CartesianAOs(bs, {0: 0, 1: 1, 2: 2})
    live = settings("isodensity").cavity.build(
        basis=aos.gaussian_basis, context=moist.Context(verbosity=0)
    )
    aos.bind_layout(*live.isodensity_layout())
    rng = np.random.default_rng(4)
    pts = rng.normal(size=(30, 3)) * 1.5
    vals = aos.values(pts, 3)
    ref = np.array([np.asarray(bs.compute_phi(*p)) for p in pts])
    np.testing.assert_allclose(vals[0, 0, 0] @ aos.T, ref, atol=1e-12 * np.abs(ref).max())
    h = 1e-5
    for key, v in vals.items():
        if sum(key) == 0:
            continue
        axis = next(a for a in range(3) if key[a] > 0)
        lower = (key[0] - (axis == 0), key[1] - (axis == 1), key[2] - (axis == 2))
        step = np.zeros(3)
        step[axis] = h
        fd = (aos.values(pts + step, 2)[lower] - aos.values(pts - step, 2)[lower]) / (2 * h)
        np.testing.assert_allclose(v, fd, atol=1e-6 * np.abs(v).max())


def test_cartesian_aos_blocks_add_up(monkeypatch: pytest.MonkeyPatch) -> None:
    """The isodensity Fock term and gradient are the same with the grid in many blocks."""
    bs = core.BasisSet.build(psi4_molecule("water"), "ORBITAL", "def2-svp")
    aos = CartesianAOs(bs, {0: 0, 1: 1, 2: 2})
    live = settings("isodensity").cavity.build(
        basis=aos.gaussian_basis, context=moist.Context(verbosity=0)
    )
    aos.bind_layout(*live.isodensity_layout())
    rng = np.random.default_rng(5)
    pts = rng.normal(size=(40, 3)) * 2.0
    w2 = rng.normal(size=(40, 3, 3))
    item = SimpleNamespace(
        w_rho=rng.normal(size=40), w_grad_rho=rng.normal(size=(40, 3)), w_hess_rho=w2 + w2.mT
    )
    D = rng.normal(size=(aos.ncart, aos.ncart))
    D = D + D.T
    F, g = aos.fock(pts, item), aos.gradient(pts, D, item, 3)
    monkeypatch.setattr(moist_host, "BLOCK_BYTES", 32 * aos.ncart * 8 * 7)  # 7 points
    assert len(list(aos._blocks(len(pts), 32))) == 6
    np.testing.assert_allclose(aos.fock(pts, item), F, rtol=1e-12, atol=1e-12 * abs(F).max())
    np.testing.assert_allclose(aos.gradient(pts, D, item, 3), g, rtol=1e-12, atol=1e-12)


def test_settings_are_checked() -> None:
    cavity = CAVITIES["svdw"]()
    with pytest.raises(TypeError, match="cavity"):
        MoistSettings(cavity=object(), components=(COMPONENTS["cpcm"](),))
    with pytest.raises(TypeError, match="components"):
        MoistSettings(cavity=cavity, components=())
    with pytest.raises(TypeError, match="components"):
        MoistSettings(cavity=cavity, components=(object(),))


@pytest.mark.parametrize("cavity", list(CAVITIES))
def test_cavity_is_built_on_the_pcm_spheres(cavity: str) -> None:
    """MoistSettings.radii, as PCMSettings.radii and with its default, go into moist's cavity
    in place of moist's own CPCM radii; radii=None keeps the cavity's."""
    component = (COMPONENTS["cpcm"](),)
    s = MoistSettings(cavity=CAVITIES[cavity](), components=component)
    radii = s.radii
    assert radii is not None and radii == psi4_swig_pcm.PCMSettings().radii
    assert s.cavity.radii == moist.CustomRadii(radii[1:], numbers=range(1, len(radii)))
    assert dataclasses.replace(s, nthreads=2).cavity == s.cavity
    wider = MoistSettings(
        cavity=CAVITIES[cavity](),
        components=component,
        radii=modified_bondi_radii(hydrogen=1.2),  # type: ignore[arg-type]  # any sequence
    ).radii
    assert wider is not None and wider[1] == pytest.approx(radii[1] * 1.2 / 1.1)
    assert wider[2:] == radii[2:]
    own = MoistSettings(cavity=CAVITIES[cavity](), components=component, radii=None)
    assert own.cavity.radii == moist.CPCMRadii() and own.radii is None
    bondi = dataclasses.replace(CAVITIES[cavity](), radii=moist.BondiRadii())
    with pytest.raises(ValueError, match="radii of its own"):
        MoistSettings(cavity=bondi, components=component)
    assert MoistSettings(cavity=bondi, components=component, radii=None).cavity == bondi


@pytest.mark.parametrize("cavity", list(CAVITIES))
def test_energy_runs_through_the_ddx_slot(cavity: str) -> None:
    psi4.set_options(HF_OPTIONS)
    psi4_swig_pcm.enable(settings(cavity))
    E, wfn = psi4.energy("hf", molecule=psi4_molecule("formate"), return_wfn=True)
    assert isinstance(wfn.ddx, MoistSolver)
    assert wfn.ddx.charges.B is not None  # (k|mn) stored, on a grid that moves or not
    E_solv = wfn.variable("DD SOLVATION ENERGY")
    assert -0.2 < E_solv < -0.05  # an anion in water
    psi4_swig_pcm.disable()
    E_gas = psi4.energy("hf", molecule=psi4_molecule("formate"))
    assert E_gas > E


def test_finished_scf_frees_its_wavefunction_and_solver() -> None:
    """Nothing keeps a finished SCF's wavefunction and MoistSolver alive (driver.py, 6.)."""
    psi4.set_options(HF_OPTIONS)
    psi4_swig_pcm.enable(settings("svdw"))
    _, wfn = psi4.gradient("hf", molecule=psi4_molecule("water"), return_wfn=True)
    assert isinstance(wfn.ddx, MoistSolver)
    refs = weakref.ref(wfn), weakref.ref(wfn.ddx)
    del wfn
    gc.collect()
    assert [ref() for ref in refs] == [None, None]


@pytest.mark.parametrize("cavity", ["svdw", "isodensity"])
def test_gradient_makes_one_derivative_pass(cavity: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """One pass of derivative integrals gives the gradient its dphi_dr (if moist asks: on a
    fixed cavity) and the basis-centre term."""
    psi4.set_options(HF_OPTIONS)
    psi4_swig_pcm.enable(settings(cavity))
    _, wfn = psi4.energy("hf", molecule=psi4_molecule("water"), return_wfn=True)
    calls: list[bool] = []
    original = GaussianCharges.esp_deriv1

    def counted(self: GaussianCharges, *args: Any, **kwargs: Any) -> Any:
        calls.append(True)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(GaussianCharges, "esp_deriv1", counted)
    psi4.gradient("hf", molecule=psi4_molecule("water"), ref_wfn=wfn)
    assert len(calls) == 1


@pytest.mark.parametrize(
    ("cavity", "match"),
    [("svdw", "w_dphi_dr"), ("isodensity", "d2phi_dr2")],  # _fock; _answer_all
)
def test_normal_derivative_models_are_refused(cavity: str, match: str) -> None:
    """DPCM needs w_dphi_dr and second derivatives of the potential, not here yet."""
    psi4.set_options(HF_OPTIONS)
    psi4_swig_pcm.enable(settings(cavity, "dpcm"))
    with pytest.raises(NotImplementedError, match=match) as raised:
        psi4.energy("hf", molecule=psi4_molecule("water"))
    assert "DPCM" in str(raised.value)


def test_ghost_atoms_are_refused_on_an_isodensity_cavity() -> None:
    psi4.set_options(HF_OPTIONS)
    psi4_swig_pcm.enable(settings("isodensity"))
    with pytest.raises(NotImplementedError, match="ghost"):
        psi4.energy("hf", molecule=psi4_molecule("water_ghost_water"))


@pytest.mark.slow
@pytest.mark.parametrize(
    ("cavity", "component", "name", "atoms"),
    [
        ("svdw", "cpcm", "formate", [0, 1, 3]),
        ("isodensity", "cpcm", "formate", [0, 1, 3]),
        ("svdw", "cpcm", "formyl", [1, 2]),  # UHF
        ("isodensity", "cpcm", "formyl", [1, 2]),
        ("svdw", "iefpcm", "formate", [1, 3]),
        ("isodensity", "iefpcm", "formate", [1, 3]),
        ("cfc", "cpcm", "formate", [1, 3]),
        ("iswig", "cpcm", "formate", [1, 3]),
        ("isodensity", "lpb", "formate", [1]),
        ("svdw", "cpcm", "water_ghost_water", [0, 4]),  # 4: a ghost, basis functions only
        ("svdw", "cpcm", "iodomethane", [1]),  # iodine's ECP: reduced nuclear charge
        ("isodensity", "cpcm", "iodomethane", [1]),
    ],
)
def test_gradient_matches_finite_differences(
    cavity: str, component: str, name: str, atoms: list[int]
) -> None:
    """With an ECP, Psi4's gas-phase HF gradient itself differs from finite differences by
    ~6e-8 along the C-I bond, so there the residual is taken relative to the gas one (as in
    test_gradient.py)."""
    reference = "uhf" if MOLECULES[name].multiplicity > 1 else "rhf"
    psi4.set_options({**HF_OPTIONS, "reference": reference})
    xyz = MOLECULES[name].bohr

    def energy_at(atom: int, axis: int, step: float) -> float:
        e: float = psi4.energy(
            "hf", molecule=psi4_molecule(name, displaced(xyz, atom, axis, step))
        )
        return e

    def residuals() -> dict[tuple[int, int], float]:
        g = as_array(psi4.gradient("hf", molecule=psi4_molecule(name)))
        return {
            (a, c): g[a, c] - five_point(partial(energy_at, a, c), 2e-3)
            for a in atoms
            for c in range(3)
        }

    gas = residuals() if name == "iodomethane" else {}
    psi4_swig_pcm.enable(settings(cavity, component))
    for key, resid in residuals().items():
        assert abs(resid - gas.get(key, 0.0)) < 1e-8, (key, resid)


@pytest.mark.slow
def test_dft_optimization_converges() -> None:
    psi4.set_options({"basis": "def2-svp", "scf_type": "df", "g_convergence": "gau_tight"})
    psi4_swig_pcm.enable(settings("svdw"))
    mol = psi4_molecule("formate", off_axis=True)  # an optimization (helpers.OFF_AXIS)
    psi4.optimize("pbe0", molecule=mol)
    g = as_array(psi4.gradient("pbe0", molecule=mol))
    assert np.abs(g).max() < 1e-4


@pytest.mark.slow
@pytest.mark.parametrize(
    "cavity",
    [
        "svdw",
        pytest.param(
            "isodensity",
            marks=pytest.mark.skipif(
                not os.environ.get("PSI4_SWIG_PCM_VERY_SLOW"),
                reason="~7 min (rho-DROP rebuilds its grid every SCF); "
                "set PSI4_SWIG_PCM_VERY_SLOW=1 to run it",
            ),
        ),
    ],
)
def test_frequencies_from_gradients_match_energies(cavity: str) -> None:
    """psi4.frequency at a tightly optimized geometry: finite differences of moist gradients
    against finite differences of energies (as test_driver's for the switching/Gaussian PCM)."""
    psi4.set_options({**HF_OPTIONS, "g_convergence": "gau_verytight", "geom_maxiter": 50})
    mol = psi4_molecule("formate", off_axis=True)  # an optimization (helpers.OFF_AXIS)
    psi4_swig_pcm.enable(settings(cavity))
    psi4.optimize("hf", molecule=mol)
    assert np.abs(as_array(psi4.gradient("hf", molecule=mol))).max() < 2e-6

    _, wfn_g = psi4.frequency("hf", molecule=mol, return_wfn=True)
    _, wfn_e = psi4.frequency("hf", molecule=mol, dertype=0, return_wfn=True)
    f_g = wfn_g.frequencies().to_array()
    f_e = wfn_e.frequencies().to_array()
    assert len(f_g) == 6
    assert (f_g > 0).all()
    np.testing.assert_allclose(f_g, f_e, atol=0.05)


# ------------------------------------------------------------------ against moist.pyscf


@pytest.mark.pyscf
@pytest.mark.parametrize(
    ("cavity", "component", "name"),
    [
        ("svdw", "cpcm", "formate"),
        ("svdw", "cpcm", "formyl"),  # UHF
        ("isodensity", "cpcm", "formate"),
        ("isodensity", "cpcm", "formyl"),
        ("cfc", "cpcm", "formate"),
        ("iswig", "cpcm", "formate"),
        ("svdw", "iefpcm", "formate"),
        ("iswig", "iefpcm", "formate"),
        ("svdw", "ssvpe", "formate"),
        ("isodensity", "lpb", "formate"),
        ("isodensity", "iefpcm", "formyl"),
        ("cfc", "iefpcm", "formyl"),
    ],
)
def test_matches_moist_pyscf_host(cavity: str, component: str, name: str) -> None:
    """Same moist model, two hosts: Psi4's integrals here, PySCF's in moist.pyscf.  Every
    cavity and every supported component, closed and open shell: the finite-difference tests
    check each gradient against its own energy, this checks the energy."""
    pytest.importorskip("pyscf")
    import moist.pyscf
    from pyscf import gto

    m = MOLECULES[name]
    mol = gto.M(
        atom=list(zip(m.symbols, m.bohr.tolist(), strict=True)), unit="Bohr",
        basis="def2-svp", charge=m.charge, spin=m.multiplicity - 1, verbose=0,
    )  # fmt: skip
    mf = mol.HF().MOIST(
        cavity=settings(cavity, component).cavity,  # on the same spheres
        components=[COMPONENTS[component]()],
        context=moist.Context(nthreads=4, verbosity=0),
    )
    mf.conv_tol, mf.conv_tol_grad = 1e-12, 1e-8
    e_ref = mf.kernel()
    g_ref = mf.nuc_grad_method().kernel()

    reference = "uhf" if m.multiplicity > 1 else "rhf"
    psi4.set_options({**HF_OPTIONS, "reference": reference})
    psi4_swig_pcm.enable(settings(cavity, component))
    G, wfn = psi4.gradient("hf", molecule=psi4_molecule(name), return_wfn=True)
    assert abs(wfn.energy() - e_ref) < 1e-10
    np.testing.assert_allclose(as_array(G), g_ref, atol=2e-8)  # UHF: PySCF converges to ~7e-9


@pytest.mark.parametrize("component", ["cpcm", "iefpcm"])
def test_apparent_charges_are_published(component: str) -> None:
    """moist (from 1.2) publishes a PCM component's potential adjoint, and a layer model's
    apparent surface charges q, as named results.

    `potential_adjoint` is the w_phi the host contracts.  For C-PCM that is q itself, and moist
    publishes no separate `surface_charge`; for IEF-PCM w_phi is the symmetrized adjoint and q
    (`surface_charge`) differs from it, but either gives the energy, 1/2 q . phi.
    """
    psi4.set_options(HF_OPTIONS)
    psi4_swig_pcm.enable(settings("svdw", component))
    try:
        _e, wfn = psi4.energy("hf", molecule=psi4_molecule("formate"), return_wfn=True)
        sol = wfn.ddx
        view = sol.model.components[0]
        published = {f.name for f in view.fields()}
        w_phi = sum(
            np.asarray(i.w_phi)
            for i in sol.response
            if isinstance(i, moist.PotentialAdjointResponse)
        )
        np.testing.assert_allclose(
            np.asarray(view.get("potential_adjoint")), w_phi, rtol=0, atol=1e-14
        )
        if component == "cpcm":
            assert "surface_charge" not in published
            q = np.asarray(view.get("potential_adjoint"))
        else:
            q = np.asarray(view.get("surface_charge"))
            assert np.abs(q - w_phi).max() > 1e-6 * np.abs(q).max()
        phi = sol.nuc[0] + np.asarray(sol.charges.esp(sol.basis, sol.D))
        assert 0.5 * q @ phi == pytest.approx(sol.energy, abs=1e-8)
    finally:
        psi4_swig_pcm.disable()
