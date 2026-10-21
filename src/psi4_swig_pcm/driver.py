# Copyright (C) 2026 Kent Gorday
# SPDX-License-Identifier: LGPL-3.0-only
"""Route Psi4's own driver (energy, gradient, optimize, frequency) through the PCM.

enable() makes these process-global changes, all undone by disable():

1. Psi4's DDX solvation slot (procrouting.solvent.ddx) holds the solver, and the DDX option is
   on, so every SCF includes the reaction field.  The solver is PCMSolver, or for a moist model
   (prototype) moist_host.MoistSolver.
2. Every SCF entry of psi4.driver.procedures["gradient"] (proc.select_scf_gradient) is wrapped
   to add the PCM terms.  Psi4 1.11 has no DDX gradient: without this, psi4.gradient() returns
   the solvated SCF's gradient without them, and nothing warns.
3. Every other analytic gradient (MP2, CC, OO methods, ...) and every analytic Hessian leaves
   the table, so Psi4 negotiates finite differences instead: of PCM gradients for SCF
   Hessians and frequencies, of energies for the rest.  Those analytic codes would differentiate
   the SCF part's reaction field away.  The displacements run in this process (AtomicComputer
   sets psiapi), so they see 1 and 2.
4. CFOUR energies (c4-*) are refused: CFOUR runs its own SCF, outside Psi4, without the PCM.

Correlated energies (MP2, CC, ...) on top of the PCM SCF see the reaction field only through
the reference orbitals: the correlation energy itself has no PCM response.

The SCF gradient itself runs Psi4's own run_scf_gradient with the converged symmetrized
charges frozen on an ExternalPotential (their interaction with electrons and nuclei, plus -D),
then adds the motion of the charges with their atoms and the cavity term.  For a moist model it
is Psi4's own SCF gradient at the solvated density, with no external potential, plus
MoistSolver.gradient().

With PCMSettings.nonelectrostatic, the correction G enters in one of two ways:

* self-consistent: the solver adds G and V = dG/dD every SCF iteration, so energies need
  nothing more; the gradient adds G's explicit derivative at fixed density (Psi4's
  energy-weighted density already comes from the Fock matrix that holds V).
* post-SCF (as CMIRS): the SCF is the PCM's alone and G is evaluated on its density.  SCF
  energies (5., below) get G added.  The gradient needs the density's response,
  tr(V dD/dR): it is the derivative, at lambda = 0, of the ordinary SCF gradient of an SCF with
  lambda V0 added to its Fock matrix (V0 = V at the converged density, held fixed).  Four SCFs
  at +-lambda and +-2 lambda, each from the previous orbitals and converged tightly, give it
  by Richardson extrapolation to O(lambda^4).

5. With a post-SCF correction, SCF energy entries of procedures["energy"] add G.
6. HF.set_external_cpscf_perturbation registers the "DDX" CPSCF callback through a weak
   reference to the wavefunction.  Psi4 1.11's scf_finalize_energy registers
   `lambda pert_dm: self.ddx...`, which captures the wavefunction itself; the C++ wavefunction
   then holds the Python one through its own callback, a cycle that Python's garbage collector
   cannot see.  Every SCF would keep its wavefunction and its solver (the dense npts x npts
   matrices) for the life of the process: some 20 MB per formate SCF, gigabytes over a
   finite-difference Hessian or an optimization.  Neither solver has a CPSCF response yet, so
   the callback still only raises NotImplementedError, as it did before.
"""

from __future__ import annotations

import functools
import sys
import types
import weakref
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, TypeGuard

import numpy as np
from psi4 import core
from psi4.driver import p4util, procedures
from psi4.driver.procrouting import proc, proc_util
from psi4.driver.procrouting import solvent as p4solvent
from psi4.driver.procrouting.interface_cfour import run_cfour

from . import cavity as cav
from . import solver as _solver
from .cavity import FloatArray
from .solver import PCMSettings, PCMSolver

if TYPE_CHECKING:
    from .moist_host import MoistSettings, MoistSolver

# lambda of the post-SCF gradient's perturbed SCFs (at +-lambda and +-2 lambda).  The
# central difference at lambda is off by O(lambda^2), ~1e-5 of the response term at 1e-2
# for formate; the extrapolation removes that, and leaves the SCFs' convergence error over
# lambda, which PERTURBED_CONVERGENCE keeps below ~1e-8 Eh/bohr.
FIELD_STRENGTH = 1e-2
PERTURBED_CONVERGENCE = 1e-10  # E_ and D_CONVERGENCE of the perturbed SCFs, at most

_MISSING = object()


@dataclass
class _Patch:
    settings: PCMSettings | MoistSettings
    have_ddx: Any
    ddx: Any
    options: Any  # p4util.OptionsState for DDX
    set_cpscf: Any  # core.HF's own set_external_cpscf_perturbation (6.)
    # (table, method) -> the procedures entry it replaced or removed
    entries: dict[tuple[str, str], Any] = field(default_factory=dict)


_active: _Patch | None = None


def enable(settings: PCMSettings | MoistSettings | None = None) -> None:
    """Make psi4.energy/gradient/optimize/hessian/frequency include the PCM (process-global).

    settings: PCMSettings for this package's switching/Gaussian PCM, or (prototype)
    moist_host.MoistSettings for a moist cavity and model.
    """
    global _active
    disable()
    settings = PCMSettings() if settings is None else settings
    solver_class = _solver_class(settings)
    # Recorded before anything changes, so that disable() can undo a partial enable().
    _active = patch = _Patch(
        settings=settings,
        have_ddx=p4solvent._have_ddx,
        ddx=getattr(p4solvent, "ddx", _MISSING),
        options=p4util.OptionsState(["DDX"]),
        set_cpscf=core.HF.__dict__["set_external_cpscf_perturbation"],
    )
    try:
        core.HF.set_external_cpscf_perturbation = _weak_ddx_cpscf(
            core.HF.set_external_cpscf_perturbation
        )
        p4solvent._have_ddx = True
        p4solvent.ddx = types.SimpleNamespace(
            get_ddx_options=lambda molecule: {},
            DdxInterface=functools.partial(solver_class, settings),
        )
        core.set_global_option("DDX", True)
        wrapper = _wrap_scf_gradient(proc.select_scf_gradient)
        for name, fn in list(procedures["gradient"].items()):
            patch.entries["gradient", name] = fn
            if fn is proc.select_scf_gradient:
                procedures["gradient"][name] = wrapper
            else:
                del procedures["gradient"][name]
        for name in list(procedures["hessian"]):
            patch.entries["hessian", name] = procedures["hessian"].pop(name)
        energy_wrapper = _wrap_scf_energy(proc.run_scf)
        for name, fn in list(procedures["energy"].items()):
            if fn is run_cfour:
                patch.entries["energy", name] = fn
                procedures["energy"][name] = _refuse_cfour
            elif fn is proc.run_scf:
                patch.entries["energy", name] = fn
                procedures["energy"][name] = energy_wrapper
    except BaseException:
        disable()
        raise


def disable() -> None:
    """Undo enable(); a no-op when not enabled."""
    global _active
    if _active is None:
        return
    patch, _active = _active, None
    for (table, name), fn in patch.entries.items():
        procedures[table][name] = fn
    patch.options.restore()
    core.HF.set_external_cpscf_perturbation = patch.set_cpscf
    p4solvent._have_ddx = patch.have_ddx
    if patch.ddx is _MISSING:
        if hasattr(p4solvent, "ddx"):  # absent if enable() failed before setting it
            del p4solvent.ddx
    else:
        p4solvent.ddx = patch.ddx


@contextmanager
def enabled(settings: PCMSettings | MoistSettings | None = None) -> Iterator[None]:
    """enable() for the duration of a with block."""
    enable(settings)
    try:
        yield
    finally:
        disable()


# moist_host needs moist, so it is never imported here: whoever made MoistSettings, or enabled
# them, has imported it.


def _moist_host() -> types.ModuleType | None:
    return sys.modules.get(f"{__package__}.moist_host")


def _solver_class(settings: Any) -> type:
    if isinstance(settings, PCMSettings):
        return PCMSolver
    moist_host = _moist_host()
    if moist_host is not None and isinstance(settings, moist_host.MoistSettings):
        solver: type = moist_host.MoistSolver
        return solver
    raise TypeError(f"settings must be PCMSettings or MoistSettings, not {type(settings)}")


def _is_moist_solver(sol: Any) -> TypeGuard[MoistSolver]:
    moist_host = _moist_host()
    return moist_host is not None and isinstance(sol, moist_host.MoistSolver)


def active_settings() -> PCMSettings | MoistSettings | None:
    """The settings of the current enable(), or None."""
    return None if _active is None else _active.settings


def _weak_ddx_cpscf(setter: Callable[..., None]) -> Callable[..., None]:
    """HF.set_external_cpscf_perturbation, with the "DDX" callback holding its wavefunction
    only weakly (6. above).  Psi4's own callback is
    lambda pert_dm: self.ddx.get_solvation_contributions(pert_dm, ...)[1]."""

    def set_external_cpscf_perturbation(
        self: core.HF, name: str, function: Callable[[core.Matrix], core.Matrix]
    ) -> None:
        if name == "DDX":
            function = _ddx_cpscf_response(weakref.ref(self))
        setter(self, name, function)

    return set_external_cpscf_perturbation


def _ddx_cpscf_response(
    wfn_ref: weakref.ref[core.HF],
) -> Callable[[core.Matrix], core.Matrix]:
    def response(pert_dm: core.Matrix) -> core.Matrix:
        wfn = wfn_ref()
        if wfn is None:
            raise ReferenceError("the wavefunction of this DDX CPSCF callback no longer exists")
        V: core.Matrix = wfn.ddx.get_solvation_contributions(
            pert_dm, elec_only=True, nonequilibrium=True
        )[1]
        return V

    return response


def _refuse_cfour(name: str, **kwargs: Any) -> None:
    raise NotImplementedError(
        f"{name}: CFOUR runs its own SCF without psi4-swig-pcm's PCM; disable() it first"
    )


def _wrap_scf_gradient(original: Callable[..., Any]) -> Callable[..., Any]:
    @functools.wraps(original)  # keeps the select_ name, so psi4 still probes availability here
    def scf_gradient(name: str, **kwargs: Any) -> Any:
        if kwargs.get("probe", False) or not core.get_option("SCF", "DDX"):
            return original(name, **kwargs)
        return _pcm_scf_gradient(name, **kwargs)

    return scf_gradient


def _run_scf(name: str, **kwargs: Any) -> core.Wavefunction:
    """proc.run_scf as run_scf_gradient starts it: REFERENCE set for the SCF, then the SCF."""
    optstash = proc_util.scf_set_reference_local(name, is_dft="dft_functional" in kwargs)
    try:
        return proc.run_scf(name, **kwargs)
    finally:
        optstash.restore()


def _post_scf(wfn: core.Wavefunction) -> bool:
    sol = getattr(wfn, "ddx", None)
    ne = sol.settings.nonelectrostatic if isinstance(sol, PCMSolver) else None
    return ne is not None and not ne.self_consistent


def _total_density(wfn: core.Wavefunction) -> FloatArray:
    return np.asarray(wfn.Da().np) + np.asarray(wfn.Db().np)


def _record_nonelectrostatic(wfn: core.Wavefunction, energy: float, scf_energy: float) -> None:
    """G as NONELECTROSTATIC ENERGY; CURRENT ENERGY and the wfn's energy as SCF + G."""
    for key, value in (
        ("NONELECTROSTATIC ENERGY", energy),
        ("CURRENT ENERGY", scf_energy + energy),
    ):
        wfn.set_variable(key, value)
        core.set_variable(key, value)
    wfn.set_energy(scf_energy + energy)


def _wrap_scf_energy(original: Callable[..., Any]) -> Callable[..., Any]:
    @functools.wraps(original)
    def scf_energy(name: str, **kwargs: Any) -> Any:
        wfn = original(name, **kwargs)
        sol = getattr(wfn, "ddx", None)
        if isinstance(sol, PCMSolver) and sol.settings.nonelectrostatic is not None:
            if _post_scf(wfn):
                energy = sol.terms.energy(_total_density(wfn)).total
                _record_nonelectrostatic(wfn, energy, wfn.energy())
            elif sol.e_nonelectrostatic is not None:  # already in the SCF energy
                wfn.set_variable("NONELECTROSTATIC ENERGY", sol.e_nonelectrostatic.total)
                core.set_variable("NONELECTROSTATIC ENERGY", sol.e_nonelectrostatic.total)
        return wfn

    return scf_energy


def _pcm_scf_gradient(name: str, **kwargs: Any) -> core.Wavefunction:
    ref_wfn = kwargs.pop("ref_wfn", None)
    if ref_wfn is None:  # the start of proc.run_scf_gradient
        ref_wfn = _run_scf(name, **kwargs)
    g = gradient_from_wfn(name, ref_wfn)
    if _post_scf(ref_wfn):
        g = g + post_scf_gradient(name, ref_wfn, **kwargs)
    G = core.Matrix.from_array(g)
    ref_wfn.set_gradient(G)
    total = "DFT TOTAL GRADIENT" if ref_wfn.functional().needs_xc() else "HF TOTAL GRADIENT"
    for key in ("SCF TOTAL GRADIENT", total, "CURRENT GRADIENT"):
        ref_wfn.set_variable(key, G)
        core.set_variable(key, G)
    return ref_wfn


def post_scf_gradient(name: str, ref_wfn: core.Wavefunction, **kwargs: Any) -> FloatArray:
    """dG/dR (natom, 3) of a post-SCF correction on the converged PCM SCF ref_wfn.

    G's explicit derivative at fixed density, plus tr(V dD/dR) from SCFs with +-lambda V0 and
    +-2 lambda V0 on their Fock matrices (each from the orbitals the previous SCF left in the
    scratch file).  Records SCF + G as the energy, and leaves Psi4's variables as ref_wfn's.
    """
    sol = ref_wfn.ddx
    D0 = _total_density(ref_wfn)
    energies, V0 = sol.terms.energy_and_potential(D0)
    assert V0 is not None
    g: FloatArray = sol.terms.gradient(D0)
    scf_energy = ref_wfn.energy()
    saved = dict(core.variables())
    lam = FIELD_STRENGTH
    perturbed: dict[float, FloatArray] = {}
    optstash = p4util.OptionsState(
        ["SCF", "GUESS"], ["SCF", "E_CONVERGENCE"], ["SCF", "D_CONVERGENCE"]
    )
    try:
        core.set_local_option("SCF", "GUESS", "READ")
        for key in ("E_CONVERGENCE", "D_CONVERGENCE"):
            current = core.get_option("SCF", key)
            core.set_local_option("SCF", key, min(current, PERTURBED_CONVERGENCE))
        for t in (lam, -lam, 2.0 * lam, -2.0 * lam):
            _solver._perturbation = (t, V0)
            perturbed[t] = gradient_from_wfn(name, _run_scf(name, **kwargs))
    finally:
        _solver._perturbation = None
        optstash.restore()

    def central(t: float) -> FloatArray:
        return (perturbed[t] - perturbed[-t]) / (2.0 * t)

    g += (4.0 * central(lam) - central(2.0 * lam)) / 3.0
    for key, value in saved.items():  # the perturbed SCFs set them too
        core.set_variable(key, value)
    _record_nonelectrostatic(ref_wfn, energies.total, scf_energy)
    return g


def gradient_from_wfn(name: str, wfn: core.Wavefunction) -> FloatArray:
    """Analytic PCM gradient (natm, 3) for an SCF converged with the PCM in the DDX slot."""
    sol = getattr(wfn, "ddx", None)
    if not (isinstance(sol, PCMSolver) or _is_moist_solver(sol)):
        raise ValueError("wfn was not converged with psi4_swig_pcm enabled")
    if not isinstance(wfn, core.RHF | core.UHF):
        raise NotImplementedError(f"PCM gradients for {type(wfn).__name__} references")
    if wfn.external_pot() is not None:
        raise NotImplementedError("PCM gradients with another external potential on the wfn")
    if not isinstance(sol, PCMSolver):
        return _moist_gradient(name, wfn, sol)
    ep = sol.charges.external_potential(sol.q_sym)
    wfn.set_external_potential(ep)
    try:
        # SCF gradient with the charges frozen: their interaction with electrons and nuclei
        # (Psi4's C++), plus Psi4's own -D gradient.
        g = _psi4_scf_gradient(name, wfn)
        g_pts = sol.charges.point_gradients(ep)  # the charges move with their atoms
    finally:
        wfn.set_external_potential(None)
    # Per cavity atom, i.e. per real atom: ghost atoms have no sphere.
    g_pts_atom = np.array([g_pts[p0:p1].sum(0) for p0, p1 in sol.surf.atom_slices])
    g[sol.atoms] += g_pts_atom + cav.cavity_gradient(sol.op, sol.phi, sol.q)
    ne = sol.settings.nonelectrostatic
    if ne is not None and ne.self_consistent:
        g += sol.terms.gradient(_total_density(wfn))
    return g


def _psi4_scf_gradient(name: str, wfn: core.Wavefunction) -> FloatArray:
    # run_scf_gradient sets REFERENCE to RKS/UKS for scfgrad, but only knows a custom (dict)
    # functional is DFT from dft_functional.
    kwargs = {"dft_functional": True} if wfn.functional().needs_xc() else {}
    g: FloatArray = np.array(
        proc.run_scf_gradient(name.lower(), ref_wfn=wfn, **kwargs).gradient().np
    )
    return g


def _moist_gradient(name: str, wfn: core.Wavefunction, sol: MoistSolver) -> FloatArray:
    """Psi4's SCF gradient at the solvated density, plus moist's part (no external potential).

    Psi4's energy-weighted density comes from the converged orbital energies, which include
    moist's Fock contribution, so its Pulay term is already the solvated one.
    """
    D = wfn.Da().clone()
    D.add(wfn.Db())
    return _psi4_scf_gradient(name, wfn) + sol.gradient(D)
