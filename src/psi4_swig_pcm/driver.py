# Copyright (C) 2026 Kent Gorday
# SPDX-License-Identifier: LGPL-3.0-only
"""Route Psi4's own driver (energy, gradient, optimize, frequency) through the PCM.

enable() makes these process-global changes, all undone by disable():

1. Psi4's DDX solvation slot (procrouting.solvent.ddx) holds PCMSolver and the DDX option is
   on, so every SCF includes the reaction field.
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
then adds the motion of the charges with their atoms and the cavity term.
"""

from __future__ import annotations

import functools
import types
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from psi4 import core
from psi4.driver import p4util, procedures
from psi4.driver.procrouting import proc, proc_util
from psi4.driver.procrouting import solvent as p4solvent
from psi4.driver.procrouting.interface_cfour import run_cfour

from . import cavity as cav
from .cavity import FloatArray
from .solver import PCMSettings, PCMSolver

_MISSING = object()


@dataclass
class _Patch:
    settings: PCMSettings
    have_ddx: Any
    ddx: Any
    options: Any  # p4util.OptionsState for DDX
    # (table, method) -> the procedures entry it replaced or removed
    entries: dict[tuple[str, str], Any] = field(default_factory=dict)


_active: _Patch | None = None


def enable(settings: PCMSettings | None = None) -> None:
    """Make psi4.energy/gradient/optimize/hessian/frequency include the PCM (process-global)."""
    global _active
    disable()
    settings = PCMSettings() if settings is None else settings
    # Recorded before anything changes, so that disable() can undo a partial enable().
    _active = patch = _Patch(
        settings=settings,
        have_ddx=p4solvent._have_ddx,
        ddx=getattr(p4solvent, "ddx", _MISSING),
        options=p4util.OptionsState(["DDX"]),
    )
    try:
        p4solvent._have_ddx = True
        p4solvent.ddx = types.SimpleNamespace(
            get_ddx_options=lambda molecule: {},
            DdxInterface=functools.partial(PCMSolver, settings),
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
        for name, fn in list(procedures["energy"].items()):
            if fn is run_cfour:
                patch.entries["energy", name] = fn
                procedures["energy"][name] = _refuse_cfour
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
    p4solvent._have_ddx = patch.have_ddx
    if patch.ddx is _MISSING:
        if hasattr(p4solvent, "ddx"):  # absent if enable() failed before setting it
            del p4solvent.ddx
    else:
        p4solvent.ddx = patch.ddx


@contextmanager
def enabled(settings: PCMSettings | None = None) -> Iterator[None]:
    """enable() for the duration of a with block."""
    enable(settings)
    try:
        yield
    finally:
        disable()


def active_settings() -> PCMSettings | None:
    """The settings of the current enable(), or None."""
    return None if _active is None else _active.settings


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


def _pcm_scf_gradient(name: str, **kwargs: Any) -> core.Wavefunction:
    ref_wfn = kwargs.pop("ref_wfn", None)
    if ref_wfn is None:  # the start of proc.run_scf_gradient: REFERENCE for the SCF, then the SCF
        optstash = proc_util.scf_set_reference_local(name, is_dft="dft_functional" in kwargs)
        try:
            ref_wfn = proc.run_scf(name, **kwargs)
        finally:
            optstash.restore()
    G = core.Matrix.from_array(gradient_from_wfn(name, ref_wfn))
    ref_wfn.set_gradient(G)
    total = "DFT TOTAL GRADIENT" if ref_wfn.functional().needs_xc() else "HF TOTAL GRADIENT"
    for key in ("SCF TOTAL GRADIENT", total, "CURRENT GRADIENT"):
        ref_wfn.set_variable(key, G)
        core.set_variable(key, G)
    return ref_wfn


def gradient_from_wfn(name: str, wfn: core.Wavefunction) -> FloatArray:
    """Analytic PCM gradient (natm, 3) for an SCF converged with the PCM in the DDX slot."""
    sol = getattr(wfn, "ddx", None)
    if not isinstance(sol, PCMSolver):
        raise ValueError("wfn was not converged with psi4_swig_pcm enabled")
    if not isinstance(wfn, core.RHF | core.UHF):
        raise NotImplementedError(f"PCM gradients for {type(wfn).__name__} references")
    if wfn.external_pot() is not None:
        raise NotImplementedError("PCM gradients with another external potential on the wfn")
    ep = sol.charges.external_potential(sol.q_sym)
    wfn.set_external_potential(ep)
    try:
        # SCF gradient with the charges frozen: their interaction with electrons and nuclei
        # (Psi4's C++), plus Psi4's own -D gradient.  run_scf_gradient sets REFERENCE to
        # RKS/UKS for scfgrad, but only knows a custom (dict) functional is DFT from
        # dft_functional.
        kwargs = {"dft_functional": True} if wfn.functional().needs_xc() else {}
        g = np.array(proc.run_scf_gradient(name.lower(), ref_wfn=wfn, **kwargs).gradient().np)
        g_pts = sol.charges.point_gradients(ep)  # the charges move with their atoms
    finally:
        wfn.set_external_potential(None)
    # Per cavity atom, i.e. per real atom: ghost atoms have no sphere.
    g_pts_atom = np.array([g_pts[p0:p1].sum(0) for p0, p1 in sol.surf.atom_slices])
    g[sol.atoms] += g_pts_atom + cav.cavity_gradient(sol.op, sol.phi, sol.q)
    return g
