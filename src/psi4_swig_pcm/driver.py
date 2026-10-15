# Copyright (C) 2026 Kent Gorday
# SPDX-License-Identifier: LGPL-3.0-only
"""Route Psi4's own driver (energy, gradient, optimize, frequency) through the PCM.

enable() makes three process-global changes, all undone by disable():

1. Psi4's DDX solvation slot (procrouting.solvent.ddx) holds PCMSolver and the DDX option is
   on, so every SCF includes the reaction field.
2. Every SCF entry of psi4.driver.procedures["gradient"] (proc.select_scf_gradient) is wrapped
   to add the PCM terms.  Psi4 1.11 has no DDX gradient: without this, psi4.gradient() returns
   the solvated SCF's gradient without them, and nothing warns.
3. The analytic SCF Hessians (proc.run_scf_hessian) leave procedures["hessian"], so Psi4
   negotiates finite differences of the PCM gradient for hessian() and frequency().  Those
   displacements run in this process (AtomicComputer sets psiapi), so they see 1 and 2.

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
    gradient: dict[str, Any] = field(default_factory=dict)
    hessian: dict[str, Any] = field(default_factory=dict)


_active: _Patch | None = None


def enable(settings: PCMSettings | None = None) -> None:
    """Make psi4.energy/gradient/optimize/hessian/frequency include the PCM (process-global)."""
    global _active
    disable()
    settings = PCMSettings() if settings is None else settings
    patch = _Patch(
        settings=settings,
        have_ddx=p4solvent._have_ddx,
        ddx=getattr(p4solvent, "ddx", _MISSING),
        options=p4util.OptionsState(["DDX"]),
    )
    p4solvent._have_ddx = True
    p4solvent.ddx = types.SimpleNamespace(
        get_ddx_options=lambda molecule: {},
        DdxInterface=functools.partial(PCMSolver, settings),
    )
    core.set_global_option("DDX", True)
    wrapper = _wrap_scf_gradient(proc.select_scf_gradient)
    for name, fn in list(procedures["gradient"].items()):
        if fn is proc.select_scf_gradient:
            patch.gradient[name] = fn
            procedures["gradient"][name] = wrapper
    for name, fn in list(procedures["hessian"].items()):
        if fn is proc.run_scf_hessian:
            patch.hessian[name] = procedures["hessian"].pop(name)
    _active = patch


def disable() -> None:
    """Undo enable(); a no-op when not enabled."""
    global _active
    if _active is None:
        return
    patch, _active = _active, None
    procedures["gradient"].update(patch.gradient)
    procedures["hessian"].update(patch.hessian)
    patch.options.restore()
    p4solvent._have_ddx = patch.have_ddx
    if patch.ddx is _MISSING:
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
    ep = sol.charges.external_potential(sol.q_sym)
    wfn.set_external_potential(ep)
    # SCF gradient with the charges frozen: their interaction with electrons and nuclei (Psi4's
    # C++), plus Psi4's own -D gradient.  run_scf_gradient sets REFERENCE to RKS/UKS for
    # scfgrad, but it only knows that a custom (dict) functional is DFT from dft_functional.
    kwargs = {"dft_functional": True} if wfn.functional().needs_xc() else {}
    g = np.array(proc.run_scf_gradient(name.lower(), ref_wfn=wfn, **kwargs).gradient().np)
    g_pts = sol.charges.point_gradients(ep)  # the charges move with their atoms
    for a, (p0, p1) in enumerate(sol.surf.atom_slices):
        g[a] += g_pts[p0:p1].sum(0)
    g += cav.cavity_gradient(sol.op, sol.phi, sol.q)
    return g
