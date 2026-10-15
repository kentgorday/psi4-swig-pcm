# Copyright (C) 2026 Kent Gorday
# SPDX-License-Identifier: LGPL-3.0-only
"""The PCM solver Psi4's SCF calls every iteration, through its DDX solvation slot.

With the SCF option DDX on, scf_helper builds ``procrouting.solvent.ddx.DdxInterface(molecule=,
options=, basisset=)`` and scf_iterator calls ``get_solvation_contributions(D, state)`` with
the total density, adding the returned energy and pushing the returned matrix onto the Fock
matrix.  PCMSolver provides that interface; driver.enable() puts it in the slot.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, ClassVar

import numpy as np
from psi4 import core

from . import cavity as cav
from .cavity import FloatArray, Method
from .charges import GaussianCharges

log = logging.getLogger(__name__)

# |relative error| of the nuclear surface charge tolerated before warning
# (C-PCM / IEF-PCM reach ~1e-3 at Lebedev order 29).
GAUSS_TOLERANCE = 1e-2


@dataclass(frozen=True)
class PCMSettings:
    method: Method = "IEF-PCM"  # SS(V)PE (symmetrized K) is the same continuum model
    eps: float = 78.3553  # water
    lebedev_order: int = 29  # 302 points per sphere (PySCF default)
    check: bool = True  # per-geometry numerical checks, as PCMSolver does
    incore_mb: float = 2000.0  # store (k|mn) per geometry if it fits (MB); 0 = always direct
    radii: FloatArray = field(default_factory=cav.modified_bondi_radii, compare=False, hash=False)

    def __post_init__(self) -> None:
        if self.method not in cav.METHODS:
            raise ValueError(f"method must be one of {cav.METHODS}, not {self.method!r}")
        if self.lebedev_order not in cav.LEBEDEV_POINTS:
            raise ValueError(f"lebedev_order must be one of {sorted(cav.LEBEDEV_POINTS)}")


def check_diagnostics(d: cav.Diagnostics, method: Method) -> None:
    """Refuse an unusable discretization; warn on a Gauss's-law violation (PCMSolver's checks)."""
    if not d.s_positive_definite:
        raise RuntimeError(f"PCM S matrix is not positive-definite: {d}")
    # SS(V)PE's symmetrized K violates Gauss's law by 4-24% on multi-sphere cavities by
    # construction, so only C-PCM / COSMO / IEF-PCM are held to it.
    if abs(d.gauss_nuclear_relerr) > GAUSS_TOLERANCE and method != "SS(V)PE":
        log.warning(
            "PCM nuclear surface charge violates Gauss's law by %.2e (%s)",
            d.gauss_nuclear_relerr,
            d,
        )


class PCMSolver:
    """Duck-types psi4's DdxInterface: get_solvation_contributions(D, state) -> (E, V, state)."""

    last: ClassVar[PCMSolver | None] = None  # the solver of the most recent SCF

    def __init__(
        self,
        settings: PCMSettings,
        molecule: core.Molecule,
        options: Any,
        basisset: core.BasisSet,
    ) -> None:
        self.settings, self.basis = settings, basisset
        natm = molecule.natom()
        xyz = np.asarray(molecule.geometry().np)
        true_Z = [molecule.true_atomic_number(i) for i in range(natm)]
        # Electrons see ECP-reduced nuclear charges; psi4 has already applied them to Z().
        Z = [molecule.Z(i) for i in range(natm)]
        t0 = time.perf_counter()
        self.surf = cav.make_surface(xyz, true_Z, settings.radii, settings.lebedev_order)
        self.op = cav.PCMOperator(self.surf, settings.method, settings.eps)
        self.phi_nuc = cav.nuclear_potential(self.surf, Z)
        self.diagnostics = self.op.diagnostics(Z) if settings.check else None
        if self.diagnostics is not None:
            check_diagnostics(self.diagnostics, settings.method)
        self.timings = {"cavity": time.perf_counter() - t0, "phi": 0.0, "solve": 0.0, "V": 0.0}
        self.calls = 0
        t1 = time.perf_counter()
        self.charges = GaussianCharges(
            self.surf, core.get_num_threads(), basisset, settings.incore_mb
        )
        self.timings["integrals_incore"] = (
            time.perf_counter() - t1 if self.charges.B is not None else 0.0
        )
        self.phi = self.q = self.q_sym = np.zeros(self.surf.npts)
        PCMSolver.last = self

    def get_solvation_contributions(
        self,
        D: core.Matrix,
        state: Any = None,
        elec_only: bool = False,
        nonequilibrium: bool = False,
    ) -> tuple[float, core.Matrix, Any]:
        if nonequilibrium:
            raise NotImplementedError("nonequilibrium (optical dielectric) response")
        t0 = time.perf_counter()
        phi = self.charges.esp(self.basis, D)
        if not elec_only:
            phi = phi + self.phi_nuc
        t1 = time.perf_counter()
        q, q_sym = self.op.charges(phi)
        t2 = time.perf_counter()
        V = self.charges.potential_matrix(self.basis, q_sym)
        self.timings["phi"] += t1 - t0
        self.timings["solve"] += t2 - t1
        self.timings["V"] += time.perf_counter() - t2
        self.calls += 1
        if not elec_only:
            self.phi, self.q, self.q_sym = phi, q, q_sym
        return float(0.5 * q_sym @ phi), V, state
