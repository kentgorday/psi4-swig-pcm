# Copyright (C) 2026 Kent Gorday
# SPDX-License-Identifier: LGPL-3.0-only
"""The PCM solver Psi4's SCF calls every iteration, through its DDX solvation slot.

With the SCF option DDX on, scf_helper builds ``procrouting.solvent.ddx.DdxInterface(molecule=,
options=, basisset=)`` and scf_iterator calls ``get_solvation_contributions(D, state)`` with
the total density, adding the returned energy and pushing the returned matrix onto the Fock
matrix.  PCMSolver provides that interface; driver.enable() puts it in the slot.

The cavity is built from the real atoms only: ghost atoms (counterpoise monomers in the dimer
basis) carry basis functions but no sphere, so a monomer sits in its own cavity.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import ArrayLike
from psi4 import core

from . import cavity as cav
from .cavity import FloatArray, IntArray, Method, Switching
from .charges import GaussianCharges
from .nonelectrostatic import Energies, NonelectrostaticSettings, Terms

log = logging.getLogger(__name__)

# |relative error| of the nuclear surface charge tolerated before warning
# (C-PCM / IEF-PCM reach ~1e-3 at Lebedev order 29).
GAUSS_TOLERANCE = 1e-2


def _default_radii() -> tuple[float, ...]:
    return tuple(float(r) for r in cav.modified_bondi_radii())


@dataclass(frozen=True)
class PCMSettings:
    """PCM model and numerical settings.

    radii: cavity radius in bohr by atomic number (index 0 unused), any sequence; stored as a
    tuple so that the settings stay immutable, comparable and hashable.  The default is
    cavity.modified_bondi_radii(): PySCF's modified Bondi x 1.2, hydrogen 1.10 A.
    switching: the switching function that fades out points buried in neighbouring spheres,
    "SWIG" (PySCF's default) or "ISWIG" (cavity module docstring).  ISWIG keeps ~10% more
    points, whose weights fall off more gently.
    incore_mb: memory (MB) the stored (k|mn) integrals may take per geometry, on top of what
    Psi4 is given with set_memory; 0 computes them directly every SCF iteration instead.
    nonelectrostatic: the nonelectrostatic correction (psi4_swig_pcm.nonelectrostatic), or
    None for the electrostatic PCM alone.
    """

    method: Method = "IEF-PCM"  # SS(V)PE (symmetrized K) is the same continuum model
    eps: float = 78.3553  # water; float("inf") for a conductor
    lebedev_order: int = 29  # 302 points per sphere (PySCF default)
    switching: Switching = "SWIG"
    check: bool = True  # per-geometry numerical checks, as PCMSolver does
    incore_mb: float = 2000.0
    radii: tuple[float, ...] = field(default_factory=_default_radii)
    nonelectrostatic: NonelectrostaticSettings | None = None

    def __post_init__(self) -> None:
        if self.method not in cav.METHODS:
            raise ValueError(f"method must be one of {cav.METHODS}, not {self.method!r}")
        if self.lebedev_order not in cav.LEBEDEV_POINTS:
            raise ValueError(f"lebedev_order must be one of {sorted(cav.LEBEDEV_POINTS)}")
        if self.switching not in cav.SWITCHINGS:
            raise ValueError(f"switching must be one of {cav.SWITCHINGS}, not {self.switching!r}")
        if not self.eps >= 1.0:  # also rejects NaN
            raise ValueError(f"eps must be >= 1 (inf for a conductor), not {self.eps!r}")
        radii: ArrayLike = self.radii
        object.__setattr__(self, "radii", tuple(float(r) for r in np.asarray(radii).ravel()))


def check_diagnostics(d: cav.Diagnostics, method: Method) -> None:
    """Refuse an unusable discretization; warn on a Gauss's-law violation (PCMSolver's checks)."""
    if not d.s_positive_definite:
        raise RuntimeError(f"PCM S matrix is not positive-definite: {d}")
    # SS(V)PE's symmetrized K violates Gauss's law by 4-24% on multi-sphere cavities by
    # construction, so only C-PCM / COSMO / IEF-PCM are held to it.
    if abs(d.gauss_nuclear_relerr) > GAUSS_TOLERANCE and method != "SS(V)PE":
        message = (
            f"PCM nuclear surface charge violates Gauss's law by {d.gauss_nuclear_relerr:.2e}"
        )
        log.warning("%s (%s)", message, d)
        core.print_out(f"    Warning: {message}\n")


# A fixed Fock-matrix perturbation lambda * V0 (and energy lambda tr(V0 D)) that every SCF
# solved while it is set includes; driver.py's post-SCF gradient sets it for the two
# perturbed SCFs that give the density's response to the nonelectrostatic correction.
_perturbation: tuple[float, FloatArray] | None = None


class PCMSolver:
    """Duck-types psi4's DdxInterface: get_solvation_contributions(D, state) -> (E, V, state)."""

    def __init__(
        self,
        settings: PCMSettings,
        molecule: core.Molecule,
        options: Any,
        basisset: core.BasisSet,
    ) -> None:
        self.settings, self.basis = settings, basisset
        # Ghost atoms have Z() == 0 but keep their element in true_atomic_number().
        self.atoms: IntArray = np.array(
            [i for i in range(molecule.natom()) if molecule.Z(i) != 0], dtype=np.intp
        )
        if len(self.atoms) == 0:
            raise ValueError("the PCM needs at least one real (non-ghost) atom")
        xyz = np.asarray(molecule.geometry().np)[self.atoms]
        true_Z = [molecule.true_atomic_number(int(i)) for i in self.atoms]
        # Electrons see ECP-reduced nuclear charges; psi4 has already applied them to Z().
        Z = [molecule.Z(int(i)) for i in self.atoms]
        self.surf = cav.make_surface(
            xyz, true_Z, np.array(settings.radii), settings.lebedev_order, settings.switching
        )
        self.op = cav.PCMOperator(self.surf, settings.method, settings.eps)
        self.phi_nuc = cav.nuclear_potential(self.surf, Z)
        self.charges = GaussianCharges(
            self.surf, core.get_num_threads(), basisset, settings.incore_mb
        )
        self.calls = 0
        self.phi = self.q = self.q_sym = np.zeros(self.surf.npts)
        self.molecule = molecule
        self._terms: Terms | None = None
        self.e_nonelectrostatic: Energies | None = None
        self._print_header(molecule.natom())
        self.diagnostics = self.op.diagnostics(Z) if settings.check else None
        if self.diagnostics is not None:
            d = self.diagnostics
            core.print_out(
                f"    Nuclear surface charge vs Gauss's law: {d.gauss_nuclear_relerr:+.2e}"
                f" (relative); rcond(K) {d.rcond_K:.1e}\n"
            )
            check_diagnostics(d, settings.method)

    def _print_header(self, natom: int) -> None:
        s = self.settings
        ghosts = natom - len(self.atoms)
        stored = self.charges.B
        integrals = (
            f"stored in core ({stored.nbytes / 2**20:.0f} MB)"
            if stored is not None
            else "computed directly each iteration"
        )
        core.print_out(
            f"\n  ==> psi4-swig-pcm: {s.method}, eps = {s.eps:.4f} <==\n\n"
            f"    Cavity: {len(self.atoms)} atoms"
            + (f" ({ghosts} ghost atoms left out)" if ghosts else "")
            + f", Lebedev order {s.lebedev_order}, {s.switching} switching,"
            f" {self.surf.npts} surface points\n"
            f"    Surface-charge integrals (k|mn): {integrals}\n"
        )

    def get_solvation_contributions(
        self,
        D: core.Matrix,
        state: Any = None,
        elec_only: bool = False,
        nonequilibrium: bool = False,
    ) -> tuple[float, core.Matrix, Any]:
        if nonequilibrium:
            raise NotImplementedError("nonequilibrium (optical dielectric) response")
        phi = self.charges.esp(self.basis, D)
        if not elec_only:
            phi = phi + self.phi_nuc
        q, q_sym = self.op.charges(phi)
        V = self.charges.potential_matrix(self.basis, q_sym)
        energy = float(0.5 * q_sym @ phi)
        self.calls += 1
        if not elec_only:
            self.phi, self.q, self.q_sym = phi, q, q_sym
            ne = self.settings.nonelectrostatic
            if ne is not None and ne.self_consistent:
                energies, Vg = self.terms.energy_and_potential(np.asarray(D.np))
                assert Vg is not None
                self.e_nonelectrostatic = energies
                energy += energies.total
                V.add(core.Matrix.from_array(Vg))
            if _perturbation is not None:
                lam, V0 = _perturbation
                energy += lam * float(np.sum(V0 * np.asarray(D.np)))
                V.add(core.Matrix.from_array(lam * V0))
        return energy, V, state

    @property
    def terms(self) -> Terms:
        """The nonelectrostatic correction at this geometry (built on first use)."""
        ne = self.settings.nonelectrostatic
        if ne is None:
            raise ValueError("these PCMSettings have no nonelectrostatic correction")
        if self._terms is None:
            radii = np.array(
                [self.settings.radii[self.molecule.true_atomic_number(int(i))] for i in self.atoms]
            )
            self._terms = Terms(
                self.molecule,
                self.basis,
                self.surf,
                self.atoms,
                radii,
                ne,
                core.get_num_threads(),
                store_field_integrals=ne.self_consistent,
                store_mb=self.settings.incore_mb,
            )
        return self._terms
