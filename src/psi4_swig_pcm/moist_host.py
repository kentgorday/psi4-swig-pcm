# Copyright (C) 2026 Kent Gorday
# SPDX-License-Identifier: LGPL-3.0-only
#
# The nuclear potential and the contractions of moist's density response follow moist's own
# PySCF host; they are in moist_contractions.py, under moist's license (MPL-2.0).  See NOTICE.
"""Prototype: moist's cavities and solvation models in Psi4, with analytic nuclear gradients.

moist (https://github.com/lukaswittmann/moist, MPL-2.0) builds smooth, differentiable cavities
(SvdW-DROP, CFC-DROP, the isodensity rho-DROP, iSwiG) and solves the PCM on them.  It talks to
a quantum-chemistry host through its "coupling protocol": moist asks for quantities on its
cavity grid (the molecular potential and its derivatives, the density matrix), and hands back
coefficients (a "response") that the host contracts with its own integrals into the Fock
matrix and the nuclear gradient.  moist itself takes care of everything that moves with the
cavity.  This module is that host for Psi4, as moist.pyscf is for PySCF:

    import moist
    import psi4_swig_pcm
    from psi4_swig_pcm.moist_host import MoistSettings

    psi4_swig_pcm.enable(MoistSettings(
        cavity=moist.DROP(lsf=moist.SvdW(), parameters=moist.DROPParameters(nleb=194)),
        components=(moist.ModelComponentCPCM(78.3553),),
    ))
    psi4.optimize("b3lyp-d3bj")

The cavity is built on MoistSettings.radii, by default the switching/Gaussian cavity's spheres
(cavity.modified_bondi_radii(), as PCMSettings.radii), not moist's own CPCM radii.

The plumbing into Psi4 is driver.enable(): the solver sits in Psi4's DDX slot, and the SCF
gradient is Psi4's own gas-phase-form gradient at the solvated density plus gradient().

What Psi4 supplies, with its own libint2 integrals (csrc/gcharge.cpp, charges.GaussianCharges)
and, for the isodensity cavity, numpy:

  request / item            Psi4 side
  ------------------------  ------------------------------------------------------------------
  phi                       (k|mn) D_mn with unit Gaussian charges, + nuclei (erf(xi r)/r)
  dphi_dr                   d(k|mn)/dr_k D_mn (ERI first derivatives), + nuclei; in the
                            gradient, from the same pass as w_phi's basis-centre term
  dphi_dxi                  three-centre overlaps (k m n) D_mn, + nuclei
  w_phi (Fock)              sum_k w_k (k|mn)
  w_phi (gradient)          sum_k w_k d(k|mn)/dR_A D_mn, basis-function centres only
  density_matrix (rho-DROP) D in moist's Cartesian-monomial layout (CartesianAOs)
  density item (rho-DROP)   basis functions and derivatives (to 3rd) on the grid, numpy

The (k|mn) are stored for a grid when they fit incore_mb.  An isodensity cavity has a new grid
every SCF iteration; stored, they still serve both its potential and its Fock matrix.

Not supported yet (NotImplementedError): the models that use the normal derivative of the
potential (moist's DPCM in both forms, SLIC and the nonlocal model: w_dphi_dr and second
derivatives of the potential), GOSTSHYP's Gaussian moments, ghost atoms with an isodensity
cavity, ROHF gradients (ROHF energies work; the driver refuses the gradient), and the CPSCF
(nonequilibrium) response.  C-PCM, COSMO, IEF-PCM, SS(V)PE and LPB
work on every cavity.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

import moist
import numpy as np
from psi4 import core

from . import solver as _solver
from .cavity import FloatArray, IntArray
from .charges import GaussianCharges, GaussianPoints
from .moist_contractions import density_fock, density_gradient, nuclear_potential
from .nonelectrostatic import Energies, NonelectrostaticSettings
from .nonelectrostatic_drop import DropTerms

#: Psi4 may store the (k|mn) of a cavity grid in core up to this size.
DEFAULT_INCORE_MB = 2000.0

#: The isodensity cavity's terms evaluate the basis on moist's grid this many bytes at a time.
BLOCK_BYTES = 64 * 2**20


@dataclass(frozen=True)
class MoistSettings:
    """A moist model for Psi4: the cavity and the components (both moist configurations).

    cavity: moist.DROP(lsf=moist.SvdW() | moist.CFC() | moist.Isodensity(), ...) or moist.ISwiG().
    components: e.g. (moist.ModelComponentCPCM(78.3553),).
    nthreads: moist's OpenMP threads; 0 takes Psi4's.
    verbosity: moist's own printout, which goes to the process's stdout (0 = silent).
    incore_mb: as PCMSettings.incore_mb: the (k|mn) of a grid are stored if they fit.  An
    isodensity cavity's grid moves every SCF iteration, so they are rebuilt each time.
    nonelectrostatic: the nonelectrostatic correction on moist's surface
    (nonelectrostatic_drop), post-SCF or self-consistent as its settings say, or None.
    radii: the cavity's sphere radii in bohr by atomic number (index 0 unused), any sequence,
    as PCMSettings.radii and with its default, cavity.modified_bondi_radii() (PySCF's modified
    Bondi x 1.2, hydrogen 1.10 A): moist's cavity is built on them (as moist.CustomRadii)
    rather than on moist's own CPCM radii, so that it has the same spheres as the
    switching/Gaussian cavity.  The cavity must then be left on moist's default radii
    (moist.DROP(...) without radii=, which moist cannot tell from radii=moist.CPCMRadii());
    a cavity given other radii is refused, so that the radii come from one place.  None
    keeps the cavity's own radii, e.g. one of moist's sets.  An isodensity surface is the
    density's contour; there the radii only place its points.
    """

    cavity: Any
    components: tuple[Any, ...]
    nthreads: int = 0
    verbosity: int = 0
    incore_mb: float = DEFAULT_INCORE_MB
    nonelectrostatic: NonelectrostaticSettings | None = None
    radii: tuple[float, ...] | None = field(default_factory=_solver._default_radii)

    def __post_init__(self) -> None:
        if not isinstance(self.cavity, moist.CavityConfiguration):
            raise TypeError("cavity must be a moist cavity configuration, e.g. moist.DROP(...)")
        components = tuple(self.components)
        if not components or not all(
            isinstance(c, moist.SolvationModelComponent) for c in components
        ):
            raise TypeError("components must be a nonempty sequence of moist components")
        object.__setattr__(self, "components", components)
        if self.radii is not None:
            radii = tuple(float(r) for r in self.radii)
            cavity: Any = self.cavity
            takes_radii = dataclasses.is_dataclass(cavity) and hasattr(cavity, "radii")
            if not takes_radii:
                raise TypeError(
                    f"{type(self.cavity).__name__} takes no radii; pass radii=None to keep its own"
                )
            custom = moist.CustomRadii(radii[1:], numbers=range(1, len(radii)))
            # (custom: these settings again, e.g. through dataclasses.replace)
            if self.cavity.radii not in (moist.CPCMRadii(), custom):
                raise ValueError(
                    f"the cavity has radii of its own ({self.cavity.radii!r}): give them as "
                    "MoistSettings.radii (by atomic number, in bohr), or pass radii=None to "
                    "keep the cavity's"
                )
            object.__setattr__(self, "radii", radii)
            object.__setattr__(self, "cavity", dataclasses.replace(cavity, radii=custom))


# ------------------------------------------------------------------ the basis as moist sees it


@dataclass
class _Shell:
    center: FloatArray
    atom: int  # index into the moist Structure (real atoms)
    exps: FloatArray
    coefs: FloatArray
    l: int  # noqa: E741
    cart: slice = slice(0)  # its Cartesian monomials in moist's layout
    powers: IntArray = field(default_factory=lambda: np.zeros((0, 3), dtype=np.intp))
    func: slice = slice(0)  # its Psi4 basis functions


class CartesianAOs:
    """Psi4's basis as moist's Cartesian monomials, for moist's internal isodensity evaluator.

    Each Psi4 shell becomes x^a y^b z^c sum_k c_k exp(-alpha_k r^2) with Psi4's own normalized
    contraction coefficients.  T (ncart, nbf) maps Psi4's functions onto them (chi = T^T chi_cart;
    the 2l+1 solid harmonics of a pure shell, in Psi4's order and normalization).  T is fitted
    shell by shell to Psi4's own basis-function values (BasisSet.compute_phi), so no convention
    has to be copied; the fit is exact to rounding and is checked.  Then

        D_cart = T D T^T,    F_psi4 = T^T F_cart T.

    values() evaluates the monomials and their derivatives (to any order; the gradient of the
    isodensity term needs 3) on the cavity grid.  fock() and gradient() take the grid in blocks
    of about BLOCK_BYTES of values.
    """

    FIT_TOL = 1e-10

    def __init__(self, basis: core.BasisSet, structure_index: dict[int, int]) -> None:
        self.basis = basis
        geom = np.asarray(basis.molecule().geometry().np)
        self.shells: list[_Shell] = []
        for s in range(basis.nshell()):
            sh = basis.shell(s)
            n = sh.nprimitive
            f0 = basis.shell_to_basis_function(s)
            self.shells.append(
                _Shell(
                    center=geom[sh.ncenter].copy(),
                    atom=structure_index[sh.ncenter],
                    exps=np.array([sh.exp(k) for k in range(n)]),
                    coefs=np.array([sh.coef(k) for k in range(n)]),
                    l=sh.am,
                    func=slice(f0, f0 + sh.nfunction),
                )
            )
        self.gaussian_basis = moist.GaussianBasis(
            shell_atom=[s.atom for s in self.shells],
            shell_l=[s.l for s in self.shells],
            shell_nprim=[len(s.exps) for s in self.shells],
            exponents=np.concatenate([s.exps for s in self.shells]),
            coefficients=np.concatenate([s.coefs for s in self.shells]),
        )
        self.ncart = self.gaussian_basis.ncart
        self.T = np.zeros((self.ncart, basis.nbf()))

    def bind_layout(self, offsets: Any, powers: Any) -> None:
        """Take moist's monomial order (cavity.isodensity_layout()) and fit T to it."""
        offsets = np.asarray(offsets, dtype=np.intp).ravel()  # nshell + 1: shell s is [o_s, o_s+1)
        powers = np.asarray(powers, dtype=np.intp)
        rng = np.random.default_rng(20261020)
        for sh, c0, c1 in zip(self.shells, offsets[:-1], offsets[1:], strict=True):
            sh.cart, sh.powers = slice(int(c0), int(c1)), powers[c0:c1]
            if not np.all(sh.powers.sum(1) == sh.l):
                raise RuntimeError("moist's isodensity layout does not match the basis")
            # Points around the shell's centre where its functions are well above zero
            npts = 4 * (c1 - c0) + 8
            u = rng.normal(size=(npts, 3))
            r = rng.uniform(0.2, 1.5, npts) / np.sqrt(sh.exps.min())
            pts = sh.center + u / np.linalg.norm(u, axis=1)[:, None] * r[:, None]
            psi4_vals = np.array([np.asarray(self.basis.compute_phi(*p))[sh.func] for p in pts])
            cart_vals = self._shell_values(sh, pts, 0)[(0, 0, 0)]
            Ts, *_ = np.linalg.lstsq(cart_vals, psi4_vals, rcond=None)
            resid = np.abs(cart_vals @ Ts - psi4_vals).max() / np.abs(psi4_vals).max()
            if resid > self.FIT_TOL:
                raise RuntimeError(f"Psi4 shell (l={sh.l}) is not in moist's span ({resid:.1e})")
            self.T[sh.cart, sh.func] = Ts

    def cartesian_density(self, D: FloatArray) -> FloatArray:
        Dc: FloatArray = self.T @ D @ self.T.T
        return 0.5 * (Dc + Dc.T)

    def to_psi4(self, F_cart: FloatArray) -> FloatArray:
        F: FloatArray = self.T.T @ F_cart @ self.T
        return 0.5 * (F + F.T)

    @staticmethod
    def _keys(order: int) -> list[tuple[int, int, int]]:
        return [
            (a, b, n - a - b) for n in range(order + 1) for a in range(n, -1, -1)
            for b in range(n - a, -1, -1)
        ]  # fmt: skip

    def _shell_values(
        self, sh: _Shell, points: FloatArray, order: int
    ) -> dict[tuple[int, int, int], FloatArray]:
        """d^(a,b,c) of the shell's monomials at points: {(a, b, c): (npts, ncomp)}."""
        d = points - sh.center
        r2 = np.einsum("ix,ix->i", d, d)
        nmax = sh.l + order
        keys = self._keys(order)
        out = {k: np.zeros((len(points), len(sh.powers))) for k in keys}
        for alpha, coef in zip(sh.exps, sh.coefs, strict=True):
            e = coef * np.exp(-alpha * r2)
            # p[t, k, n]: polynomial factor of d^k/dx_t^k [x_t^n exp(-alpha x_t^2)], from
            # d/dx g_n = n g_(n-1) - 2 alpha g_(n+1)
            p = np.zeros((3, order + 1, nmax + 1, len(points)))
            p[:, 0, 0] = 1.0
            for n in range(1, nmax + 1):
                p[:, 0, n] = p[:, 0, n - 1] * d.T
            for k in range(order):
                for n in range(nmax - k):
                    p[:, k + 1, n] = -2.0 * alpha * p[:, k, n + 1]
                    if n > 0:
                        p[:, k + 1, n] += n * p[:, k, n - 1]
            lx, ly, lz = sh.powers.T
            for a, b, c in keys:
                out[a, b, c] += e[:, None] * (p[0, a, lx] * p[1, b, ly] * p[2, c, lz]).T
        return out

    def values(self, points: FloatArray, order: int) -> dict[tuple[int, int, int], FloatArray]:
        """All monomials and derivatives to order at points: {(a, b, c): (npts, ncart)}."""
        out = {k: np.zeros((len(points), self.ncart)) for k in self._keys(order)}
        for sh in self.shells:
            for k, v in self._shell_values(sh, points, order).items():
                out[k][:, sh.cart] = v
        return out

    def _blocks(self, npts: int, narrays: int) -> Iterator[slice]:
        """The points in slices of about BLOCK_BYTES in narrays arrays of (points, ncart)."""
        step = max(1, BLOCK_BYTES // (narrays * self.ncart * 8))
        return (slice(p, min(p + step, npts)) for p in range(0, npts, step))

    def fock(self, points: FloatArray, item: Any) -> FloatArray:
        """The density weights of moist.DensityResponse against d rho/dD (Psi4 basis)."""
        w0, w1, w2 = (np.asarray(item.w_rho), np.asarray(item.w_grad_rho),
                      np.asarray(item.w_hess_rho))  # fmt: skip
        F = np.zeros((self.ncart, self.ncart))
        for p in self._blocks(len(points), 12):  # 10 arrays of values, and products
            F += density_fock(self.values(points[p], 2), w0[p], w1[p], w2[p])
        return self.to_psi4(F)

    def gradient(
        self, points: FloatArray, D_cart: FloatArray, item: Any, natom: int
    ) -> FloatArray:
        """The density weights against d rho/dR_A at fixed D (basis centres move), (natom, 3)."""
        w0, w1, w2 = (np.asarray(item.w_rho), np.asarray(item.w_grad_rho),
                      np.asarray(item.w_hess_rho))  # fmt: skip
        per_cart = np.zeros((3, self.ncart))
        for p in self._blocks(len(points), 32):  # 20 arrays of values, 10 of products with D
            per_cart += density_gradient(self.values(points[p], 3), D_cart, w0[p], w1[p], w2[p])
        g = np.zeros((natom, 3))
        for sh in self.shells:
            g[sh.atom] += per_cart[:, sh.cart].sum(1)
        return g


# ------------------------------------------------------------------ the solver in the DDX slot


class MoistSolver:
    """Duck-types psi4's DdxInterface for a moist model: get_solvation_contributions(D, state).

    One instance per SCF (Psi4 builds it in scf_helper), i.e. per geometry.  Each call runs the
    coupling protocol at the total density D: the build phase (every call for an isodensity
    cavity, once otherwise), the energy phase and the response phase, whose items make the Fock
    contribution.  gradient() runs the gradient phase at the last density.
    """

    # The current cavity grid, set by _build: the points, their Gaussians, and the nuclei's
    # potential there (phi, dphi_dr, dphi_dxi).
    points: GaussianPoints
    charges: GaussianCharges
    nuc: tuple[FloatArray, FloatArray, FloatArray]

    def __init__(
        self,
        settings: MoistSettings,
        molecule: core.Molecule,
        options: Any,
        basisset: core.BasisSet,
    ) -> None:
        self.settings, self.basis = settings, basisset
        self.natom = molecule.natom()
        self.atoms: IntArray = np.array(
            [i for i in range(self.natom) if molecule.Z(i) != 0], dtype=np.intp
        )
        if len(self.atoms) == 0:
            raise ValueError("moist needs at least one real (non-ghost) atom")
        self.density_dependent = bool(settings.cavity.density_dependent)
        if self.density_dependent and len(self.atoms) < self.natom:
            raise NotImplementedError("ghost atoms with a density-dependent (isodensity) cavity")
        self.xyz = np.asarray(molecule.geometry().np)[self.atoms]
        # Electrons see ECP-reduced nuclear charges; the structure keeps the elements.
        self.Z = np.array([molecule.Z(int(i)) for i in self.atoms], dtype=np.float64)
        numbers = np.array([molecule.true_atomic_number(int(i)) for i in self.atoms])
        self.nthreads = core.get_num_threads()
        self.context = moist.Context(
            nthreads=settings.nthreads or self.nthreads, verbosity=settings.verbosity
        )
        self.aos: CartesianAOs | None = None
        if self.density_dependent:
            self.aos = CartesianAOs(basisset, {int(a): i for i, a in enumerate(self.atoms)})
            live = settings.cavity.build(basis=self.aos.gaussian_basis)
            self.aos.bind_layout(*live.isodensity_layout())
        else:
            live = settings.cavity.build()
        self.model = moist.SolvationModel(self.context, live, settings.components)
        self.model.update(moist.Structure(numbers, self.xyz, charge=molecule.molecular_charge()))
        self.coupling = self.model.new_coupling()
        self.D: core.Matrix | None = None
        self.response: Any = None
        self.energy = 0.0
        self.calls = 0
        self._terms: DropTerms | None = None
        self.e_nonelectrostatic: Energies | None = None
        if not self.density_dependent:
            self._build(None)
        self._print_header()

    def _print_header(self) -> None:
        cav = self.settings.cavity
        lsf = getattr(cav, "lsf", None)
        name = cav.name + (f"({lsf.name})" if lsf is not None else "")
        comps = ", ".join(c.name for c in self.model.components)
        if self.density_dependent:
            cavity = "rebuilt at every SCF density"
            integrals = (
                f"for each new grid; stored while they fit {self.settings.incore_mb:.0f} MB"
            )
        else:
            stored = self.charges.B
            cavity = f"{self.points.npts} surface points"
            integrals = (
                f"stored in core ({stored.nbytes / 2**20:.0f} MB)"
                if stored is not None
                else "computed directly each iteration"
            )
        core.print_out(
            f"\n  ==> psi4-swig-pcm / moist {moist.__version__}: {name}, {comps} <==\n\n"
            f"    Cavity: {len(self.atoms)} atoms, {cavity}\n"
            f"    Surface-charge integrals (k|mn): {integrals}\n"
        )

    # -------------------------------------------------------------- the coupling protocol

    def _build(self, D: core.Matrix | None) -> None:
        """Build phase: moist (re)builds the cavity; then the Gaussians on its new grid."""
        self.model.prepare_build(self.coupling)
        for request in self.coupling:
            if isinstance(request, moist.DensityMatrixRequest):
                if D is None or self.aos is None:
                    raise RuntimeError("moist asked for the density before there was one")
                self.coupling.answer(dcart=self.aos.cartesian_density(np.asarray(D.np)))
            elif isinstance(request, moist.NuclearChargesRequest):
                self.coupling.answer(z=self.Z)
            else:
                raise NotImplementedError(f"moist build request {request.name!r}")
        self.model.build(self.coupling)
        cav = self.model.cavity
        self.points = GaussianPoints(np.array(cav.xyz), np.array(cav.xi0))
        if hasattr(self, "charges"):
            del self.charges  # the last grid's stored integrals go before this grid's are stored
        self.charges = GaussianCharges(
            self.points, self.nthreads, self.basis, self.settings.incore_mb
        )
        self.nuc = nuclear_potential(self.points.coords, self.points.xi, self.xyz, self.Z)

    def _answer_all(self, D: core.Matrix, field: FloatArray | None = None) -> None:
        """Answer every missing output of the staged phase on the current grid at D.

        field: the electronic dphi_dr at D, when it is already known.
        """
        for request in self.coupling:
            if isinstance(request, moist.GaussianPotentialRequest):
                missing, out = request.missing, {}
                phi_nuc, dphi_dr_nuc, dphi_dxi_nuc = self.nuc
                if "phi" in missing:
                    out["phi"] = phi_nuc + self.charges.esp(self.basis, D)
                if "dphi_dr" in missing:
                    if field is None:
                        field = self.charges.esp_deriv1(self.basis, D)[0]
                    out["dphi_dr"] = dphi_dr_nuc + field
                if "dphi_dxi" in missing:
                    out["dphi_dxi"] = dphi_dxi_nuc + self.charges.esp_dxi(self.basis, D)
                if unsupported := missing - out.keys():
                    raise NotImplementedError(
                        f"moist asks for {sorted(unsupported)}: second derivatives of the "
                        "potential, which this prototype does not compute yet.  The models "
                        "that use the normal derivative of the potential (DPCM, SLIC, "
                        "nonlocal) need them."
                    )
                self.coupling.answer(**out)
            elif isinstance(request, moist.NuclearChargesRequest):
                self.coupling.answer(z=self.Z)
            else:
                raise NotImplementedError(f"moist request {request.name!r}")

    def _evaluate(self, D: core.Matrix) -> None:
        """The build (isodensity), energy and response phases at D; the results are kept with D."""
        # No result until this one is complete: a failure leaves none, rather than the last
        # density's paired with a new grid.
        self.D = self.response = None
        if self.density_dependent:
            self._build(D)
        self.model.prepare_energy(self.coupling)
        self._answer_all(D)
        energy = np.array(0.0)
        self.model.get_energy(self.coupling, energy)
        self.model.prepare_response(self.coupling)
        self._answer_all(D)
        self.response = self.model.get_response(self.coupling)
        self.energy, self.D = float(energy), D.clone()
        self.calls += 1

    def _fock(self) -> core.Matrix:
        V = np.zeros((self.basis.nbf(), self.basis.nbf()))
        for item in self.response:
            if isinstance(item, moist.PotentialAdjointResponse):
                if item.w_dphi_dr is not None:
                    raise NotImplementedError(
                        "moist's w_dphi_dr: the models that use the normal derivative of the "
                        "potential (DPCM, SLIC, nonlocal) are not supported yet"
                    )
                V += np.asarray(
                    self.charges.potential_matrix(self.basis, np.asarray(item.w_phi)).np
                )
            elif isinstance(item, moist.DensityResponse) and self.aos is not None:
                V += self.aos.fock(self.points.coords, item)
            else:
                raise NotImplementedError(f"moist response item {item.name!r}")
        return core.Matrix.from_array(V)

    def get_solvation_contributions(
        self,
        D: core.Matrix,
        state: Any = None,
        elec_only: bool = False,
        nonequilibrium: bool = False,
    ) -> tuple[float, core.Matrix, Any]:
        if elec_only or nonequilibrium:
            raise NotImplementedError("moist: electronic-only / nonequilibrium (CPSCF) response")
        self._evaluate(D)
        energy, V = self.energy, self._fock()
        ne = self.settings.nonelectrostatic
        if ne is not None and ne.self_consistent:
            energies, Vg = self.terms.energy_and_potential(np.asarray(D.np))
            self.e_nonelectrostatic = energies
            energy += energies.total
            V.add(core.Matrix.from_array(Vg))
        if _solver._perturbation is not None:  # driver.post_scf_gradient's perturbed SCFs
            lam, V0 = _solver._perturbation
            energy += lam * float(np.sum(V0 * np.asarray(D.np)))
            V.add(core.Matrix.from_array(lam * V0))
        return energy, V, state

    @property
    def terms(self) -> DropTerms:
        """The nonelectrostatic correction on this geometry's cavity (built on first use)."""
        ne = self.settings.nonelectrostatic
        if ne is None:
            raise ValueError("these MoistSettings have no nonelectrostatic correction")
        if self._terms is None:
            self._terms = DropTerms(self, ne, self.nthreads)
        return self._terms

    def gradient(self, D: core.Matrix) -> FloatArray:
        """moist's part of the nuclear gradient at the total density D, (natom, 3).

        The rest is Psi4's own SCF gradient at the same density, whose energy-weighted density
        already includes moist's Fock contribution.  Three routes add up here:
        model     moist's own terms: the cavity moving with the atoms, the nuclear potential;
        potential the potential adjoint against d(k|mn)/dR at fixed points;
        density   (isodensity) the response phase's density weights against d rho/dR.

        One pass of the derivative integrals gives the potential route and, on a fixed cavity,
        the field (dphi_dr) the gradient phase asks for.
        """
        if self.D is None or not np.array_equal(np.asarray(D.np), np.asarray(self.D.np)):
            self._evaluate(D)
        response = self.response  # the response phase's items: only they carry `density`
        w_phi = [
            np.asarray(item.w_phi)
            for item in response
            if isinstance(item, moist.PotentialAdjointResponse)
        ]
        field, g_potential = (
            self.charges.esp_deriv1(self.basis, D, np.sum(w_phi, axis=0)) if w_phi else (None, 0.0)
        )
        self.model.prepare_gradient(self.coupling)
        self._answer_all(D, field)
        g_model = np.zeros((len(self.atoms), 3))
        self.model.get_gradient(self.coupling, g_model)
        g = np.zeros((self.natom, 3)) + g_potential
        g[self.atoms] += g_model
        for item in response:
            if isinstance(item, moist.PotentialAdjointResponse):
                continue  # g_potential
            if isinstance(item, moist.DensityResponse) and self.aos is not None:
                D_cart = self.aos.cartesian_density(np.asarray(D.np))
                g[self.atoms] += self.aos.gradient(
                    self.points.coords, D_cart, item, len(self.atoms)
                )
            else:
                raise NotImplementedError(f"moist response item {item.name!r}")
        ne = self.settings.nonelectrostatic
        if ne is not None and ne.self_consistent:
            g += self.terms.gradient(np.asarray(D.np))
        return g
