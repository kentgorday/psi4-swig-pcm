# Copyright (C) 2026 Kent Gorday
# SPDX-License-Identifier: LGPL-3.0-only
"""The nonelectrostatic correction to the PCM, as a function of the density matrix.

The model and its numerics are tautomator's (tautomator/nonelectrostatic.py and
pyscf_nonelectrostatic.py, docs/nonelectrostatic.md), reimplemented here on Psi4's basis
functions and libint2, so that Psi4 runs need neither PySCF nor tautomator:

    G = A * dispersion + B * exchange + sum_i A_i f(F_i)

    dispersion  sum_{p inside} w_p e(rho, |grad rho|^2) I(r_p),  e = rho / (w (w + sqrt(rho_s))),
                w = sqrt(rho + (3C/4pi) |grad rho|^4 / rho^4), I the damped R^-6 kernel to the
                solvent outside the cavity
    exchange    sum_{p outside} w_p |grad rho|
    short range f(F) = c- |F|^n- (F < 0), c+ F^n+ (F > 0), F_i the solute's own normal field
                (nuclei + electrons) at surface element i, A_i its area

The cavity is the PCM's union of spheres; its inside and outside are integrated on rigid
per-atom ray grids with a smooth (C^2) partition among the atoms, so G is smooth in the
geometry.  `Terms` gives G, its potential V = dG/dD (the same for both spins) and its explicit
nuclear gradient at fixed density matrix: the basis functions, the grids, the surface elements
and the nuclei moving with their atoms.

Everything is in atomic units; D is the total (alpha + beta) AO density matrix, symmetric.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from psi4 import core
from scipy.integrate import lebedev_rule

from . import _nonelectrostatic as _ne
from .cavity import FloatArray, IntArray, Surface, dF_dA

Region = Literal["inside", "outside"]

# The defaults and floors of tautomator's implementation; the cross-check against it
# (tests/test_nonelectrostatic.py, pyscf marker) holds them equal.
DELTA = 7.0  # damping length of the dispersion kernel (bohr), CMIRS's value
VV_C = 0.0093  # gradient-correction constant of the VV10 local frequency
# (Lebedev order, radial nodes per direction) of the inside and outside grids
INSIDE_GRID: tuple[int, int] = (23, 12)
OUTSIDE_GRID: tuple[int, int] = (41, 12)
STEP_WIDTH = 2.0  # bohr; width of the partition's hand-over at a sphere's surface
EXTERIOR_MAP = 0.5  # bohr; t = r_a + L (1 + x)/(1 - x) on the outside rays
RHO_FLOOR = 1e-14  # densities at or below this are left out of the dispersion integrand
GRAD_FLOOR = 1e-14  # |grad rho| is taken as x^2 / sqrt(x^2 + floor^2)
FIELD_FLOOR = 1e-8  # and |F| likewise
_MIN_WEIGHT = 1e-14
_AO_BLOCK_BYTES = 200e6


@dataclass(frozen=True)
class NonelectrostaticSettings:
    """The correction's parameters (atomic units) and how it enters the SCF.

    self_consistent: add G to the SCF energy and V to the Fock matrix every iteration (the
    density relaxes with the correction), rather than evaluate G once on the converged PCM
    density (post-SCF, as CMIRS does).  solvent_rho: the liquid's mean electron density
    (e/bohr^3).
    """

    dispersion: float
    exchange: float
    neg_coeff: float
    neg_exponent: float
    pos_coeff: float
    pos_exponent: float
    solvent_rho: float
    self_consistent: bool = False
    delta: float = DELTA
    vv_c: float = VV_C
    inside_grid: tuple[int, int] = INSIDE_GRID
    outside_grid: tuple[int, int] = OUTSIDE_GRID


@dataclass(frozen=True)
class Energies:
    dispersion: float
    exchange: float
    short_range: float

    @property
    def total(self) -> float:
        return self.dispersion + self.exchange + self.short_range


# ------------------------------------------------------------------------------- integrands


def dispersion_density(
    rho: FloatArray, grad_sq: FloatArray, solvent_rho: float, vv_c: float = VV_C
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """rho / (w (w + sqrt(rho_s))) and its derivatives in rho and |grad rho|^2."""
    use = rho > RHO_FLOOR
    r = np.where(use, rho, 1.0)
    s = np.where(use, grad_sq, 0.0)
    c = 3.0 * vv_c / (4.0 * np.pi)
    q = np.sqrt(solvent_rho)
    w = np.sqrt(r + c * s**2 / r**4)
    denom = w * (w + q)
    d_w = -r * (2.0 * w + q) / denom**2
    value = np.where(use, r / denom, 0.0)
    d_rho = np.where(use, 1.0 / denom + d_w * (1.0 - 4.0 * c * s**2 / r**5) / (2.0 * w), 0.0)
    d_grad_sq = np.where(use, d_w * c * s / (r**4 * w), 0.0)
    return value, d_rho, d_grad_sq


def exchange_density(grad_sq: FloatArray) -> tuple[FloatArray, FloatArray]:
    """|grad rho| (regularized) and its derivative in |grad rho|^2."""
    shifted = grad_sq + GRAD_FLOOR**2
    root = np.sqrt(shifted)
    return grad_sq / root, (grad_sq + 2.0 * GRAD_FLOOR**2) / (2.0 * shifted * root)


def field_power(
    field: FloatArray, exponent: float, sign: Literal[-1, 1]
) -> tuple[FloatArray, FloatArray]:
    """|F|^exponent on the elements whose field has the sign, 0 elsewhere, and d/dF."""
    shifted = field**2 + FIELD_FLOOR**2
    root = np.sqrt(shifted)
    magnitude = field**2 / root
    d_magnitude = field * (field**2 + 2.0 * FIELD_FLOOR**2) / (shifted * root)
    mask = field * sign > 0
    with np.errstate(divide="ignore", invalid="ignore"):
        d = exponent * magnitude ** (exponent - 1.0) * d_magnitude
    return np.where(mask, magnitude**exponent, 0.0), np.where(mask, d, 0.0)


# ------------------------------------------------------------------------------- cavity grids


@dataclass(frozen=True)
class CavityGrid:
    """Points on rigid per-atom rays, weights = base weights x partition (owners: cavity atoms)."""

    coords: FloatArray
    weights: FloatArray
    owners: IntArray
    base_weights: FloatArray
    region: Region


def _pair_distances(centres: FloatArray) -> FloatArray:
    d: FloatArray = np.linalg.norm(centres[:, None] - centres[None], axis=2)
    np.fill_diagonal(d, 1.0)
    return d


def build_cavity_grid(
    centres: FloatArray,
    radii: FloatArray,
    region: Region,
    lebedev_order: int | None = None,
    n_radial: int | None = None,
    nthreads: int = 1,
) -> CavityGrid:
    default = INSIDE_GRID if region == "inside" else OUTSIDE_GRID
    lebedev_order = default[0] if lebedev_order is None else lebedev_order
    n_radial = default[1] if n_radial is None else n_radial
    directions, w_ang = lebedev_rule(lebedev_order)
    unit = np.ascontiguousarray(directions.T)
    x, w = np.polynomial.legendre.leggauss(n_radial)
    coords, base, owners = [], [], []
    for a, (centre, radius) in enumerate(zip(centres, radii, strict=True)):
        if region == "inside":
            t = 0.5 * radius * (x + 1.0)
            wt = 0.5 * radius * w
        else:
            t = radius + EXTERIOR_MAP * (1.0 + x) / (1.0 - x)
            wt = 2.0 * EXTERIOR_MAP * w / (1.0 - x) ** 2
        coords.append((centre + t[:, None, None] * unit[None]).reshape(-1, 3))
        base.append(((wt * t**2)[:, None] * w_ang[None]).ravel())
        owners.append(np.full(len(t) * len(unit), a, dtype=np.intp))
    points = np.ascontiguousarray(np.vstack(coords))
    base_weights = np.concatenate(base)
    owner = np.concatenate(owners)
    partition = _ne.cavity_partition(
        points,
        owner,
        centres,
        radii,
        _pair_distances(centres),
        region == "inside",
        STEP_WIDTH,
        nthreads,
    )
    weights = base_weights * partition
    keep = weights > _MIN_WEIGHT
    return CavityGrid(points[keep], weights[keep], owner[keep], base_weights[keep], region)


def partition_gradient(
    grid: CavityGrid, centres: FloatArray, radii: FloatArray, values: FloatArray, nthreads: int
) -> FloatArray:
    """d/dR (cavity atoms) of sum_p values_p weights_p, values held fixed."""
    out: FloatArray = _ne.cavity_partition_gradient(
        grid.coords,
        grid.owners,
        centres,
        radii,
        _pair_distances(centres),
        grid.region == "inside",
        STEP_WIDTH,
        np.ascontiguousarray(grid.base_weights * values),
        nthreads,
    )
    return out


# ------------------------------------------------------------------------------- basis functions

_SECOND = (("XX", 0, 0), ("XY", 0, 1), ("XZ", 0, 2), ("YY", 1, 1), ("YZ", 1, 2), ("ZZ", 2, 2))


def ao_values(basis: core.BasisSet, points: FloatArray, deriv: int) -> FloatArray:
    """phi (and derivatives) at points: (1, 4 or 10, n, nbf), Psi4's AO order.

    Components 0, x, y, z, then xx, xy, xz, yy, yz, zz for deriv 2 (PySCF's eval_ao order).
    """
    nbf, n = basis.nbf(), len(points)
    extents = core.BasisExtents(basis, 0.0)  # every function at every point
    bf = core.BasisFunctions(basis, n, nbf)
    bf.set_deriv(deriv)
    block = core.BlockOPoints(
        *(core.Vector.from_array(np.ascontiguousarray(points[:, k])) for k in range(3)),
        core.Vector.from_array(np.ones(n)),
        extents,
    )
    bf.compute_functions(block)
    values = bf.basis_values()
    local = np.asarray(block.functions_local_to_global(), dtype=np.intp)
    names = ["PHI"] + (["PHI_X", "PHI_Y", "PHI_Z"] if deriv >= 1 else [])
    names += [f"PHI_{c}" for c, _, _ in _SECOND] if deriv >= 2 else []
    out = np.zeros((len(names), n, nbf))
    for k, name in enumerate(names):
        out[k][:, local] = np.asarray(values[name].np)[:n, : len(local)]
    return out


def basis_motion_gradient(
    ao: FloatArray, D: FloatArray, coeff: FloatArray, ao_atom: IntArray, natom: int
) -> FloatArray:
    """sum_p sum_k coeff_k d u_k / dR as the basis functions move, u = (rho[, grad rho]).

    Same as tautomator's pyscf_nonelectrostatic.basis_motion_gradient.
    """
    c0 = ao[0] @ D
    T = np.einsum("kpi,p,pi->ki", ao[1:4], coeff[0], c0)
    if len(coeff) == 4:
        c1 = np.einsum("kpi,ij->kpj", ao[1:4], D)
        T += np.einsum("kpi,yp,ypi->ki", ao[1:4], coeff[1:], c1)
        for comp, (_name, k, kk) in enumerate(_SECOND, start=4):
            T[k] += np.einsum("pi,p,pi->i", ao[comp], coeff[1 + kk], c0)
            if k != kk:
                T[kk] += np.einsum("pi,p,pi->i", ao[comp], coeff[1 + k], c0)
    g = np.empty((natom, 3))
    for x in range(3):
        g[:, x] = -2.0 * np.bincount(ao_atom, T[x], minlength=natom)
    return g


def _blocks(n: int, values_per_point: int) -> list[slice]:
    size = max(256, int(_AO_BLOCK_BYTES / (8 * max(1, values_per_point))))
    return [slice(p0, min(n, p0 + size)) for p0 in range(0, n, size)]


# ------------------------------------------------------------------------------- the terms


class Terms:
    """The correction at one geometry, for any density matrix D (see the module docstring).

    surf, atoms and radii are the PCM's: the cavity's surface, the molecule's atoms that carry
    a sphere (ghost atoms do not), and those spheres' radii (bohr).  Nuclei enter the field
    with the charges the electrons see (ECP-reduced).
    """

    def __init__(
        self,
        molecule: core.Molecule,
        basis: core.BasisSet,
        surf: Surface,
        atoms: IntArray,
        radii: FloatArray,
        settings: NonelectrostaticSettings,
        nthreads: int = 1,
        store_field_integrals: bool = False,
        store_mb: float = 2000.0,
    ) -> None:
        self.basis, self.settings, self.nthreads = basis, settings, nthreads
        self.surf, self.atoms = surf, np.asarray(atoms, dtype=np.intp)
        self.natom = molecule.natom()
        self.centres = np.ascontiguousarray(surf.atom_coords, dtype=float)
        self.radii = np.ascontiguousarray(radii, dtype=float)
        self.coords = np.ascontiguousarray(surf.coords)
        self.normals = np.ascontiguousarray(surf.normals)
        self.areas = np.ascontiguousarray(surf.area)
        self.element_atoms = self.atoms[surf.atom_of_point]
        geometry = np.asarray(molecule.geometry().np)
        self.nuclei = geometry
        self.charges = np.array([molecule.Z(i) for i in range(self.natom)], dtype=float)
        s = settings
        self.inside = build_cavity_grid(
            self.centres, self.radii, "inside", *s.inside_grid, nthreads
        )
        self.outside = build_cavity_grid(
            self.centres, self.radii, "outside", *s.outside_grid, nthreads
        )
        J = _ne.dispersion_kernel(
            self.inside.coords, self.coords, self.normals, self.areas, s.delta, None, nthreads
        )[0]
        self.kernel = 2.0 * np.pi**2 / (3.0 * s.delta**3) - np.asarray(J)
        self.ao_atom = np.array([basis.function_to_center(i) for i in range(basis.nbf())])
        nbf = basis.nbf()
        self.tril = np.tril_indices(nbf)
        self.offdiag = self.tril[0] != self.tril[1]
        self.field_matrix: FloatArray | None = None
        if store_field_integrals and len(self.areas) * nbf * (nbf + 1) / 2 * 8 / 2**20 <= store_mb:
            self.field_matrix = np.empty((len(self.areas), nbf * (nbf + 1) // 2))
            _ne.field_matrix_packed_into(
                basis, self.coords, self.normals, self.field_matrix, nthreads
            )

    # -- pieces ---------------------------------------------------------------------------

    def _integrand(
        self, grid: CavityGrid, sl: slice, rho: FloatArray, grad_sq: FloatArray
    ) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
        s = self.settings
        if grid.region == "inside":
            e, e_r, e_s = dispersion_density(rho, grad_sq, s.solvent_rho, s.vv_c)
            return s.dispersion * e, s.dispersion * e_r, s.dispersion * e_s, self.kernel[sl]
        e, e_s = exchange_density(grad_sq)
        return s.exchange * e, np.zeros_like(e), s.exchange * e_s, np.ones_like(e)

    def _surface_function(self, F: FloatArray) -> tuple[FloatArray, FloatArray]:
        s = self.settings
        neg, d_neg = field_power(F, s.neg_exponent, -1)
        pos, d_pos = field_power(F, s.pos_exponent, 1)
        return s.neg_coeff * neg + s.pos_coeff * pos, s.neg_coeff * d_neg + s.pos_coeff * d_pos

    def _nuclear_field(self) -> tuple[FloatArray, FloatArray]:
        sep = self.coords[:, None] - self.nuclei[None]
        dist = np.linalg.norm(sep, axis=2)
        with np.errstate(divide="ignore", invalid="ignore"):
            field = np.einsum("a,pax->px", self.charges, np.nan_to_num(sep / dist[..., None] ** 3))
            tensor = (
                np.eye(3)[None, None] / dist[..., None, None] ** 3
                - 3.0 * np.einsum("pax,pay->paxy", sep, sep) / dist[..., None, None] ** 5
            )
        return field, np.nan_to_num(tensor)

    def _normal_field(self, D: FloatArray) -> FloatArray:
        nuclear, _ = self._nuclear_field()
        F: FloatArray = np.einsum("px,px->p", nuclear, self.normals)
        if self.field_matrix is not None:
            d = D[self.tril].copy()
            d[self.offdiag] *= 2.0
            return F + self.field_matrix @ d
        return F + np.asarray(
            _ne.normal_field(self.basis, self.coords, self.normals, D, self.nthreads)
        )

    # -- energy and potential ----------------------------------------------------------------

    def energy(self, D: FloatArray) -> Energies:
        return self.energy_and_potential(D, potential=False)[0]

    def energy_and_potential(
        self, D: FloatArray, potential: bool = True
    ) -> tuple[Energies, FloatArray | None]:
        D = np.asarray(D, dtype=float)
        nbf = self.basis.nbf()
        V = np.zeros((nbf, nbf)) if potential else None
        grid_energy: dict[str, float] = {}
        for grid in (self.inside, self.outside):
            total = 0.0
            for sl in _blocks(len(grid.weights), 4 * nbf):
                ao = ao_values(self.basis, grid.coords[sl], 1)
                c0 = ao[0] @ D
                rho = np.einsum("pi,pi->p", ao[0], c0)
                grad = 2.0 * np.einsum("kpi,pi->kp", ao[1:4], c0)
                e, e_r, e_s, kernel = self._integrand(grid, sl, rho, np.sum(grad**2, axis=0))
                w = grid.weights[sl] * kernel
                total += float(w @ e)
                if V is not None:
                    wv = np.empty((4, len(w)))
                    wv[0] = 0.5 * w * e_r
                    wv[1:] = 2.0 * w * e_s * grad
                    V += ao[0].T @ np.einsum("kpi,kp->pi", ao, wv)
            grid_energy[grid.region] = total
        if V is not None:
            V = V + V.T
        F = self._normal_field(D)
        f, f_prime = self._surface_function(F)
        if V is not None:
            c = self.areas * f_prime
            if self.field_matrix is not None:
                packed = c @ self.field_matrix
                Vs = np.zeros((nbf, nbf))
                Vs[self.tril] = packed
                Vs.T[self.tril] = packed
                V += Vs
            else:
                V += _ne.normal_field_matrix(
                    self.basis, self.coords, self.normals, c, self.nthreads
                )
        energies = Energies(grid_energy["inside"], grid_energy["outside"], float(self.areas @ f))
        return energies, V

    # -- explicit nuclear gradient -------------------------------------------------------------

    def gradient(self, D: FloatArray) -> FloatArray:
        """dG/dR (natom, 3) at fixed D: basis functions, grids, surface and nuclei move."""
        D = np.asarray(D, dtype=float)
        nbf, natom = self.basis.nbf(), self.natom
        g = np.zeros((natom, 3))
        g_cavity = np.zeros((len(self.atoms), 3))  # per cavity atom, mapped at the end
        dispersion_values = np.zeros(len(self.inside.weights))
        for grid in (self.inside, self.outside):
            values = np.zeros(len(grid.weights))
            for sl in _blocks(len(grid.weights), 14 * nbf):
                ao = ao_values(self.basis, grid.coords[sl], 2)
                c0 = ao[0] @ D
                c1 = np.einsum("kpi,ij->kpj", ao[1:4], D)
                rho = np.einsum("pi,pi->p", ao[0], c0)
                grad = 2.0 * np.einsum("kpi,pi->kp", ao[1:4], c0)
                hess = np.empty((3, 3, len(rho)))
                for comp, (_name, k, kk) in enumerate(_SECOND, start=4):
                    h = 2.0 * (
                        np.einsum("pi,pi->p", ao[comp], c0)
                        + np.einsum("pi,pi->p", ao[1 + k], c1[kk])
                    )
                    hess[k, kk] = hess[kk, k] = h
                e, e_r, e_s, kernel = self._integrand(grid, sl, rho, np.sum(grad**2, axis=0))
                values[sl] = e * kernel
                if grid.region == "inside":
                    dispersion_values[sl] = e
                w = grid.weights[sl]
                motion = (w * kernel)[:, None] * (
                    e_r[:, None] * grad.T
                    + 2.0 * e_s[:, None] * np.einsum("klp,lp->pk", hess, grad)
                )
                for x in range(3):
                    g_cavity[:, x] += np.bincount(
                        grid.owners[sl], motion[:, x], minlength=len(self.atoms)
                    )
                coeff = np.empty((4, len(w)))
                coeff[0] = w * kernel * e_r
                coeff[1:] = 2.0 * (w * kernel * e_s)[None] * grad
                g += basis_motion_gradient(ao, D, coeff, self.ao_atom, natom)
            g_cavity += partition_gradient(grid, self.centres, self.radii, values, self.nthreads)

        s = self.settings
        _J, dJ, dE_ds, dE_dA = (
            np.asarray(a)
            for a in _ne.dispersion_kernel(
                self.inside.coords,
                self.coords,
                self.normals,
                self.areas,
                s.delta,
                np.ascontiguousarray(self.inside.weights * dispersion_values),
                self.nthreads,
            )
        )
        motion = -(self.inside.weights * dispersion_values)[:, None] * dJ
        for x in range(3):
            g_cavity[:, x] += np.bincount(
                self.inside.owners, motion[:, x], minlength=len(self.atoms)
            )
        element_force = -dE_ds
        area_force = -dE_dA

        F = self._normal_field(D)
        f, f_prime = self._surface_function(F)
        area_force += f
        q = np.ascontiguousarray((self.areas * f_prime)[:, None] * self.normals)
        g_basis, g_points = _ne.field_gradient(self.basis, self.coords, D, q, natom, self.nthreads)
        g += np.asarray(g_basis)
        element_force += np.asarray(g_points)
        _nuclear, tensor = self._nuclear_field()
        element_force += np.einsum("px,a,paxy->py", q, self.charges, tensor)
        g -= np.einsum("a,px,paxy->ay", self.charges, q, tensor)
        for x in range(3):
            g_cavity[:, x] += np.bincount(
                self.surf.atom_of_point, element_force[:, x], minlength=len(self.atoms)
            )
        _dF, dA = dF_dA(self.surf)
        g_cavity += np.einsum("i,iax->ax", area_force, dA)
        g[self.atoms] += g_cavity
        return g
