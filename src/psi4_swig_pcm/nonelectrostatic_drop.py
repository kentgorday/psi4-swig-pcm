# Copyright (C) 2026 Kent Gorday
# SPDX-License-Identifier: LGPL-3.0-only
"""The nonelectrostatic correction on moist's DROP cavities (moist_host), on Psi4's basis.

tautomator's pyscf_nonelectrostatic_drop and surface_quadrature, reimplemented here as
nonelectrostatic.py reimplements its sphere-cavity version: the same three terms

    G = A int_inside u I + B int_outside |grad rho| + sum_i a_i f(n_i . E(s_i))

on the surface moist builds for its PCM (SvdW-DROP, or rho-DROP, which follows the density).
The kernel I is the surface sum of the sphere-cavity version over moist's points, normals and
areas.  The outside is a weighted point set on rays from the nuclei through every surface
point (the divergence theorem, each atom taking its share through the Stratmann-Scuseria-Frisch
partition); the inside is an all-space atom grid less the outside.  G then depends on the
cavity only through its points s_i, normals n_i and areas a_i, and moist contracts dG/d(s, n,
a) into the nuclear gradient and, on rho-DROP, into density weights (moist_contract_cavity_
surface_gradient and _density, which packages/moist's patch 0002 exports):

    Fock      V = dG/dD at a fixed surface, plus on rho-DROP the density weights against
              d rho/dD (CartesianAOs.fock);
    gradient  dG/dR at a fixed surface and density matrix, plus moist's contraction, plus on
              rho-DROP the density weights against d rho/dR (CartesianAOs.gradient).

`DropTerms` is the correction at one geometry, on a MoistSolver's cavity.  D is the total
(alpha + beta) AO density matrix; everything is in atomic units.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import moist
import numpy as np
from scipy.integrate import lebedev_rule

from . import _nonelectrostatic as _ne
from .cavity import FloatArray, IntArray
from .nonelectrostatic import (
    _SECOND,
    Energies,
    NonelectrostaticSettings,
    _blocks,
    ao_values,
    basis_motion_gradient,
    dispersion_density,
    exchange_density,
    field_power,
)

if TYPE_CHECKING:
    from .moist_host import MoistSolver

# Stratmann-Scuseria-Frisch's a (tautomator's surface_quadrature.SSF_A)
SSF_A = 0.64


def mapped_legendre(n: int, length: float) -> tuple[FloatArray, FloatArray]:
    """Gauss-Legendre on [0, infinity) after t = L (1 + x)/(1 - x): nodes and weights."""
    x, w = np.polynomial.legendre.leggauss(n)
    t: FloatArray = length * (1.0 + x) / (1.0 - x)
    wt: FloatArray = w * 2.0 * length / (1.0 - x) ** 2
    return t, wt


# ------------------------------------------------------------------------------- quadrature


@dataclass(frozen=True)
class VolumeGrid:
    """An all-space grid around the cavity atoms, each atom's points moving with it."""

    coords: FloatArray
    weights: FloatArray
    owners: IntArray  # cavity-atom index
    base_weights: FloatArray


def build_volume_grid(
    centres: FloatArray, rule: tuple[int, int, float], nthreads: int
) -> VolumeGrid:
    order, n_radial, length = rule
    directions, w_ang = lebedev_rule(order)
    t, w_rad = mapped_legendre(n_radial, length)
    local = (t[:, None, None] * directions.T[None]).reshape(-1, 3)
    w_local = (w_rad[:, None] * t[:, None] ** 2 * w_ang[None]).ravel()
    m = len(centres)
    coords = np.ascontiguousarray((centres[:, None, :] + local[None]).reshape(-1, 3))
    owners = np.repeat(np.arange(m, dtype=np.intp), len(local))
    base = np.tile(w_local, m)
    p = _ne.ssf_weights(coords, owners, centres, SSF_A, nthreads)
    keep = p > 0.0
    return VolumeGrid(
        np.ascontiguousarray(coords[keep]),
        np.ascontiguousarray(base[keep] * p[keep]),
        np.ascontiguousarray(owners[keep]),
        np.ascontiguousarray(base[keep]),
    )


def volume_grid_gradient(
    grid: VolumeGrid, centres: FloatArray, values: FloatArray, dvalues: FloatArray, nthreads: int
) -> FloatArray:
    """d/dR (cavity atoms) of sum_p weights_p v_p, v's gradient dvalues (n, 3) carried by the
    points moving with their owners."""
    gR, gy = _ne.ssf_gradient(
        grid.coords, grid.owners, centres, SSF_A, grid.base_weights * values, nthreads
    )
    gy = gy + grid.weights[:, None] * dvalues
    for x in range(3):
        gR[:, x] += np.bincount(grid.owners, gy[:, x], minlength=len(centres))
    return gR


@dataclass(frozen=True)
class ExteriorRays:
    """The outside of a closed surface as weighted nodes on rays from the nuclei."""

    coords: FloatArray
    weights: FloatArray
    point: IntArray
    atom: IntArray  # cavity-atom index
    tau: FloatArray
    w_radial: FloatArray
    partition: FloatArray

    def adjoint(
        self,
        xyz: FloatArray,
        normals: FloatArray,
        areas: FloatArray,
        centres: FloatArray,
        values: FloatArray,
        dvalues: FloatArray,
        nthreads: int,
    ) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
        """(dE/ds, dE/dn, dE/da, dE/dR) of E = sum_k weights_k G_k, G and dG/dy at the nodes."""
        base = self.weights / self.partition
        gR, gy = _ne.ssf_gradient(self.coords, self.atom, centres, SSF_A, base * values, nthreads)
        gy = np.ascontiguousarray(gy + self.weights[:, None] * dvalues)
        ds, dn, da, dc = _ne.ray_adjoint(
            xyz,
            normals,
            areas,
            centres,
            self.point,
            self.atom,
            self.tau,
            self.w_radial,
            self.partition,
            np.ascontiguousarray(values),
            gy,
            nthreads,
        )
        return ds, dn, da, gR + dc


def build_exterior_rays(
    xyz: FloatArray,
    normals: FloatArray,
    areas: FloatArray,
    centres: FloatArray,
    rule: tuple[int, float],
    nthreads: int,
) -> ExteriorRays:
    pairs_i, pairs_A = np.nonzero(_ne.ssf_cells_hold(xyz, centres, SSF_A, nthreads))
    n_radial, length = rule
    tau, w_rad = mapped_legendre(n_radial, length)
    R = xyz[pairs_i] - centres[pairs_A]
    r = np.linalg.norm(R, axis=1)
    e = R / r[:, None]
    ne = np.einsum("px,px->p", normals[pairs_i], e)
    y = (xyz[pairs_i][:, None, :] + tau[None, :, None] * e[:, None, :]).reshape(-1, 3)
    s = (r[:, None] + tau[None]) / r[:, None]
    base = ((areas[pairs_i] * ne)[:, None] * s**2 * w_rad[None]).ravel()
    point = np.repeat(pairs_i, n_radial).astype(np.intp)
    atom = np.repeat(pairs_A, n_radial).astype(np.intp)
    y = np.ascontiguousarray(y)
    p = _ne.ssf_weights(y, atom, centres, SSF_A, nthreads)
    keep = p > 0.0
    return ExteriorRays(
        np.ascontiguousarray(y[keep]),
        np.ascontiguousarray(base[keep] * p[keep]),
        np.ascontiguousarray(point[keep]),
        np.ascontiguousarray(atom[keep]),
        np.ascontiguousarray(np.tile(tau, len(pairs_i))[keep]),
        np.ascontiguousarray(np.tile(w_rad, len(pairs_i))[keep]),
        np.ascontiguousarray(p[keep]),
    )


# ------------------------------------------------------------------------------- the terms


@dataclass
class _Surface:
    key: bytes
    xyz: FloatArray
    normals: FloatArray
    areas: FloatArray
    owners: IntArray  # cavity-atom index of each point
    rays: ExteriorRays
    kernel_volume: FloatArray
    kernel_rays: FloatArray


@dataclass
class _Pass:
    energies: Energies
    V: FloatArray | None = None
    w_xyz: FloatArray | None = None
    w_n: FloatArray | None = None
    w_a: FloatArray | None = None
    explicit: FloatArray | None = None  # (natom, 3), every atom of the molecule


class DropTerms:
    """The correction on a MoistSolver's cavity, as a function of the total density matrix.

    energy(D), energy_and_potential(D) (V = dG/dD, the surface's response included) and
    gradient(D) (dG/dR at fixed D, the cavity's motion included), all atoms of the molecule.
    """

    def __init__(
        self, solver: MoistSolver, settings: NonelectrostaticSettings, nthreads: int
    ) -> None:
        self.solver, self.settings, self.nthreads = solver, settings, nthreads
        self.basis = solver.basis
        self.natom = solver.natom
        self.atoms = solver.atoms  # molecule indices of the cavity atoms
        self.centres = np.ascontiguousarray(solver.xyz, dtype=float)
        self.charges = np.asarray(solver.Z, dtype=float)
        self.volume = build_volume_grid(self.centres, settings.volume_rule, nthreads)
        self.ao_atom = np.array(
            [self.basis.function_to_center(i) for i in range(self.basis.nbf())], dtype=np.intp
        )
        self._cache: _Surface | None = None

    # -- the surface ---------------------------------------------------------------------------

    def _surface(self, D: FloatArray) -> _Surface:
        sol = self.solver
        if sol.D is None or not np.array_equal(np.asarray(sol.D.np), D):
            from psi4 import core

            sol._evaluate(core.Matrix.from_array(D))
        snap = sol.model.cavity.snapshot()
        key = np.ascontiguousarray(snap.xyz).tobytes() + np.ascontiguousarray(snap.a).tobytes()
        if self._cache is not None and self._cache.key == key:
            return self._cache
        s = self.settings
        xyz = np.ascontiguousarray(snap.xyz, dtype=float)
        normals = np.ascontiguousarray(snap.normal0, dtype=float)
        areas = np.ascontiguousarray(snap.a, dtype=float)
        rays = build_exterior_rays(xyz, normals, areas, self.centres, s.ray_rule, self.nthreads)
        whole = 2.0 * np.pi**2 / (3.0 * s.delta**3)

        def kernel(points: FloatArray) -> FloatArray:
            J = _ne.dispersion_kernel(points, xyz, normals, areas, s.delta, None, self.nthreads)[0]
            out: FloatArray = whole - np.asarray(J)
            return out

        self._cache = _Surface(
            key,
            xyz,
            normals,
            areas,
            np.asarray(snap.owner, dtype=np.intp),
            rays,
            kernel(self.volume.coords),
            kernel(rays.coords),
        )
        return self._cache

    # -- pieces ---------------------------------------------------------------------------

    def _density(
        self, D: FloatArray, coords: FloatArray, deriv: int
    ) -> tuple[FloatArray, FloatArray, FloatArray | None]:
        nbf, n = self.basis.nbf(), len(coords)
        rho, grad = np.empty(n), np.empty((3, n))
        hess = np.empty((3, 3, n)) if deriv >= 2 else None
        for sl in _blocks(n, (10 if deriv >= 2 else 4) * nbf):
            ao = ao_values(self.basis, coords[sl], deriv)
            c0 = ao[0] @ D
            rho[sl] = np.einsum("pi,pi->p", ao[0], c0)
            grad[:, sl] = 2.0 * np.einsum("kpi,pi->kp", ao[1:4], c0)
            if hess is not None:
                c1 = np.einsum("kpi,ij->kpj", ao[1:4], D)
                for comp, (_name, k, kk) in enumerate(_SECOND, start=4):
                    h = 2.0 * (
                        np.einsum("pi,pi->p", ao[comp], c0)
                        + np.einsum("pi,pi->p", ao[1 + k], c1[kk])
                    )
                    hess[k, kk, sl] = h
                    hess[kk, k, sl] = h
        return rho, grad, hess

    def _gga(
        self,
        D: FloatArray,
        coords: FloatArray,
        c_rho: FloatArray,
        c_grad: FloatArray,
        potential: bool,
        gradient: bool,
    ) -> tuple[FloatArray | None, FloatArray | None]:
        """V and the basis functions' motion for sum_p (c_rho rho + c_grad . grad rho)."""
        if not (potential or gradient):
            return None, None
        nbf = self.basis.nbf()
        V = np.zeros((nbf, nbf)) if potential else None
        g = np.zeros((self.natom, 3)) if gradient else None
        deriv = 2 if gradient else 1
        for sl in _blocks(len(coords), (10 if gradient else 4) * nbf):
            ao = ao_values(self.basis, coords[sl], deriv)
            if V is not None:
                wv = np.empty((4, sl.stop - sl.start))
                wv[0] = 0.5 * c_rho[sl]
                wv[1:] = c_grad[:, sl]
                V += ao[0].T @ np.einsum("kpi,kp->pi", ao[:4], wv)
            if g is not None:
                coeff = np.empty((4, sl.stop - sl.start))
                coeff[0] = c_rho[sl]
                coeff[1:] = c_grad[:, sl]
                g += basis_motion_gradient(ao, D, coeff, self.ao_atom, self.natom)
        if V is not None:
            V = V + V.T
        return V, g

    def _surface_function(self, F: FloatArray) -> tuple[FloatArray, FloatArray]:
        s = self.settings
        neg, d_neg = field_power(F, s.neg_exponent, -1)
        pos, d_pos = field_power(F, s.pos_exponent, 1)
        return s.neg_coeff * neg + s.pos_coeff * pos, s.neg_coeff * d_neg + s.pos_coeff * d_pos

    def _field(self, surf: _Surface, D: FloatArray) -> tuple[FloatArray, FloatArray]:
        """The solute's field at the points (n, 3), and the nuclei's dE/ds tensors."""
        sep = surf.xyz[:, None] - self.centres[None]
        dist = np.linalg.norm(sep, axis=2)
        field = np.einsum("a,pax->px", self.charges, sep / dist[..., None] ** 3)
        tensor = (
            np.eye(3)[None, None] / dist[..., None, None] ** 3
            - 3.0 * np.einsum("pax,pay->paxy", sep, sep) / dist[..., None, None] ** 5
        )
        n = len(surf.areas)
        for k in range(3):
            unit = np.zeros((n, 3))
            unit[:, k] = 1.0
            field[:, k] += _ne.normal_field(self.basis, surf.xyz, unit, D, self.nthreads)
        return field, tensor

    # -- one evaluation --------------------------------------------------------------------------

    def _evaluate(
        self, D: FloatArray, potential: bool = False, adjoint: bool = False, gradient: bool = False
    ) -> _Pass:
        D = np.asarray(D, dtype=float)
        s = self.settings
        A, B = s.dispersion, s.exchange
        surf = self._surface(D)
        rays, vol = surf.rays, self.volume
        derivs = adjoint or gradient
        deriv = 2 if derivs else 1

        rho_v, grad_v, hess_v = self._density(D, vol.coords, deriv)
        rho_r, grad_r, hess_r = self._density(D, rays.coords, deriv)
        u_v, ur_v, us_v = dispersion_density(
            rho_v, np.sum(grad_v**2, axis=0), s.solvent_rho, s.vv_c
        )
        u_r, ur_r, us_r = dispersion_density(
            rho_r, np.sum(grad_r**2, axis=0), s.solvent_rho, s.vv_c
        )
        x_r, xs_r = exchange_density(np.sum(grad_r**2, axis=0))
        I_v, I_r = surf.kernel_volume, surf.kernel_rays
        dispersion = float(vol.weights @ (u_v * I_v) - rays.weights @ (u_r * I_r))
        exchange = float(rays.weights @ x_r)

        cv_rho = A * vol.weights * I_v * ur_v
        cv_grad = 2.0 * (A * vol.weights * I_v * us_v)[None] * grad_v
        cr_rho = -A * rays.weights * I_r * ur_r
        cr_grad = 2.0 * (rays.weights * (-A * I_r * us_r + B * xs_r))[None] * grad_r
        V1, gb_v = self._gga(D, vol.coords, cv_rho, cv_grad, potential, gradient)
        V2, gb_r = self._gga(D, rays.coords, cr_rho, cr_grad, potential, gradient)

        field, tensor = self._field(surf, D)
        F = np.einsum("px,px->p", field, surf.normals)
        f, f_prime = self._surface_function(F)
        energies = Energies(A * dispersion, B * exchange, float(surf.areas @ f))
        out = _Pass(energies)
        c = surf.areas * f_prime
        if potential:
            assert V1 is not None and V2 is not None
            out.V = (
                V1
                + V2
                + _ne.normal_field_matrix(self.basis, surf.xyz, surf.normals, c, self.nthreads)
            )
        if not derivs:
            return out

        assert hess_v is not None and hess_r is not None
        q = np.ascontiguousarray(c[:, None] * surf.normals)  # dG/dE at s_i
        g_basis, g_points = _ne.field_gradient(
            self.basis, surf.xyz, D, q, self.natom, self.nthreads
        )
        f_xyz = np.asarray(g_points) + np.einsum("px,a,paxy->py", q, self.charges, tensor)

        n_v = len(vol.weights)
        points = np.ascontiguousarray(np.concatenate([vol.coords, rays.coords]))
        c_kernel = np.ascontiguousarray(
            np.concatenate([-A * vol.weights * u_v, A * rays.weights * u_r])
        )
        _J, dJ, kj_xyz, kj_a, kj_n = (
            np.asarray(a)
            for a in _ne.dispersion_kernel(
                points, surf.xyz, surf.normals, surf.areas, s.delta, c_kernel, self.nthreads
            )
        )
        dI_v, dI_r = -dJ[:n_v].T, -dJ[n_v:].T

        du_r = ur_r[None] * grad_r + 2.0 * us_r[None] * np.einsum("klp,lp->kp", hess_r, grad_r)
        dx_r = 2.0 * xs_r[None] * np.einsum("klp,lp->kp", hess_r, grad_r)
        G_r = -A * u_r * I_r + B * x_r
        dG_r = -A * (du_r * I_r[None] + u_r[None] * dI_r) + B * dx_r
        r_xyz, r_n, r_a, r_R = rays.adjoint(
            surf.xyz,
            surf.normals,
            surf.areas,
            self.centres,
            G_r,
            np.ascontiguousarray(dG_r.T),
            self.nthreads,
        )
        out.w_xyz = r_xyz + kj_xyz + f_xyz
        out.w_n = r_n + kj_n + c[:, None] * field
        out.w_a = r_a + kj_a + f
        if not gradient:
            return out

        assert gb_v is not None and gb_r is not None
        du_v = ur_v[None] * grad_v + 2.0 * us_v[None] * np.einsum("klp,lp->kp", hess_v, grad_v)
        G_v = A * u_v * I_v
        dG_v = A * (du_v * I_v[None] + u_v[None] * dI_v)
        g_cavity = volume_grid_gradient(
            vol, self.centres, G_v, np.ascontiguousarray(dG_v.T), self.nthreads
        )
        g_cavity += r_R
        g_cavity -= np.einsum("a,px,paxy->ay", self.charges, q, tensor)  # the nuclei's field
        g = gb_v + gb_r + np.asarray(g_basis)
        g[self.atoms] += g_cavity
        out.explicit = g
        return out

    # -- the public interface --------------------------------------------------------------------

    def energy(self, D: FloatArray) -> Energies:
        return self._evaluate(D).energies

    def _require_contractions(self) -> None:
        """The surface derivatives need moist's reverse contractions (packages/moist's patch
        0002); energies do not."""
        if not hasattr(self.solver.model.cavity, "contract_surface_gradient"):
            raise RuntimeError(
                f"moist {moist.__version__} does not export the surface contractions the "
                "nonelectrostatic correction's gradient and self-consistent Fock matrix need: "
                "build it from packages/moist, whose patches/0002 adds them"
            )

    def _density_weights(self, out: _Pass) -> Any:
        self._require_contractions()
        return self.solver.model.cavity.contract_surface_density(
            w_xyz=out.w_xyz, w_n=out.w_n, w_a=out.w_a
        )

    def energy_and_potential(self, D: FloatArray) -> tuple[Energies, FloatArray]:
        dependent = self.solver.density_dependent
        out = self._evaluate(D, potential=True, adjoint=dependent)
        assert out.V is not None
        V = out.V
        if dependent:
            weights = self._density_weights(out)
            aos = self.solver.aos
            if weights is not None and aos is not None:
                V = V + aos.fock(self.solver.points.coords, weights)
        return out.energies, V

    def gradient(self, D: FloatArray) -> FloatArray:
        self._require_contractions()
        out = self._evaluate(D, gradient=True)
        assert out.explicit is not None
        g = out.explicit.copy()
        cavity = self.solver.model.cavity
        g[self.atoms] += cavity.contract_surface_gradient(
            w_xyz=out.w_xyz, w_n=out.w_n, w_a=out.w_a
        )
        aos = self.solver.aos
        if self.solver.density_dependent and aos is not None:
            weights = self._density_weights(out)
            if weights is not None:
                D_cart = aos.cartesian_density(np.asarray(D, dtype=float))
                g[self.atoms] += aos.gradient(
                    self.solver.points.coords, D_cart, weights, len(self.atoms)
                )
        return g
