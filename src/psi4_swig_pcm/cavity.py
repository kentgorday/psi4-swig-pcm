# Copyright 2014-2023 The PySCF Developers. All Rights Reserved.
# Copyright (C) 2026 Kent Gorday (modifications).
# SPDX-License-Identifier: Apache-2.0
#
# Derived from PySCF v2.11, pyscf/solvent/pcm.py and pyscf/solvent/grad/pcm.py (Author:
# Xiaojie Wu <wxj6000@gmail.com>); PySCF's gradient module was itself copied from GPU4PySCF
# with modifications.  Unlike the rest of psi4-swig-pcm this file stays under the Apache
# License, Version 2.0 (LICENSES/Apache-2.0.txt); PySCF's attribution notices are in NOTICE.
#
# Changes from PySCF: numpy/scipy only (Lebedev grids from scipy), radii indexed by true
# nuclear charge, the system held as S, D and the LU factors of K (no dense K or R), S and D
# built with a few n x n temporaries, the cavity gradient evaluated in row blocks,
# PCMSolver-style numerical diagnostics, and type annotations.
"""Switching/Gaussian PCM cavity (Lange & Herbert, JCP 133, 244111 (2010)) and its gradient.

Everything is in bohr.  The surface is a union of scaled vdW spheres with Lebedev points; each
point carries a unit Gaussian charge of exponent xi**2 and a smooth switching weight F in
[0, 1], the product over the other spheres J of f_iJ.  Two switching functions, as in PySCF
(2.13 and later):

  SWIG    f_iJ = h((r_iJ - R_in,J) / R_sw,J), h the smoothstep switch_h over a band of width
          R_sw,J = R_J (14 / N_leb)^(1/2) around R_J.
  ISWIG   f_iJ = 1 - [erf(xi_i (R_J - r_iJ)) + erf(xi_i (R_J + r_iJ))] / 2, the fraction of
          point i's own Gaussian outside sphere J.

The models share this surface:

  C-PCM / COSMO   K = S                                 R = -f I
  IEF-PCM         K = S - f/(2 pi) D A S                R = -f (I - D A / (2 pi))   (asymmetric)
  SS(V)PE         K = S - f/(4 pi) (D A S + S A D^T)    R = -f (I - D A / (2 pi))   (symmetrized)

with q = K^-1 R phi.  R is never symmetric for IEF-PCM / SS(V)PE, so the energy
E = 1/2 phi^T K^-1 R phi has dE/dphi = q_sym = (q + R^T K^-T phi) / 2.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal, get_args

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.integrate import lebedev_rule
from scipy.linalg import LinAlgError, cholesky, lu_factor, lu_solve
from scipy.linalg.lapack import dgecon
from scipy.spatial.distance import cdist
from scipy.special import erf

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.intp]
Method = Literal["C-PCM", "COSMO", "IEF-PCM", "SS(V)PE"]
METHODS: tuple[Method, ...] = get_args(Method)
Switching = Literal["SWIG", "ISWIG"]
SWITCHINGS: tuple[Switching, ...] = get_args(Switching)

BOHR = 0.52917721092  # PySCF's value; radii are converted with it for parity

# Bondi vdW radii (Angstrom) by atomic number, as pyscf.data.radii.VDW; index 0 = ghost.
# _U: PySCF's radius for elements without a Bondi value, 1.999999 rather than 2.0 (parity).
_U = 1.999999
# fmt: off
VDW_ANGSTROM = np.array([
    _U, 1.2, 1.4, 1.82, 1.53, 1.92, 1.7, 1.55, 1.52, 1.47, 1.54, 2.27, 1.73, 1.84, 2.1, 1.8, 1.8,
    1.75, 1.88, 2.75, 2.31, _U, _U, _U, _U, _U, _U, _U, 1.63, 1.4, 1.39, 1.87, 2.11, 1.85,
    1.9, 1.85, 2.02, 3.03, 2.49, _U, _U, _U, _U, _U, _U, _U, 1.63, 1.72, 1.58, 1.93, 2.17,
    2.06, 2.06, 1.98, 2.16, 3.43, 2.49, _U, _U, _U, _U, _U, _U, _U, _U, _U, _U, _U,
    _U, _U, _U, _U, _U, _U, _U, _U, _U, _U, 1.75, 1.66, 1.55, 1.96, 2.02, 2.07, 1.97,
    2.02, 2.2, 3.48, 2.83, _U, _U, _U, 1.86, _U, _U, _U, _U, _U, _U, _U, _U, _U, _U,
    _U,
])
# fmt: on

# TABLE II, J. Chem. Phys. 122, 194110 (2005): Gaussian width factor per Lebedev point count
XI = {
    6: 4.84566077868, 14: 4.86458714334, 26: 4.85478226219, 38: 4.90105812685,
    50: 4.89250673295, 86: 4.89741372580, 110: 4.90101060987, 146: 4.89825187392,
    170: 4.90685517725, 194: 4.90337644248, 302: 4.90498088169, 350: 4.86879474832,
    434: 4.90567349080, 590: 4.90624071359, 770: 4.90656435779, 974: 4.90685167998,
    1202: 4.90704098216, 1454: 4.90721023869, 1730: 4.90733270691, 2030: 4.90744499142,
    2354: 4.90753082825, 2702: 4.90760972766, 3074: 4.90767282394, 3470: 4.90773141371,
    3890: 4.90777965981, 4334: 4.90782469526, 4802: 4.90749125553, 5294: 4.90762073452,
    5810: 4.90792902522,
}  # fmt: skip
# Lebedev orders offered, with their point counts.  Orders 13, 25 and 27 (74, 230 and 266
# points) are left out: their rules have negative weights, and the 2005 table gives no XI
# for them.
LEBEDEV_POINTS = {
    5: 14, 7: 26, 9: 38, 11: 50, 15: 86, 17: 110, 19: 146, 21: 170, 23: 194, 29: 302, 31: 350,
    35: 434, 41: 590, 47: 770, 53: 974, 59: 1202,
}  # fmt: skip


def modified_bondi_radii(scale: float = 1.2, hydrogen: float = 1.10) -> FloatArray:
    """Cavity radii (bohr) by atomic number: modified Bondi x 1.2, H 1.10 A (PySCF default)."""
    r = VDW_ANGSTROM.copy()
    r[1] = hydrogen
    radii: FloatArray = scale * r / BOHR
    return radii


def switch_h(x: FloatArray) -> FloatArray:
    """Smoothstep switching function, eq. 3.19 of JCP 133, 244111 (2010), as PySCF corrects it."""
    y = x**3 * (10.0 - 15.0 * x + 6.0 * x**2)
    return np.where(x < 0, 0.0, np.where(x > 1, 1.0, y))


def grad_switch_h(x: FloatArray) -> FloatArray:
    dy = 30.0 * x**2 - 60.0 * x**3 + 30.0 * x**4
    return np.where((x < 0) | (x > 1), 0.0, dy)


def iswig_f(xi: FloatArray, R: FloatArray, r: FloatArray) -> FloatArray:
    """ISWIG's f for a point of exponent parameter xi at distance r from a sphere of radius R."""
    f: FloatArray = 1.0 - 0.5 * (erf(xi * (R - r)) + erf(xi * (R + r)))
    return f


def iswig_dlogf_dr(xi: FloatArray, R: FloatArray, r: FloatArray) -> FloatArray:
    """d ln f / dr for iswig_f."""
    g: FloatArray = (
        xi / np.pi**0.5 * (np.exp(-((xi * (R - r)) ** 2)) - np.exp(-((xi * (R + r)) ** 2)))
    )
    return g / iswig_f(xi, R, r)


@dataclass
class Surface:
    atom_coords: FloatArray  # (natm, 3)
    coords: FloatArray  # (npts, 3) point positions
    normals: FloatArray  # (npts, 3) outward unit normals (sphere directions)
    weights: FloatArray  # Lebedev weights (sum 4 pi per sphere)
    xi: FloatArray  # Gaussian exponent parameter; the charge is exp(-(xi r)^2)
    switch: FloatArray  # switching function F
    R_vdw: FloatArray  # sphere radius of each point's atom
    area: FloatArray  # w R^2 F
    atom_slices: list[tuple[int, int]]  # [(p0, p1)] points of each atom
    R_in: FloatArray  # per-atom inner switching radius (SWIG)
    R_sw: FloatArray  # per-atom switching width (SWIG)
    R_atom: FloatArray  # per-atom sphere radius
    switching: Switching = "SWIG"

    @property
    def npts(self) -> int:
        return len(self.xi)

    @property
    def atom_of_point(self) -> IntArray:
        return np.concatenate(
            [np.full(p1 - p0, a, dtype=np.intp) for a, (p0, p1) in enumerate(self.atom_slices)]
        )


def make_surface(
    atom_coords: ArrayLike,
    Z: Sequence[int],
    radii: FloatArray,
    lebedev_order: int = 29,
    switching: Switching = "SWIG",
) -> Surface:
    """Switching/Gaussian surface.  atom_coords in bohr; Z = true (not ECP-reduced) charges."""
    if switching not in SWITCHINGS:
        raise ValueError(f"switching must be one of {SWITCHINGS}, not {switching!r}")
    ng = LEBEDEV_POINTS[lebedev_order]
    x, w = lebedev_rule(lebedev_order)
    unit: FloatArray = x.T  # w sums to 4 pi
    xyz = np.asarray(atom_coords, dtype=float)
    R_J = np.asarray([radii[int(z)] for z in Z])
    R_sw = R_J * (14.0 / ng) ** 0.5
    alpha = 0.5 + R_J / R_sw - ((R_J / R_sw) ** 2 - 1.0 / 28) ** 0.5
    R_in = R_J - alpha * R_sw
    parts: list[tuple[FloatArray, ...]] = []
    slices: list[tuple[int, int]] = []
    p1 = 0
    for ia, r in enumerate(R_J):
        pts = r * unit + xyz[ia]
        xi = XI[ng] / (r * w**0.5)
        if switching == "SWIG":
            d = (cdist(pts, xyz) - R_in) / R_sw
            d[:, ia] = 1.0
            d[d < 1e-8] = 0.0
            f = switch_h(d)
        else:
            f = iswig_f(xi[:, None], R_J[None], cdist(pts, xyz))
            f[:, ia] = 1.0
            f[f < 1e-8] = 0.0
        F = np.prod(f, axis=1)
        keep = w * F > 1e-16
        n = int(keep.sum())
        slices.append((p1, p1 + n))
        p1 += n
        parts.append((pts[keep], unit[keep], w[keep], xi[keep], F[keep], np.full(n, r)))
    coords, normals, weights, xi_all, F_all, Rv = (
        np.concatenate(c) for c in zip(*parts, strict=True)
    )
    area = weights * Rv**2 * F_all
    return Surface(
        xyz, coords, normals, weights, xi_all, F_all, Rv, area, slices, R_in, R_sw, R_J, switching
    )


def nuclear_potential(surf: Surface, charges: ArrayLike) -> FloatArray:
    """Potential of point nuclei (charges may be ECP-reduced) at each Gaussian charge."""
    d = cdist(surf.coords, surf.atom_coords)
    per_atom = np.asarray(charges, dtype=float)[None] * erf(surf.xi[:, None] * d) / d
    phi: FloatArray = per_atom.sum(1)
    return phi


def _pair_geometry(
    surf: Surface, rows: IntArray, cols: IntArray
) -> tuple[FloatArray, FloatArray, FloatArray, NDArray[np.bool_]]:
    """rij, xi_ij and (ri - rj) for a block of point pairs; diagonal distances set to 1 later."""
    xr, xc = surf.xi[rows], surf.xi[cols]
    xi_ij = xr[:, None] * xc[None] / np.sqrt(xr[:, None] ** 2 + xc[None] ** 2)
    dr = surf.coords[rows][:, None] - surf.coords[cols][None]
    rij = np.linalg.norm(dr, axis=-1)
    diag = rows[:, None] == cols[None]
    return rij, xi_ij, dr, diag


def S_D(surf: Surface, with_D: bool) -> tuple[FloatArray, FloatArray | None]:
    """Dense S (and D) matrices, JCP 133, 244111 (2010).

    Built in place, so the peak is five n x n arrays (S, D and three temporaries) rather
    than the n x n x 3 separation vectors.
    """
    coords = surf.coords - surf.coords.mean(0)  # small coordinates keep r_i.n_j - r_j.n_j exact
    rij = cdist(coords, coords)
    np.fill_diagonal(rij, 1.0)
    inv_xi2 = 1.0 / surf.xi**2
    xr = np.add.outer(inv_xi2, inv_xi2)  # xi_ij = (1/xi_i^2 + 1/xi_j^2)^(-1/2)
    np.sqrt(xr, out=xr)
    np.reciprocal(xr, out=xr)
    xr *= rij
    S: FloatArray = erf(xr)
    S /= rij
    np.fill_diagonal(S, surf.xi * (2.0 / np.pi) ** 0.5 / surf.switch)
    if not with_D:
        return S, None
    # D_ij = n_j.(r_i - r_j) / r_ij^2 * (S_ij - 2 xr exp(-xr^2) / (sqrt(pi) r_ij))
    D: FloatArray = np.square(xr)
    np.negative(D, out=D)
    np.exp(D, out=D)
    D *= xr
    D *= 2.0 / np.pi**0.5
    D /= rij
    np.subtract(S, D, out=D)
    del xr
    nrij = coords @ surf.normals.T
    nrij -= np.einsum("jx,jx->j", coords, surf.normals)[None]
    nrij /= rij
    nrij /= rij
    D *= nrij
    del nrij
    np.fill_diagonal(D, -surf.xi * (2.0 / np.pi) ** 0.5 / (2.0 * surf.R_vdw))
    return S, D


def f_epsilon(method: Method, eps: float) -> float:
    if np.isinf(eps):
        return 1.0
    if method == "C-PCM":
        return (eps - 1.0) / eps
    if method == "COSMO":
        return (eps - 1.0) / (eps + 0.5)
    return (eps - 1.0) / (eps + 1.0)  # IEF-PCM and SS(V)PE


@dataclass(frozen=True)
class Diagnostics:
    """Numerical health of one discretization, in the spirit of PCMSolver's checks.

    s_positive_definite   S must be SPD (Cholesky succeeds); PCMSolver aborts otherwise.
    rcond_K               LAPACK 1-norm reciprocal condition estimate of K (from the LU).
    gauss_nuclear_relerr  Gauss's law for the nuclei alone: point charges strictly inside the
                          cavity must induce total charge -(eps-1)/eps * sum(Z)
                          (-(eps-1)/(eps+1/2) for COSMO).  PCMSolver's "total nuclear ASC"
                          check; a hole or a leak in the cavity shows up here.
    solve_residual        ||K q - R phi|| / ||R phi|| for the nuclear potential.
    min_switch            smallest switching value kept (tiny F -> huge S_ii).
    """

    s_positive_definite: bool
    rcond_K: float
    gauss_nuclear_relerr: float
    solve_residual: float
    min_switch: float
    npts: int


class PCMOperator:
    """The PCM system K q = R phi for one geometry: charges and energy for a potential.

    Held as S, D (IEF-PCM / SS(V)PE) and the LU factors of K: R and K act through S, D and
    the areas, so neither is stored.  dense_K() and dense_R() build them for inspection.
    """

    def __init__(self, surf: Surface, method: Method = "IEF-PCM", eps: float = 78.3553) -> None:
        if method not in METHODS:
            raise ValueError(f"method must be one of {METHODS}, not {method!r}")
        self.surf, self.method, self.eps = surf, method, eps
        self.f = f_epsilon(method, eps)
        self.with_D = method in ("IEF-PCM", "SS(V)PE")
        self.S, self.D = S_D(surf, self.with_D)
        K = self.dense_K()
        self.anorm_K = float(np.abs(K).sum(0).max())  # for the condition estimate
        self.lu = lu_factor(K, overwrite_a=True, check_finite=False)

    def dense_K(self) -> FloatArray:
        if self.D is None:
            return self.S.copy()
        K: FloatArray = (self.D * self.surf.area) @ self.S  # D A S
        if self.method == "IEF-PCM":
            K *= -self.f / (2 * np.pi)
        else:
            K += K.T.copy()
            K *= -self.f / (4 * np.pi)
        K += self.S
        return K

    def dense_R(self) -> FloatArray:
        n = self.surf.npts
        if self.D is None:
            return -self.f * np.eye(n)
        R: FloatArray = -self.f * (np.eye(n) - self.D * self.surf.area / (2 * np.pi))
        return R

    def apply_K(self, q: FloatArray) -> FloatArray:
        Sq: FloatArray = self.S @ q
        if self.D is None:
            return Sq
        A = self.surf.area
        DASq = self.D @ (A * Sq)
        if self.method == "IEF-PCM":
            Kq: FloatArray = Sq - self.f / (2 * np.pi) * DASq
            return Kq
        Kq = Sq - self.f / (4 * np.pi) * (DASq + self.S @ (A * (self.D.T @ q)))
        return Kq

    def apply_R(self, x: FloatArray) -> FloatArray:
        if self.D is None:
            return -self.f * x
        Rx: FloatArray = -self.f * (x - self.D @ (self.surf.area * x) / (2 * np.pi))
        return Rx

    def apply_RT(self, y: FloatArray) -> FloatArray:
        if self.D is None:
            return -self.f * y
        RTy: FloatArray = -self.f * (y - self.surf.area * (self.D.T @ y) / (2 * np.pi))
        return RTy

    def charges(self, phi: FloatArray) -> tuple[FloatArray, FloatArray]:
        """(q, q_sym) for the total potential phi at the surface points."""
        q: FloatArray = lu_solve(self.lu, self.apply_R(phi), check_finite=False)
        v = lu_solve(self.lu, phi, trans=1, check_finite=False)
        q_sym: FloatArray = 0.5 * (q + self.apply_RT(v))
        return q, q_sym

    def energy(self, phi: FloatArray) -> tuple[float, FloatArray, FloatArray]:
        q, q_sym = self.charges(phi)
        return float(0.5 * q_sym @ phi), q, q_sym

    def diagnostics(self, nuclear_charges: ArrayLike) -> Diagnostics:
        try:
            cholesky(self.S, lower=True, check_finite=False)
            s_pd = True
        except LinAlgError:
            s_pd = False
        rcond, _ = dgecon(self.lu[0], self.anorm_K, norm="1")
        Z = np.asarray(nuclear_charges, dtype=float)
        phi = nuclear_potential(self.surf, Z)
        q, _ = self.charges(phi)
        if np.isinf(self.eps):
            f_exact = 1.0
        elif self.method == "COSMO":
            f_exact = (self.eps - 1.0) / (self.eps + 0.5)
        else:
            f_exact = (self.eps - 1.0) / self.eps
        expected = -f_exact * Z.sum()
        rphi = self.apply_R(phi)
        return Diagnostics(
            s_positive_definite=s_pd,
            rcond_K=float(rcond),
            gauss_nuclear_relerr=float((q.sum() - expected) / expected),
            solve_residual=float(np.linalg.norm(self.apply_K(q) - rphi) / np.linalg.norm(rphi)),
            min_switch=float(self.surf.switch.min()),
            npts=self.surf.npts,
        )


# ---------------------------------------------------------------- geometry derivative


def dF_dA(surf: Surface) -> tuple[FloatArray, FloatArray]:
    """d(switching)/dR_atom and d(area)/dR_atom, (npts, natm, 3).  JCP 133, 244111, App. C."""
    n, natm = surf.npts, len(surf.atom_coords)
    dF = np.zeros((n, natm, 3))
    dA = np.zeros((n, natm, 3))
    for ia, (p0, p1) in enumerate(surf.atom_slices):
        rJ = surf.coords[p0:p1, None] - surf.atom_coords[None]
        r = np.linalg.norm(rJ, axis=-1)
        if surf.switching == "SWIG":
            d = (r - surf.R_in) / surf.R_sw
            d[:, ia] = 1.0
            d[d < 1e-8] = 0.0
            rJ[d < 1e-8] = 0.0
            dlogf_dr = grad_switch_h(d) / (switch_h(d) * surf.R_sw)
        else:
            dlogf_dr = iswig_dlogf_dr(surf.xi[p0:p1, None], surf.R_atom[None], r)
        rJ[:, ia] = 0.0  # the own sphere does not switch (its r is R, so f is finite there)
        df = (dlogf_dr / r)[..., None] * rJ
        F, A = surf.switch[p0:p1, None], surf.area[p0:p1, None]
        g = df.sum(1)
        dF[p0:p1, ia] += F * g  # the point moves with its own atom
        dA[p0:p1, ia] += A * g
        dF[p0:p1] -= F[:, None] * df  # neighbouring atoms move
        dA[p0:p1] -= A[:, None] * df
    return dF, dA


def _dS_dD_rows(
    surf: Surface, rows: IntArray, with_D: bool
) -> tuple[FloatArray, FloatArray | None]:
    """dS_ij/dr_i and dD_ij/dr_i for i in rows, all j: arrays (len(rows), npts, 3)."""
    cols = np.arange(surf.npts)
    rij, xi_ij, dr, diag = _pair_geometry(surf, rows, cols)
    xr = xi_ij * rij
    rij[diag] = 1.0
    dS_dr = -(erf(xr) - 2.0 * xr / np.pi**0.5 * np.exp(-(xr**2))) / rij**2
    dS_dr[diag] = 0.0
    u = dr / rij[..., None]
    dS: FloatArray = dS_dr[..., None] * u
    if not with_D:
        return dS, None
    nrij = np.einsum("ijx,jx->ij", dr, surf.normals)
    dD_dr = 4.0 * xr**2 * xi_ij / np.pi**0.5 * np.exp(-(xr**2)) * nrij / rij**3
    dD_dr[diag] = 0.0
    dD: FloatArray = dD_dr[..., None] * u + dS_dr[..., None] * (
        -surf.normals[None] / rij[..., None] + 3.0 * (nrij / rij**2)[..., None] * u
    )
    return dS, dD


def _atom_sum(surf: Surface, per_point: FloatArray) -> FloatArray:
    return np.array([per_point[p0:p1].sum(0) for p0, p1 in surf.atom_slices])


PairTerms = dict[str, list[tuple[FloatArray, FloatArray]]]


def _contract_pair_terms(
    surf: Surface, with_D: bool, pairs: PairTerms, block_mb: float
) -> dict[str, list[FloatArray]]:
    """P(a, M, b) = sum_ij a_i b_j dM_ij/dR for M in S, D, D^T, per atom (natm, 3).

    M_ij depends on r_i - r_j and the points move with their atoms, so a_i (dM_ij/dr_i) b_j is
    accumulated on point i and its negative on point j.  dS, dD are built block_mb at a time.
    """
    n = surf.npts
    acc = {k: [np.zeros((n, 3)) for _ in pairs[k]] for k in pairs}
    rows_per_block = max(1, int(block_mb * 2**20 / (8 * 3 * n * (3 if with_D else 1))))
    idx = np.arange(n)
    for r0 in range(0, n, rows_per_block):
        rows = idx[r0 : r0 + rows_per_block]
        dS, dD = _dS_dD_rows(surf, rows, with_D)
        for kind, G in (("S", dS), ("D", dD)):
            if G is None:
                continue
            for k, (a, b) in enumerate(pairs[kind]):
                acc[kind][k][rows] += 0.5 * a[rows, None] * np.einsum("ijx,j->ix", G, b)
                acc[kind][k] -= 0.5 * np.einsum("i,ijx->jx", a[rows], G) * b[:, None]
        if pairs["DT"]:
            # dD holds d(D_rj)/dr_r for r in rows; as (D^T)_jr, d/dr_j of it is -dD[r, j].
            assert dD is not None
            G = -dD
            for k, (a, b) in enumerate(pairs["DT"]):
                acc["DT"][k] += 0.5 * a[:, None] * np.einsum("rjx,r->jx", G, b[rows])
                acc["DT"][k][rows] -= 0.5 * np.einsum("j,rjx->rx", a, G) * b[rows, None]
    return {k: [_atom_sum(surf, x) for x in acc[k]] for k in acc}


def cavity_gradient(
    op: PCMOperator, phi: FloatArray, q: FloatArray, block_mb: float = 256
) -> FloatArray:
    """dE/dR at fixed phi: 1/2 phi^T d(K^-1 R) phi = 1/2 v^T (dR phi - dK q), v = K^-T phi.

    Same terms as pyscf.solvent.grad.pcm.grad_solver, but the S/D derivatives are built
    block_mb at a time, so memory stays O(npts^2) instead of several npts^2 x 3 arrays.
    """
    surf, f = op.surf, op.f
    A, S = surf.area, op.S
    v: FloatArray = lu_solve(op.lu, phi, trans=1)
    dF, dA = dF_dA(surf)
    dSii = (-surf.xi * (2.0 / np.pi) ** 0.5 / surf.switch**2)[:, None, None] * dF  # (n, natm, 3)

    def diag_term(c: FloatArray) -> FloatArray:  # 1/2 sum_i c_i dSii_i
        out: FloatArray = 0.5 * np.einsum("i,inx->nx", c, dSii)
        return out

    def area_term(c: FloatArray) -> FloatArray:  # 1/2 sum_j c_j dA_j
        out: FloatArray = 0.5 * np.einsum("j,jnx->nx", c, dA)
        return out

    if op.D is None:  # C-PCM, COSMO
        P = _contract_pair_terms(surf, False, {"S": [(v, q)], "D": [], "DT": []}, block_mb)
        g_cpcm: FloatArray = -P["S"][0] - diag_term(v * q)
        return g_cpcm

    D = op.D
    fac_R = f / (2 * np.pi)
    fac_K = f / (2 * np.pi) if op.method == "IEF-PCM" else f / (4 * np.pi)
    vD, vS = v @ D, v @ S
    vDA = vD * A
    Sq = S @ q
    ssvpe = op.method == "SS(V)PE"
    DTq = D.T @ q
    pairs: PairTerms = {
        "S": [(v, q), (vDA, q)] + ([(v, A * DTq)] if ssvpe else []),
        "D": [(v, A * phi), (v, A * Sq)],
        "DT": [(vS * A, q)] if ssvpe else [],
    }
    P = _contract_pair_terms(surf, True, pairs, block_mb)
    de_dR = fac_R * (P["D"][0] + area_term(vD * phi))
    de_dS0 = P["S"][0] + diag_term(v * q)
    terms = P["D"][1] + area_term(vD * Sq) + P["S"][1] + diag_term(vDA * q)
    if ssvpe:
        terms = terms + P["DT"][0] + area_term(vS * DTq) + P["S"][2] + diag_term(v * A * DTq)
    g: FloatArray = de_dR - (de_dS0 - fac_K * terms)
    return g
