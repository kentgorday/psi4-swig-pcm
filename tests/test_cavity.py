# Copyright (C) 2026 Kent Gorday
# SPDX-License-Identifier: LGPL-3.0-only
"""The switching/Gaussian cavity on its own: exact limits, Gauss's law, and its gradient."""

from __future__ import annotations

from functools import partial

import numpy as np
import pytest

from psi4_swig_pcm import PCMSettings
from psi4_swig_pcm import cavity as cav
from psi4_swig_pcm.cavity import METHODS, FloatArray, Method
from tests.helpers import EPS, displaced, five_point
from tests.molecules import MOLECULES


def surface(name: str, xyz: FloatArray | None = None) -> cav.Surface:
    mol = MOLECULES[name]
    return cav.make_surface(mol.bohr if xyz is None else xyz, mol.Z, cav.modified_bondi_radii())


def test_born_ion_exact() -> None:
    """A unit charge at the centre of one sphere: every model gives the Born energy."""
    surf = cav.make_surface(np.zeros((1, 3)), [17], cav.modified_bondi_radii(), 29)
    exact = -(1 - 1 / EPS) / (2 * surf.R_vdw[0])
    for method in ("C-PCM", "IEF-PCM", "SS(V)PE"):
        phi = cav.nuclear_potential(surf, [1.0])
        e, q, _ = cav.PCMOperator(surf, method, EPS).energy(phi)
        assert abs(e / exact - 1) < 1e-7
        assert abs(q.sum() + (1 - 1 / EPS)) < 1e-7


@pytest.mark.parametrize("name", ["formate", "glycine_zwitterion", "iodomethane"])
def test_diagnostics(name: str) -> None:
    """S is SPD, solves are exact, and C-PCM / IEF-PCM obey Gauss's law (SS(V)PE does not)."""
    surf = surface(name)
    Z = MOLECULES[name].Z
    cases: list[tuple[Method, float | None]] = [
        ("C-PCM", 5e-3),
        ("IEF-PCM", 5e-3),
        ("SS(V)PE", None),
    ]
    for method, tol in cases:
        d = cav.PCMOperator(surf, method, EPS).diagnostics(Z)
        assert d.s_positive_definite
        assert d.solve_residual < 1e-12
        if tol is not None:
            assert abs(d.gauss_nuclear_relerr) < tol, d


def test_conductor_limit_is_finite() -> None:
    op = cav.PCMOperator(surface("formate"), "SS(V)PE", float("inf"))
    assert op.f == 1.0
    assert np.isfinite(op.dense_K()).all()


def test_ecp_atoms_use_true_z_for_radii() -> None:
    """The surface depends on true Z; PySCF 2.11 uses ECP-reduced charges for switching radii."""
    surf = surface("iodomethane")
    p0, p1 = surf.atom_slices[1]
    assert np.allclose(surf.R_vdw[p0:p1], cav.modified_bondi_radii()[53])


def test_invalid_settings_are_rejected() -> None:
    with pytest.raises(ValueError, match="method"):
        PCMSettings(method="PCM")  # type: ignore[arg-type]
    for order in (30, 13, 25, 27):  # 13, 25, 27: rules with negative weights
        with pytest.raises(ValueError, match="lebedev_order"):
            PCMSettings(lebedev_order=order)
    for eps in (0.5, -5.0, float("nan")):
        with pytest.raises(ValueError, match="eps"):
            PCMSettings(eps=eps)
    assert PCMSettings(eps=1.0).eps == 1.0  # vacuum is allowed
    assert PCMSettings(eps=float("inf")).eps == float("inf")


def test_settings_radii_are_immutable_and_compared() -> None:
    other = cav.modified_bondi_radii(hydrogen=1.3)
    settings = PCMSettings(radii=other)  # type: ignore[arg-type]  # any sequence is accepted
    assert isinstance(settings.radii, tuple)
    assert settings != PCMSettings()
    assert settings == PCMSettings(radii=tuple(other))
    assert hash(settings) == hash(PCMSettings(radii=tuple(other)))
    other[1] = 9.9  # the caller's array is not shared
    assert settings.radii[1] == pytest.approx(1.2 * 1.3 / cav.BOHR)


def test_every_lebedev_order_offered_has_positive_weights_and_xi() -> None:
    from scipy.integrate import lebedev_rule

    for order, npts in cav.LEBEDEV_POINTS.items():
        _, w = lebedev_rule(order)
        assert len(w) == npts
        assert w.min() > 0, order
        assert npts in cav.XI, order


def test_S_D_match_the_direct_formulas() -> None:
    """The in-place construction against JCP 133, 244111 written out directly."""
    from scipy.special import erf

    surf = surface("formate")
    S, D = cav.S_D(surf, with_D=True)
    assert D is not None
    dr = surf.coords[:, None] - surf.coords[None]
    rij = np.linalg.norm(dr, axis=-1)
    np.fill_diagonal(rij, 1.0)
    xi_ij = surf.xi[:, None] * surf.xi[None] / np.hypot(surf.xi[:, None], surf.xi[None])
    xr = xi_ij * rij
    S_ref = erf(xr) / rij
    nrij = np.einsum("ijx,jx->ij", dr, surf.normals)
    D_ref = S_ref * nrij / rij**2 - 2 * xr / np.pi**0.5 * np.exp(-(xr**2)) * nrij / rij**3
    off = ~np.eye(surf.npts, dtype=bool)
    np.testing.assert_allclose(S[off], S_ref[off], rtol=1e-13)
    np.testing.assert_allclose(D[off], D_ref[off], rtol=1e-9, atol=1e-12)


@pytest.mark.parametrize("method", METHODS)
def test_matrix_free_products_match_dense(method: Method) -> None:
    op = cav.PCMOperator(surface("formate"), method, EPS)
    x = np.random.default_rng(3).standard_normal(op.surf.npts)
    K, R = op.dense_K(), op.dense_R()
    np.testing.assert_allclose(op.apply_K(x), K @ x, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(op.apply_R(x), R @ x, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(op.apply_RT(x), R.T @ x, rtol=1e-12, atol=1e-12)
    q, q_sym = op.charges(x)
    np.testing.assert_allclose(q, np.linalg.solve(K, R @ x), rtol=1e-9, atol=1e-12)
    q_sym_ref = 0.5 * (q + R.T @ np.linalg.solve(K.T, x))
    np.testing.assert_allclose(q_sym, q_sym_ref, rtol=1e-9, atol=1e-12)


@pytest.mark.parametrize("method", METHODS)
def test_cavity_gradient_matches_finite_differences(method: Method) -> None:
    """dE/dR at fixed per-point potential: 1/2 phi^T K^-1 R phi as the surface moves."""
    name = "formate"
    xyz = MOLECULES[name].bohr
    surf = surface(name)
    rng = np.random.default_rng(0)
    phi = 0.1 * cav.nuclear_potential(surf, MOLECULES[name].Z) + 0.01 * rng.standard_normal(
        surf.npts
    )
    op = cav.PCMOperator(surf, method, EPS)
    _, q, _ = op.energy(phi)
    g = cav.cavity_gradient(op, phi, q)

    def energy_at(atom: int, axis: int, step: float) -> float:
        s = surface(name, displaced(xyz, atom, axis, step))
        assert s.npts == surf.npts  # the same points survive the switching threshold
        return cav.PCMOperator(s, method, EPS).energy(phi)[0]

    for atom in range(len(xyz)):
        for axis in range(3):
            fd = five_point(partial(energy_at, atom, axis), 1e-3)
            assert abs(g[atom, axis] - fd) < 1e-9, (atom, axis, g[atom, axis], fd)
    assert np.abs(g.sum(0)).max() < 1e-10  # translation moves nothing relative to the cavity


@pytest.mark.parametrize("method", ["C-PCM", "SS(V)PE"])
def test_blocked_cavity_gradient_is_independent_of_block_size(method: Method) -> None:
    surf = surface("glycine_zwitterion")
    op = cav.PCMOperator(surf, method, EPS)
    phi = cav.nuclear_potential(surf, MOLECULES["glycine_zwitterion"].Z)
    _, q, _ = op.energy(phi)
    np.testing.assert_allclose(
        cav.cavity_gradient(op, phi, q, block_mb=0.05),
        cav.cavity_gradient(op, phi, q),
        rtol=1e-12,
        atol=1e-14,
    )
