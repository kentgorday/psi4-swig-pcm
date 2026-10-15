# Copyright (C) 2026 Kent Gorday
# SPDX-License-Identifier: LGPL-3.0-only
"""Shared test helpers."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import psi4

from psi4_swig_pcm.cavity import FloatArray
from tests.molecules import MOLECULES

EPS = 78.3553
HF_OPTIONS = {
    "basis": "def2-svp",
    "scf_type": "pk",
    "e_convergence": 1e-12,
    "d_convergence": 1e-10,
}


def psi4_molecule(name: str, xyz_bohr: FloatArray | None = None) -> Any:
    """A Psi4 Molecule in a fixed frame (bohr), C1."""
    mol = MOLECULES[name]
    xyz = mol.bohr if xyz_bohr is None else xyz_bohr
    lines = [
        f"{mol.charge} {mol.multiplicity}",
        "units bohr",
        "symmetry c1",
        "no_com",
        "no_reorient",
    ]
    lines += [
        f"{s} {x:.12f} {y:.12f} {z:.12f}" for s, (x, y, z) in zip(mol.symbols, xyz, strict=True)
    ]
    return psi4.geometry("\n".join(lines))


def five_point(f: Callable[[float], float], h: float) -> float:
    """df/dx at 0 from f(+-h), f(+-2h): O(h^4); central differences alone are ~1e-6 off here."""
    return (8 * (f(h) - f(-h)) - (f(2 * h) - f(-2 * h))) / (12 * h)


def displaced(xyz: FloatArray, atom: int, axis: int, step: float) -> FloatArray:
    out = xyz.copy()
    out[atom, axis] += step
    return out


def as_array(m: Any) -> FloatArray:
    out: FloatArray = np.array(m.np)
    return out
