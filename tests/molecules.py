# Copyright (C) 2026 Kent Gorday
# SPDX-License-Identifier: LGPL-3.0-only
"""Test molecules (plain data, importable without psi4)."""

from __future__ import annotations

from typing import NamedTuple

import numpy as np

from psi4_swig_pcm.cavity import BOHR, FloatArray


class Molecule(NamedTuple):
    charge: int
    multiplicity: int
    symbols: tuple[str, ...]
    angstrom: tuple[tuple[float, float, float], ...]

    @property
    def bohr(self) -> FloatArray:
        xyz: FloatArray = np.array(self.angstrom) / BOHR
        return xyz

    @property
    def Z(self) -> list[int]:
        return [Z_OF[s.lstrip("@")] for s in self.symbols]


Z_OF = {"H": 1, "C": 6, "N": 7, "O": 8, "I": 53}

MOLECULES = {
    "formate": Molecule(
        -1,
        1,
        ("C", "O", "O", "H"),
        ((0.0, 0.0, 0.52), (1.13, 0.0, -0.05), (-1.13, 0.0, -0.05), (0.0, 0.0, 1.66)),
    ),
    "methylammonium": Molecule(
        1,
        1,
        ("C", "N", "H", "H", "H", "H", "H", "H"),
        (
            (0.0, 0.0, 0.0),
            (1.49, 0.0, 0.0),
            (1.86, 0.95, 0.05),
            (1.86, -0.52, 0.80),
            (1.86, -0.43, -0.85),
            (-0.36, 1.03, 0.0),
            (-0.36, -0.52, 0.89),
            (-0.36, -0.51, -0.90),
        ),
    ),
    "glycine_zwitterion": Molecule(
        0,
        1,
        ("N", "C", "C", "O", "O", "H", "H", "H", "H", "H"),
        (
            (-1.45, 0.30, 0.10),
            (-0.25, -0.55, -0.05),
            (1.05, 0.25, 0.05),
            (1.00, 1.50, 0.15),
            (2.10, -0.45, 0.00),
            (-1.40, 0.95, -0.70),
            (-2.30, -0.25, 0.05),
            (-1.45, 0.90, 0.95),
            (-0.25, -1.10, -1.00),
            (-0.25, -1.25, 0.80),
        ),
    ),
    "acetone": Molecule(
        0,
        1,
        ("C", "O", "C", "C", "H", "H", "H", "H", "H", "H"),
        (
            (0.000, 0.000, 0.000),
            (0.000, 0.000, 1.215),
            (1.287, 0.000, -0.786),
            (-1.287, 0.000, -0.786),
            (1.157, 0.000, -1.868),
            (1.867, 0.885, -0.506),
            (1.867, -0.885, -0.506),
            (-1.157, 0.000, -1.868),
            (-1.867, 0.885, -0.506),
            (-1.867, -0.885, -0.506),
        ),
    ),
    "water": Molecule(
        0,
        1,
        ("O", "H", "H"),
        ((0.0, 0.0, 0.1173), (0.0, 0.7572, -0.4692), (0.0, -0.7572, -0.4692)),
    ),
    # Water with a ghost water (counterpoise monomer in the dimer basis); "@" marks ghosts.
    "water_ghost_water": Molecule(
        0,
        1,
        ("O", "H", "H", "@O", "@H", "@H"),
        (
            (0.0, 0.0, 0.1173),
            (0.0, 0.7572, -0.4692),
            (0.0, -0.7572, -0.4692),
            (2.9, 0.0, 0.25),
            (3.3, 0.0, -0.62),
            (1.95, 0.0, 0.1),
        ),
    ),
    "iodomethane": Molecule(  # def2 basis sets put an ECP on iodine
        0,
        1,
        ("C", "I", "H", "H", "H"),
        (
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 2.14),
            (1.03, 0.0, -0.36),
            (-0.515, 0.892, -0.36),
            (-0.515, -0.892, -0.36),
        ),
    ),
    # A doublet without degenerate singly occupied orbitals (OH, with its two pi orbitals,
    # stalls UHF DIIS near 1e-10 once the cavity breaks their degeneracy).
    "formyl": Molecule(
        0, 2, ("C", "O", "H"), ((0.0, 0.0, 0.0), (1.18, 0.0, 0.0), (-0.627, 0.916, 0.0))
    ),
    # Formate near its PCM minimum (from an optking run): the switching/Gaussian surface here
    # has an exactly symmetric 4-point set of near-coincident partners, on which Psi4's
    # point-group detection fails unless the anchor Gaussian breaks the symmetry.
    "formate_symmetric": Molecule(
        -1,
        1,
        ("C", "O", "O", "H"),
        (
            (-0.352857766051, -0.000000001064, -0.000000001329),
            (0.178786873800, -1.108514302762, 0.000000000533),
            (0.178786868607, 1.108514303220, 0.000000000529),
            (-1.473537806557, 0.000000005415, -0.000000001022),
        ),
    ),
}
