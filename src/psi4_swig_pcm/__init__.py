# Copyright (C) 2026 Kent Gorday
# SPDX-License-Identifier: LGPL-3.0-only
"""Switching/Gaussian (Lange-Herbert) PCM with analytic nuclear gradients for Psi4.

    import psi4_swig_pcm
    psi4_swig_pcm.enable(psi4_swig_pcm.PCMSettings(method="IEF-PCM", eps=78.3553))
    psi4.optimize("wb97x-3c")      # Psi4's own driver, now with the PCM in energy and gradient
    psi4.frequency("wb97x-3c")     # finite differences of the analytic PCM gradient

The solvation energy of the last SCF is the Psi4 variable "DD SOLVATION ENERGY".
"""

from __future__ import annotations

from importlib.metadata import version as _dist_version

import psi4

from ._build_info import PSI4_VERSION


def _major_minor(version: str) -> tuple[str, ...]:
    return tuple(version.split(".")[:2])


# The extension uses Psi4's C++ classes directly, so it only works with the Psi4 it was built
# against.  The conda package pins that; this catches a stale build in a changed environment.
if _major_minor(psi4.__version__) != _major_minor(PSI4_VERSION):
    raise ImportError(
        f"psi4_swig_pcm was compiled against Psi4 {PSI4_VERSION}, but Psi4 {psi4.__version__} "
        "is installed: rebuild psi4-swig-pcm"
    )

from .cavity import METHODS, Method  # noqa: E402
from .driver import active_settings, disable, enable, enabled, gradient_from_wfn  # noqa: E402
from .solver import PCMSettings, PCMSolver  # noqa: E402

__version__ = _dist_version("psi4-swig-pcm")

__all__ = [
    "METHODS",
    "PSI4_VERSION",
    "Method",
    "PCMSettings",
    "PCMSolver",
    "__version__",
    "active_settings",
    "disable",
    "enable",
    "enabled",
    "gradient_from_wfn",
]
