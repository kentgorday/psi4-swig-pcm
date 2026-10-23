# Copyright (C) 2026 Kent Gorday
# SPDX-License-Identifier: LGPL-3.0-only
"""The moist package from packages/moist: its extension takes BLAS/LAPACK from the environment.

Built against conda-forge's OpenBLAS variants of libblas/liblapack, the extension records
libopenblas, which an environment on another implementation does not install (MKL in tautomator
on linux-64, or Accelerate): it then fails to import, or loads an OpenBLAS from outside the
environment (a system one).  Runs in the moist environment (pixi run -e moist test-moist).
"""

from __future__ import annotations

import ctypes
import importlib.util
import os
import re
import shutil
import subprocess
import sys

import pytest

# Skip only where moist is not installed; a moist that is installed but cannot load fails.
if importlib.util.find_spec("moist") is None:
    pytest.skip("moist is installed only in the moist environments", allow_module_level=True)

import moist
import moist._libmoist
import numpy as np

pytestmark = pytest.mark.moist

_BLAS_NAME = re.compile(r"lib(open)?blas|liblapack|libmkl|libcblas")


def _linked_libraries(path: str) -> list[str] | None:
    """The shared libraries a binary names (ELF NEEDED, Mach-O load commands); None: no tool."""
    if sys.platform == "darwin":
        if shutil.which("otool") is None:
            return None
        out = subprocess.run(["otool", "-L", path], capture_output=True, text=True, check=True)
        return [line.split()[0] for line in out.stdout.splitlines()[1:] if line.strip()]
    if shutil.which("readelf") is None:
        return None
    out = subprocess.run(["readelf", "-d", path], capture_output=True, text=True, check=True)
    return re.findall(r"\(NEEDED\)\s+Shared library: \[(.+?)\]", out.stdout)


def _loaded_libraries() -> list[str]:
    """Every shared library this process has loaded, by path."""
    if sys.platform == "darwin":
        dyld = ctypes.CDLL("/usr/lib/libSystem.B.dylib")
        dyld._dyld_image_count.restype = ctypes.c_uint32
        dyld._dyld_get_image_name.restype = ctypes.c_char_p
        dyld._dyld_get_image_name.argtypes = [ctypes.c_uint32]
        return [dyld._dyld_get_image_name(i).decode() for i in range(dyld._dyld_image_count())]
    with open("/proc/self/maps") as maps:
        return sorted({line.split()[-1] for line in maps if "/" in line.split()[-1]})


def test_extension_links_the_generic_blas_and_lapack() -> None:
    linked = _linked_libraries(moist._libmoist.__file__)
    if linked is None:
        pytest.skip("neither readelf nor otool is installed")
    names = [os.path.basename(lib) for lib in linked]
    assert not [n for n in names if "openblas" in n], names
    assert any(n.startswith("liblapack.") for n in names), names
    assert any(n.startswith("libblas.") for n in names), names


def test_blas_and_lapack_come_from_the_environment() -> None:
    prefix = os.path.realpath(sys.prefix)
    blas = [
        lib
        for lib in _loaded_libraries()
        if _BLAS_NAME.search(os.path.basename(lib))
        and not lib.startswith("/System/")  # macOS's own Accelerate
    ]
    assert blas, "no BLAS/LAPACK loaded after importing moist"
    outside = [lib for lib in blas if not os.path.realpath(lib).startswith(prefix + os.sep)]
    assert not outside, f"BLAS/LAPACK loaded from outside {prefix}: {outside}"


# ---------------------------------------------------------------------------------------------
# patches/0002: the surface-adjoint contractions host functionals need


_WATER = np.array([[0.13, 0.0, 0.2217], [0.0, 1.4309, -0.8867], [0.0, -1.4309, -0.8867]])


def _water_cavity(shift: tuple[int, int, float] | None = None) -> moist.CavityDROP:
    """SvdW-DROP around water (bohr), its oxygen off the plane of symmetry so that no
    gradient component vanishes; ``shift`` moves one coordinate of one atom."""
    xyz = _WATER.copy()
    if shift is not None:
        atom, k, h = shift
        xyz[atom, k] += h
    cavity = moist.CavityDROP(lsf=moist.SvdW(), context=moist.Context(nthreads=1, verbosity=0))
    cavity.update(moist.Structure([8, 1, 1], xyz))
    return cavity


def test_surface_contractions_are_exported() -> None:
    assert hasattr(moist._libmoist.lib, "moist_contract_cavity_surface_gradient")
    assert hasattr(moist._libmoist.lib, "moist_contract_cavity_surface_density")


def test_total_area_and_volume_gradients() -> None:
    """dA/dR and dV/dR through moist_contract_cavity_surface_gradient, against the cavity
    rebuilt at displaced nuclei.  moist's volume is sum_i a_i (r_i - c) . n_i / 3 with c the
    centroid of the atoms, so it takes the area, position, normal and origin channels."""
    cav = _water_cavity()
    s = cav.snapshot()
    g_area = cav.contract_surface_gradient(w_a=np.ones(s.ngrid))
    r = s.xyz - _WATER.mean(axis=0)
    g_volume = cav.contract_surface_gradient(
        w_a=np.einsum("ix,ix->i", r, s.normal0) / 3.0,
        w_xyz=s.a[:, None] * s.normal0 / 3.0,
        w_n=s.a[:, None] * r / 3.0,
        w_c=-np.einsum("i,ix->x", s.a, s.normal0) / 3.0,
    )
    assert cav.contract_surface_density(w_a=np.ones(s.ngrid)) is None  # a fixed surface

    def difference(atom: int, k: int, h: float) -> np.ndarray:
        plus, minus = _water_cavity((atom, k, h)), _water_cavity((atom, k, -h))
        return np.array([plus.area - minus.area, plus.volume - minus.volume]) / (2 * h)

    for atom in range(3):
        for k in range(3):
            # Richardson: the cavity is small and curved, so steps of 1e-4 leave 1e-6
            fd = (4 * difference(atom, k, 1e-4) - difference(atom, k, 2e-4)) / 3
            assert g_area[atom, k] == pytest.approx(fd[0], rel=1e-7, abs=1e-8)
            assert g_volume[atom, k] == pytest.approx(fd[1], rel=1e-7, abs=1e-8)
