# Copyright (C) 2026 Kent Gorday
# SPDX-License-Identifier: LGPL-3.0-only
"""The moist package from packages/moist: its extension takes BLAS/LAPACK from the environment.

Built against conda-forge's OpenBLAS variants of libblas/liblapack, the extension records
libopenblas, which the environments that use it do not install (Accelerate on osx-arm64, MKL in
tautomator on linux-64): it then fails to import, or loads an OpenBLAS from outside the
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

import moist._libmoist

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
