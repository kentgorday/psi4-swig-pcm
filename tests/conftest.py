# Copyright (C) 2026 Kent Gorday
# SPDX-License-Identifier: LGPL-3.0-only
from __future__ import annotations

from collections.abc import Iterator

# psi4 before pyscf (crosscheck environment): the PyPI pyscf wheel's bundled libxc otherwise
# shadows Psi4's, and Psi4 then fails with "Could not find required LibXC functional".
import psi4
import pytest

import psi4_swig_pcm


@pytest.fixture(scope="session", autouse=True)
def _psi4_session() -> None:
    psi4.core.be_quiet()
    psi4.set_memory("2 GB")
    psi4.set_num_threads(4)


@pytest.fixture(autouse=True)
def _clean_psi4() -> Iterator[None]:
    """Every test starts and ends with Psi4's defaults and the PCM disabled."""
    psi4.core.clean()
    psi4.core.clean_options()
    psi4.core.clean_variables()
    yield
    psi4_swig_pcm.disable()
    psi4.core.clean()
    psi4.core.clean_options()
    psi4.core.clean_variables()
