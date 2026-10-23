# Copyright (C) 2026 Kent Gorday
# SPDX-License-Identifier: LGPL-3.0-only
"""The nonelectrostatic correction on moist's DROP cavities (nonelectrostatic_drop, moist_host).

A distorted formate at HF/def2-SVP (exact integrals), IEF-PCM on SvdW-DROP and rho-DROP: the
energies against tautomator's PySCF implementation on the same calculation (its
pyscf_nonelectrostatic_drop, with which this one agrees to ~1e-10 Eh), and both models'
gradients against five-point energy differences.  moist's PCM solves its charges iteratively
to 1e-8 by default, which leaves noise in the SCF density that the post-SCF energy (not
variational) carries into differences; these use its dense solver.  Runs in the moist
environment.
"""

from __future__ import annotations

from functools import partial
from typing import Any

import numpy as np
import psi4
import pytest

import psi4_swig_pcm
from psi4_swig_pcm.nonelectrostatic import NonelectrostaticSettings
from tests.helpers import HF_OPTIONS, displaced, five_point, psi4_molecule

moist = pytest.importorskip("moist")
from psi4_swig_pcm.moist_host import MoistSettings  # noqa: E402

pytestmark = pytest.mark.moist

FORMATE = np.array(
    [
        [6.150766787413e-05, 1.493727687542e-02, 9.689506920057e-01],
        [2.090860928821e00, -2.273353925859e-02, -1.440686339781e-01],
        [-2.132383340629e00, 6.701076227773e-02, -1.190966321558e-01],
        [-3.102374499100e-02, 2.449210250926e-02, 3.154789717186e00],
    ]
)  # bohr

# tautomator's pyscf_nonelectrostatic_drop on the same calculation: (total energy, (dispersion,
# exchange, short range)) for the self-consistent and the post-SCF model.  Made on the cavity
# _settings builds: moist's own default grid, nleb 194, with pruning level 4.  tautomator's
# pyscf_moist.drop_cavity defaults to nleb 434 since then, so regenerating these from it needs
# drop_cavity(name, nleb=194) (and IEF-PCM with PCMParameters(solver="DENSE"), the same
# formate and HF/def2-SVP with exact integrals, and the same correction settings).
PYSCF: dict[tuple[str, bool], tuple[float, tuple[float, float, float]]] = {
    ("svdw", True): (
        -188.238004715353,
        (-3.182248184475e-03, 3.424947538641e-03, -1.003031028594e-01),
    ),
    ("svdw", False): (
        -188.227791998990,
        (-3.268405431691e-03, 3.732847140768e-03, -7.862833111409e-02),
    ),
    ("rho", True): (
        -188.233302457221,
        (-3.161950107420e-03, 3.081148177519e-03, -9.777583374847e-02),
    ),
    ("rho", False): (
        -188.223707126550,
        (-3.250745762221e-03, 3.168247320235e-03, -7.649048795413e-02),
    ),
}
LSF = {"svdw": lambda: moist.SvdW(), "rho": lambda: moist.Isodensity()}


def _settings(cavity: str, self_consistent: bool) -> MoistSettings:
    # Larger than any fit will give, so that every term's derivatives show
    ne = NonelectrostaticSettings(
        dispersion=-0.0036,
        exchange=0.0105,
        neg_coeff=-0.5,
        neg_exponent=2.5,
        pos_coeff=-0.3,
        pos_exponent=2.5,
        solvent_rho=0.0494,
        self_consistent=self_consistent,
    )
    return MoistSettings(
        cavity=moist.DROP(lsf=LSF[cavity](), parameters=moist.DROPParameters(wleb_prune_level=4)),
        components=(
            moist.ModelComponentIEFPCM(78.3553, parameters=moist.PCMParameters(solver="DENSE")),
        ),
        nonelectrostatic=ne,
    )


@pytest.fixture(autouse=True)
def _psi4_options() -> Any:
    psi4.core.be_quiet()
    psi4.set_options(HF_OPTIONS)
    yield
    psi4_swig_pcm.disable()
    psi4.core.clean()


@pytest.mark.parametrize("self_consistent", [True, False])
@pytest.mark.parametrize("cavity", ["svdw", "rho"])
def test_energy_matches_pyscf(cavity: str, self_consistent: bool) -> None:
    psi4_swig_pcm.enable(_settings(cavity, self_consistent))
    e, wfn = psi4.energy("hf", molecule=psi4_molecule("formate", FORMATE), return_wfn=True)
    total, terms = PYSCF[(cavity, self_consistent)]
    assert e == pytest.approx(total, abs=1e-9)
    D = np.asarray(wfn.Da().np) + np.asarray(wfn.Db().np)
    energies = wfn.ddx.terms.energy(D)
    assert (energies.dispersion, energies.exchange, energies.short_range) == pytest.approx(
        terms, rel=1e-8
    )
    assert psi4.core.variable("NONELECTROSTATIC ENERGY") == pytest.approx(sum(terms), abs=1e-9)


@pytest.mark.slow
@pytest.mark.parametrize("self_consistent", [True, False])
@pytest.mark.parametrize("cavity", ["svdw", "rho"])
def test_gradient_matches_finite_differences(cavity: str, self_consistent: bool) -> None:
    psi4_swig_pcm.enable(_settings(cavity, self_consistent))
    g = np.asarray(psi4.gradient("hf", molecule=psi4_molecule("formate", FORMATE)).np)
    assert g.sum(axis=0) == pytest.approx(0.0, abs=1e-8)

    def energy_at(atom: int, axis: int, step: float) -> float:
        mol = psi4_molecule("formate", displaced(FORMATE, atom, axis, step))
        return float(psi4.energy("hf", molecule=mol))

    for atom, axis in ((0, 0), (1, 1), (2, 2), (3, 0)):
        fd = five_point(partial(energy_at, atom, axis), 2e-3)
        assert g[atom, axis] == pytest.approx(fd, abs=5e-8)
