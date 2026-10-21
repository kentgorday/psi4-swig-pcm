# psi4-swig-pcm

Switching/Gaussian ("SWIG", Lange & Herbert, JCP 133, 244111 (2010)) polarizable continuum
models for [Psi4](https://psicode.org) 1.11, with analytic nuclear gradients. Psi4's own PCM
(PCMSolver) has none, so Psi4 falls back to finite differences of energies.

- C-PCM, COSMO, IEF-PCM (the default) and SS(V)PE, with the same surface, radii and
  discretization as PySCF 2.11's `pyscf.solvent.pcm`. Energies and gradients match PySCF to
  1e-9 Eh and 5e-8 Eh/bohr.
- SWIG's switching function (the default, as in PySCF) or ISWIG's, PySCF 2.13's
  `surface_discretization_method = "ISWIG"`: the surface, its derivatives and the PCM energy
  match PySCF 2.13.1 to 3e-15 and 1e-13 Eh.
- It works through Psi4's own driver. `psi4.energy`, `psi4.gradient`, `psi4.optimize` (optking
  or geomeTRIC), `psi4.hessian` and `psi4.frequency` all include the PCM once it is enabled.
- It needs no change to Psi4's source and no PySCF at run time.

The repository also holds a build recipe for [moist](https://github.com/lukaswittmann/moist),
which tautomator uses under PySCF, without Psi4: see [The moist package](#the-moist-package).

## Use

```python
import psi4
import psi4_swig_pcm

psi4_swig_pcm.enable(psi4_swig_pcm.PCMSettings(method="IEF-PCM", eps=78.3553))
E, wfn = psi4.optimize("wb97x-3c", return_wfn=True)
E_solv = wfn.variable("DD SOLVATION ENERGY")
freqs = psi4.frequency("wb97x-3c", return_wfn=True)[1].frequencies().to_array()
psi4_swig_pcm.disable()
```

`with psi4_swig_pcm.enabled(settings): ...` does the same for one block. The same lines work
in a Psi4 input file run by the `psi4` executable.

`PCMSettings` fields:

| field | default | meaning |
|---|---|---|
| `method` | `"IEF-PCM"` | `"C-PCM"`, `"COSMO"`, `"IEF-PCM"` or `"SS(V)PE"` |
| `eps` | 78.3553 | dielectric constant (water), at least 1; `float("inf")` for a conductor |
| `lebedev_order` | 29 | 302 points per sphere; orders with negative weights are not offered |
| `switching` | `"SWIG"` | `"SWIG"` or `"ISWIG"`: how points buried in neighbouring spheres fade out. ISWIG's erf switch keeps ~10% more points |
| `radii` | modified Bondi × 1.2, H 1.10 Å | cavity radii in bohr by atomic number (any sequence; stored as a tuple) |
| `check` | `True` | per-geometry checks: S positive definite, Gauss's law (as PCMSolver does) |
| `incore_mb` | 2000 | store the (k\|mn) integrals for a geometry if they fit; 0 = always direct |
| `nonelectrostatic` | `None` | a `NonelectrostaticSettings`: the nonelectrostatic correction (below) |

Each SCF writes a short block to Psi4's output: the model, the cavity's atoms and points, how
the integrals are handled, and the Gauss's-law check.

- **SCF references:** RHF/RKS and UHF/UKS. The UHF gradient is tested against finite
  differences; ROHF and CUHF gradients raise `NotImplementedError`. Psi4 1.11 has no analytic
  gradients for VV10 functionals, so for those it uses finite differences of PCM energies.
  Those are correct, but slow.
- **Correlated methods (MP2, CC, ...):** the energy sees the PCM only through the SCF
  reference (no PCM response of the correlation energy). Psi4's analytic correlated gradients
  would leave the PCM terms out, so their gradients become finite differences of these
  energies. CFOUR methods (`c4-*`) run their own SCF outside Psi4 and are refused.
- **Ghost atoms (counterpoise):** they carry basis functions but no cavity sphere, so a
  monomer in the dimer basis sits in its own cavity.

### Nonelectrostatic correction

`PCMSettings(nonelectrostatic=NonelectrostaticSettings(...))` adds the CMIRS-style correction
of [tautomator](https://github.com/kentgorday/qm-pka) (`docs/nonelectrostatic.md` there):
dispersion and exchange integrals of the solute density inside and outside the cavity, and a
surface integral of powers of the solute's normal field. It is the same implementation on
Psi4's basis functions and libint2; on the same calculation the two agree to 1e-10 Eh.

```python
from psi4_swig_pcm.nonelectrostatic import NonelectrostaticSettings

ne = NonelectrostaticSettings(dispersion=A, exchange=B, neg_coeff=c_minus, neg_exponent=n_minus,
                              pos_coeff=c_plus, pos_exponent=n_plus, solvent_rho=0.0494)
psi4_swig_pcm.enable(psi4_swig_pcm.PCMSettings(eps=78.3553, nonelectrostatic=ne))
```

- **Post-SCF (default):** the SCF is the PCM's alone, and G is evaluated on its density, as in
  CMIRS. `psi4.energy` returns SCF + G (`NONELECTROSTATIC ENERGY` holds G). The gradient
  includes the density's response: four extra SCFs with ±λ·∂G/∂D and ±2λ·∂G/∂D on the Fock
  matrix, each started from the previous orbitals, give it by Richardson extrapolation
  (agreement with energy differences ~1e-8 Eh/bohr). An exact Z-vector would need Psi4's
  JK gradient for an arbitrary density, an XC-kernel derivative and ECP derivative integrals,
  none of which Psi4 1.11 exports.
- **Self-consistent (`self_consistent=True`):** G and its potential enter every SCF iteration,
  so the density relaxes with the correction; the gradient adds G's explicit derivative.
- SCF references: tested with RHF. UHF/UKS go through the same code (the potential is the
  same for both spins) but have no test yet.

## How it plugs into Psi4

`enable()` makes these process-global changes, and `disable()` undoes them:

1. **The solver goes into Psi4's DDX slot, and the `DDX` option is turned on.**
   `procrouting.solvent.ddx` holds the solver. On every SCF iteration, `scf_iterator` calls
   `get_solvation_contributions(D)`; it returns the reaction-field energy and Fock matrix,
   computed from surface Gaussian charges with Psi4's own libint2 integrals
   (`csrc/gcharge.cpp`).
2. **The SCF gradient entries of `psi4.driver.procedures["gradient"]` are wrapped.** Psi4 1.11
   has no DDX gradient, so without this `psi4.gradient` would silently return a gradient
   without the PCM terms. The wrapper:
   - freezes the converged charges on an `ExternalPotential`;
   - lets Psi4's `run_scf_gradient` add their interaction with electrons and nuclei, plus the
     dispersion gradient;
   - adds the motion of the charges with their atoms, and the cavity term.
3. **Every other analytic derivative is removed from the table:** the correlated gradients,
   and all Hessians. Psi4 then picks finite differences: of the PCM gradient for SCF Hessians
   and frequencies, of energies for the rest. Those run in the same process, so they see the
   patches.
4. **CFOUR energies are refused.**
5. **With a post-SCF nonelectrostatic correction, SCF energy entries of
   `procedures["energy"]` add G.**

These are Psi4 internals. `tests/test_psi4_interfaces.py` checks each assumption separately,
so a Psi4 upgrade names what changed.

## Install

The package builds a C++ extension against Psi4's headers, through Psi4's CMake plugin
interface. It only works with the Psi4 it was built against, so it is a conda package built
by [pixi-build](https://pixi.sh). That package declares `psi4 >=1.11,<1.12` (Psi4
run-exports it), and the extension also refuses to import under a different Psi4.

From another pixi workspace, for example the `psi4` environment of tautomator, the pKa
workflow this package was written for:

```toml
[workspace]
preview = ["pixi-build"]

[feature.psi4.dependencies]
# add rev = "<commit>" to pin it; pixi.lock records the commit either way
psi4-swig-pcm = { git = "https://github.com/kentgorday/psi4-swig-pcm.git" }
```

In a conda environment that already has Psi4 1.11, a C++ compiler, pybind11, Eigen 5,
libboost-headers, CMake and scikit-build-core:

```bash
pip install --no-build-isolation .
```

## Develop

```bash
pixi install                # builds the package and installs it editable
pixi run test               # fast tests (~10 s)
pixi run test-slow          # finite differences, optimizations, frequencies (~2 min)
pixi run -e crosscheck crosscheck   # against PySCF 2.11 (~3 min)
pixi run -e moist test-moist        # the moist package (compiles moist the first time)
pixi run check              # ruff, ruff format, mypy --strict
```

## The moist package

`packages/moist` builds [moist](https://github.com/lukaswittmann/moist) (MPL-2.0,
pre-release), which has no conda-forge or PyPI package, as a conda package with pixi-build:
a rattler-build recipe that pins a moist release and the Fortran libraries its meson wraps
name. The recipe fetches all of them, so meson itself downloads nothing; moving to a new
moist release means copying the revisions from its `subprojects/*.wrap` too.

It is the one part of this repository that is not about Psi4. tautomator's `moist`
environment runs moist under PySCF, with no Psi4 in it, and takes moist from here, so that
one recipe serves both repositories:

```toml
moist = { git = "https://github.com/kentgorday/psi4-swig-pcm.git", subdirectory = "packages/moist" }
```

A change to the recipe has to suit both:

- **No Psi4.** The package must not depend on it.
- **One Python.** The recipe pins the Python both repositories use (3.13).
- **BLAS/LAPACK from the environment.** The recipe builds against the netlib variants of
  `libblas`/`liblapack`, as conda-forge's own builds do, so moist runs on whatever the
  environment selects: Accelerate on osx-arm64; OpenBLAS here and MKL in tautomator on
  linux-64. Built against the default OpenBLAS variants, it would link `libopenblas`, then
  fail to import beside Accelerate or load a system OpenBLAS beside MKL.
  `tests/test_moist_build.py` checks the extension's links and the libraries it loads.
- **Patches.** `patches/` adds an `#include <algorithm>` to moist's copy of FINUFFT that
  conda-forge's clang 23 (macOS) needs.

`pixi run -e moist test-moist` builds and checks it here. tautomator's `pixi.lock` pins the
commit it builds from; after a change here, `pixi update moist` there moves it.

## Memory

Per geometry the PCM holds three dense npts × npts matrices: S, D and the LU factors of K
(two for C-PCM and COSMO), and it briefly needs about four to five while building them. A
38-atom chain at Lebedev order 29 has 4,820 points and peaks at 0.7 GB above the surface.
The cost grows with npts², at roughly 100–150 points per atom: some 5–6 GB for 100 atoms. The
(k|mn) integrals add npts · nbf(nbf + 1)/2 doubles when they fit `incore_mb`. None of this
counts against `psi4.set_memory`, so leave room beside Psi4's own memory, or lower
`incore_mb` or `lebedev_order`.

## Known issues handled here

- **Close points.** Switching/Gaussian points can nearly coincide, and a Psi4 `Molecule`
  rejects atoms closer than 0.1 bohr. The Gaussians are split into several sets.
- **Symmetric point sets.** With a fixed frame, Psi4's point-group detection fails
  ("Unrecognized point group bits") on some exactly symmetric point sets. Every set carries
  one zero-charge "anchor" Gaussian far away, which breaks the symmetry.
- **Boost at configure time.** `libint2-config.cmake` (loaded by `psi4Config.cmake`) asks for
  Boost, but no Boost header reaches the extension. `CMakeLists.txt` declares the header-only
  target instead of adding `libboost` as a run dependency.

## License

psi4-swig-pcm is licensed under the GNU Lesser General Public License, version 3 only
(`LGPL-3.0-only`), like Psi4. The LGPL is a set of additional permissions on top of the GNU
GPL, so both texts are included: [COPYING.LESSER](COPYING.LESSER) and [COPYING](COPYING).

`src/psi4_swig_pcm/cavity.py` is derived from PySCF and remains under the Apache License 2.0
([LICENSES/Apache-2.0.txt](LICENSES/Apache-2.0.txt)). PySCF's attribution notices are in
[NOTICE](NOTICE), which also credits the Psi4 code `charges.py` adapts.

Built packages also contain pybind11 code, compiled into the extension, under the BSD
3-Clause License ([LICENSES/pybind11-BSD-3-Clause.txt](LICENSES/pybind11-BSD-3-Clause.txt)).
That applies to the binary only; this repository's source contains no pybind11 code.
