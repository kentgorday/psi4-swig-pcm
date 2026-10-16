// Copyright (C) 2026 Kent Gorday
// SPDX-License-Identifier: LGPL-3.0-only
//
// This file is part of psi4-swig-pcm; see NOTICE.
//
// Contractions between an AO density/charges and a set of unit Gaussian s "charges", using
// Psi4's own libmints/libint2 integrals:
//
//   phi_el[k] = sum_mn D_mn (k|mn)          electronic potential at Gaussian k
//   V[m,n]    = sum_k  q_k  (k|mn)          Fock contribution of Gaussian charges q
//   B[k, mn]  = (k|mn), m >= n              the integrals themselves, for in-core reuse
//
// The aux basis must be built the way psi4's proc.py builds "diffuse" external charges
// (ghost He s-shells + negative_gaussian_normalization_to_coefficients), so (k|mn) already
// carries the electron sign: phi_el is the electrons' potential (negative) and V is the matrix
// to add to the Fock matrix.
//
// Only the M >= N shell pairs are computed, so the density is taken to be symmetric (as every
// SCF density is): gaussian_esp counts each off-diagonal pair twice.  A non-symmetric density
// (a response or transition density) would need both triangles.
//
// Parallel over aux shells (one per surface point); each thread owns a clone of the integral
// object.  Psi4's ExternalPotential::computePotentialMatrix does the same V contraction but
// opens one OpenMP region per point.

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <algorithm>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "psi4/libmints/basisset.h"
#include "psi4/libmints/integral.h"
#include "psi4/libmints/matrix.h"
#include "psi4/libmints/twobody.h"
#include "psi4/libmints/vector.h"

namespace py = pybind11;

namespace psi {
namespace gcharge {

namespace {

std::vector<std::shared_ptr<TwoBodyAOInt>> make_ints(std::shared_ptr<BasisSet> aux, std::shared_ptr<BasisSet> basis,
                                                     int nthreads) {
    auto zero = BasisSet::zero_ao_basis_set();
    auto fact = std::make_shared<IntegralFactory>(aux, zero, basis, basis);
    std::vector<std::shared_ptr<TwoBodyAOInt>> eri(nthreads);
    eri[0] = std::shared_ptr<TwoBodyAOInt>(fact->eri());
    for (int t = 1; t < nthreads; t++) eri[t] = std::shared_ptr<TwoBodyAOInt>(eri[0]->clone());
    return eri;
}

int thread_rank() {
#ifdef _OPENMP
    return omp_get_thread_num();
#else
    return 0;
#endif
}

// Calls visit(buf, oM, nM, oN, nN, same_shell) for every significant shell pair M >= N of
// (P 0|M N), where buf holds the integrals as [p][m][n] over the functions of P, M and N.
template <typename Visit>
void for_each_shell_pair(TwoBodyAOInt& eri, const BasisSet& basis, int P, Visit&& visit) {
    for (const auto& [M, N] : eri.shell_pairs()) {
        if (eri.compute_shell(P, 0, M, N) == 0) continue;
        const auto& sM = basis.shell(M);
        const auto& sN = basis.shell(N);
        visit(eri.buffer(), sM.function_index(), sM.nfunction(), sN.function_index(), sN.nfunction(), M == N);
    }
}

}  // namespace

SharedVector gaussian_esp(std::shared_ptr<BasisSet> aux, std::shared_ptr<BasisSet> basis, SharedMatrix D,
                          int nthreads) {
    auto eri = make_ints(aux, basis, nthreads);
    auto phi = std::make_shared<Vector>("Gaussian ESP", aux->nbf());
    double* phip = phi->pointer();
    double** Dp = D->pointer();

#pragma omp parallel for schedule(dynamic) num_threads(nthreads)
    for (int P = 0; P < aux->nshell(); P++) {
        const int nP = aux->shell(P).nfunction(), oP = aux->shell(P).function_index();
        std::vector<double> acc(nP, 0.0);
        for_each_shell_pair(*eri[thread_rank()], *basis, P,
                            [&](const double* buf, int oM, int nM, int oN, int nN, bool same) {
                                const double fac = same ? 1.0 : 2.0;  // D symmetric: (n, m) too
                                for (int p = 0, idx = 0; p < nP; p++) {
                                    double s = 0.0;
                                    for (int m = 0; m < nM; m++)
                                        for (int n = 0; n < nN; n++, idx++) s += Dp[oM + m][oN + n] * buf[idx];
                                    acc[p] += fac * s;
                                }
                            });
        for (int p = 0; p < nP; p++) phip[oP + p] = acc[p];
    }
    return phi;
}

SharedMatrix gaussian_potential_matrix(std::shared_ptr<BasisSet> aux, std::shared_ptr<BasisSet> basis, SharedVector q,
                                       int nthreads) {
    auto eri = make_ints(aux, basis, nthreads);
    const int nbf = basis->nbf();
    const double* qp = q->pointer();
    std::vector<SharedMatrix> Vt(nthreads);
    for (int t = 0; t < nthreads; t++) {
        Vt[t] = std::make_shared<Matrix>("V", nbf, nbf);
        Vt[t]->zero();
    }

#pragma omp parallel for schedule(dynamic) num_threads(nthreads)
    for (int P = 0; P < aux->nshell(); P++) {
        const int rank = thread_rank();
        const int nP = aux->shell(P).nfunction(), oP = aux->shell(P).function_index();
        double** Vp = Vt[rank]->pointer();
        for_each_shell_pair(*eri[rank], *basis, P, [&](const double* buf, int oM, int nM, int oN, int nN, bool same) {
            for (int p = 0, idx = 0; p < nP; p++) {
                const double c = qp[oP + p];
                for (int m = 0; m < nM; m++)
                    for (int n = 0; n < nN; n++, idx++)
                        if (!same || n <= m) Vp[oM + m][oN + n] += c * buf[idx];  // lower triangle only
            }
        });
    }
    auto V = std::make_shared<Matrix>("Gaussian charge potential", nbf, nbf);
    for (int t = 0; t < nthreads; t++) V->add(Vt[t]);
    // Only the lower triangle was filled (M >= N shell pairs, m >= n inside diagonal blocks).
    double** Vp = V->pointer();
    for (int i = 0; i < nbf; i++)
        for (int j = 0; j < i; j++) Vp[j][i] = Vp[i][j];
    return V;
}

// In-core store: out[rows[k], mn] = (k|mn) for m >= n, packed row-major as m*(m+1)/2 + n, so
// that each SCF iteration is two BLAS matrix-vector products:
//   phi = B d   with d_mn = D_mn (m == n) or 2 D_mn (m > n)
//   V   = unpack(B^T q)
// Written straight into the caller's array, rows[k] being the row of aux function k; aux
// functions past the end of rows (the zero-charge anchor) are not stored.
void gaussian_3c_packed_into(std::shared_ptr<BasisSet> aux, std::shared_ptr<BasisSet> basis,
                             py::array_t<double, py::array::c_style> out,
                             py::array_t<std::int64_t, py::array::c_style | py::array::forcecast> rows,
                             int nthreads) {
    const long nbf = basis->nbf(), npack = nbf * (nbf + 1) / 2;
    if (out.ndim() != 2 || out.shape(1) != npack) throw std::invalid_argument("out must be (nrows, nbf*(nbf+1)/2)");
    if (rows.ndim() != 1 || rows.shape(0) > aux->nbf()) throw std::invalid_argument("rows: one per aux function");
    const long nrows = rows.shape(0), nout = out.shape(0);
    const std::int64_t* rp = rows.data();
    for (long k = 0; k < nrows; k++)
        if (rp[k] < 0 || rp[k] >= nout) throw std::out_of_range("rows index outside out");
    double* Bp = out.mutable_data();
    {
        py::gil_scoped_release release;
        for (long k = 0; k < nrows; k++) std::fill(Bp + rp[k] * npack, Bp + (rp[k] + 1) * npack, 0.0);
        auto eri = make_ints(aux, basis, nthreads);
#pragma omp parallel for schedule(dynamic) num_threads(nthreads)
        for (int P = 0; P < aux->nshell(); P++) {
            const int nP = aux->shell(P).nfunction(), oP = aux->shell(P).function_index();
            for_each_shell_pair(*eri[thread_rank()], *basis, P,
                                [&](const double* buf, int oM, int nM, int oN, int nN, bool /*same*/) {
                                    for (int p = 0, idx = 0; p < nP; p++) {
                                        const long k = oP + p;
                                        if (k >= nrows) {  // the anchor: skip its integrals
                                            idx += nM * nN;
                                            continue;
                                        }
                                        double* row = Bp + rp[k] * npack;
                                        for (int m = 0; m < nM; m++)
                                            for (int n = 0; n < nN; n++, idx++) {
                                                const long i = oM + m, j = oN + n;
                                                if (i >= j) row[i * (i + 1) / 2 + j] = buf[idx];
                                            }
                                    }
                                });
        }
    }
}

}  // namespace gcharge
}  // namespace psi

PYBIND11_MODULE(_gcharge, m) {
    py::module_::import("psi4.core");  // register psi4 types before our signatures use them
    // noconvert: a converted copy of `out` would be filled and then thrown away.
    m.def("gaussian_3c_packed_into", &psi::gcharge::gaussian_3c_packed_into, py::arg("aux"), py::arg("basis"),
          py::arg("out").noconvert(), py::arg("rows"), py::arg("nthreads"));
    m.def("gaussian_esp", &psi::gcharge::gaussian_esp, py::arg("aux"), py::arg("basis"), py::arg("D"),
          py::arg("nthreads"), py::call_guard<py::gil_scoped_release>());
    m.def("gaussian_potential_matrix", &psi::gcharge::gaussian_potential_matrix, py::arg("aux"), py::arg("basis"),
          py::arg("q"), py::arg("nthreads"), py::call_guard<py::gil_scoped_release>());
}
