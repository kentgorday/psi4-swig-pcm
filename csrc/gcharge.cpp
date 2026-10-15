// Copyright (C) 2026 Kent Gorday
// SPDX-License-Identifier: LGPL-3.0-only
//
// This file is part of psi4-swig-pcm; see NOTICE.
//
// Direct (no-storage) contractions between an AO density/charges and a set of
// unit Gaussian s "charges", using Psi4's own libmints/libint2 integrals.
//
//   phi_el[k] = sum_mn D_mn (k|mn)          electronic potential at Gaussian k
//   V[m,n]    = sum_k  q_k  (k|mn)          Fock contribution of Gaussian charges q
//
// The aux basis must be built the way psi4's proc.py builds "diffuse" external
// charges (ghost He s-shells + negative_gaussian_normalization_to_coefficients),
// so (k|mn) already carries the electron sign: phi_el is the electrons'
// potential (negative) and V is the matrix to add to the Fock matrix.
//
// Parallel over aux shells (one per surface point); each thread owns a clone of
// the integral object.  Psi4's ExternalPotential::computePotentialMatrix does the
// same V contraction but opens one OpenMP region per point.

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <algorithm>
#include <memory>
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

}  // namespace

SharedVector gaussian_esp(std::shared_ptr<BasisSet> aux, std::shared_ptr<BasisSet> basis, SharedMatrix D,
                          int nthreads) {
    auto eri = make_ints(aux, basis, nthreads);
    const auto& pairs = eri[0]->shell_pairs();  // significant (M >= N) pairs
    const long npairs = pairs.size();
    auto phi = std::make_shared<Vector>("Gaussian ESP", aux->nbf());
    double* phip = phi->pointer();
    double** Dp = D->pointer();

#pragma omp parallel for schedule(dynamic) num_threads(nthreads)
    for (int P = 0; P < aux->nshell(); P++) {
        int rank = 0;
#ifdef _OPENMP
        rank = omp_get_thread_num();
#endif
        const int nP = aux->shell(P).nfunction();
        const int oP = aux->shell(P).function_index();
        std::vector<double> acc(nP, 0.0);
        for (long MN = 0; MN < npairs; MN++) {
            const auto [M, N] = pairs[MN];
            if (eri[rank]->compute_shell(P, 0, M, N) == 0) continue;
            const double* buf = eri[rank]->buffer();
            const int nM = basis->shell(M).nfunction(), oM = basis->shell(M).function_index();
            const int nN = basis->shell(N).nfunction(), oN = basis->shell(N).function_index();
            const double fac = (M == N) ? 1.0 : 2.0;
            for (int p = 0, idx = 0; p < nP; p++) {
                double s = 0.0;
                for (int m = 0; m < nM; m++)
                    for (int n = 0; n < nN; n++, idx++) s += Dp[oM + m][oN + n] * buf[idx];
                acc[p] += fac * s;
            }
        }
        for (int p = 0; p < nP; p++) phip[oP + p] = acc[p];
    }
    return phi;
}

SharedMatrix gaussian_potential_matrix(std::shared_ptr<BasisSet> aux, std::shared_ptr<BasisSet> basis, SharedVector q,
                                       int nthreads) {
    auto eri = make_ints(aux, basis, nthreads);
    const auto& pairs = eri[0]->shell_pairs();
    const long npairs = pairs.size();
    const int nbf = basis->nbf();
    const double* qp = q->pointer();
    std::vector<SharedMatrix> Vt(nthreads);
    for (int t = 0; t < nthreads; t++) {
        Vt[t] = std::make_shared<Matrix>("V", nbf, nbf);
        Vt[t]->zero();
    }

#pragma omp parallel for schedule(dynamic) num_threads(nthreads)
    for (int P = 0; P < aux->nshell(); P++) {
        int rank = 0;
#ifdef _OPENMP
        rank = omp_get_thread_num();
#endif
        const int nP = aux->shell(P).nfunction();
        const int oP = aux->shell(P).function_index();
        double** Vp = Vt[rank]->pointer();
        for (long MN = 0; MN < npairs; MN++) {
            const auto [M, N] = pairs[MN];
            if (eri[rank]->compute_shell(P, 0, M, N) == 0) continue;
            const double* buf = eri[rank]->buffer();
            const int nM = basis->shell(M).nfunction(), oM = basis->shell(M).function_index();
            const int nN = basis->shell(N).nfunction(), oN = basis->shell(N).function_index();
            for (int p = 0, idx = 0; p < nP; p++) {
                const double c = qp[oP + p];
                for (int m = 0; m < nM; m++)
                    for (int n = 0; n < nN; n++, idx++)
                        if (M != N || n <= m) Vp[oM + m][oN + n] += c * buf[idx];  // lower triangle only
            }
        }
    }
    auto V = std::make_shared<Matrix>("Gaussian charge potential", nbf, nbf);
    for (int t = 0; t < nthreads; t++) V->add(Vt[t]);
    // Only the lower triangle was filled (M >= N shell pairs, m >= n inside diagonal blocks).
    double** Vp = V->pointer();
    for (int i = 0; i < nbf; i++)
        for (int j = 0; j < i; j++) Vp[j][i] = Vp[i][j];
    return V;
}

// In-core variant: B[k, mn] = (k|mn) for m >= n, packed row-major as m*(m+1)/2 + n, so that
//   phi = B d   with d_mn = D_mn (m == n) or 2 D_mn (m > n)
//   V   = unpack(B^T q)
// are two BLAS matrix-vector products per SCF iteration.  Memory: naux * nbf (nbf + 1) / 2 doubles.
py::array_t<double> gaussian_3c_packed(std::shared_ptr<BasisSet> aux, std::shared_ptr<BasisSet> basis,
                                       int nthreads) {
    const long nbf = basis->nbf(), npack = nbf * (nbf + 1) / 2, naux = aux->nbf();
    py::array_t<double> B({naux, npack});
    double* Bp = B.mutable_data();
    {
        py::gil_scoped_release release;
        std::fill(Bp, Bp + naux * npack, 0.0);
        auto eri = make_ints(aux, basis, nthreads);
        const auto& pairs = eri[0]->shell_pairs();
        const long npairs = pairs.size();
#pragma omp parallel for schedule(dynamic) num_threads(nthreads)
        for (int P = 0; P < aux->nshell(); P++) {
            int rank = 0;
#ifdef _OPENMP
            rank = omp_get_thread_num();
#endif
            const int nP = aux->shell(P).nfunction(), oP = aux->shell(P).function_index();
            for (long MN = 0; MN < npairs; MN++) {
                const auto [M, N] = pairs[MN];
                if (eri[rank]->compute_shell(P, 0, M, N) == 0) continue;
                const double* buf = eri[rank]->buffer();
                const int nM = basis->shell(M).nfunction(), oM = basis->shell(M).function_index();
                const int nN = basis->shell(N).nfunction(), oN = basis->shell(N).function_index();
                for (int p = 0, idx = 0; p < nP; p++) {
                    double* row = Bp + (oP + p) * npack;
                    for (int m = 0; m < nM; m++)
                        for (int n = 0; n < nN; n++, idx++) {
                            const long i = oM + m, j = oN + n;
                            if (i >= j) row[i * (i + 1) / 2 + j] = buf[idx];  // M >= N shell pairs cover i >= j
                        }
                }
            }
        }
    }
    return B;
}

}  // namespace gcharge
}  // namespace psi

PYBIND11_MODULE(_gcharge, m) {
    py::module_::import("psi4.core");  // register psi4 types before our signatures use them
    m.def("gaussian_3c_packed", &psi::gcharge::gaussian_3c_packed, py::arg("aux"), py::arg("basis"),
          py::arg("nthreads"));
    m.def("gaussian_esp", &psi::gcharge::gaussian_esp, py::arg("aux"), py::arg("basis"), py::arg("D"),
          py::arg("nthreads"), py::call_guard<py::gil_scoped_release>());
    m.def("gaussian_potential_matrix", &psi::gcharge::gaussian_potential_matrix, py::arg("aux"), py::arg("basis"),
          py::arg("q"), py::arg("nthreads"), py::call_guard<py::gil_scoped_release>());
}
