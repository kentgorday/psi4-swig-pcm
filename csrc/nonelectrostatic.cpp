// Copyright (C) 2026 Kent Gorday
// SPDX-License-Identifier: LGPL-3.0-only
//
// This file is part of psi4-swig-pcm; see NOTICE.
//
// The compiled parts of the nonelectrostatic correction (psi4_swig_pcm.nonelectrostatic):
//
// * The solute's electronic field normal to the cavity surface, from libint2 nuclear-attraction
//   integrals with one unit point charge at each surface point s_i:
//
//     U_mn(s) = int phi_m phi_n / |r - s|,   S_i,mn = n_i . grad_s U_mn(s_i),
//
//   so that n_i . E_el(s_i) = sum_mn D_mn S_i,mn (the electrons' field: E = grad_s int rho/|r-s|).
//   libint2 returns V = -U (an attractive unit charge) and its derivatives with respect to the
//   bra, ket and charge centres; Psi4's basis sets provide the libint2 shells, normalization
//   embedded, in Psi4's AO order.  Their second derivatives give the nuclear gradient of
//   sum_i q_i . E_el(s_i) at fixed density: the basis functions moving with their atoms, and the
//   points moving (field gradient at s_i).
//
// * The dispersion kernel's surface sum J(r) = sum_i A_i (R . n_i) arctan(t)/(3 delta^3 R^3),
//   R = s_i - r, t = R^3/delta^3, and its derivatives (tautomator's dispersion_kernel.py).
//
// * The cavity grids' smooth partition of unity among the atoms and its derivative
//   (tautomator's cavity_quadrature.py).
//
// Shell pairs M >= N only, D taken symmetric, as in gcharge.cpp.  Parallel over points.

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <libint2.hpp>

#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "psi4/libmints/basisset.h"
#include "psi4/libmints/gshell.h"

namespace py = pybind11;
using Array = py::array_t<double, py::array::c_style | py::array::forcecast>;
using IntArray = py::array_t<std::int64_t, py::array::c_style | py::array::forcecast>;

namespace psi {
namespace nonelectrostatic {

namespace {

int thread_rank() {
#ifdef _OPENMP
    return omp_get_thread_num();
#else
    return 0;
#endif
}

struct Shells {
    std::vector<libint2::Shell> shells;
    std::vector<int> offset, size, center;
    // order[M][f]: the position, in libint2's output for shell M, of Psi4's function f.
    // libint2 here orders solid harmonics m = -l..l (LIBINT_SHGSHELL_ORDERING_STANDARD);
    // Psi4 orders them m = 0, +1, -1, +2, -2, ...  Cartesian shells share one order.
    std::vector<std::vector<int>> order;
    int nbf = 0;
    std::size_t max_nprim = 0;
    int max_l = 0;
};

std::vector<int> psi4_order(const libint2::Shell& sh) {
    const int l = sh.contr[0].l;
    const int nf = static_cast<int>(sh.size());
    std::vector<int> order(nf);
    if (!sh.contr[0].pure || l == 0) {
        for (int f = 0; f < nf; f++) order[f] = f;
        return order;
    }
    order[0] = l;  // m = 0
    for (int j = 1; j <= l; j++) {
        order[2 * j - 1] = l + j;  // m = +j
        order[2 * j] = l - j;      // m = -j
    }
    return order;
}

Shells shells_of(const BasisSet& basis) {
    Shells s;
    s.nbf = basis.nbf();
    for (int i = 0; i < basis.nshell(); i++) {
        const auto& sh = basis.l2_shell(i);
        if (static_cast<int>(sh.size()) != basis.shell(i).nfunction())
            throw std::runtime_error("libint2 and Psi4 disagree on the size of a shell");
        s.shells.push_back(sh);
        s.order.push_back(psi4_order(sh));
        s.offset.push_back(basis.shell(i).function_index());
        s.size.push_back(basis.shell(i).nfunction());
        s.center.push_back(basis.shell_to_center(i));
        s.max_nprim = std::max(s.max_nprim, sh.nprim());
        s.max_l = std::max(s.max_l, sh.contr[0].l);
    }
    return s;
}

std::vector<libint2::Engine> engines(const Shells& s, int deriv, int nthreads) {
    libint2::Engine e(libint2::Operator::nuclear, s.max_nprim, s.max_l, deriv);
    return std::vector<libint2::Engine>(nthreads, e);
}

void set_point(libint2::Engine& e, const double* p) {
    std::vector<std::pair<double, std::array<double, 3>>> charge{{1.0, {p[0], p[1], p[2]}}};
    e.set_params(charge);
}

void check_points(const Array& points, const char* name) {
    if (points.ndim() != 2 || points.shape(1) != 3) throw std::invalid_argument(std::string(name) + " must be (n, 3)");
}

// Index of the second derivative (i, j), i <= j, among 9 coordinates (bra, ket, charge),
// libint2's upper-triangle order.
constexpr int hess(int i, int j) { return i * 9 - i * (i - 1) / 2 + (j - i); }

// Calls visit(i, j, k) for every function pair of the shell pair (M, N): i in M and j in N
// are Psi4 AO indices, k the matching index into libint2's buffer.
template <typename F>
void each_function(const Shells& s, int M, int N, F&& visit) {
    const int oM = s.offset[M], oN = s.offset[N], nN = s.size[N];
    const auto &rM = s.order[M], &rN = s.order[N];
    for (int m = 0; m < s.size[M]; m++)
        for (int n = 0; n < nN; n++) visit(oM + m, oN + n, rM[m] * nN + rN[n]);
}

}  // namespace

// out[i, p] = S_i,mn for m >= n, packed p = m (m + 1)/2 + n.
void field_matrix_packed_into(std::shared_ptr<BasisSet> basis, Array points, Array normals,
                              py::array_t<double, py::array::c_style> out, int nthreads) {
    check_points(points, "points");
    check_points(normals, "normals");
    const auto s = shells_of(*basis);
    const long n = points.shape(0), npack = static_cast<long>(s.nbf) * (s.nbf + 1) / 2;
    if (out.ndim() != 2 || out.shape(0) != n || out.shape(1) != npack)
        throw std::invalid_argument("out must be (npoints, nbf*(nbf+1)/2)");
    const double *P = points.data(), *Nv = normals.data();
    double* O = out.mutable_data();
    py::gil_scoped_release release;
    auto eng = engines(s, 1, nthreads);
    const int nsh = static_cast<int>(s.shells.size());
#pragma omp parallel for schedule(dynamic) num_threads(nthreads)
    for (long i = 0; i < n; i++) {
        auto& e = eng[thread_rank()];
        set_point(e, P + 3 * i);
        const double* nv = Nv + 3 * i;
        double* row = O + i * npack;
        std::fill(row, row + npack, 0.0);
        const auto& buf = e.results();
        for (int M = 0; M < nsh; M++)
            for (int N = 0; N <= M; N++) {
                e.compute(s.shells[M], s.shells[N]);
                if (buf[6] == nullptr) continue;
                each_function(s, M, N, [&](int a, int b, int k) {
                    if (a < b) return;
                    // n . grad_s U = -n . dV/dC
                    row[static_cast<long>(a) * (a + 1) / 2 + b] =
                        -(nv[0] * buf[6][k] + nv[1] * buf[7][k] + nv[2] * buf[8][k]);
                });
            }
    }
}

// F[i] = sum_mn D_mn S_i,mn, without storing S.
py::array_t<double> normal_field(std::shared_ptr<BasisSet> basis, Array points, Array normals, Array D,
                                 int nthreads) {
    check_points(points, "points");
    check_points(normals, "normals");
    const auto s = shells_of(*basis);
    if (D.ndim() != 2 || D.shape(0) != s.nbf || D.shape(1) != s.nbf) throw std::invalid_argument("D must be (nbf, nbf)");
    const long n = points.shape(0);
    py::array_t<double> F(n);
    const double *P = points.data(), *Nv = normals.data(), *Dp = D.data();
    double* Fp = F.mutable_data();
    {
        py::gil_scoped_release release;
        auto eng = engines(s, 1, nthreads);
        const int nsh = static_cast<int>(s.shells.size());
        const int nbf = s.nbf;
#pragma omp parallel for schedule(dynamic) num_threads(nthreads)
        for (long i = 0; i < n; i++) {
            auto& e = eng[thread_rank()];
            set_point(e, P + 3 * i);
            const double* nv = Nv + 3 * i;
            const auto& buf = e.results();
            double acc = 0.0;
            for (int M = 0; M < nsh; M++)
                for (int N = 0; N <= M; N++) {
                    e.compute(s.shells[M], s.shells[N]);
                    if (buf[6] == nullptr) continue;
                    const double fac = (M == N) ? 1.0 : 2.0;
                    each_function(s, M, N, [&](int a, int b, int k) {
                        acc -= fac * Dp[a * nbf + b] * (nv[0] * buf[6][k] + nv[1] * buf[7][k] + nv[2] * buf[8][k]);
                    });
                }
            Fp[i] = acc;
        }
    }
    return F;
}

// V = sum_i c_i S_i (symmetric), without storing S.
py::array_t<double> normal_field_matrix(std::shared_ptr<BasisSet> basis, Array points, Array normals, Array c,
                                        int nthreads) {
    check_points(points, "points");
    check_points(normals, "normals");
    const auto s = shells_of(*basis);
    const long n = points.shape(0);
    if (c.ndim() != 1 || c.shape(0) != n) throw std::invalid_argument("c must be (npoints,)");
    const int nbf = s.nbf;
    py::array_t<double> V({nbf, nbf});
    const double *P = points.data(), *Nv = normals.data(), *cp = c.data();
    double* Vp = V.mutable_data();
    {
        py::gil_scoped_release release;
        auto eng = engines(s, 1, nthreads);
        std::vector<std::vector<double>> Vt(nthreads, std::vector<double>(static_cast<std::size_t>(nbf) * nbf, 0.0));
        const int nsh = static_cast<int>(s.shells.size());
#pragma omp parallel for schedule(dynamic) num_threads(nthreads)
        for (long i = 0; i < n; i++) {
            auto& e = eng[thread_rank()];
            auto& Vl = Vt[thread_rank()];
            set_point(e, P + 3 * i);
            const double* nv = Nv + 3 * i;
            const double ci = cp[i];
            const auto& buf = e.results();
            for (int M = 0; M < nsh; M++)
                for (int N = 0; N <= M; N++) {
                    e.compute(s.shells[M], s.shells[N]);
                    if (buf[6] == nullptr) continue;
                    each_function(s, M, N, [&](int a, int b, int k) {
                        if (a < b) return;
                        Vl[static_cast<std::size_t>(a) * nbf + b] +=
                            -ci * (nv[0] * buf[6][k] + nv[1] * buf[7][k] + nv[2] * buf[8][k]);
                    });
                }
        }
        for (int a = 0; a < nbf; a++)
            for (int b = 0; b <= a; b++) {
                double v = 0.0;
                for (int t = 0; t < nthreads; t++) v += Vt[t][static_cast<std::size_t>(a) * nbf + b];
                Vp[a * nbf + b] = v;
                Vp[b * nbf + a] = v;
            }
    }
    return V;
}

// The nuclear gradient of sum_i q_i . E_el(s_i) at fixed density matrix D:
//   atoms[A]  = sum_i q_i . dE_el(s_i)/dR_A, the basis functions on A moving (points fixed);
//   points[i] = q_i . grad_s E_el(s_i), the point moving.
// E_el,x = sum D_mn dU_mn/ds_x, so both are second derivatives of U = -V.
py::tuple field_gradient(std::shared_ptr<BasisSet> basis, Array points, Array D, Array q, int natom, int nthreads) {
    check_points(points, "points");
    check_points(q, "q");
    const auto s = shells_of(*basis);
    const long n = points.shape(0);
    if (q.shape(0) != n) throw std::invalid_argument("q must be (npoints, 3)");
    if (D.ndim() != 2 || D.shape(0) != s.nbf || D.shape(1) != s.nbf) throw std::invalid_argument("D must be (nbf, nbf)");
    py::array_t<double> atoms({natom, 3}), pts({static_cast<py::ssize_t>(n), static_cast<py::ssize_t>(3)});
    const double *P = points.data(), *Dp = D.data(), *Qp = q.data();
    double *Ap = atoms.mutable_data(), *Gp = pts.mutable_data();
    {
        py::gil_scoped_release release;
        auto eng = engines(s, 2, nthreads);
        std::vector<std::vector<double>> At(nthreads, std::vector<double>(3 * static_cast<std::size_t>(natom), 0.0));
        const int nsh = static_cast<int>(s.shells.size());
        const int nbf = s.nbf;
#pragma omp parallel for schedule(dynamic) num_threads(nthreads)
        for (long i = 0; i < n; i++) {
            auto& e = eng[thread_rank()];
            auto& Al = At[thread_rank()];
            set_point(e, P + 3 * i);
            const double* qi = Qp + 3 * i;
            const auto& buf = e.results();
            double gp[3] = {0.0, 0.0, 0.0};
            for (int M = 0; M < nsh; M++)
                for (int N = 0; N <= M; N++) {
                    e.compute(s.shells[M], s.shells[N]);
                    if (buf[0] == nullptr) continue;
                    const double fac = (M == N) ? 1.0 : 2.0;
                    double ga[3] = {0.0, 0.0, 0.0}, gb[3] = {0.0, 0.0, 0.0};
                    each_function(s, M, N, [&](int a, int b, int k) {
                        const double d = fac * Dp[a * nbf + b];
                        for (int x = 0; x < 3; x++) {
                            const double w = -d * qi[x];  // U = -V
                            const int cx = 6 + x;
                            for (int y = 0; y < 3; y++) {
                                // d2/dC_x dA_y, d2/dC_x dB_y, d2/dC_x dC_y (indices i <= j)
                                ga[y] += w * buf[hess(y, cx)][k];
                                gb[y] += w * buf[hess(3 + y, cx)][k];
                                const int cy = 6 + y;
                                gp[y] += w * buf[cx <= cy ? hess(cx, cy) : hess(cy, cx)][k];
                            }
                        }
                    });
                    for (int y = 0; y < 3; y++) {
                        Al[3 * s.center[M] + y] += ga[y];
                        Al[3 * s.center[N] + y] += gb[y];
                    }
                }
            for (int y = 0; y < 3; y++) Gp[3 * i + y] = gp[y];
        }
        for (int a = 0; a < 3 * natom; a++) {
            double v = 0.0;
            for (int t = 0; t < nthreads; t++) v += At[t][a];
            Ap[a] = v;
        }
    }
    return py::make_tuple(atoms, pts);
}

// ---------------------------------------------------------------------------------------------
// The dispersion kernel

namespace {

constexpr double T_SERIES = 1e-2;

// k * 3 delta^6 and grad_R k * 3 delta^6 for R = s - r and normal n.
inline void kernel_pair(double rx, double ry, double rz, const double* nv, double d3, double& k, double* g) {
    const double r2 = rx * rx + ry * ry + rz * rz;
    const double t = r2 * std::sqrt(r2) / d3;
    const double t2 = t * t;
    double atan_t, g_over_r2;
    if (t < T_SERIES) {
        atan_t = 1.0 - t2 / 3.0 + t2 * t2 / 5.0 - t2 * t2 * t2 / 7.0;
        g_over_r2 = 3.0 * (r2 * r2 / (d3 * d3)) * (-2.0 / 3.0 + 4.0 * t2 / 5.0 - 6.0 * t2 * t2 / 7.0);
    } else {
        atan_t = std::atan(t) / t;
        g_over_r2 = 3.0 * (1.0 / (1.0 + t2) - atan_t) / r2;
    }
    const double rn = rx * nv[0] + ry * nv[1] + rz * nv[2];
    k = rn * atan_t;
    const double sc = rn * g_over_r2;
    g[0] = nv[0] * atan_t + sc * rx;
    g[1] = nv[1] * atan_t + sc * ry;
    g[2] = nv[2] * atan_t + sc * rz;
}

}  // namespace

// J at each point, and if weights is given the derivatives of E = sum_p w_p J_p:
// (J, dJ/dr (n,3), dE/ds (ns,3), dE/dA (ns,)).
py::tuple dispersion_kernel(Array points, Array coords, Array normals, Array areas, double delta, py::object weights,
                            int nthreads) {
    check_points(points, "points");
    check_points(coords, "coords");
    check_points(normals, "normals");
    const long np_ = points.shape(0), ns = coords.shape(0);
    const bool deriv = !weights.is_none();
    Array w = deriv ? weights.cast<Array>() : Array(0);
    if (deriv && w.shape(0) != np_) throw std::invalid_argument("weights must be (npoints,)");
    py::array_t<double> J(np_), dJ({np_, static_cast<long>(3)}), dEs({ns, static_cast<long>(3)}), dEA(ns);
    const double *P = points.data(), *C = coords.data(), *Nv = normals.data(), *Ar = areas.data();
    const double* W = deriv ? w.data() : nullptr;
    double *Jp = J.mutable_data(), *dJp = dJ.mutable_data(), *dEsp = dEs.mutable_data(), *dEAp = dEA.mutable_data();
    {
        py::gil_scoped_release release;
        const double d3 = delta * delta * delta, scale = 1.0 / (3.0 * d3 * d3);
        std::vector<std::vector<double>> acc(deriv ? nthreads : 0, std::vector<double>(4 * ns, 0.0));
#pragma omp parallel for schedule(static) num_threads(nthreads)
        for (long p = 0; p < np_; p++) {
            const double px = P[3 * p], py_ = P[3 * p + 1], pz = P[3 * p + 2];
            double j = 0.0, gx = 0.0, gy = 0.0, gz = 0.0, k, g[3];
            const double cp = deriv ? W[p] : 0.0;
            double* a = deriv ? acc[thread_rank()].data() : nullptr;
            for (long i = 0; i < ns; i++) {
                kernel_pair(C[3 * i] - px, C[3 * i + 1] - py_, C[3 * i + 2] - pz, Nv + 3 * i, d3, k, g);
                j += Ar[i] * k;
                if (deriv) {
                    gx += Ar[i] * g[0];
                    gy += Ar[i] * g[1];
                    gz += Ar[i] * g[2];
                    a[4 * i] += cp * Ar[i] * g[0];
                    a[4 * i + 1] += cp * Ar[i] * g[1];
                    a[4 * i + 2] += cp * Ar[i] * g[2];
                    a[4 * i + 3] += cp * k;
                }
            }
            Jp[p] = j * scale;
            dJp[3 * p] = -gx * scale;
            dJp[3 * p + 1] = -gy * scale;
            dJp[3 * p + 2] = -gz * scale;
        }
        for (long i = 0; i < ns; i++) {
            double v[4] = {0, 0, 0, 0};
            for (auto& t : acc)
                for (int c = 0; c < 4; c++) v[c] += t[4 * i + c];
            for (int c = 0; c < 3; c++) dEsp[3 * i + c] = v[c] * scale;
            dEAp[i] = v[3] * scale;
        }
    }
    return py::make_tuple(J, dJ, dEs, dEA);
}

// ---------------------------------------------------------------------------------------------
// The cavity grids' partition of unity

namespace {

inline void becke_s(double mu, double& s, double& dls) {
    const double p1 = 1.5 * mu - 0.5 * mu * mu * mu;
    const double p2 = 1.5 * p1 - 0.5 * p1 * p1 * p1;
    const double p3 = 1.5 * p2 - 0.5 * p2 * p2 * p2;
    s = 0.5 * (1.0 - p3);
    if (s <= 0.0) {
        s = 0.0;
        dls = 0.0;
        return;
    }
    const double dp3 = 1.5 * (1.0 - p2 * p2) * 1.5 * (1.0 - p1 * p1) * 1.5 * (1.0 - mu * mu);
    dls = -0.5 * dp3 / s;
}

inline void log_step(double x, double& ls, double& dls) {
    if (x >= 1.0) {
        ls = 0.0;
        dls = 0.0;
    } else if (x <= 0.0) {
        ls = -std::numeric_limits<double>::infinity();
        dls = 0.0;
    } else {
        const double poly = 10.0 - 15.0 * x + 6.0 * x * x;
        ls = std::log(x * x * x * poly);
        dls = 30.0 * (1.0 - x) * (1.0 - x) / (x * poly);
    }
}

struct Partition {
    const double *C, *R, *Dcc;
    int m;
    bool inside;
    double width;
    // per point workspace
    std::vector<double> u, rhat, L, Pw;
    void logs(const double* r) {
        const double NEG_INF = -std::numeric_limits<double>::infinity();
        for (int c = 0; c < m; c++) {
            const double dx = r[0] - C[3 * c], dy = r[1] - C[3 * c + 1], dz = r[2] - C[3 * c + 2];
            const double d = std::sqrt(dx * dx + dy * dy + dz * dz);
            u[c] = d;
            rhat[3 * c] = d > 0 ? dx / d : 0.0;
            rhat[3 * c + 1] = d > 0 ? dy / d : 0.0;
            rhat[3 * c + 2] = d > 0 ? dz / d : 0.0;
        }
        for (int b = 0; b < m; b++) {
            double Lb = 0.0, s, dls;
            for (int c = 0; c < m; c++) {
                if (c == b) continue;
                becke_s((u[b] - u[c]) / Dcc[b * m + c], s, dls);
                Lb += s > 0.0 ? std::log(s) : NEG_INF;
            }
            L[b] = Lb;
        }
        double ls, dls;
        if (inside) {
            for (int b = 0; b < m; b++) {
                log_step((R[b] - u[b]) / width, ls, dls);
                L[b] += ls;
            }
        } else {
            double total = 0.0;
            int n_inf = 0, which = -1;
            for (int c = 0; c < m; c++) {
                log_step((u[c] - R[c]) / width, ls, dls);
                if (std::isinf(ls)) {
                    n_inf++;
                    which = c;
                } else {
                    total += ls;
                }
            }
            for (int b = 0; b < m; b++) {
                if (n_inf == 0) {
                    log_step((u[b] - R[b]) / width, ls, dls);
                    L[b] += total - ls;
                } else if (n_inf == 1 && b == which) {
                    L[b] += total;
                } else {
                    L[b] = NEG_INF;
                }
            }
        }
        double top = NEG_INF;
        for (int b = 0; b < m; b++) top = std::max(top, L[b]);
        if (std::isinf(top)) {
            for (int b = 0; b < m; b++) Pw[b] = 0.0;
            return;
        }
        double tot = 0.0;
        for (int b = 0; b < m; b++) {
            Pw[b] = std::exp(L[b] - top);
            tot += Pw[b];
        }
        for (int b = 0; b < m; b++) Pw[b] /= tot;
    }
};

Partition make_partition(const Array& centres, const Array& radii, const Array& dist, bool inside, double width) {
    Partition p{centres.data(), radii.data(), dist.data(), static_cast<int>(centres.shape(0)), inside, width, {}, {}, {}, {}};
    p.u.resize(p.m);
    p.rhat.resize(3 * p.m);
    p.L.resize(p.m);
    p.Pw.resize(p.m);
    return p;
}

}  // namespace

py::array_t<double> cavity_partition(Array points, IntArray owners, Array centres, Array radii, Array dist,
                                     bool inside, double width, int nthreads) {
    check_points(points, "points");
    const long n = points.shape(0);
    py::array_t<double> out(n);
    const double* Pt = points.data();
    const std::int64_t* own = owners.data();
    double* O = out.mutable_data();
    {
        py::gil_scoped_release release;
#pragma omp parallel num_threads(nthreads)
        {
            auto part = make_partition(centres, radii, dist, inside, width);
#pragma omp for schedule(static)
            for (long p = 0; p < n; p++) {
                part.logs(Pt + 3 * p);
                O[p] = part.Pw[own[p]];
            }
        }
    }
    return out;
}

// sum_p values_p d P_owner(p)(r_p) / dR, the points moving with their owners: (natm, 3).
py::array_t<double> cavity_partition_gradient(Array points, IntArray owners, Array centres, Array radii, Array dist,
                                              bool inside, double width, Array values, int nthreads) {
    check_points(points, "points");
    const long n = points.shape(0);
    const int m = static_cast<int>(centres.shape(0));
    py::array_t<double> out({m, 3});
    const double *Pt = points.data(), *Vv = values.data(), *C = centres.data(), *R = radii.data(), *Dcc = dist.data();
    const std::int64_t* own = owners.data();
    double* O = out.mutable_data();
    {
        py::gil_scoped_release release;
        std::vector<std::vector<double>> acc(nthreads, std::vector<double>(3 * m, 0.0));
#pragma omp parallel num_threads(nthreads)
        {
            auto part = make_partition(centres, radii, dist, inside, width);
            auto& g = acc[thread_rank()];
#pragma omp for schedule(static)
            for (long p = 0; p < n; p++) {
                const int a = static_cast<int>(own[p]);
                part.logs(Pt + 3 * p);
                const auto& P = part.Pw;
                const auto& u = part.u;
                const auto& rh = part.rhat;
                if (P[a] == 0.0 || Vv[p] == 0.0) continue;
                for (int b = 0; b < m; b++) {
                    const double omega = Vv[p] * P[a] * ((b == a ? 1.0 : 0.0) - P[b]);
                    if (omega == 0.0) continue;
                    for (int c = 0; c < m; c++) {
                        if (c == b) continue;
                        const double D = Dcc[b * m + c];
                        const double mu = (u[b] - u[c]) / D;
                        double s, dls;
                        becke_s(mu, s, dls);
                        if (s == 0.0) continue;
                        const double f = omega * dls / D;
                        for (int x = 0; x < 3; x++) {
                            const double e = (C[3 * b + x] - C[3 * c + x]) / D;
                            g[3 * a + x] += f * (rh[3 * b + x] - rh[3 * c + x]);
                            g[3 * b + x] += f * (-rh[3 * b + x] - mu * e);
                            g[3 * c + x] += f * (rh[3 * c + x] + mu * e);
                        }
                    }
                    double ls, dls;
                    if (inside) {
                        log_step((R[b] - u[b]) / width, ls, dls);
                        const double f = omega * dls / width;
                        for (int x = 0; x < 3; x++) {
                            g[3 * a + x] -= f * rh[3 * b + x];
                            g[3 * b + x] += f * rh[3 * b + x];
                        }
                    } else {
                        for (int c = 0; c < m; c++) {
                            if (c == b) continue;
                            log_step((u[c] - R[c]) / width, ls, dls);
                            const double f = omega * dls / width;
                            for (int x = 0; x < 3; x++) {
                                g[3 * a + x] += f * rh[3 * c + x];
                                g[3 * c + x] -= f * rh[3 * c + x];
                            }
                        }
                    }
                }
            }
        }
        for (int k = 0; k < 3 * m; k++) {
            double v = 0.0;
            for (auto& t : acc) v += t[k];
            O[k] = v;
        }
    }
    return out;
}

}  // namespace nonelectrostatic
}  // namespace psi

PYBIND11_MODULE(_nonelectrostatic, m) {
    py::module_::import("psi4.core");
    namespace ne = psi::nonelectrostatic;
    m.def("field_matrix_packed_into", &ne::field_matrix_packed_into, py::arg("basis"), py::arg("points"),
          py::arg("normals"), py::arg("out").noconvert(), py::arg("nthreads"));
    m.def("normal_field", &ne::normal_field, py::arg("basis"), py::arg("points"), py::arg("normals"), py::arg("D"),
          py::arg("nthreads"));
    m.def("normal_field_matrix", &ne::normal_field_matrix, py::arg("basis"), py::arg("points"), py::arg("normals"),
          py::arg("c"), py::arg("nthreads"));
    m.def("field_gradient", &ne::field_gradient, py::arg("basis"), py::arg("points"), py::arg("D"), py::arg("q"),
          py::arg("natom"), py::arg("nthreads"));
    m.def("dispersion_kernel", &ne::dispersion_kernel, py::arg("points"), py::arg("coords"), py::arg("normals"),
          py::arg("areas"), py::arg("delta"), py::arg("weights") = py::none(), py::arg("nthreads") = 1);
    m.def("cavity_partition", &ne::cavity_partition, py::arg("points"), py::arg("owners"), py::arg("centres"),
          py::arg("radii"), py::arg("dist"), py::arg("inside"), py::arg("width"), py::arg("nthreads"));
    m.def("cavity_partition_gradient", &ne::cavity_partition_gradient, py::arg("points"), py::arg("owners"),
          py::arg("centres"), py::arg("radii"), py::arg("dist"), py::arg("inside"), py::arg("width"),
          py::arg("values"), py::arg("nthreads"));
}
