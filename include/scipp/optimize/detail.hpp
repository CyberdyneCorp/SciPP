#pragma once
// Internal helpers for scipp::optimize: ndarray<->vector conversion and
// forward-difference gradient / Jacobian (SciPy's default numerical-derivative
// step), shared by the minimizers and least-squares solvers.

#include <cmath>
#include <vector>

#include "scipp/linalg/detail.hpp"
#include "scipp/optimize/optimize.hpp"

namespace scipp::optimize::detail {

inline std::vector<double> tov(const ndarray& a) { return scipp::linalg::detail::to_vec(a); }
inline ndarray tond(const std::vector<double>& v) { return scipp::linalg::detail::from_vec(v); }

constexpr double kSqrtEps = 1.4901161193847656e-08;  // sqrt(2^-52)

inline double fd_step(double xi) { return kSqrtEps * std::max(std::fabs(xi), 1.0); }

// Forward-difference gradient of an Rⁿ→R objective at x.
inline std::vector<double> num_gradient(const ObjFn& f, const std::vector<double>& x, int& nfev) {
  int n = static_cast<int>(x.size());
  std::vector<double> g(n);
  double f0 = f(tond(x));
  ++nfev;
  for (int i = 0; i < n; ++i) {
    double h = fd_step(x[i]);
    std::vector<double> xph = x;
    xph[i] += h;
    g[i] = (f(tond(xph)) - f0) / h;
    ++nfev;
  }
  return g;
}

// Forward-difference Jacobian (m×n, row-major) of an Rⁿ→Rᵐ map at x.
inline std::vector<double> num_jacobian(const VecFn& F, const std::vector<double>& x,
                                        const std::vector<double>& f0, int& nfev) {
  int n = static_cast<int>(x.size());
  int m = static_cast<int>(f0.size());
  std::vector<double> J(static_cast<size_t>(m) * n, 0.0);
  for (int j = 0; j < n; ++j) {
    double h = fd_step(x[j]);
    std::vector<double> xph = x;
    xph[j] += h;
    std::vector<double> fj = tov(F(tond(xph)));
    ++nfev;
    for (int i = 0; i < m; ++i) J[i * n + j] = (fj[i] - f0[i]) / h;
  }
  return J;
}

}  // namespace scipp::optimize::detail
