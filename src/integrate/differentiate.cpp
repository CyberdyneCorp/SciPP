// Finite-difference differentiation: derivative (Richardson extrapolation),
// jacobian and hessian (central differences).
#include "scypp/differentiate/differentiate.hpp"

#include <cmath>
#include <vector>

#include "numpp/core/dtype.hpp"
#include "scypp/linalg/detail.hpp"

namespace scypp::differentiate {
namespace {
namespace sd = scypp::linalg::detail;
}

DerivativeResult derivative(const ScalarFn& f, double x, double initial_step) {
  const int levels = 9;
  std::vector<std::vector<double>> T(levels, std::vector<double>(levels, 0.0));
  double h = initial_step;
  auto central = [&](double hh) { return (f(x + hh) - f(x - hh)) / (2.0 * hh); };
  T[0][0] = central(h);
  double best = T[0][0], err = std::fabs(best);
  for (int i = 1; i < levels; ++i) {
    h *= 0.5;
    T[i][0] = central(h);
    double pow4 = 4.0;
    for (int j = 1; j <= i; ++j) {
      T[i][j] = T[i][j - 1] + (T[i][j - 1] - T[i - 1][j - 1]) / (pow4 - 1.0);
      pow4 *= 4.0;
    }
    double e = std::fabs(T[i][i] - T[i - 1][i - 1]);
    if (e <= err) { err = e; best = T[i][i]; }
  }
  return {best, err, true};
}

ndarray jacobian(const VecFn& F, const ndarray& x) {
  std::vector<double> xv = sd::to_vec(x);
  int n = static_cast<int>(xv.size());
  std::vector<double> f0 = sd::to_vec(F(sd::from_vec(xv)));
  int m = static_cast<int>(f0.size());
  std::vector<double> J(static_cast<size_t>(m) * n, 0.0);
  for (int j = 0; j < n; ++j) {
    double h = 1e-6 * std::max(std::fabs(xv[j]), 1.0);
    std::vector<double> xp = xv, xm = xv;
    xp[j] += h; xm[j] -= h;
    std::vector<double> fp = sd::to_vec(F(sd::from_vec(xp)));
    std::vector<double> fm = sd::to_vec(F(sd::from_vec(xm)));
    for (int i = 0; i < m; ++i) J[i * n + j] = (fp[i] - fm[i]) / (2.0 * h);
  }
  return sd::from_mat(J, m, n);
}

ndarray hessian(const ObjFn& f, const ndarray& x) {
  std::vector<double> xv = sd::to_vec(x);
  int n = static_cast<int>(xv.size());
  const double h = 1e-4;
  auto ev = [&](const std::vector<double>& v) { return f(sd::from_vec(v)); };
  double f0 = ev(xv);
  std::vector<double> H(static_cast<size_t>(n) * n, 0.0);
  for (int i = 0; i < n; ++i) {
    std::vector<double> xpp = xv, xmm = xv;
    xpp[i] += h; xmm[i] -= h;
    H[i * n + i] = (ev(xpp) - 2.0 * f0 + ev(xmm)) / (h * h);
    for (int j = i + 1; j < n; ++j) {
      std::vector<double> a = xv, b = xv, c = xv, d = xv;
      a[i] += h; a[j] += h;
      b[i] += h; b[j] -= h;
      c[i] -= h; c[j] += h;
      d[i] -= h; d[j] -= h;
      double v = (ev(a) - ev(b) - ev(c) + ev(d)) / (4.0 * h * h);
      H[i * n + j] = v;
      H[j * n + i] = v;
    }
  }
  return sd::from_mat(H, n, n);
}

}  // namespace scypp::differentiate
