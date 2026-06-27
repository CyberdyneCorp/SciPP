// Multivariate minimization: Nelder-Mead simplex and BFGS quasi-Newton.
#include "scypp/optimize/optimize.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <vector>

#include "scypp/error.hpp"
#include "scypp/optimize/detail.hpp"

namespace scypp::optimize {
namespace {

using detail::tond;
using detail::tov;

OptimizeResult nelder_mead(const ObjFn& f, std::vector<double> x0, double tol, int maxiter) {
  const double rho = 1.0, chi = 2.0, psi = 0.5, sigma = 0.5;
  const double xatol = (tol > 0 ? tol : 1e-4), fatol = (tol > 0 ? tol : 1e-4);
  int n = static_cast<int>(x0.size());
  int nfev = 0;
  if (maxiter <= 0) maxiter = n * 200;

  std::vector<std::vector<double>> sim(n + 1, x0);
  for (int k = 0; k < n; ++k) {
    if (sim[k + 1][k] != 0.0) sim[k + 1][k] *= 1.05;
    else sim[k + 1][k] = 0.00025;
  }
  std::vector<double> fsim(n + 1);
  for (int i = 0; i <= n; ++i) { fsim[i] = f(tond(sim[i])); ++nfev; }

  auto order = [&]() {
    std::vector<int> idx(n + 1);
    std::iota(idx.begin(), idx.end(), 0);
    std::sort(idx.begin(), idx.end(), [&](int a, int b) { return fsim[a] < fsim[b]; });
    std::vector<std::vector<double>> s2(n + 1);
    std::vector<double> f2(n + 1);
    for (int i = 0; i <= n; ++i) { s2[i] = sim[idx[i]]; f2[i] = fsim[idx[i]]; }
    sim = std::move(s2); fsim = std::move(f2);
  };
  order();

  auto axpy = [](double a, const std::vector<double>& x, double b, const std::vector<double>& y) {
    std::vector<double> r(x.size());
    for (size_t i = 0; i < x.size(); ++i) r[i] = a * x[i] + b * y[i];
    return r;
  };

  int it = 1;
  for (; it < maxiter; ++it) {
    double xspread = 0.0, fspread = 0.0;
    for (int i = 1; i <= n; ++i) {
      for (int j = 0; j < n; ++j) xspread = std::max(xspread, std::fabs(sim[i][j] - sim[0][j]));
      fspread = std::max(fspread, std::fabs(fsim[i] - fsim[0]));
    }
    if (xspread <= xatol && fspread <= fatol) break;

    std::vector<double> xbar(n, 0.0);
    for (int i = 0; i < n; ++i)
      for (int j = 0; j < n; ++j) xbar[j] += sim[i][j] / n;

    std::vector<double> xr = axpy(1.0 + rho, xbar, -rho, sim[n]);
    double fxr = f(tond(xr)); ++nfev;
    bool shrink = false;
    if (fxr < fsim[0]) {
      std::vector<double> xe = axpy(1.0 + rho * chi, xbar, -rho * chi, sim[n]);
      double fxe = f(tond(xe)); ++nfev;
      if (fxe < fxr) { sim[n] = xe; fsim[n] = fxe; } else { sim[n] = xr; fsim[n] = fxr; }
    } else if (fxr < fsim[n - 1]) {
      sim[n] = xr; fsim[n] = fxr;
    } else if (fxr < fsim[n]) {
      std::vector<double> xc = axpy(1.0 + psi * rho, xbar, -psi * rho, sim[n]);
      double fxc = f(tond(xc)); ++nfev;
      if (fxc <= fxr) { sim[n] = xc; fsim[n] = fxc; } else shrink = true;
    } else {
      std::vector<double> xcc = axpy(1.0 - psi, xbar, psi, sim[n]);
      double fxcc = f(tond(xcc)); ++nfev;
      if (fxcc < fsim[n]) { sim[n] = xcc; fsim[n] = fxcc; } else shrink = true;
    }
    if (shrink) {
      for (int i = 1; i <= n; ++i) {
        sim[i] = axpy(1.0, sim[0], sigma, axpy(1.0, sim[i], -1.0, sim[0]));
        fsim[i] = f(tond(sim[i])); ++nfev;
      }
    }
    order();
  }
  OptimizeResult r;
  r.x = tond(sim[0]); r.fun = fsim[0]; r.success = it < maxiter; r.nit = it; r.nfev = nfev;
  r.message = r.success ? "Optimization terminated successfully." : "Maximum iterations reached.";
  return r;
}

OptimizeResult bfgs(const ObjFn& f, std::vector<double> x, double /*tol*/, int maxiter) {
  const double gtol = 1e-5, c1 = 1e-4;
  int n = static_cast<int>(x.size());
  int nfev = 0;
  std::vector<double> H(n * n, 0.0);
  for (int i = 0; i < n; ++i) H[i * n + i] = 1.0;
  std::vector<double> g = detail::num_gradient(f, x, nfev);
  double fx = f(tond(x)); ++nfev;

  auto dot = [&](const std::vector<double>& a, const std::vector<double>& b) {
    double s = 0; for (int i = 0; i < n; ++i) s += a[i] * b[i]; return s;
  };
  auto inf_norm = [&](const std::vector<double>& a) {
    double m = 0; for (double v : a) m = std::max(m, std::fabs(v)); return m;
  };

  int it = 0;
  for (; it < maxiter; ++it) {
    if (inf_norm(g) <= gtol) break;
    std::vector<double> p(n, 0.0);  // p = -H g
    for (int i = 0; i < n; ++i) { double s = 0; for (int j = 0; j < n; ++j) s += H[i * n + j] * g[j]; p[i] = -s; }
    double gp = dot(g, p);
    if (gp >= 0.0) {  // not a descent direction: reset to steepest descent
      for (int i = 0; i < n; ++i) p[i] = -g[i];
      gp = dot(g, p);
    }
    // Backtracking Armijo line search.
    double alpha = 1.0;
    std::vector<double> xn(n);
    double fxn;
    for (int ls = 0; ls < 60; ++ls) {
      for (int i = 0; i < n; ++i) xn[i] = x[i] + alpha * p[i];
      fxn = f(tond(xn)); ++nfev;
      if (fxn <= fx + c1 * alpha * gp) break;
      alpha *= 0.5;
    }
    std::vector<double> s(n), gn = detail::num_gradient(f, xn, nfev), y(n);
    for (int i = 0; i < n; ++i) { s[i] = xn[i] - x[i]; y[i] = gn[i] - g[i]; }
    double sy = dot(s, y);
    x = xn; fx = fxn; g = gn;
    if (sy > 1e-10) {  // BFGS inverse-Hessian update
      double rho = 1.0 / sy;
      std::vector<double> Hy(n, 0.0);
      for (int i = 0; i < n; ++i) { double t = 0; for (int j = 0; j < n; ++j) t += H[i * n + j] * y[j]; Hy[i] = t; }
      double yHy = dot(y, Hy);
      for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
          H[i * n + j] += (rho * rho * yHy + rho) * s[i] * s[j] - rho * (Hy[i] * s[j] + s[i] * Hy[j]);
    }
  }
  OptimizeResult r;
  r.x = tond(x); r.fun = fx; r.success = inf_norm(g) <= gtol; r.nit = it; r.nfev = nfev;
  r.message = r.success ? "Optimization terminated successfully." : "Maximum iterations reached.";
  return r;
}

}  // namespace

OptimizeResult minimize(const ObjFn& f, const ndarray& x0, const std::string& method, double tol,
                        int maxiter) {
  std::vector<double> x = tov(x0);
  if (method == "Nelder-Mead") return nelder_mead(f, x, tol, maxiter);
  if (method == "BFGS") return bfgs(f, x, tol, maxiter);
  throw scypp::value_error("minimize: unknown method " + method);
}

}  // namespace scypp::optimize
