// Nonlinear least squares (Levenberg-Marquardt), curve_fit, and fsolve.
#include "scypp/optimize/optimize.hpp"

#include <cmath>
#include <vector>

#include "numpp/linalg/linalg.hpp"
#include "scypp/error.hpp"
#include "scypp/linalg/detail.hpp"
#include "scypp/optimize/detail.hpp"

namespace scypp::optimize {
namespace {

using detail::tond;
using detail::tov;
namespace sd = scypp::linalg::detail;

double half_sq(const std::vector<double>& r) {
  double s = 0; for (double v : r) s += v * v; return 0.5 * s;
}
double inf_norm(const std::vector<double>& a) {
  double m = 0; for (double v : a) m = std::max(m, std::fabs(v)); return m;
}

// Solve the n×n system A x = b through NumPP.
std::vector<double> solve_system(const std::vector<double>& A, const std::vector<double>& b, int n) {
  ndarray x = numpp::linalg::solve(sd::from_mat(A, n, n), sd::from_vec(b));
  return tov(x);
}

}  // namespace

LeastSquaresResult least_squares(const VecFn& residual, const ndarray& x0, double ftol,
                                 double xtol, int max_nfev) {
  std::vector<double> x = tov(x0);
  int n = static_cast<int>(x.size());
  int nfev = 0;
  std::vector<double> r = tov(residual(tond(x))); ++nfev;
  int m = static_cast<int>(r.size());
  double cost = half_sq(r);
  double lambda = 1e-3;
  bool success = false;

  for (int it = 0; it < max_nfev && nfev < max_nfev; ++it) {
    std::vector<double> J = detail::num_jacobian(residual, x, r, nfev);  // m×n
    std::vector<double> JtJ(n * n, 0.0), Jtr(n, 0.0);
    for (int i = 0; i < n; ++i) {
      for (int k = 0; k < m; ++k) Jtr[i] += J[k * n + i] * r[k];
      for (int j = 0; j < n; ++j) {
        double s = 0; for (int k = 0; k < m; ++k) s += J[k * n + i] * J[k * n + j];
        JtJ[i * n + j] = s;
      }
    }
    if (inf_norm(Jtr) < 1e-12) { success = true; break; }

    bool accepted = false;
    for (int inner = 0; inner < 30; ++inner) {
      std::vector<double> A = JtJ;
      for (int i = 0; i < n; ++i) A[i * n + i] += lambda * JtJ[i * n + i];  // Marquardt scaling
      std::vector<double> rhs(n);
      for (int i = 0; i < n; ++i) rhs[i] = -Jtr[i];
      std::vector<double> delta;
      try { delta = solve_system(A, rhs, n); }
      catch (const numpp::linalg_error&) { lambda *= 10.0; continue; }
      std::vector<double> xn(n);
      for (int i = 0; i < n; ++i) xn[i] = x[i] + delta[i];
      std::vector<double> rn = tov(residual(tond(xn))); ++nfev;
      double costn = half_sq(rn);
      if (costn < cost) {
        double dx = inf_norm(delta), dcost = cost - costn;
        x = xn; r = rn;
        bool conv = dcost < ftol * cost || dx < xtol * (xtol + inf_norm(x));
        cost = costn;
        lambda = std::max(lambda * 0.5, 1e-12);
        accepted = true;
        if (conv) success = true;
        break;
      }
      lambda *= 10.0;
      if (lambda > 1e12) break;
    }
    if (!accepted) { success = true; break; }  // could not improve -> stationary
    if (success) break;
  }
  LeastSquaresResult res;
  res.x = tond(x); res.fun = tond(r); res.cost = cost; res.success = success; res.nfev = nfev;
  return res;
}

CurveFitResult curve_fit(const ModelFn& model, const ndarray& xdata, const ndarray& ydata,
                         const ndarray& p0, double ftol, double xtol) {
  VecFn residual = [&](const ndarray& p) -> ndarray {
    std::vector<double> y = tov(model(xdata, p));
    std::vector<double> yd = tov(ydata);
    for (size_t i = 0; i < y.size(); ++i) y[i] -= yd[i];
    return tond(y);
  };
  LeastSquaresResult ls = least_squares(residual, p0, ftol, xtol, 2000);
  std::vector<double> popt = tov(ls.x);
  int n = static_cast<int>(popt.size());

  // pcov = sigma^2 (JtJ)^-1, sigma^2 = SSR / (m - n).
  int nfev = 0;
  std::vector<double> r = tov(residual(ls.x));
  int m = static_cast<int>(r.size());
  std::vector<double> J = detail::num_jacobian(residual, popt, r, nfev);
  std::vector<double> JtJ(n * n, 0.0);
  for (int i = 0; i < n; ++i)
    for (int j = 0; j < n; ++j) {
      double s = 0; for (int k = 0; k < m; ++k) s += J[k * n + i] * J[k * n + j];
      JtJ[i * n + j] = s;
    }
  double ssr = 0; for (double v : r) ssr += v * v;
  double sigma2 = (m > n) ? ssr / (m - n) : 1.0;
  ndarray cov = numpp::linalg::inv(sd::from_mat(JtJ, n, n));
  std::vector<double> covv = tov(cov);
  for (double& v : covv) v *= sigma2;

  CurveFitResult out;
  out.popt = ls.x;
  out.pcov = sd::from_mat(covv, n, n);
  return out;
}

ndarray fsolve(const VecFn& F, const ndarray& x0, double xtol, int maxiter) {
  std::vector<double> x = tov(x0);
  int n = static_cast<int>(x.size());
  int nfev = 0;
  for (int it = 0; it < maxiter; ++it) {
    std::vector<double> f0 = tov(F(tond(x))); ++nfev;
    if (inf_norm(f0) < xtol) break;
    std::vector<double> J = detail::num_jacobian(F, x, f0, nfev);  // n×n (assume square)
    std::vector<double> rhs(n);
    for (int i = 0; i < n; ++i) rhs[i] = -f0[i];
    std::vector<double> delta;
    try { delta = solve_system(J, rhs, n); }
    catch (const numpp::linalg_error&) { break; }
    double step = 0;
    for (int i = 0; i < n; ++i) { x[i] += delta[i]; step = std::max(step, std::fabs(delta[i])); }
    if (step < xtol) break;
  }
  return tond(x);
}

}  // namespace scypp::optimize
