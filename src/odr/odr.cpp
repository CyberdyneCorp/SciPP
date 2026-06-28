// Orthogonal distance regression (scipy.odr port).
//
// The fit is cast as an augmented nonlinear least-squares problem over the
// parameter vector [beta, delta], where delta holds the per-point x-offsets:
//   r_eps_i = (f(beta, x_i + delta_i) - y_i) / sy_i      (n entries)
//   r_del_i =  delta_i / sx_i                            (n entries)
// Minimizing 1/2 sum r^2 with Levenberg-Marquardt (scipp::optimize) yields the
// orthogonal-distance solution. The parameter covariance is the top-left p-block
// of (J^T J)^-1 evaluated at the solution; res_var scales it to the standard
// errors, matching ODRPACK.
#include "scipp/odr/odr.hpp"

#include <cmath>
#include <vector>

#include "numpp/linalg/linalg.hpp"
#include "scipp/error.hpp"
#include "scipp/linalg/detail.hpp"
#include "scipp/optimize/detail.hpp"
#include "scipp/optimize/optimize.hpp"

namespace scipp::odr {
namespace {

namespace sd = scipp::linalg::detail;
using sd::from_mat;
using sd::from_vec;
using sd::to_vec;

// Expand an optional standard-deviation array into per-point inverse weights
// (1/s_i). A scalar (size 1) is broadcast; absence yields all-ones.
std::vector<double> inv_weights(const std::optional<ndarray>& s, int n) {
  std::vector<double> w(n, 1.0);
  if (!s) return w;
  std::vector<double> sv = to_vec(*s);
  if (static_cast<int>(sv.size()) == 1) {
    for (int i = 0; i < n; ++i) w[i] = 1.0 / sv[0];
  } else {
    if (static_cast<int>(sv.size()) != n)
      throw scipp::value_error("odr: sx/sy length must match data");
    for (int i = 0; i < n; ++i) w[i] = 1.0 / sv[i];
  }
  return w;
}

}  // namespace

Output ODR::run() {
  namespace opt = scipp::optimize;

  std::vector<double> xv = to_vec(data_.x);
  std::vector<double> yv = to_vec(data_.y);
  std::vector<double> b0 = to_vec(beta0_);
  const int n = static_cast<int>(xv.size());
  const int p = static_cast<int>(b0.size());
  if (static_cast<int>(yv.size()) != n)
    throw scipp::value_error("odr: x and y must have the same length");

  const std::vector<double> swe = inv_weights(data_.sy, n);  // 1/sy
  const std::vector<double> swd = inv_weights(data_.sx, n);  // 1/sx

  // Augmented residual over par = [beta(p), delta(n)] -> 2n residuals.
  opt::VecFn resid = [&](const ndarray& par) -> ndarray {
    std::vector<double> pv = to_vec(par);
    std::vector<double> beta(pv.begin(), pv.begin() + p);
    std::vector<double> xpd(n);
    for (int i = 0; i < n; ++i) xpd[i] = xv[i] + pv[p + i];
    std::vector<double> ymod = to_vec(model_.fcn(from_vec(beta), from_vec(xpd)));
    std::vector<double> r(2 * n);
    for (int i = 0; i < n; ++i) {
      r[i] = swe[i] * (ymod[i] - yv[i]);
      r[n + i] = swd[i] * pv[p + i];
    }
    return from_vec(r);
  };

  std::vector<double> par0(p + n, 0.0);
  for (int i = 0; i < p; ++i) par0[i] = b0[i];
  const int max_nfev = maxit_ * (p + n + 4) + 200;
  opt::LeastSquaresResult ls = opt::least_squares(resid, from_vec(par0), ftol_, xtol_, max_nfev);

  std::vector<double> pv = to_vec(ls.x);
  std::vector<double> beta(pv.begin(), pv.begin() + p);
  std::vector<double> delta(pv.begin() + p, pv.end());

  std::vector<double> r = to_vec(resid(ls.x));
  double sum_square = 0.0;
  for (double v : r) sum_square += v * v;
  double res_var = (n > p) ? sum_square / (n - p) : 1.0;

  // cov_beta = top-left p-block of (J^T J)^-1 at the solution.
  int nfev = 0;
  std::vector<double> J = opt::detail::num_jacobian(resid, pv, r, nfev);  // 2n x (p+n)
  const int np = p + n;
  std::vector<double> M(static_cast<size_t>(np) * np, 0.0);
  for (int i = 0; i < np; ++i)
    for (int j = 0; j < np; ++j) {
      double s = 0;
      for (int k = 0; k < 2 * n; ++k) s += J[k * np + i] * J[k * np + j];
      M[i * np + j] = s;
    }
  std::vector<double> Minv = to_vec(numpp::linalg::inv(from_mat(M, np, np)));
  std::vector<double> cov(static_cast<size_t>(p) * p, 0.0);
  std::vector<double> sd_beta(p, 0.0);
  for (int i = 0; i < p; ++i) {
    for (int j = 0; j < p; ++j) cov[i * p + j] = Minv[i * np + j];
    sd_beta[i] = std::sqrt(cov[i * p + i] * res_var);
  }

  Output o;
  o.beta = from_vec(beta);
  o.delta = from_vec(delta);
  o.sd_beta = from_vec(sd_beta);
  o.cov_beta = from_mat(cov, p, p);
  o.res_var = res_var;
  o.sum_square = sum_square;
  o.success = ls.success;
  o.nfev = ls.nfev;
  return o;
}

}  // namespace scipp::odr
