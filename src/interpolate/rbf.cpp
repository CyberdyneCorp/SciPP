// Radial-basis-function interpolation (scattered data) with a polynomial tail.
#include "scipp/interpolate/interpolate.hpp"

#include <cmath>
#include <vector>

#include "numpp/core/dtype.hpp"
#include "numpp/linalg/linalg.hpp"
#include "scipp/error.hpp"
#include "scipp/linalg/detail.hpp"

namespace scipp::interpolate {
namespace {
namespace sd = scipp::linalg::detail;

double kernel_value(const std::string& k, double r) {  // r already scaled by epsilon
  if (k == "linear") return -r;
  if (k == "cubic") return r * r * r;
  if (k == "quintic") return -r * r * r * r * r;
  if (k == "thin_plate_spline") return (r == 0.0) ? 0.0 : r * r * std::log(r);
  if (k == "multiquadric") return -std::sqrt(r * r + 1.0);
  if (k == "inverse_multiquadric") return 1.0 / std::sqrt(r * r + 1.0);
  if (k == "inverse_quadratic") return 1.0 / (r * r + 1.0);
  if (k == "gaussian") return std::exp(-r * r);
  throw scipp::value_error("RBFInterpolator: unknown kernel " + k);
}

int min_degree(const std::string& k) {
  if (k == "multiquadric" || k == "linear") return 0;
  if (k == "thin_plate_spline" || k == "cubic") return 1;
  if (k == "quintic") return 2;
  return -1;  // gaussian / inverse_* : no minimum
}

// Monomials of x (dim components) up to total degree, in a fixed order.
std::vector<double> monomials(const double* x, int dim, int degree) {
  std::vector<double> p;
  p.push_back(1.0);                                  // degree 0
  if (degree >= 1)
    for (int i = 0; i < dim; ++i) p.push_back(x[i]);  // degree 1
  if (degree >= 2) {
    for (int i = 0; i < dim; ++i) p.push_back(x[i] * x[i]);
    for (int i = 0; i < dim; ++i)
      for (int j = i + 1; j < dim; ++j) p.push_back(x[i] * x[j]);
  }
  return p;
}

double dist(const double* a, const double* b, int dim) {
  double s = 0.0;
  for (int d = 0; d < dim; ++d) { double e = a[d] - b[d]; s += e * e; }
  return std::sqrt(s);
}
}  // namespace

RBFInterpolator::RBFInterpolator(const ndarray& y, const ndarray& d, std::string kernel,
                                 double epsilon, int degree)
    : kernel_(std::move(kernel)), epsilon_(epsilon) {
  numpp::ndarray Y = y.astype(numpp::kFloat64).ascontiguousarray();
  n_ = static_cast<int>(Y.shape()[0]);
  dim_ = (Y.ndim() == 1) ? 1 : static_cast<int>(Y.shape()[1]);
  const double* yp = Y.typed_data<double>();
  y_.assign(yp, yp + static_cast<size_t>(n_) * dim_);
  std::vector<double> dv = sd::to_vec(d);

  degree_ = (degree >= 0) ? degree : std::max(min_degree(kernel_), 0);
  npoly_ = static_cast<int>(monomials(y_.data(), dim_, degree_).size());

  int N = n_ + npoly_;
  std::vector<double> A(static_cast<size_t>(N) * N, 0.0), rhs(N, 0.0);
  auto a = [&](int r, int c) -> double& { return A[static_cast<size_t>(r) * N + c]; };
  for (int i = 0; i < n_; ++i) {
    for (int j = 0; j < n_; ++j)
      a(i, j) = kernel_value(kernel_, epsilon_ * dist(&y_[i * dim_], &y_[j * dim_], dim_));
    std::vector<double> P = monomials(&y_[i * dim_], dim_, degree_);
    for (int k = 0; k < npoly_; ++k) { a(i, n_ + k) = P[k]; a(n_ + k, i) = P[k]; }
    rhs[i] = dv[i];
  }
  std::vector<double> sol = sd::to_vec(numpp::linalg::solve(sd::from_mat(A, N, N), sd::from_vec(rhs)));
  w_.assign(sol.begin(), sol.begin() + n_);
  poly_.assign(sol.begin() + n_, sol.end());
}

ndarray RBFInterpolator::operator()(const ndarray& x) const {
  numpp::ndarray X = x.astype(numpp::kFloat64).ascontiguousarray();
  int64_t m = (X.ndim() == 1) ? 1 : X.shape()[0];
  const double* xp = X.typed_data<double>();
  std::vector<double> out(m);
  for (int64_t r = 0; r < m; ++r) {
    const double* xq = xp + r * dim_;
    double acc = 0.0;
    for (int i = 0; i < n_; ++i)
      acc += w_[i] * kernel_value(kernel_, epsilon_ * dist(xq, &y_[i * dim_], dim_));
    std::vector<double> P = monomials(xq, dim_, degree_);
    for (int k = 0; k < npoly_; ++k) acc += poly_[k] * P[k];
    out[r] = acc;
  }
  return sd::from_vec(out);
}

}  // namespace scipp::interpolate
