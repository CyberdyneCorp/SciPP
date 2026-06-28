// Basic operations — delegate to numpp::linalg, adapting to SciPy conventions.
#include "scipp/linalg/linalg.hpp"

#include <cmath>

#include "numpp/core/dtype.hpp"
#include "numpp/linalg/linalg.hpp"
#include "scipp/error.hpp"
#include "scipp/linalg/detail.hpp"

namespace scipp::linalg {
namespace nl = numpp::linalg;
namespace {
double scalar(const numpp::ndarray& a) {
  return a.astype(numpp::kFloat64).ascontiguousarray().typed_data<double>()[0];
}
}  // namespace

ndarray inv(const ndarray& a) { return nl::inv(a); }
double  det(const ndarray& a) { return scalar(nl::det(a)); }
ndarray solve(const ndarray& a, const ndarray& b) { return nl::solve(a, b); }
ndarray pinv(const ndarray& a) { return nl::pinv(a); }

LstsqResult lstsq(const ndarray& a, const ndarray& b) {
  nl::LstsqResult r = nl::lstsq(a, b);
  return {r.solution, r.residuals, r.rank, r.singular_values};
}

// Hermitian pseudo-inverse via the symmetric eigendecomposition.
ndarray pinvh(const ndarray& a) {
  EighResult e = eigh(a);
  int64_t n = e.eigenvectors.shape()[0];
  std::vector<double> V = detail::to_vec(e.eigenvectors);  // n*n row-major
  std::vector<double> w = detail::to_vec(e.eigenvalues);
  double wmax = 0.0;
  for (double x : w) wmax = std::max(wmax, std::fabs(x));
  double tol = wmax * static_cast<double>(n) * 2.220446049250313e-16;
  std::vector<double> dinv(n);
  for (int64_t i = 0; i < n; ++i) dinv[i] = (std::fabs(w[i]) > tol) ? 1.0 / w[i] : 0.0;
  std::vector<double> out(n * n, 0.0);
  for (int64_t i = 0; i < n; ++i)
    for (int64_t j = 0; j < n; ++j) {
      double s = 0.0;
      for (int64_t k = 0; k < n; ++k) s += V[i * n + k] * dinv[k] * V[j * n + k];
      out[i * n + j] = s;
    }
  return detail::from_mat(out, n, n);
}

double norm(const ndarray& a) { return scalar(nl::norm(a)); }

ndarray norm(const ndarray& a, const std::string& ord) {
  if (ord == "fro" || ord == "2") return nl::norm(a);
  int64_t r, c;
  std::vector<double> m = detail::to_mat(a, r, c);
  double best = 0.0;
  if (ord == "1") {  // max column sum
    for (int64_t j = 0; j < c; ++j) {
      double s = 0.0;
      for (int64_t i = 0; i < r; ++i) s += std::fabs(m[i * c + j]);
      best = std::max(best, s);
    }
  } else if (ord == "inf") {  // max row sum
    for (int64_t i = 0; i < r; ++i) {
      double s = 0.0;
      for (int64_t j = 0; j < c; ++j) s += std::fabs(m[i * c + j]);
      best = std::max(best, s);
    }
  } else {
    throw scipp::value_error("unsupported norm order: " + ord);
  }
  numpp::ndarray out(numpp::Shape{}, numpp::kFloat64);
  out.typed_data<double>()[0] = best;
  return out;
}

}  // namespace scipp::linalg
