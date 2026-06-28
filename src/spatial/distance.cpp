// Distance computations: pdist, cdist, squareform, distance_matrix + metrics.
#include "scipp/spatial/spatial.hpp"

#include <cmath>
#include <functional>
#include <vector>

#include "numpp/backend/backend.hpp"
#include "numpp/core/dtype.hpp"
#include "scipp/error.hpp"
#include "scipp/linalg/detail.hpp"

namespace scipp::spatial {
namespace {
namespace sd = scipp::linalg::detail;
thread_local Backend g_last = Backend::Cpu;

Backend map_backend(numpp::Backend b) {
  return (b == numpp::Backend::Cpu || b == numpp::Backend::Auto) ? Backend::Cpu
                                                                 : Backend::Device;
}

bool is_euclidean_family(const std::string& m) {
  return m == "euclidean" || m == "sqeuclidean";
}

using Metric = std::function<double(const double*, const double*, int64_t)>;

Metric make_metric(const std::string& name, double p) {
  if (name == "euclidean") return [](const double* u, const double* v, int64_t d) { double s = 0; for (int64_t i = 0; i < d; ++i) { double e = u[i] - v[i]; s += e * e; } return std::sqrt(s); };
  if (name == "sqeuclidean") return [](const double* u, const double* v, int64_t d) { double s = 0; for (int64_t i = 0; i < d; ++i) { double e = u[i] - v[i]; s += e * e; } return s; };
  if (name == "cityblock") return [](const double* u, const double* v, int64_t d) { double s = 0; for (int64_t i = 0; i < d; ++i) s += std::fabs(u[i] - v[i]); return s; };
  if (name == "chebyshev") return [](const double* u, const double* v, int64_t d) { double m = 0; for (int64_t i = 0; i < d; ++i) m = std::max(m, std::fabs(u[i] - v[i])); return m; };
  if (name == "minkowski") return [p](const double* u, const double* v, int64_t d) { double s = 0; for (int64_t i = 0; i < d; ++i) s += std::pow(std::fabs(u[i] - v[i]), p); return std::pow(s, 1.0 / p); };
  if (name == "cosine") return [](const double* u, const double* v, int64_t d) { double uv = 0, uu = 0, vv = 0; for (int64_t i = 0; i < d; ++i) { uv += u[i] * v[i]; uu += u[i] * u[i]; vv += v[i] * v[i]; } return 1.0 - uv / (std::sqrt(uu) * std::sqrt(vv)); };
  if (name == "correlation") return [](const double* u, const double* v, int64_t d) {
    double mu = 0, mv = 0; for (int64_t i = 0; i < d; ++i) { mu += u[i]; mv += v[i]; } mu /= d; mv /= d;
    double uv = 0, uu = 0, vv = 0; for (int64_t i = 0; i < d; ++i) { double a = u[i] - mu, b = v[i] - mv; uv += a * b; uu += a * a; vv += b * b; }
    return 1.0 - uv / (std::sqrt(uu) * std::sqrt(vv)); };
  if (name == "hamming") return [](const double* u, const double* v, int64_t d) { double c = 0; for (int64_t i = 0; i < d; ++i) c += (u[i] != v[i]); return c / d; };
  if (name == "jaccard") return [](const double* u, const double* v, int64_t d) {
    double neq = 0, nz = 0; for (int64_t i = 0; i < d; ++i) { if (u[i] != v[i]) neq += 1; if (u[i] != 0 || v[i] != 0) nz += 1; }
    return nz == 0 ? 0.0 : neq / nz; };
  throw scipp::value_error("unknown metric: " + name);
}
}  // namespace

Backend last_backend() { return g_last; }

ndarray pdist(const ndarray& X, const std::string& metric, double p) {
  numpp::ndarray Xc = X.astype(numpp::kFloat64).ascontiguousarray();
  int64_t n = Xc.shape()[0], d = Xc.shape()[1];
  if (is_euclidean_family(metric)) {
    // Delegate to NumPP's accelerable kernel, then extract the condensed upper
    // triangle (i<j). last_backend() reflects NumPP's actual choice.
    numpp::ndarray D = numpp::cdist_euclidean(Xc, Xc, metric == "sqeuclidean")
                           .astype(numpp::kFloat64)
                           .ascontiguousarray();
    g_last = map_backend(numpp::last_backend());
    const double* dm = D.typed_data<double>();
    std::vector<double> out;
    out.reserve(static_cast<size_t>(n) * (n - 1) / 2);
    for (int64_t i = 0; i < n; ++i)
      for (int64_t j = i + 1; j < n; ++j) out.push_back(dm[i * n + j]);
    return sd::from_vec(out);
  }
  const double* x = Xc.typed_data<double>();
  Metric f = make_metric(metric, p);
  std::vector<double> out;
  for (int64_t i = 0; i < n; ++i)
    for (int64_t j = i + 1; j < n; ++j) out.push_back(f(x + i * d, x + j * d, d));
  g_last = Backend::Cpu;
  return sd::from_vec(out);
}

ndarray cdist(const ndarray& XA, const ndarray& XB, const std::string& metric, double p,
              Backend forced) {
  numpp::ndarray A = XA.astype(numpp::kFloat64).ascontiguousarray();
  numpp::ndarray B = XB.astype(numpp::kFloat64).ascontiguousarray();
  int64_t m = A.shape()[0], n = B.shape()[0], d = A.shape()[1];
  if (is_euclidean_family(metric)) {
    // Delegate euclidean/sqeuclidean to NumPP's accelerable kernel, which owns
    // device offload + CPU fallback. `forced==Cpu` pins the CPU path; otherwise
    // `Auto` lets NumPP choose by size/availability. last_backend() then
    // reflects NumPP's actual choice (CPU locally as NumPP is CPU-only).
    numpp::Backend nb = (forced == Backend::Cpu) ? numpp::Backend::Cpu : numpp::Backend::Auto;
    numpp::ndarray D = numpp::cdist_euclidean(A, B, metric == "sqeuclidean", nb);
    g_last = map_backend(numpp::last_backend());
    return D.astype(numpp::kFloat64).ascontiguousarray();
  }
  const double* a = A.typed_data<double>();
  const double* b = B.typed_data<double>();
  g_last = Backend::Cpu;  // other metrics: portable CPU kernel
  Metric f = make_metric(metric, p);
  std::vector<double> out(m * n);
  for (int64_t i = 0; i < m; ++i)
    for (int64_t j = 0; j < n; ++j) out[i * n + j] = f(a + i * d, b + j * d, d);
  return sd::from_mat(out, m, n);
}

ndarray squareform(const ndarray& X) {
  numpp::ndarray Xc = X.astype(numpp::kFloat64).ascontiguousarray();
  if (Xc.ndim() == 1) {  // condensed -> square
    int64_t k = Xc.size();
    int64_t n = static_cast<int64_t>((1 + std::sqrt(1.0 + 8.0 * k)) / 2);
    const double* c = Xc.typed_data<double>();
    std::vector<double> sq(n * n, 0.0);
    int64_t idx = 0;
    for (int64_t i = 0; i < n; ++i)
      for (int64_t j = i + 1; j < n; ++j) { sq[i * n + j] = sq[j * n + i] = c[idx++]; }
    return sd::from_mat(sq, n, n);
  }
  int64_t n = Xc.shape()[0];  // square -> condensed
  const double* s = Xc.typed_data<double>();
  std::vector<double> out;
  for (int64_t i = 0; i < n; ++i)
    for (int64_t j = i + 1; j < n; ++j) out.push_back(s[i * n + j]);
  return sd::from_vec(out);
}

ndarray distance_matrix(const ndarray& X, const ndarray& Y, double p) {
  return cdist(X, Y, "minkowski", p);
}

}  // namespace scipp::spatial
