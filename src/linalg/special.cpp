// Special-matrix constructors (scipy.linalg special_matrices), built by direct
// row-major fill.
#include "scypp/linalg/linalg.hpp"

#include <vector>

#include "numpp/core/dtype.hpp"
#include "numpp/linalg/linalg.hpp"
#include "scypp/error.hpp"
#include "scypp/linalg/detail.hpp"

namespace scypp::linalg {

ndarray toeplitz(const ndarray& c, const ndarray& r) {
  std::vector<double> cv = detail::to_vec(c), rv = detail::to_vec(r);
  int64_t m = cv.size(), n = rv.size();
  std::vector<double> t(m * n);
  for (int64_t i = 0; i < m; ++i)
    for (int64_t j = 0; j < n; ++j) t[i * n + j] = (i >= j) ? cv[i - j] : rv[j - i];
  return detail::from_mat(t, m, n);
}
ndarray toeplitz(const ndarray& c) { return toeplitz(c, c); }

ndarray circulant(const ndarray& c) {
  std::vector<double> cv = detail::to_vec(c);
  int64_t n = cv.size();
  std::vector<double> m(n * n);
  for (int64_t i = 0; i < n; ++i)
    for (int64_t j = 0; j < n; ++j) m[i * n + j] = cv[((i - j) % n + n) % n];
  return detail::from_mat(m, n, n);
}

ndarray hankel(const ndarray& c, const ndarray& r) {
  std::vector<double> cv = detail::to_vec(c), rv = detail::to_vec(r);
  int64_t m = cv.size(), n = rv.size();
  std::vector<double> h(m * n);
  for (int64_t i = 0; i < m; ++i)
    for (int64_t j = 0; j < n; ++j) {
      int64_t s = i + j;
      h[i * n + j] = (s < m) ? cv[s] : rv[s - m + 1];
    }
  return detail::from_mat(h, m, n);
}
ndarray hankel(const ndarray& c) {
  std::vector<double> cv = detail::to_vec(c);
  numpp::ndarray zeros(numpp::Shape{static_cast<int64_t>(cv.size())}, numpp::kFloat64);
  double* z = zeros.typed_data<double>();
  for (size_t i = 0; i < cv.size(); ++i) z[i] = 0.0;
  return hankel(c, zeros);
}

ndarray tri(int64_t n, int64_t m, int64_t k) {
  if (m < 0) m = n;
  std::vector<double> t(n * m, 0.0);
  for (int64_t i = 0; i < n; ++i)
    for (int64_t j = 0; j < m; ++j) t[i * m + j] = (j <= i + k) ? 1.0 : 0.0;
  return detail::from_mat(t, n, m);
}

ndarray block_diag(const std::vector<ndarray>& blocks) {
  int64_t R = 0, C = 0;
  std::vector<std::vector<double>> bufs;
  std::vector<std::pair<int64_t, int64_t>> dims;
  for (const auto& b : blocks) {
    int64_t r, c;
    bufs.push_back(detail::to_mat(b, r, c));
    dims.push_back({r, c});
    R += r;
    C += c;
  }
  std::vector<double> out(R * C, 0.0);
  int64_t ro = 0, co = 0;
  for (size_t k = 0; k < blocks.size(); ++k) {
    auto [r, c] = dims[k];
    for (int64_t i = 0; i < r; ++i)
      for (int64_t j = 0; j < c; ++j) out[(ro + i) * C + (co + j)] = bufs[k][i * c + j];
    ro += r;
    co += c;
  }
  return detail::from_mat(out, R, C);
}

ndarray companion(const ndarray& a) {
  std::vector<double> av = detail::to_vec(a);
  int64_t n = av.size() - 1;
  if (n < 1 || av[0] == 0.0) throw scypp::value_error("companion: invalid polynomial");
  std::vector<double> m(n * n, 0.0);
  for (int64_t j = 0; j < n; ++j) m[0 * n + j] = -av[j + 1] / av[0];
  for (int64_t i = 1; i < n; ++i) m[i * n + (i - 1)] = 1.0;
  return detail::from_mat(m, n, n);
}

ndarray leslie(const ndarray& f, const ndarray& s) {
  std::vector<double> fv = detail::to_vec(f), sv = detail::to_vec(s);
  int64_t n = fv.size();
  if (static_cast<int64_t>(sv.size()) != n - 1)
    throw scypp::value_error("leslie: len(s) must be len(f)-1");
  std::vector<double> m(n * n, 0.0);
  for (int64_t j = 0; j < n; ++j) m[0 * n + j] = fv[j];
  for (int64_t i = 1; i < n; ++i) m[i * n + (i - 1)] = sv[i - 1];
  return detail::from_mat(m, n, n);
}

ndarray kron(const ndarray& a, const ndarray& b) { return numpp::kron(a, b); }

ndarray hilbert(int64_t n) {
  std::vector<double> h(n * n);
  for (int64_t i = 0; i < n; ++i)
    for (int64_t j = 0; j < n; ++j) h[i * n + j] = 1.0 / static_cast<double>(i + j + 1);
  return detail::from_mat(h, n, n);
}

ndarray hadamard(int64_t n) {
  if (n < 1 || (n & (n - 1)) != 0) throw scypp::value_error("hadamard: n must be a power of 2");
  std::vector<double> h(n * n, 0.0);
  h[0] = 1.0;
  for (int64_t k = 1; k < n; k <<= 1) {  // Sylvester doubling
    for (int64_t i = 0; i < k; ++i)
      for (int64_t j = 0; j < k; ++j) {
        double v = h[i * n + j];
        h[i * n + (j + k)] = v;
        h[(i + k) * n + j] = v;
        h[(i + k) * n + (j + k)] = -v;
      }
  }
  return detail::from_mat(h, n, n);
}

ndarray pascal(int64_t n, const std::string& kind) {
  std::vector<std::vector<double>> binom(n + n, std::vector<double>(n + n, 0.0));
  for (int64_t i = 0; i < n + n; ++i) {
    binom[i][0] = 1.0;
    for (int64_t j = 1; j <= i; ++j) binom[i][j] = binom[i - 1][j - 1] + binom[i - 1][j];
  }
  std::vector<double> m(n * n, 0.0);
  for (int64_t i = 0; i < n; ++i)
    for (int64_t j = 0; j < n; ++j) {
      if (kind == "symmetric") m[i * n + j] = binom[i + j][i];
      else if (kind == "lower") m[i * n + j] = (j <= i) ? binom[i][j] : 0.0;
      else if (kind == "upper") m[i * n + j] = (i <= j) ? binom[j][i] : 0.0;
      else throw scypp::value_error("pascal: kind must be symmetric/lower/upper");
    }
  return detail::from_mat(m, n, n);
}

}  // namespace scypp::linalg
