// Matrix functions: expm (Padé scaling-and-squaring) and polar (SVD-based).
// Dense products go through numpp::backend::matmul so a BLAS/GPU NumPP build
// accelerates them; the final Padé system uses numpp::linalg::solve.
#include "scipp/linalg/linalg.hpp"

#include <cmath>
#include <vector>

#include "numpp/backend/backend.hpp"
#include "numpp/core/dtype.hpp"
#include "numpp/linalg/linalg.hpp"
#include "scipp/error.hpp"
#include "scipp/linalg/detail.hpp"

namespace scipp::linalg {
namespace {

ndarray mm(const ndarray& a, const ndarray& b) { return numpp::matmul(a, b); }

// Linear combination of n*n matrices plus an optional scalar on the diagonal.
ndarray lin(const std::vector<std::pair<double, ndarray>>& terms, double diag = 0.0) {
  int64_t n = terms.front().second.shape()[0];
  std::vector<double> acc(n * n, 0.0);
  for (const auto& [coeff, M] : terms) {
    std::vector<double> m = detail::to_vec(M);
    for (int64_t i = 0; i < n * n; ++i) acc[i] += coeff * m[i];
  }
  for (int64_t i = 0; i < n; ++i) acc[i * n + i] += diag;
  return detail::from_mat(acc, n, n);
}

double one_norm(const std::vector<double>& a, int64_t n) {
  double best = 0.0;
  for (int64_t j = 0; j < n; ++j) {
    double s = 0.0;
    for (int64_t i = 0; i < n; ++i) s += std::fabs(a[i * n + j]);
    best = std::max(best, s);
  }
  return best;
}

}  // namespace

ndarray expm(const ndarray& A) {
  int64_t n, c;
  std::vector<double> a0 = detail::to_mat(A, n, c);
  if (n != c) throw scipp::value_error("expm: matrix must be square");

  // Order-13 Padé coefficients (Higham, 2005).
  static const double b[14] = {
      64764752532480000.0, 32382376266240000.0, 7771770303897600.0, 1187353796428800.0,
      129060195264000.0, 10559470521600.0, 670442572800.0, 33522128640.0,
      1323241920.0, 40840800.0, 960960.0, 16380.0, 182.0, 1.0};
  const double theta13 = 5.371920351148152;

  double normA = one_norm(a0, n);
  int s = 0;
  if (normA > theta13) s = static_cast<int>(std::ceil(std::log2(normA / theta13)));
  ndarray Ac = (s > 0) ? lin({{std::ldexp(1.0, -s), detail::from_mat(a0, n, n)}})
                       : detail::from_mat(a0, n, n);

  ndarray I = lin({{0.0, Ac}}, 1.0);
  ndarray A2 = mm(Ac, Ac), A4 = mm(A2, A2), A6 = mm(A2, A4);
  ndarray U = mm(Ac, lin({{1.0, mm(A6, lin({{b[13], A6}, {b[11], A4}, {b[9], A2}}))},
                          {b[7], A6}, {b[5], A4}, {b[3], A2}}, b[1]));
  ndarray V = lin({{1.0, mm(A6, lin({{b[12], A6}, {b[10], A4}, {b[8], A2}}))},
                   {b[6], A6}, {b[4], A4}, {b[2], A2}}, b[0]);
  ndarray P = lin({{1.0, V}, {1.0, U}});   // V + U
  ndarray Q = lin({{1.0, V}, {-1.0, U}});  // V - U
  ndarray R = numpp::linalg::solve(Q, P);
  for (int i = 0; i < s; ++i) R = mm(R, R);
  return R;
}

PolarResult polar(const ndarray& A, const std::string& side) {
  numpp::linalg::SVDResult s = numpp::linalg::svd(A, /*full_matrices=*/false);
  ndarray U = mm(s.u, s.vh);  // nearest orthonormal factor
  int64_t k = s.s.shape()[0];
  std::vector<double> sv = detail::to_vec(s.s);
  if (side == "right") {
    int64_t nn = s.vh.shape()[1];
    std::vector<double> vh = detail::to_vec(s.vh);  // k x nn
    std::vector<double> P(nn * nn, 0.0);
    for (int64_t i = 0; i < nn; ++i)
      for (int64_t j = 0; j < nn; ++j) {
        double acc = 0.0;
        for (int64_t l = 0; l < k; ++l) acc += vh[l * nn + i] * sv[l] * vh[l * nn + j];
        P[i * nn + j] = acc;
      }
    return {U, detail::from_mat(P, nn, nn)};
  }
  // side == "left": A = P U, P = (u*s) @ uᵀ
  int64_t m = s.u.shape()[0];
  std::vector<double> u = detail::to_vec(s.u);  // m x k
  std::vector<double> P(m * m, 0.0);
  for (int64_t i = 0; i < m; ++i)
    for (int64_t j = 0; j < m; ++j) {
      double acc = 0.0;
      for (int64_t l = 0; l < k; ++l) acc += u[i * k + l] * sv[l] * u[j * k + l];
      P[i * m + j] = acc;
    }
  return {U, detail::from_mat(P, m, m)};
}

}  // namespace scipp::linalg
