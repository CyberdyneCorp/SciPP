// Nonnegative least squares via the Lawson-Hanson active-set algorithm,
// matching scipy.optimize.nnls.
#include "scipp/optimize/optimize.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

#include "numpp/core/creation.hpp"
#include "scipp/linalg/detail.hpp"

namespace scipp::optimize {
namespace {
namespace sd = scipp::linalg::detail;

// Solve the unconstrained least-squares problem restricted to the columns in
// `cols` via normal equations with Gaussian elimination; result written into z.
void solve_subset(const std::vector<double>& A, int m, int n,
                  const std::vector<double>& b, const std::vector<int>& cols,
                  std::vector<double>& z) {
  int k = static_cast<int>(cols.size());
  std::vector<double> G(k * k, 0.0), rhs(k, 0.0);
  for (int p = 0; p < k; ++p) {
    int cp = cols[p];
    for (int q = 0; q < k; ++q) {
      int cq = cols[q];
      double s = 0.0;
      for (int i = 0; i < m; ++i) s += A[i * n + cp] * A[i * n + cq];
      G[p * k + q] = s;
    }
    double s = 0.0;
    for (int i = 0; i < m; ++i) s += A[i * n + cp] * b[i];
    rhs[p] = s;
  }
  // Gaussian elimination with partial pivoting.
  for (int col = 0; col < k; ++col) {
    int piv = col;
    for (int r = col + 1; r < k; ++r)
      if (std::fabs(G[r * k + col]) > std::fabs(G[piv * k + col])) piv = r;
    if (piv != col) {
      for (int j = 0; j < k; ++j) std::swap(G[col * k + j], G[piv * k + j]);
      std::swap(rhs[col], rhs[piv]);
    }
    double d = G[col * k + col];
    if (std::fabs(d) < 1e-300) continue;
    for (int r = 0; r < k; ++r) {
      if (r == col) continue;
      double f = G[r * k + col] / d;
      for (int j = 0; j < k; ++j) G[r * k + j] -= f * G[col * k + j];
      rhs[r] -= f * rhs[col];
    }
  }
  std::fill(z.begin(), z.end(), 0.0);
  for (int p = 0; p < k; ++p) {
    double d = G[p * k + p];
    z[cols[p]] = std::fabs(d) < 1e-300 ? 0.0 : rhs[p] / d;
  }
}
}  // namespace

NNLSResult nnls(const ndarray& A_in, const ndarray& b_in, int maxiter) {
  int64_t m64, n64;
  std::vector<double> A = sd::to_mat(A_in, m64, n64);
  std::vector<double> b = sd::to_vec(b_in);
  int m = static_cast<int>(m64), n = static_cast<int>(n64);
  if (maxiter < 0) maxiter = 3 * n;

  std::vector<double> x(n, 0.0), w(n, 0.0), z(n, 0.0);
  std::vector<char> in_P(n, 0);  // passive (active-value) set membership
  const double tol = 10.0 * std::numeric_limits<double>::epsilon() *
                     (m > n ? m : n);

  int iter = 0;
  while (iter++ < maxiter * 3) {
    // w = Aᵀ(b - A x): gradient of the residual.
    std::vector<double> resid(m);
    for (int i = 0; i < m; ++i) {
      double s = b[i];
      for (int j = 0; j < n; ++j) s -= A[i * n + j] * x[j];
      resid[i] = s;
    }
    for (int j = 0; j < n; ++j) {
      double s = 0.0;
      for (int i = 0; i < m; ++i) s += A[i * n + j] * resid[i];
      w[j] = s;
    }
    // Pick the steepest-ascent index not yet in the passive set.
    int t = -1;
    double wmax = tol;
    for (int j = 0; j < n; ++j)
      if (!in_P[j] && w[j] > wmax) { wmax = w[j]; t = j; }
    if (t < 0) break;  // KKT satisfied
    in_P[t] = 1;

    // Inner loop: solve LS on passive set, drop any nonpositive components.
    for (int inner = 0; inner < maxiter * 3; ++inner) {
      std::vector<int> P;
      for (int j = 0; j < n; ++j)
        if (in_P[j]) P.push_back(j);
      solve_subset(A, m, n, b, P, z);
      double zmin = 0.0;
      for (int j : P) zmin = std::min(zmin, z[j]);
      if (zmin > 0) {
        x = z;
        break;
      }
      // Move along x -> z as far as feasibility allows.
      double alpha = 1.0;
      for (int j : P)
        if (z[j] <= 0) alpha = std::min(alpha, x[j] / (x[j] - z[j]));
      for (int j = 0; j < n; ++j) x[j] += alpha * (z[j] - x[j]);
      for (int j : P)
        if (x[j] <= tol) { in_P[j] = 0; x[j] = 0.0; }
    }
  }

  NNLSResult res;
  res.x = numpp::zeros(numpp::Shape{n}, numpp::kFloat64);
  double* xp = res.x.typed_data<double>();
  for (int j = 0; j < n; ++j) xp[j] = x[j];
  double rn = 0.0;
  for (int i = 0; i < m; ++i) {
    double s = b[i];
    for (int j = 0; j < n; ++j) s -= A[i * n + j] * x[j];
    rn += s * s;
  }
  res.rnorm = std::sqrt(rn);
  return res;
}

}  // namespace scipp::optimize
