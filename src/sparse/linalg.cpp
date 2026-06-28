// Sparse linear algebra: spsolve (direct), cg / gmres (matrix-free), norm.
#include "scipp/sparse/sparse.hpp"

#include <cmath>
#include <vector>

#include "numpp/linalg/linalg.hpp"
#include "scipp/sparse/detail.hpp"

namespace scipp::sparse {
namespace d = detail;
namespace {
double dot(const std::vector<double>& a, const std::vector<double>& b) {
  double s = 0; for (size_t i = 0; i < a.size(); ++i) s += a[i] * b[i]; return s;
}
}  // namespace

ndarray spsolve(const CsrMatrix& A, const ndarray& b) {
  return numpp::linalg::solve(A.toarray(), b);  // dense factorization (sparse LU is a follow-up)
}

ndarray cg(const CsrMatrix& A, const ndarray& b, double tol, int maxiter) {
  std::vector<double> bv = d::dv(b);
  int n = static_cast<int>(bv.size());
  std::vector<double> x(n, 0.0), r = bv, p = bv;
  double rs = dot(r, r), bnorm = std::sqrt(dot(bv, bv));
  if (bnorm == 0) return d::from_dv(x);
  for (int it = 0; it < maxiter; ++it) {
    std::vector<double> Ap = d::dv(A.spmv(d::from_dv(p)));
    double alpha = rs / dot(p, Ap);
    for (int i = 0; i < n; ++i) { x[i] += alpha * p[i]; r[i] -= alpha * Ap[i]; }
    double rs_new = dot(r, r);
    if (std::sqrt(rs_new) / bnorm < tol) break;
    double beta = rs_new / rs;
    for (int i = 0; i < n; ++i) p[i] = r[i] + beta * p[i];
    rs = rs_new;
  }
  return d::from_dv(x);
}

ndarray gmres(const CsrMatrix& A, const ndarray& b, double tol, int maxiter) {
  std::vector<double> bv = d::dv(b);
  int n = static_cast<int>(bv.size());
  double bnorm = std::sqrt(dot(bv, bv));
  if (bnorm == 0) return d::from_dv(std::vector<double>(n, 0.0));
  std::vector<double> x(n, 0.0);
  int restart = std::min(n, 50);
  for (int outer = 0; outer < maxiter; ++outer) {
    std::vector<double> r = bv, Ax = d::dv(A.spmv(d::from_dv(x)));
    for (int i = 0; i < n; ++i) r[i] -= Ax[i];
    double beta = std::sqrt(dot(r, r));
    if (beta / bnorm < tol) break;
    std::vector<std::vector<double>> V;
    V.push_back(r); for (double& v : V[0]) v /= beta;
    std::vector<std::vector<double>> H(restart + 1, std::vector<double>(restart, 0.0));
    std::vector<double> cs(restart, 0), sn(restart, 0), g(restart + 1, 0.0);
    g[0] = beta;
    int k = 0;
    for (; k < restart; ++k) {
      std::vector<double> w = d::dv(A.spmv(d::from_dv(V[k])));
      for (int j = 0; j <= k; ++j) { H[j][k] = dot(w, V[j]); for (int i = 0; i < n; ++i) w[i] -= H[j][k] * V[j][i]; }
      H[k + 1][k] = std::sqrt(dot(w, w));
      if (H[k + 1][k] > 1e-14) { for (double& v : w) v /= H[k + 1][k]; V.push_back(w); }
      for (int j = 0; j < k; ++j) {  // apply previous Givens rotations
        double t = cs[j] * H[j][k] + sn[j] * H[j + 1][k];
        H[j + 1][k] = -sn[j] * H[j][k] + cs[j] * H[j + 1][k];
        H[j][k] = t;
      }
      double rho = std::hypot(H[k][k], H[k + 1][k]);
      cs[k] = H[k][k] / rho; sn[k] = H[k + 1][k] / rho;
      H[k][k] = rho; H[k + 1][k] = 0.0;
      g[k + 1] = -sn[k] * g[k]; g[k] = cs[k] * g[k];
      if (std::fabs(g[k + 1]) / bnorm < tol) { ++k; break; }
    }
    std::vector<double> yk(k, 0.0);  // back-substitution
    for (int i = k - 1; i >= 0; --i) {
      double s = g[i];
      for (int j = i + 1; j < k; ++j) s -= H[i][j] * yk[j];
      yk[i] = s / H[i][i];
    }
    for (int i = 0; i < n; ++i) for (int j = 0; j < k; ++j) x[i] += yk[j] * V[j][i];
  }
  return d::from_dv(x);
}

double norm(const CsrMatrix& A, const std::string& ord) {
  auto da = d::dv(A.data());
  if (ord == "fro") { double s = 0; for (double v : da) s += v * v; return std::sqrt(s); }
  auto ip = d::iv(A.indptr()), id = d::iv(A.indices());
  std::vector<double> colsum(A.cols(), 0.0), rowsum(A.rows(), 0.0);
  for (int64_t i = 0; i < A.rows(); ++i)
    for (int64_t k = ip[i]; k < ip[i + 1]; ++k) { rowsum[i] += std::fabs(da[k]); colsum[id[k]] += std::fabs(da[k]); }
  double best = 0;
  if (ord == "1") for (double c : colsum) best = std::max(best, c);
  else if (ord == "inf") for (double r : rowsum) best = std::max(best, r);
  return best;
}

}  // namespace scipp::sparse
