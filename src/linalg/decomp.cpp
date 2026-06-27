// Matrix decompositions. lu/lu_factor/lu_solve and the Cholesky solves are
// hand-written (SciPy-only / convention-specific); qr/svd delegate to NumPP.
#include "scypp/linalg/linalg.hpp"

#include <cmath>
#include <vector>

#include "numpp/core/dtype.hpp"
#include "numpp/linalg/linalg.hpp"
#include "scypp/error.hpp"
#include "scypp/linalg/detail.hpp"

namespace scypp::linalg {
namespace {

// Partial-pivot LU in place (row-major n*n). Fills piv (0-based row swaps,
// LAPACK getrf style) and perm (row permutation s.t. (LU)[i] == A[perm[i]]).
// Returns false if singular.
bool lu_inplace(std::vector<double>& a, int64_t n, std::vector<int64_t>& piv,
                std::vector<int64_t>& perm) {
  piv.assign(n, 0);
  perm.resize(n);
  for (int64_t i = 0; i < n; ++i) perm[i] = i;
  for (int64_t k = 0; k < n; ++k) {
    int64_t p = k;
    double best = std::fabs(a[k * n + k]);
    for (int64_t i = k + 1; i < n; ++i) {
      double v = std::fabs(a[i * n + k]);
      if (v > best) { best = v; p = i; }
    }
    piv[k] = p;
    if (best == 0.0) return false;
    if (p != k) {
      for (int64_t j = 0; j < n; ++j) std::swap(a[k * n + j], a[p * n + j]);
      std::swap(perm[k], perm[p]);
    }
    double akk = a[k * n + k];
    for (int64_t i = k + 1; i < n; ++i) {
      double f = a[i * n + k] / akk;
      a[i * n + k] = f;
      for (int64_t j = k + 1; j < n; ++j) a[i * n + j] -= f * a[k * n + j];
    }
  }
  return true;
}

// Solve using a packed LU factor + pivots. b is n x nrhs (row-major).
std::vector<double> lu_backsolve(const std::vector<double>& lu, int64_t n,
                                 const std::vector<int64_t>& piv,
                                 std::vector<double> b, int64_t nrhs) {
  for (int64_t k = 0; k < n; ++k)  // apply row swaps to b
    if (piv[k] != k)
      for (int64_t c = 0; c < nrhs; ++c) std::swap(b[k * nrhs + c], b[piv[k] * nrhs + c]);
  for (int64_t c = 0; c < nrhs; ++c) {  // forward solve L y = Pb (unit diagonal)
    for (int64_t i = 0; i < n; ++i)
      for (int64_t j = 0; j < i; ++j) b[i * nrhs + c] -= lu[i * n + j] * b[j * nrhs + c];
    for (int64_t i = n - 1; i >= 0; --i) {  // back solve U x = y
      for (int64_t j = i + 1; j < n; ++j) b[i * nrhs + c] -= lu[i * n + j] * b[j * nrhs + c];
      b[i * nrhs + c] /= lu[i * n + i];
    }
  }
  return b;
}

}  // namespace

LUResult lu(const ndarray& a) {
  int64_t n, c;
  std::vector<double> work = detail::to_mat(a, n, c);
  if (n != c) throw scypp::value_error("lu: matrix must be square");
  std::vector<int64_t> piv, perm;
  lu_inplace(work, n, piv, perm);  // singular A is allowed for lu (no throw)
  std::vector<double> L(n * n, 0.0), U(n * n, 0.0), P(n * n, 0.0);
  for (int64_t i = 0; i < n; ++i) {
    L[i * n + i] = 1.0;
    for (int64_t j = 0; j < n; ++j) {
      if (j < i) L[i * n + j] = work[i * n + j];
      else U[i * n + j] = work[i * n + j];
    }
    P[perm[i] * n + i] = 1.0;  // P @ L @ U == A
  }
  return {detail::from_mat(P, n, n), detail::from_mat(L, n, n), detail::from_mat(U, n, n)};
}

LUFactor lu_factor(const ndarray& a) {
  int64_t n, c;
  std::vector<double> work = detail::to_mat(a, n, c);
  if (n != c) throw scypp::value_error("lu_factor: matrix must be square");
  std::vector<int64_t> piv, perm;
  lu_inplace(work, n, piv, perm);
  numpp::ndarray pivarr(numpp::Shape{n}, numpp::kInt64);
  int64_t* pp = pivarr.typed_data<int64_t>();
  for (int64_t i = 0; i < n; ++i) pp[i] = piv[i];
  return {detail::from_mat(work, n, n), pivarr};
}

ndarray lu_solve(const LUFactor& f, const ndarray& b) {
  int64_t n, c;
  std::vector<double> lu_ = detail::to_mat(f.lu, n, c);
  std::vector<int64_t> piv(n);
  {
    numpp::ndarray pa = f.piv.astype(numpp::kInt64).ascontiguousarray();
    const int64_t* pp = pa.typed_data<int64_t>();
    for (int64_t i = 0; i < n; ++i) piv[i] = pp[i];
  }
  bool vec = (b.ndim() == 1);
  int64_t nrhs = vec ? 1 : b.shape()[1];
  std::vector<double> rhs = vec ? detail::to_vec(b) : detail::to_mat(b, n, nrhs);
  std::vector<double> x = lu_backsolve(lu_, n, piv, rhs, nrhs);
  return vec ? detail::from_vec(x) : detail::from_mat(x, n, nrhs);
}

QRResult qr(const ndarray& a, const std::string& mode) {
  std::string m = (mode == "economic") ? "reduced" : (mode == "full") ? "complete" : mode;
  numpp::linalg::QRResult r = numpp::linalg::qr(a, m);
  return {r.q, r.r};
}

SVDResult svd(const ndarray& a, bool full_matrices) {
  numpp::linalg::SVDResult r = numpp::linalg::svd(a, full_matrices);
  return {r.u, r.s, r.vh};
}

ndarray svdvals(const ndarray& a) { return numpp::linalg::svdvals(a); }

ndarray cholesky(const ndarray& a, bool lower) {
  ndarray L = numpp::linalg::cholesky(a);  // NumPP returns the lower factor
  if (lower) return L;
  return L.transpose().ascontiguousarray();  // SciPy upper default: R = Lᵀ
}

CholFactor cho_factor(const ndarray& a, bool lower) { return {cholesky(a, lower), lower}; }

ndarray cho_solve(const CholFactor& f, const ndarray& b) {
  int64_t n, c;
  std::vector<double> fac = detail::to_mat(f.c, n, c);
  // Normalize to the lower factor L (A = L Lᵀ).
  std::vector<double> L(n * n, 0.0);
  if (f.lower) L = fac;
  else for (int64_t i = 0; i < n; ++i) for (int64_t j = 0; j < n; ++j) L[i * n + j] = fac[j * n + i];

  bool vec = (b.ndim() == 1);
  int64_t nrhs = vec ? 1 : b.shape()[1];
  std::vector<double> x = vec ? detail::to_vec(b) : detail::to_mat(b, n, nrhs);
  for (int64_t col = 0; col < nrhs; ++col) {
    for (int64_t i = 0; i < n; ++i) {  // forward: L y = b
      for (int64_t j = 0; j < i; ++j) x[i * nrhs + col] -= L[i * n + j] * x[j * nrhs + col];
      x[i * nrhs + col] /= L[i * n + i];
    }
    for (int64_t i = n - 1; i >= 0; --i) {  // back: Lᵀ z = y
      for (int64_t j = i + 1; j < n; ++j) x[i * nrhs + col] -= L[j * n + i] * x[j * nrhs + col];
      x[i * nrhs + col] /= L[i * n + i];
    }
  }
  return vec ? detail::from_vec(x) : detail::from_mat(x, n, nrhs);
}

}  // namespace scypp::linalg
