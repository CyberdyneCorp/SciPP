// Oracle tests for scypp::linalg against frozen SciPy golden data, plus
// reconstruction checks for the decompositions.
#include <vector>

#include "golden.hpp"
#include "numpp/backend/backend.hpp"
#include "numpp/core/dtype.hpp"
#include "numpp/core/error.hpp"
#include "numpp/core/ndarray.hpp"
#include "scypp/linalg/linalg.hpp"
#include "scypp_test.hpp"

namespace ll = scypp::linalg;

namespace {
constexpr double R = 1e-9, A = 1e-11;

numpp::ndarray g2(const double* d, int r, int c) {
  numpp::ndarray m(numpp::Shape{r, c}, numpp::kFloat64);
  double* p = m.typed_data<double>();
  for (int i = 0; i < r * c; ++i) p[i] = d[i];
  return m;
}
numpp::ndarray mm(const numpp::ndarray& a, const numpp::ndarray& b) { return numpp::matmul(a, b); }

numpp::ndarray diagmat(const numpp::ndarray& s) {
  numpp::ndarray sc = s.astype(numpp::kFloat64).ascontiguousarray();
  int64_t k = sc.size();
  const double* sv = sc.typed_data<double>();
  numpp::ndarray d(numpp::Shape{k, k}, numpp::kFloat64);
  double* p = d.typed_data<double>();
  for (int64_t i = 0; i < k * k; ++i) p[i] = 0.0;
  for (int64_t i = 0; i < k; ++i) p[i * k + i] = sv[i];
  return d;
}

void check_mat(const numpp::ndarray& got, const double* exp, int r, int c,
               double rtol = R, double atol = A) {
  numpp::ndarray gc = got.astype(numpp::kFloat64).ascontiguousarray();
  CHECK(gc.size() == static_cast<int64_t>(r) * c);
  const double* g = gc.typed_data<double>();
  for (int i = 0; i < r * c; ++i) CHECK_CLOSE(g[i], exp[i], rtol, atol);
}
}  // namespace

#define G(name) golden::name##_d, golden::name##_r, golden::name##_c

TEST_CASE("basic operations") {
  auto Asq = g2(G(la_Asq));
  auto b4 = g2(G(la_b4));
  CHECK_CLOSE(ll::det(Asq), golden::la_det_Asq, R, A);
  check_mat(ll::inv(Asq), G(la_inv_Asq));
  check_mat(ll::solve(Asq, b4), G(la_solve_Asq_b4));
  CHECK_CLOSE(ll::norm(Asq), golden::la_norm_fro, R, A);
  CHECK_CLOSE(ll::norm(Asq, "1").typed_data<double>()[0], golden::la_norm_1, R, A);
  CHECK_CLOSE(ll::norm(Asq, "inf").typed_data<double>()[0], golden::la_norm_inf, R, A);

  auto Atall = g2(G(la_Atall));
  auto b5 = g2(G(la_b5));
  check_mat(ll::lstsq(Atall, b5).x, G(la_lstsq_x));
  check_mat(ll::pinv(Atall), G(la_pinv_Atall));
  check_mat(ll::pinvh(g2(G(la_Aspd))), G(la_pinvh_Aspd));
}

TEST_CASE("singular matrix raises") {
  double sd[] = {1.0, 2.0, 2.0, 4.0};
  auto S = g2(sd, 2, 2);
  CHECK_THROWS_AS(ll::inv(S), numpp::linalg_error);
}

TEST_CASE("LU decomposition") {
  auto Asq = g2(G(la_Asq));
  ll::LUResult r = ll::lu(Asq);
  check_mat(mm(mm(r.p, r.l), r.u), G(la_Asq));  // P @ L @ U == A
  numpp::ndarray Lc = r.l.astype(numpp::kFloat64).ascontiguousarray();
  numpp::ndarray Uc = r.u.astype(numpp::kFloat64).ascontiguousarray();
  const double* L = Lc.typed_data<double>();
  const double* U = Uc.typed_data<double>();
  int n = golden::la_Asq_r;
  for (int i = 0; i < n; ++i) {
    CHECK_CLOSE(L[i * n + i], 1.0, R, A);             // unit lower
    for (int j = i + 1; j < n; ++j) CHECK_CLOSE(L[i * n + j], 0.0, R, A);
    for (int j = 0; j < i; ++j) CHECK_CLOSE(U[i * n + j], 0.0, R, A);  // upper
  }
  // lu_factor / lu_solve round-trip: A x = b
  auto b4 = g2(G(la_b4));
  auto x = ll::lu_solve(ll::lu_factor(Asq), b4);
  check_mat(mm(Asq, x), G(la_b4));
}

TEST_CASE("QR and SVD reconstruct") {
  auto Asq = g2(G(la_Asq));
  ll::QRResult qr = ll::qr(Asq);
  check_mat(mm(qr.q, qr.r), G(la_Asq), 1e-9, 1e-10);
  ll::SVDResult s = ll::svd(Asq, /*full_matrices=*/false);
  check_mat(mm(mm(s.u, diagmat(s.s)), s.vh), G(la_Asq), 1e-9, 1e-10);
}

TEST_CASE("Cholesky") {
  auto Aspd = g2(G(la_Aspd));
  auto Rup = ll::cholesky(Aspd);                       // upper default
  check_mat(Rup, G(la_chol_upper));
  check_mat(mm(Rup.transpose().ascontiguousarray(), Rup), G(la_Aspd), 1e-9, 1e-10);
  auto b3d = std::vector<double>{1.0, 2.0, 3.0};
  auto b3 = g2(b3d.data(), 3, 1);
  auto x = ll::cho_solve(ll::cho_factor(Aspd), b3);
  check_mat(mm(Aspd, x), b3d.data(), 3, 1);
  // non-PD input raises
  double npd[] = {1.0, 2.0, 2.0, 1.0};
  CHECK_THROWS_AS(ll::cholesky(g2(npd, 2, 2)), numpp::linalg_error);
}

TEST_CASE("Hermitian eigenproblem") {
  auto Aspd = g2(G(la_Aspd));
  check_mat(ll::eigvalsh(Aspd), G(la_eigvalsh_Aspd));  // ascending
  ll::EighResult e = ll::eigh(Aspd);
  // reconstruct A ≈ V diag(w) Vᵀ
  auto V = e.eigenvectors;
  auto recon = mm(mm(V, diagmat(e.eigenvalues)), V.transpose().ascontiguousarray());
  check_mat(recon, G(la_Aspd), 1e-9, 1e-10);
}

TEST_CASE("matrix functions") {
  check_mat(ll::expm(g2(G(la_Asq))), G(la_expm_Asq), 1e-8, 1e-10);
  double rot[] = {0.0, 1.0, -1.0, 0.0};
  check_mat(ll::expm(g2(rot, 2, 2)), G(la_expm_rot), 1e-10, 1e-12);
  double z[] = {0.0, 0.0, 0.0, 0.0};
  double I2[] = {1.0, 0.0, 0.0, 1.0};
  check_mat(ll::expm(g2(z, 2, 2)), I2, 2, 2);

  auto Asq = g2(G(la_Asq));
  ll::PolarResult p = ll::polar(Asq);
  check_mat(p.u, G(la_polar_u));
  check_mat(p.p, G(la_polar_p));
  check_mat(mm(p.u, p.p), G(la_Asq), 1e-9, 1e-10);
}

TEST_CASE("special matrices") {
  double c4[] = {1, 2, 3, 4};
  check_mat(ll::toeplitz(g2(c4, 4, 1)), G(la_toeplitz));
  double c3[] = {1, 2, 3}, r3[] = {3, 4, 5};
  check_mat(ll::circulant(g2(c3, 3, 1)), G(la_circulant));
  check_mat(ll::hankel(g2(c3, 3, 1), g2(r3, 3, 1)), G(la_hankel));
  check_mat(ll::hilbert(4), G(la_hilbert));
  check_mat(ll::hadamard(4), G(la_hadamard));
  check_mat(ll::pascal(4), G(la_pascal));
  double poly[] = {1, -6, 11, -6};
  check_mat(ll::companion(g2(poly, 4, 1)), G(la_companion));
  check_mat(ll::tri(4), G(la_tri));
  double f[] = {0.1, 2.0, 1.0, 0.1}, s[] = {0.2, 0.8, 0.7};
  check_mat(ll::leslie(g2(f, 4, 1), g2(s, 3, 1)), G(la_leslie));
  std::vector<numpp::ndarray> blocks;
  double b1[] = {1, 2, 3, 4}, b2[] = {5}, b3[] = {6, 7};
  blocks.push_back(g2(b1, 2, 2));
  blocks.push_back(g2(b2, 1, 1));
  blocks.push_back(g2(b3, 1, 2));
  check_mat(ll::block_diag(blocks), G(la_blockdiag));
}
