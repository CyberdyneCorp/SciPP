// Oracle tests for scypp::sparse against frozen SciPy golden data.
#include <cmath>
#include <vector>

#include "golden.hpp"
#include "numpp/core/dtype.hpp"
#include "numpp/core/ndarray.hpp"
#include "scypp/sparse/sparse.hpp"
#include "scypp_test.hpp"

namespace sp = scypp::sparse;
namespace {
numpp::ndarray vec(const double* d, int n) {
  numpp::ndarray a(numpp::Shape{n}, numpp::kFloat64);
  double* p = a.typed_data<double>();
  for (int i = 0; i < n; ++i) p[i] = d[i];
  return a;
}
numpp::ndarray mat(const double* d, int r, int c) {
  numpp::ndarray a(numpp::Shape{r, c}, numpp::kFloat64);
  double* p = a.typed_data<double>();
  for (int i = 0; i < r * c; ++i) p[i] = d[i];
  return a;
}
std::vector<double> tov(const numpp::ndarray& a) {
  numpp::ndarray c = a.astype(numpp::kFloat64).ascontiguousarray();
  const double* p = c.typed_data<double>();
  return std::vector<double>(p, p + c.size());
}
void cv(const numpp::ndarray& got, const double* exp, int n, double rtol = 1e-9, double atol = 1e-11) {
  auto g = tov(got);
  for (int i = 0; i < n && i < (int)g.size(); ++i) CHECK_CLOSE(g[i], exp[i], rtol, atol);
}
}  // namespace
#define M(name) mat(golden::name##_d, golden::name##_r, golden::name##_c)
#define G(name) golden::name, golden::name##_n

TEST_CASE("formats and constructors") {
  auto A = sp::CsrMatrix::from_dense(M(sp_A));
  cv(A.toarray(), golden::sp_A_toarray_d, golden::sp_A_toarray_r * golden::sp_A_toarray_c);
  CHECK(A.nnz() == 8);
  cv(A.diagonal(), G(sp_diag));
  cv(sp::eye(4).toarray(), golden::sp_eye_d, 16);
  double d0[] = {1, 2, 3, 4}, d1[] = {5, 6, 7};
  std::vector<numpp::ndarray> diags{vec(d0, 4), vec(d1, 3)};
  cv(sp::diags(diags, {0, 1}, 4).toarray(), golden::sp_diags_d, 16);
  cv(A.add(sp::eye(4)).toarray(), golden::sp_add_d, 16);
  // CSC round-trip via transpose
  auto At = A.transpose();
  CHECK(At.nnz() == 8);
}

TEST_CASE("products and backend dispatch") {
  auto A = sp::CsrMatrix::from_dense(M(sp_A));
  cv(A.spmv(vec(G(sp_x))), G(sp_spmv));
  cv(A.spmm(M(sp_X)), golden::sp_spmm_d, golden::sp_spmm_r * golden::sp_spmm_c);
  // dispatch: device-reference path equals CPU path, backend reported
  auto ycpu = tov(sp::spmv(A, vec(G(sp_x)), sp::Backend::Cpu));
  CHECK(sp::last_backend() == sp::Backend::Cpu);
  auto ydev = tov(sp::spmv(A, vec(G(sp_x)), sp::Backend::Device));
  CHECK(sp::last_backend() == sp::Backend::Device);
  for (size_t i = 0; i < ycpu.size(); ++i) CHECK_CLOSE(ycpu[i], ydev[i], 1e-12, 1e-12);
}

TEST_CASE("sparse solvers") {
  auto A = sp::CsrMatrix::from_dense(M(sp_A));
  auto b = vec(G(sp_b));
  cv(sp::spsolve(A, b), G(sp_spsolve), 1e-9, 1e-11);
  // iterative solvers drive the residual to zero (A is SPD here)
  auto xcg = sp::cg(A, b);
  cv(A.spmv(xcg), G(sp_b), 1e-5, 1e-7);
  auto xg = sp::gmres(A, b);
  cv(A.spmv(xg), G(sp_b), 1e-5, 1e-7);
  CHECK_CLOSE(sp::norm(A), golden::sp_norm_fro, 1e-9, 1e-11);
}

TEST_CASE("csgraph") {
  auto G = sp::CsrMatrix::from_dense(M(sp_G));
  cv(sp::csgraph::dijkstra(G, true), golden::sp_dijkstra_d, golden::sp_dijkstra_r * golden::sp_dijkstra_c);
  cv(sp::csgraph::floyd_warshall(G, true), golden::sp_floyd_d, golden::sp_floyd_r * golden::sp_floyd_c);
  cv(sp::csgraph::bellman_ford(G, true), golden::sp_dijkstra_d, golden::sp_dijkstra_r * golden::sp_dijkstra_c);
  auto cc = sp::csgraph::connected_components(sp::CsrMatrix::from_dense(M(sp_Gu)), false);
  CHECK(cc.n_components == static_cast<int>(golden::sp_ncomp));
  auto mst = sp::csgraph::minimum_spanning_tree(sp::CsrMatrix::from_dense(M(sp_Gm)));
  auto t = tov(mst.toarray());
  double w = 0; for (double v : t) w += v;
  CHECK_CLOSE(w, golden::sp_mst_weight, 1e-9, 1e-11);
}
