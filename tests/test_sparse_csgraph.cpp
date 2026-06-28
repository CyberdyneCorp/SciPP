// Oracle tests for scypp::sparse::csgraph traversal + flow algorithms.
#include <cmath>
#include <vector>

#include "golden.hpp"
#include "numpp/core/dtype.hpp"
#include "numpp/core/ndarray.hpp"
#include "scypp/sparse/sparse.hpp"
#include "scypp_test.hpp"

namespace sp = scypp::sparse;
namespace {
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
// Exact integer/index comparison against a golden vector.
void ci(const numpp::ndarray& got, const double* exp, int n) {
  auto g = tov(got);
  CHECK(static_cast<int>(g.size()) == n);
  for (int i = 0; i < n && i < (int)g.size(); ++i) CHECK(std::llround(g[i]) == std::llround(exp[i]));
}
void cd(const numpp::ndarray& got, const double* exp, int n, double rtol = 1e-12, double atol = 1e-12) {
  auto g = tov(got);
  for (int i = 0; i < n && i < (int)g.size(); ++i) {
    if (std::isinf(exp[i])) CHECK(std::isinf(g[i]) && (g[i] > 0) == (exp[i] > 0));
    else CHECK_CLOSE(g[i], exp[i], rtol, atol);
  }
}
}  // namespace
#define M(name) mat(golden::name##_d, golden::name##_r, golden::name##_c)

TEST_CASE("csgraph traversal order") {
  auto Gt = sp::CsrMatrix::from_dense(M(sp_Gt));
  auto bfs = sp::csgraph::breadth_first_order(Gt, 0, true);
  ci(bfs.node_array, golden::sp_bfs_node, golden::sp_bfs_node_n);
  ci(bfs.predecessors, golden::sp_bfs_pred, golden::sp_bfs_pred_n);
  auto dfs = sp::csgraph::depth_first_order(Gt, 0, true);
  ci(dfs.node_array, golden::sp_dfs_node, golden::sp_dfs_node_n);
  ci(dfs.predecessors, golden::sp_dfs_pred, golden::sp_dfs_pred_n);

  auto Gtu = sp::CsrMatrix::from_dense(M(sp_Gtu));
  auto bfsu = sp::csgraph::breadth_first_order(Gtu, 0, false);
  ci(bfsu.node_array, golden::sp_bfsu_node, golden::sp_bfsu_node_n);
  ci(bfsu.predecessors, golden::sp_bfsu_pred, golden::sp_bfsu_pred_n);
  auto dfsu = sp::csgraph::depth_first_order(Gtu, 0, false);
  ci(dfsu.node_array, golden::sp_dfsu_node, golden::sp_dfsu_node_n);
  ci(dfsu.predecessors, golden::sp_dfsu_pred, golden::sp_dfsu_pred_n);
}

TEST_CASE("csgraph johnson") {
  auto Gj = sp::CsrMatrix::from_dense(M(sp_Gj));
  cd(sp::csgraph::johnson(Gj, true), golden::sp_johnson_d, golden::sp_johnson_r * golden::sp_johnson_c);
}

TEST_CASE("csgraph maximum flow") {
  auto Gf = sp::CsrMatrix::from_dense(M(sp_Gf));
  auto r = sp::csgraph::maximum_flow(Gf, 0, 6);
  CHECK(r.flow_value == std::llround(golden::sp_flow_value));
  // flow conservation: net flow leaving the source equals the flow value
  auto f = tov(r.flow.toarray());
  int n = golden::sp_Gf_r;
  double out0 = 0;
  for (int j = 0; j < n; ++j) out0 += f[0 * n + j];
  CHECK_CLOSE(out0, static_cast<double>(r.flow_value), 1e-12, 1e-12);
}

TEST_CASE("csgraph maximum bipartite matching") {
  auto Gb = sp::CsrMatrix::from_dense(M(sp_Gb));
  ci(sp::csgraph::maximum_bipartite_matching(Gb, "row"), golden::sp_bm_row, golden::sp_bm_row_n);
  ci(sp::csgraph::maximum_bipartite_matching(Gb, "column"), golden::sp_bm_col, golden::sp_bm_col_n);
  auto Gb2 = sp::CsrMatrix::from_dense(M(sp_Gb2));
  ci(sp::csgraph::maximum_bipartite_matching(Gb2, "column"), golden::sp_bm2_col, golden::sp_bm2_col_n);
}
