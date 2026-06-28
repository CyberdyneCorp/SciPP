// Oracle tests for scypp::optimize linear programming and NNLS.
#include <vector>

#include "golden.hpp"
#include "numpp/core/dtype.hpp"
#include "numpp/core/ndarray.hpp"
#include "scypp/optimize/optimize.hpp"
#include "scypp_test.hpp"

namespace opt = scypp::optimize;
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
void check_vec(const numpp::ndarray& got, const double* want, int n, double rt, double at) {
  CHECK(got.size() == n);
  const double* g = got.typed_data<double>();
  for (int i = 0; i < n; ++i) CHECK_CLOSE(g[i], want[i], rt, at);
}
}  // namespace
#define MAT(name) mat(golden::name##_d, golden::name##_r, golden::name##_c)
#define VEC(name) vec(golden::name, golden::name##_n)

TEST_CASE("linprog problems") {
  // LP1: min -x0-2x1 s.t. x0+x1<=4, x0+3x1<=6
  double A1[] = {1, 1, 1, 3};
  double b1[] = {4, 6};
  auto lp1 = opt::linprog(vec((const double[]){-1, -2}, 2), mat(A1, 2, 2), vec(b1, 2));
  CHECK(lp1.success);
  CHECK_CLOSE(lp1.fun, golden::lp1_fun, 1e-9, 1e-11);
  check_vec(lp1.x, golden::lp1_x, golden::lp1_x_n, 1e-9, 1e-9);

  // LP2: min x0+x1 s.t. -x0-2x1<=-3, x0+x1=2
  double A2u[] = {-1, -2};
  double b2u[] = {-3};
  double A2e[] = {1, 1};
  double b2e[] = {2};
  auto lp2 = opt::linprog(vec((const double[]){1, 1}, 2), mat(A2u, 1, 2), vec(b2u, 1),
                          mat(A2e, 1, 2), vec(b2e, 1));
  CHECK(lp2.success);
  CHECK_CLOSE(lp2.fun, golden::lp2_fun, 1e-9, 1e-11);
  check_vec(lp2.x, golden::lp2_x, golden::lp2_x_n, 1e-9, 1e-9);

  // LP3: min -3x0-2x1-4x2
  double A3[] = {1, 1, 1, 2, 1, 0, 0, 1, 2};
  double b3[] = {10, 8, 12};
  auto lp3 = opt::linprog(vec((const double[]){-3, -2, -4}, 3), mat(A3, 3, 3), vec(b3, 3));
  CHECK(lp3.success);
  CHECK_CLOSE(lp3.fun, golden::lp3_fun, 1e-9, 1e-11);
  check_vec(lp3.x, golden::lp3_x, golden::lp3_x_n, 1e-9, 1e-9);
}

TEST_CASE("linprog infeasible / unbounded") {
  // x0<=1 and -x0<=-3 (x0>=3) is infeasible
  double Ai[] = {1, -1};
  double bi[] = {1, -3};
  auto inf = opt::linprog(vec((const double[]){1}, 1), mat(Ai, 2, 1), vec(bi, 2));
  CHECK(!inf.success);
  CHECK(inf.status == 2);
  // min -x0 with no upper bound is unbounded
  auto unb = opt::linprog(vec((const double[]){-1}, 1));
  CHECK(!unb.success);
  CHECK(unb.status == 3);
}

TEST_CASE("nnls Lawson-Hanson") {
  auto n1 = opt::nnls(MAT(nnls_A), VEC(nnls_b));
  check_vec(n1.x, golden::nnls_x, golden::nnls_x_n, 1e-8, 1e-9);
  CHECK_CLOSE(n1.rnorm, golden::nnls_rnorm, 1e-9, 1e-11);
  auto n2 = opt::nnls(MAT(nnls_A2), VEC(nnls_b2));
  check_vec(n2.x, golden::nnls_x2, golden::nnls_x2_n, 1e-8, 1e-9);
  CHECK_CLOSE(n2.rnorm, golden::nnls_rnorm2, 1e-9, 1e-11);
}
