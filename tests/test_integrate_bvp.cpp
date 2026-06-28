// Oracle/regression tests for scypp::integrate::solve_bvp on two-point boundary
// value problems with known analytic solutions.
#include <vector>

#include "golden.hpp"
#include "numpp/core/dtype.hpp"
#include "numpp/core/ndarray.hpp"
#include "scypp/integrate/integrate.hpp"
#include "scypp_test.hpp"

namespace ig = scypp::integrate;
namespace {
numpp::ndarray vec(const double* d, int n) {
  numpp::ndarray a(numpp::Shape{n}, numpp::kFloat64);
  double* p = a.typed_data<double>();
  for (int i = 0; i < n; ++i) p[i] = d[i];
  return a;
}
numpp::ndarray vec(std::vector<double> d) { return vec(d.data(), (int)d.size()); }

// Build an (n, m) initial guess that is constant `val` for component c.
numpp::ndarray guess2(int m, double v0, double v1) {
  numpp::ndarray a(numpp::Shape{2, m}, numpp::kFloat64);
  double* p = a.typed_data<double>();
  for (int i = 0; i < m; ++i) { p[i] = v0; p[m + i] = v1; }
  return a;
}
}  // namespace

TEST_CASE("solve_bvp harmonic oscillator y''=-y -> sin x") {
  // y0' = y1, y1' = -y0 ; bc y0(0)=0, y0(pi/2)=1 ; analytic y0=sin x, y1=cos x.
  auto fun = [](double, const numpp::ndarray& y) {
    const double* p = y.typed_data<double>();
    return vec({p[1], -p[0]});
  };
  auto bc = [](const numpp::ndarray& ya, const numpp::ndarray& yb) {
    return vec({ya.typed_data<double>()[0], yb.typed_data<double>()[0] - 1.0});
  };
  int m = golden::bvp_sin_x_n;
  auto x = vec(golden::bvp_sin_x, m);
  auto r = ig::solve_bvp(fun, bc, x, guess2(m, 0.5, 0.5));
  CHECK(r.success);
  CHECK(r.y.shape()[0] == 2);
  CHECK(r.y.shape()[1] == m);
  const double* y = r.y.typed_data<double>();
  for (int i = 0; i < m; ++i) {
    // 11-node mesh: discretization error ~1e-6 (matches scipy at the nodes).
    CHECK_CLOSE(y[i], golden::bvp_sin_y0[i], 1e-5, 5e-6);          // sin x
    CHECK_CLOSE(y[m + i], golden::bvp_sin_y1[i], 1e-5, 5e-6);      // cos x
    // cross-check against scipy.solve_bvp's own solution at the nodes
    CHECK_CLOSE(y[i], golden::bvp_sin_scipy_d[i], 1e-4, 1e-5);
    CHECK_CLOSE(y[m + i], golden::bvp_sin_scipy_d[m + i], 1e-4, 1e-5);
  }
}

TEST_CASE("solve_bvp polynomial y''=6x -> x^3") {
  // y0' = y1, y1' = 6x ; bc y0(0)=0, y0(1)=1 ; analytic y0=x^3, y1=3x^2.
  auto fun = [](double x, const numpp::ndarray& y) {
    const double* p = y.typed_data<double>();
    return vec({p[1], 6.0 * x});
  };
  auto bc = [](const numpp::ndarray& ya, const numpp::ndarray& yb) {
    return vec({ya.typed_data<double>()[0], yb.typed_data<double>()[0] - 1.0});
  };
  int m = golden::bvp_cubic_x_n;
  auto x = vec(golden::bvp_cubic_x, m);
  auto r = ig::solve_bvp(fun, bc, x, guess2(m, 0.5, 1.0));
  CHECK(r.success);
  const double* y = r.y.typed_data<double>();
  // 4th-order collocation is exact for cubics; only Newton's finite-difference
  // Jacobian limits accuracy, leaving residual ~1e-10.
  for (int i = 0; i < m; ++i) {
    CHECK_CLOSE(y[i], golden::bvp_cubic_y0[i], 1e-7, 1e-9);        // x^3
    CHECK_CLOSE(y[m + i], golden::bvp_cubic_y1[i], 1e-7, 1e-9);    // 3x^2
  }
}
