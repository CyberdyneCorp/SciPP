// Oracle tests for scipp::integrate::solve_ivp method="BDF" on stiff systems.
// Validated against analytic solutions (not SciPy's exact trajectory).
#include <vector>

#include "golden.hpp"
#include "numpp/core/dtype.hpp"
#include "numpp/core/ndarray.hpp"
#include "scipp/integrate/integrate.hpp"
#include "scipp_test.hpp"

namespace ig = scipp::integrate;
namespace {
numpp::ndarray vec(const double* d, int n) {
  numpp::ndarray a(numpp::Shape{n}, numpp::kFloat64);
  double* p = a.typed_data<double>();
  for (int i = 0; i < n; ++i) p[i] = d[i];
  return a;
}
numpp::ndarray vec(std::vector<double> d) { return vec(d.data(), (int)d.size()); }
}  // namespace

TEST_CASE("solve_ivp BDF stiff scalar") {
  // y' = -20 y, y0 = 1  (exact exp(-20 t)).
  auto f = [](double, const numpp::ndarray& y) {
    return vec({-20.0 * y.typed_data<double>()[0]});
  };
  auto te = vec(golden::ode_bdf1_t, golden::ode_bdf1_t_n);
  auto r = ig::solve_ivp(f, {0.0, 1.0}, vec({1.0}), "BDF", te, 1e-8, 1e-11);
  CHECK(r.success);
  const double* y = r.y.typed_data<double>();
  int m = r.y.shape()[1];
  CHECK(m == golden::ode_bdf1_y_n);
  for (int j = 0; j < m; ++j)
    CHECK_CLOSE(y[j], golden::ode_bdf1_y[j], 1e-5, 1e-9);
}

TEST_CASE("solve_ivp BDF stiff 2-D system") {
  // y1' = -100 y1 + y2, y2' = -y2, y0 = [1, 1].
  auto f = [](double, const numpp::ndarray& y) {
    const double* p = y.typed_data<double>();
    return vec({-100.0 * p[0] + p[1], -p[1]});
  };
  auto te = vec(golden::ode_bdf2_t, golden::ode_bdf2_t_n);
  auto r = ig::solve_ivp(f, {0.0, 1.0}, vec({1.0, 1.0}), "BDF", te, 1e-9, 1e-12);
  CHECK(r.success);
  const double* y = r.y.typed_data<double>();
  int rows = golden::ode_bdf2_y_r, cols = golden::ode_bdf2_y_c;
  CHECK(r.y.shape()[0] == rows);
  CHECK(r.y.shape()[1] == cols);
  for (int i = 0; i < rows; ++i)
    for (int j = 0; j < cols; ++j)
      CHECK_CLOSE(y[i * cols + j], golden::ode_bdf2_y_d[i * cols + j], 1e-5, 1e-8);
}
