// Oracle tests for scipp::interpolate::griddata (scattered 2-D interpolation).
#include <cmath>

#include "golden.hpp"
#include "numpp/core/dtype.hpp"
#include "numpp/core/ndarray.hpp"
#include "scipp/interpolate/interpolate.hpp"
#include "scipp_test.hpp"

namespace ip = scipp::interpolate;
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
}  // namespace
#define MAT(name) mat(golden::name##_d, golden::name##_r, golden::name##_c)
#define VEC(name) vec(golden::name, golden::name##_n)

TEST_CASE("griddata linear") {
  auto pts = MAT(gd_pts);
  auto xi = MAT(gd_xi);
  auto r = ip::griddata(pts, VEC(gd_val_lin), xi, "linear");
  const double* g = r.typed_data<double>();
  CHECK(r.size() == golden::gd_linear_n);
  for (int i = 0; i < golden::gd_linear_n; ++i) {
    if (std::isnan(golden::gd_linear[i])) {
      CHECK(std::isnan(g[i]));  // outside the convex hull -> fill_value
    } else {
      CHECK_CLOSE(g[i], golden::gd_linear[i], 1e-9, 1e-11);
    }
  }
}

TEST_CASE("griddata nearest") {
  auto pts = MAT(gd_pts);
  auto xi = MAT(gd_xi);
  auto r = ip::griddata(pts, VEC(gd_val_nl), xi, "nearest");
  const double* g = r.typed_data<double>();
  CHECK(r.size() == golden::gd_nearest_n);
  for (int i = 0; i < golden::gd_nearest_n; ++i)
    CHECK_CLOSE(g[i], golden::gd_nearest[i], 1e-12, 1e-12);
}
