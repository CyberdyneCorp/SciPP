// Oracle tests for nested/extended quadrature: romberg, quad_vec, dblquad,
// tplquad, nquad.
#include <cmath>
#include <vector>

#include "golden.hpp"
#include "numpp/core/dtype.hpp"
#include "numpp/core/ndarray.hpp"
#include "scypp/integrate/integrate.hpp"
#include "scypp_test.hpp"

namespace ig = scypp::integrate;

TEST_CASE("romberg") {
  CHECK_CLOSE(ig::romberg([](double x) { return std::sin(x); }, 0.0, M_PI),
              golden::romberg_sin, 1e-9, 1e-11);
  CHECK_CLOSE(ig::romberg([](double x) { return std::exp(-x * x); }, 0.0, 1.0),
              golden::romberg_gauss, 1e-9, 1e-11);
}

TEST_CASE("quad_vec") {
  auto r = ig::quad_vec(
      [](double x) {
        numpp::ndarray v(numpp::Shape{4}, numpp::kFloat64);
        double* p = v.typed_data<double>();
        p[0] = 1.0; p[1] = x; p[2] = x * x; p[3] = std::sin(x);
        return v;
      },
      0.0, 1.0);
  CHECK(r.size() == 4);
  const double* g = r.typed_data<double>();
  for (int i = 0; i < 4; ++i) CHECK_CLOSE(g[i], golden::quadvec[i], 1e-8, 1e-10);
}

TEST_CASE("dblquad") {
  // ∫_0^1 ∫_0^2 x*y dy dx ; func(y, x)
  auto d1 = ig::dblquad([](double y, double x) { return x * y; }, 0, 1,
                        [](double) { return 0.0; }, [](double) { return 2.0; });
  CHECK_CLOSE(d1.value, golden::dblquad_xy, 1e-9, 1e-11);
  // x-dependent upper bound: ∫_0^1 ∫_0^x (x+y) dy dx
  auto d2 = ig::dblquad([](double y, double x) { return x + y; }, 0, 1,
                        [](double) { return 0.0; }, [](double x) { return x; });
  CHECK_CLOSE(d2.value, golden::dblquad_var, 1e-9, 1e-11);
}

TEST_CASE("tplquad") {
  auto t = ig::tplquad([](double z, double y, double x) { return x * y * z; }, 0, 1,
                       [](double) { return 0.0; }, [](double) { return 2.0; },
                       [](double, double) { return 0.0; }, [](double, double) { return 3.0; });
  CHECK_CLOSE(t.value, golden::tplquad_xyz, 1e-9, 1e-11);
}

TEST_CASE("nquad") {
  auto n = ig::nquad(
      [](const std::vector<double>& x) { return x[0] * x[1] * x[2]; },
      {{0, 1}, {0, 2}, {0, 3}});
  CHECK_CLOSE(n.value, golden::nquad_3d, 1e-9, 1e-11);
}
