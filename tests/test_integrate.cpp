// Oracle tests for scypp::integrate and scypp::differentiate.
#include <cmath>
#include <vector>

#include "golden.hpp"
#include "numpp/core/dtype.hpp"
#include "numpp/core/ndarray.hpp"
#include "scypp/differentiate/differentiate.hpp"
#include "scypp/integrate/integrate.hpp"
#include "scypp_test.hpp"

namespace ig = scypp::integrate;
namespace df = scypp::differentiate;

namespace {
constexpr double kPi = 3.141592653589793238462643383279502884;

numpp::ndarray vec(std::vector<double> v) {
  numpp::ndarray a(numpp::Shape{static_cast<int64_t>(v.size())}, numpp::kFloat64);
  double* p = a.typed_data<double>();
  for (size_t i = 0; i < v.size(); ++i) p[i] = v[i];
  return a;
}
std::vector<double> tov(const numpp::ndarray& a) {
  numpp::ndarray c = a.astype(numpp::kFloat64).ascontiguousarray();
  const double* p = c.typed_data<double>();
  return std::vector<double>(p, p + c.size());
}
}  // namespace

TEST_CASE("fixed-sample quadrature") {
  auto x = vec(std::vector<double>(golden::ig_x, golden::ig_x + golden::ig_x_n));
  auto y = vec(std::vector<double>(golden::ig_y, golden::ig_y + golden::ig_y_n));
  CHECK_CLOSE(ig::trapezoid(y, x), golden::ig_trapz, 1e-12, 1e-12);
  CHECK_CLOSE(ig::simpson(y, x), golden::ig_simpson, 1e-10, 1e-12);
  auto ct = tov(ig::cumulative_trapezoid(y, x, 1.0, 0.0));
  for (int i = 0; i < golden::ig_cumtrapz_n; ++i) CHECK_CLOSE(ct[i], golden::ig_cumtrapz[i], 1e-12, 1e-12);
  CHECK_CLOSE(ct.back(), ig::trapezoid(y, x), 1e-12, 1e-12);
}

TEST_CASE("adaptive and fixed quadrature") {
  auto pi_integrand = [](double t) { return 4.0 / (1.0 + t * t); };
  ig::QuadResult q = ig::quad(pi_integrand, 0.0, 1.0);
  CHECK_CLOSE(q.value, kPi, 1e-10, 1e-12);
  CHECK_CLOSE(q.value, golden::ig_quad_pi, 1e-10, 1e-12);
  CHECK(std::fabs(q.value - kPi) <= std::max(q.abserr, 1e-12));  // error estimate bounds true error
  CHECK_CLOSE(ig::quad([](double t) { return std::exp(-t * t); }, 0.0, 2.0).value,
              golden::ig_quad_exp, 1e-9, 1e-11);
  CHECK_CLOSE(ig::fixed_quad([](double t) { return t * t * t; }, 0.0, 2.0, 5), 4.0, 1e-10, 1e-11);
}

TEST_CASE("solve_ivp scalar exponential decay") {
  auto f = [](double, const numpp::ndarray& y) { return vec({-tov(y)[0]}); };
  std::vector<double> te;
  for (int i = 0; i <= 10; ++i) te.push_back(0.5 * i);
  for (const char* m : {"RK45", "RK23"}) {
    auto r = ig::solve_ivp(f, {0.0, 5.0}, vec({1.0}), m, vec(te), 1e-8, 1e-10);
    CHECK(r.success);
    auto yv = tov(r.y);
    for (size_t i = 0; i < te.size(); ++i) CHECK_CLOSE(yv[i], std::exp(-te[i]), 1e-6, 1e-8);
  }
}

TEST_CASE("solve_ivp harmonic oscillator system") {
  // y0' = y1, y1' = -y0  ->  y0 = cos t, y1 = -sin t
  auto f = [](double, const numpp::ndarray& y) {
    auto v = tov(y);
    return vec({v[1], -v[0]});
  };
  std::vector<double> te = {0.0, 1.0, 2.0, 3.0, 4.0};
  auto r = ig::solve_ivp(f, {0.0, 4.0}, vec({1.0, 0.0}), "RK45", vec(te), 1e-9, 1e-11);
  CHECK(r.success);
  auto yv = tov(r.y);  // shape (2, 5), row-major
  int T = static_cast<int>(te.size());
  for (int i = 0; i < T; ++i) {
    CHECK_CLOSE(yv[0 * T + i], std::cos(te[i]), 1e-6, 1e-8);
    CHECK_CLOSE(yv[1 * T + i], -std::sin(te[i]), 1e-6, 1e-8);
  }
}

TEST_CASE("differentiation") {
  CHECK_CLOSE(df::derivative([](double x) { return std::sin(x); }, 1.0).df, std::cos(1.0), 1e-9, 1e-11);
  CHECK_CLOSE(df::derivative([](double x) { return x * x * x; }, 2.0).df, 12.0, 1e-8, 1e-9);

  auto F = [](const numpp::ndarray& x) {
    auto v = tov(x);
    return vec({v[0] * v[0], v[0] * v[1]});
  };
  auto J = tov(df::jacobian(F, vec({3.0, 2.0})));  // [[2x0,0],[x1,x0]] = [[6,0],[2,3]]
  CHECK_CLOSE(J[0], 6.0, 1e-6, 1e-7); CHECK_CLOSE(J[1], 0.0, 1e-6, 1e-7);
  CHECK_CLOSE(J[2], 2.0, 1e-6, 1e-7); CHECK_CLOSE(J[3], 3.0, 1e-6, 1e-7);

  auto g = [](const numpp::ndarray& x) {
    auto v = tov(x);
    return v[0] * v[0] + v[0] * v[1] + 2.0 * v[1] * v[1];
  };
  auto H = tov(df::hessian(g, vec({1.0, 1.0})));  // [[2,1],[1,4]]
  CHECK_CLOSE(H[0], 2.0, 1e-4, 1e-5); CHECK_CLOSE(H[1], 1.0, 1e-4, 1e-5);
  CHECK_CLOSE(H[2], 1.0, 1e-4, 1e-5); CHECK_CLOSE(H[3], 4.0, 1e-4, 1e-5);
}
