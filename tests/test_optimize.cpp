// Oracle tests for scypp::optimize against frozen SciPy golden data.
#include <cmath>
#include <vector>

#include "golden.hpp"
#include "numpp/core/dtype.hpp"
#include "numpp/core/ndarray.hpp"
#include "scypp/error.hpp"
#include "scypp/optimize/optimize.hpp"
#include "scypp_test.hpp"

namespace op = scypp::optimize;

namespace {
constexpr double R = 1e-6, A = 1e-8;

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
double at(const numpp::ndarray& a, int i) { return tov(a)[i]; }
}  // namespace

TEST_CASE("scalar root finding") {
  auto sq2 = [](double x) { return x * x - 2.0; };
  CHECK_CLOSE(op::brentq(sq2, 0.0, 2.0), golden::opt_brentq, 1e-10, 1e-12);
  CHECK_CLOSE(op::bisect(sq2, 0.0, 2.0), golden::opt_bisect, 1e-10, 1e-11);
  CHECK_CLOSE(op::newton([](double x) { return x * x * x - 2 * x - 5; }, 2.0),
              golden::opt_newton, 1e-9, 1e-11);
  // root identity and invalid bracket
  CHECK(std::fabs(sq2(op::brentq(sq2, 0.0, 2.0))) < 1e-10);
  CHECK_THROWS_AS(op::brentq(sq2, 2.0, 3.0), scypp::value_error);
}

TEST_CASE("scalar minimization") {
  auto m1 = op::minimize_scalar([](double x) { return std::cosh(x - 1.7); }, "brent");
  CHECK_CLOSE(m1.x, golden::opt_ms_brent_x, 1e-6, 1e-7);
  CHECK_CLOSE(m1.fun, golden::opt_ms_brent_f, 1e-8, 1e-9);
  auto m2 = op::minimize_scalar([](double x) { return (x - 2.0) * (x - 2.0); }, "bounded",
                                std::make_pair(0.0, 1.0));
  CHECK_CLOSE(m2.x, golden::opt_ms_bounded_x, 1e-5, 1e-6);
}

TEST_CASE("multivariate minimization") {
  auto rosen = [](const numpp::ndarray& p) {
    auto v = tov(p);
    return (1 - v[0]) * (1 - v[0]) + 100.0 * (v[1] - v[0] * v[0]) * (v[1] - v[0] * v[0]);
  };
  auto nm = op::minimize(rosen, vec({-1.2, 1.0}), "Nelder-Mead");
  CHECK(nm.success);
  CHECK_CLOSE(at(nm.x, 0), 1.0, 1e-3, 1e-3);
  CHECK_CLOSE(at(nm.x, 1), 1.0, 1e-3, 1e-3);

  auto quad = [](const numpp::ndarray& p) {
    auto v = tov(p);
    return (v[0] - 3.0) * (v[0] - 3.0) + (v[1] + 1.0) * (v[1] + 1.0);
  };
  auto bf = op::minimize(quad, vec({0.0, 0.0}), "BFGS");
  CHECK(bf.success);
  CHECK_CLOSE(at(bf.x, 0), golden::opt_bfgs_quad_d[0], 1e-4, 1e-5);
  CHECK_CLOSE(at(bf.x, 1), golden::opt_bfgs_quad_d[1], 1e-4, 1e-5);
  CHECK_CLOSE(bf.fun, 0.0, 1e-8, 1e-8);
}

TEST_CASE("curve_fit recovers parameters") {
  auto model = [](const numpp::ndarray& x, const numpp::ndarray& p) {
    auto xv = tov(x); auto pv = tov(p);
    std::vector<double> y(xv.size());
    for (size_t i = 0; i < xv.size(); ++i) y[i] = pv[0] * std::exp(-pv[1] * xv[i]) + pv[2];
    return vec(y);
  };
  auto xd = vec(std::vector<double>(golden::cf_xdata, golden::cf_xdata + golden::cf_xdata_n));
  auto yd = vec(std::vector<double>(golden::cf_ydata, golden::cf_ydata + golden::cf_ydata_n));
  auto cf = op::curve_fit(model, xd, yd, vec({1.0, 1.0, 1.0}));
  for (int i = 0; i < 3; ++i) {
    CHECK_CLOSE(at(cf.popt, i), golden::opt_curvefit_true_d[i], 1e-4, 1e-5);
    CHECK_CLOSE(at(cf.popt, i), golden::opt_curvefit_popt_d[i], 1e-4, 1e-5);
  }
  CHECK(cf.pcov.shape() == (numpp::Shape{3, 3}));
}

TEST_CASE("least_squares drives residual to zero") {
  auto resid = [](const numpp::ndarray& p) {
    auto v = tov(p);
    return vec({v[0] - 3.0, v[1] + 1.0, v[0] + v[1] - 2.0});
  };
  auto ls = op::least_squares(resid, vec({0.0, 0.0}));
  CHECK(ls.cost < 0.5 + 1e-6);  // overdetermined; least-squares optimum
  // exactly determined system has zero residual
  auto resid2 = [](const numpp::ndarray& p) {
    auto v = tov(p);
    return vec({v[0] - 3.0, v[1] + 1.0});
  };
  auto ls2 = op::least_squares(resid2, vec({0.0, 0.0}));
  CHECK_CLOSE(at(ls2.x, 0), 3.0, 1e-6, 1e-7);
  CHECK_CLOSE(at(ls2.x, 1), -1.0, 1e-6, 1e-7);
}

TEST_CASE("fsolve finds a vector root") {
  auto F = [](const numpp::ndarray& v) {
    auto x = tov(v);
    return vec({x[0] * x[0] + x[1] * x[1] - 4.0, x[0] - x[1]});
  };
  auto r = op::fsolve(F, vec({1.0, 1.0}));
  CHECK_CLOSE(at(r, 0), golden::opt_fsolve_d[0], 1e-7, 1e-9);
  CHECK_CLOSE(at(r, 1), golden::opt_fsolve_d[1], 1e-7, 1e-9);
  auto fr = tov(F(r));
  CHECK(std::fabs(fr[0]) < 1e-8 && std::fabs(fr[1]) < 1e-8);
}
