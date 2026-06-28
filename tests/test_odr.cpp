// Oracle tests for scypp::odr against frozen scipy.odr golden data.
#include <cmath>
#include <vector>

#include "golden.hpp"
#include "numpp/core/dtype.hpp"
#include "numpp/core/ndarray.hpp"
#include "scypp/odr/odr.hpp"
#include "scypp_test.hpp"

namespace odr = scypp::odr;

namespace {

numpp::ndarray vec(const double* p, int n) {
  numpp::ndarray a(numpp::Shape{static_cast<int64_t>(n)}, numpp::kFloat64);
  double* d = a.typed_data<double>();
  for (int i = 0; i < n; ++i) d[i] = p[i];
  return a;
}
numpp::ndarray vec(std::vector<double> v) { return vec(v.data(), static_cast<int>(v.size())); }

std::vector<double> tov(const numpp::ndarray& a) {
  numpp::ndarray c = a.astype(numpp::kFloat64).ascontiguousarray();
  const double* p = c.typed_data<double>();
  return std::vector<double>(p, p + c.size());
}

}  // namespace

TEST_CASE("odr linear fit matches scipy") {
  auto lin = [](const numpp::ndarray& b, const numpp::ndarray& x) {
    auto bv = tov(b);
    auto xv = tov(x);
    std::vector<double> y(xv.size());
    for (size_t i = 0; i < xv.size(); ++i) y[i] = bv[0] + bv[1] * xv[i];
    return vec(y);
  };
  odr::Data data(vec(golden::odr_lin_x, golden::odr_lin_x_n),
                 vec(golden::odr_lin_y, golden::odr_lin_y_n));
  odr::Output o = odr::ODR(data, odr::Model(lin), vec({1.0, 1.0})).run();
  auto beta = tov(o.beta);
  CHECK(o.success);
  CHECK_CLOSE(beta[0], golden::odr_lin_beta[0], 1e-5, 1e-5);
  CHECK_CLOSE(beta[1], golden::odr_lin_beta[1], 1e-5, 1e-5);
  CHECK_CLOSE(o.res_var, golden::odr_lin_resvar, 1e-4, 1e-4);
  CHECK_CLOSE(o.sum_square, golden::odr_lin_ss, 1e-4, 1e-4);
  auto sd = tov(o.sd_beta);
  CHECK_CLOSE(sd[0], golden::odr_lin_sdbeta[0], 1e-2, 1e-3);
  CHECK_CLOSE(sd[1], golden::odr_lin_sdbeta[1], 1e-2, 1e-3);
}

TEST_CASE("odr nonlinear fit matches scipy") {
  auto expm = [](const numpp::ndarray& b, const numpp::ndarray& x) {
    auto bv = tov(b);
    auto xv = tov(x);
    std::vector<double> y(xv.size());
    for (size_t i = 0; i < xv.size(); ++i) y[i] = bv[0] * std::exp(bv[1] * xv[i]);
    return vec(y);
  };
  odr::Data data(vec(golden::odr_nl_x, golden::odr_nl_x_n),
                 vec(golden::odr_nl_y, golden::odr_nl_y_n));
  odr::Output o = odr::ODR(data, odr::Model(expm), vec({1.0, 1.0})).run();
  auto beta = tov(o.beta);
  CHECK(o.success);
  CHECK_CLOSE(beta[0], golden::odr_nl_beta[0], 1e-5, 1e-5);
  CHECK_CLOSE(beta[1], golden::odr_nl_beta[1], 1e-5, 1e-5);
  CHECK_CLOSE(o.res_var, golden::odr_nl_resvar, 1e-4, 1e-4);
  CHECK_CLOSE(o.sum_square, golden::odr_nl_ss, 1e-4, 1e-4);
  auto sd = tov(o.sd_beta);
  CHECK_CLOSE(sd[0], golden::odr_nl_sdbeta[0], 1e-2, 1e-3);
  CHECK_CLOSE(sd[1], golden::odr_nl_sdbeta[1], 1e-2, 1e-3);
}

TEST_CASE("odr weighted fit matches scipy") {
  auto lin = [](const numpp::ndarray& b, const numpp::ndarray& x) {
    auto bv = tov(b);
    auto xv = tov(x);
    std::vector<double> y(xv.size());
    for (size_t i = 0; i < xv.size(); ++i) y[i] = bv[0] + bv[1] * xv[i];
    return vec(y);
  };
  odr::Data data(vec(golden::odr_w_x, golden::odr_w_x_n),
                 vec(golden::odr_w_y, golden::odr_w_y_n),
                 vec(golden::odr_w_sx, golden::odr_w_sx_n),
                 vec(golden::odr_w_sy, golden::odr_w_sy_n));
  odr::Output o = odr::ODR(data, odr::Model(lin), vec({0.0, 1.0})).run();
  auto beta = tov(o.beta);
  CHECK(o.success);
  CHECK_CLOSE(beta[0], golden::odr_w_beta[0], 1e-5, 1e-5);
  CHECK_CLOSE(beta[1], golden::odr_w_beta[1], 1e-5, 1e-5);
  CHECK_CLOSE(o.res_var, golden::odr_w_resvar, 1e-4, 1e-4);
  CHECK_CLOSE(o.sum_square, golden::odr_w_ss, 1e-4, 1e-4);
  auto sd = tov(o.sd_beta);
  CHECK_CLOSE(sd[0], golden::odr_w_sdbeta[0], 1e-2, 1e-3);
  CHECK_CLOSE(sd[1], golden::odr_w_sdbeta[1], 1e-2, 1e-3);
}
