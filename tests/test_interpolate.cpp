// Oracle tests for scypp::interpolate against frozen SciPy golden data.
#include <cmath>
#include <vector>

#include "golden.hpp"
#include "numpp/core/dtype.hpp"
#include "numpp/core/ndarray.hpp"
#include "scypp/interpolate/interpolate.hpp"
#include "scypp_test.hpp"

namespace ip = scypp::interpolate;

namespace {
constexpr double R = 1e-9, A = 1e-11;

numpp::ndarray v1(const double* d, int n) {
  numpp::ndarray a(numpp::Shape{n}, numpp::kFloat64);
  double* p = a.typed_data<double>();
  for (int i = 0; i < n; ++i) p[i] = d[i];
  return a;
}
numpp::ndarray m2(const double* d, int r, int c) {
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
void check_vec(const numpp::ndarray& got, const double* exp, int n, double rtol = R, double atol = A) {
  auto g = tov(got);
  CHECK(static_cast<int>(g.size()) == n);
  for (int i = 0; i < n && i < static_cast<int>(g.size()); ++i) CHECK_CLOSE(g[i], exp[i], rtol, atol);
}
}  // namespace

#define X v1(golden::ip_x, golden::ip_x_n)
#define Y v1(golden::ip_y, golden::ip_y_n)
#define Q v1(golden::ip_q, golden::ip_q_n)
#define VG(name) golden::name, golden::name##_n

TEST_CASE("interp1d") {
  ip::Interp1d lin(X, Y, "linear");
  check_vec(lin(Q), VG(ip_linear));
  ip::Interp1d nr(X, Y, "nearest");
  check_vec(nr(Q), VG(ip_nearest));
  // reproduces samples
  CHECK_CLOSE(lin(2.0), std::sin(2.0), 1e-12, 1e-12);
  // out-of-bounds fill
  ip::Interp1d f(X, Y, "linear", -99.0);
  CHECK_CLOSE(f(100.0), -99.0, 0, 0);
}

TEST_CASE("CubicSpline") {
  ip::CubicSpline cs(X, Y);  // not-a-knot default
  check_vec(cs(Q), VG(ip_cubic));
  check_vec(cs(Q, 1), VG(ip_cubic_d1));
  ip::CubicSpline nat(X, Y, "natural");
  check_vec(nat(Q), VG(ip_cubic_nat));
  // passes through the nodes
  CHECK_CLOSE(cs(3.0), std::sin(3.0), 1e-10, 1e-11);
}

TEST_CASE("PCHIP and Akima") {
  check_vec(ip::PchipInterpolator(X, Y)(Q), VG(ip_pchip));
  check_vec(ip::Akima1DInterpolator(X, Y)(Q), VG(ip_akima));
}

TEST_CASE("RegularGridInterpolator and interpn") {
  double gx[] = {0, 1, 2, 3}, gy[] = {0, 1, 2};
  std::vector<numpp::ndarray> pts{v1(gx, 4), v1(gy, 3)};
  auto vals = m2(golden::ip_gv_d, golden::ip_gv_r, golden::ip_gv_c);
  auto q = m2(golden::ip_rgi_q_d, golden::ip_rgi_q_r, golden::ip_rgi_q_c);
  ip::RegularGridInterpolator lin(pts, vals, "linear");
  check_vec(lin(q), VG(ip_rgi_lin), 1e-10, 1e-11);
  ip::RegularGridInterpolator nr(pts, vals, "nearest");
  check_vec(nr(q), VG(ip_rgi_near), 1e-10, 1e-11);
  // interpn wrapper agrees
  check_vec(ip::interpn(pts, vals, q, "linear"), VG(ip_rgi_lin), 1e-10, 1e-11);
}

TEST_CASE("RBFInterpolator") {
  auto y = m2(golden::ip_rbf_y_d, golden::ip_rbf_y_r, golden::ip_rbf_y_c);
  auto d = v1(golden::ip_rbf_d, golden::ip_rbf_d_n);
  auto q = m2(golden::ip_rbf_q_d, golden::ip_rbf_q_r, golden::ip_rbf_q_c);
  check_vec(ip::RBFInterpolator(y, d, "thin_plate_spline")(q), VG(ip_rbf_tps), 1e-7, 1e-9);
  check_vec(ip::RBFInterpolator(y, d, "cubic")(q), VG(ip_rbf_cubic), 1e-7, 1e-9);
  check_vec(ip::RBFInterpolator(y, d, "linear")(q), VG(ip_rbf_lin), 1e-7, 1e-9);
  // reproduces values at the centers
  auto self = tov(ip::RBFInterpolator(y, d, "thin_plate_spline")(y));
  auto dv = tov(d);
  for (size_t i = 0; i < dv.size(); ++i) CHECK_CLOSE(self[i], dv[i], 1e-7, 1e-8);
}
