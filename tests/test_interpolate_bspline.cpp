// Oracle tests for scipp::interpolate B-splines against frozen SciPy golden data.
#include <cmath>
#include <vector>

#include "golden.hpp"
#include "numpp/core/dtype.hpp"
#include "numpp/core/ndarray.hpp"
#include "scipp/error.hpp"
#include "scipp/interpolate/interpolate.hpp"
#include "scipp_test.hpp"

namespace ip = scipp::interpolate;

namespace {
numpp::ndarray vv(const double* d, int n) {
  numpp::ndarray a(numpp::Shape{n}, numpp::kFloat64);
  double* p = a.typed_data<double>();
  for (int i = 0; i < n; ++i) p[i] = d[i];
  return a;
}
std::vector<double> tov(const numpp::ndarray& a) {
  numpp::ndarray c = a.astype(numpp::kFloat64).ascontiguousarray();
  const double* p = c.typed_data<double>();
  return std::vector<double>(p, p + c.size());
}
void check_vec(const numpp::ndarray& got, const double* exp, int n, double r, double a) {
  auto g = tov(got);
  CHECK(static_cast<int>(g.size()) == n);
  for (int i = 0; i < n && i < static_cast<int>(g.size()); ++i) CHECK_CLOSE(g[i], exp[i], r, a);
}

struct Tck {
  const double* t;
  int tn;
  const double* c;
  int cn;
  int k;
  const double* eval;
  const double* splev;
};
}  // namespace

#define BX vv(golden::bs_x, golden::bs_x_n)
#define BY vv(golden::bs_y, golden::bs_y_n)
#define BXS vv(golden::bs_xs, golden::bs_xs_n)

TEST_CASE("BSpline de Boor matches scipy") {
  const int nx = golden::bs_xs_n;
  Tck tcks[] = {
      {golden::bs_t1, golden::bs_t1_n, golden::bs_c1, golden::bs_c1_n, 1, golden::bs_eval1, golden::bs_splev1},
      {golden::bs_t2, golden::bs_t2_n, golden::bs_c2, golden::bs_c2_n, 2, golden::bs_eval2, golden::bs_splev2},
      {golden::bs_t3, golden::bs_t3_n, golden::bs_c3, golden::bs_c3_n, 3, golden::bs_eval3, golden::bs_splev3},
  };
  for (const auto& tk : tcks) {
    ip::BSpline sp(vv(tk.t, tk.tn), vv(tk.c, tk.cn), tk.k);
    CHECK(sp.k() == tk.k);
    CHECK(static_cast<int>(tov(sp.t()).size()) == tk.tn);
    // Pure de Boor on scipy's own (t, c, k): near machine precision.
    check_vec(sp(BXS), tk.eval, nx, 1e-9, 1e-11);
    // splev wrapper agrees with scipy.interpolate.splev on the same tck.
    check_vec(ip::splev(BXS, sp), tk.splev, nx, 1e-9, 1e-11);
  }
}

TEST_CASE("make_interp_spline matches scipy") {
  const int nx = golden::bs_xs_n;
  const double* evals[] = {golden::bs_eval1, golden::bs_eval2, golden::bs_eval3};
  for (int k = 1; k <= 3; ++k) {
    ip::BSpline sp = ip::make_interp_spline(BX, BY, k);
    // Reconstructs scipy's evaluation (including extrapolation region) to ~1e-7.
    check_vec(sp(BXS), evals[k - 1], nx, 1e-7, 1e-8);
    // Interpolates the data nodes exactly.
    auto g = tov(sp(BX));
    auto yv = tov(BY);
    for (size_t i = 0; i < yv.size(); ++i) CHECK_CLOSE(g[i], yv[i], 1e-9, 1e-10);
  }
}

TEST_CASE("BSpline input validation") {
  double t[] = {0, 0, 1, 2, 3, 3};
  double c[] = {1, 2, 3, 4};
  CHECK_THROWS_AS(ip::BSpline(vv(t, 6), vv(c, 4), -1), scipp::value_error);
  // len(t) too short for the claimed degree/coeffs.
  double tshort[] = {0, 1, 2};
  CHECK_THROWS_AS(ip::BSpline(vv(tshort, 3), vv(c, 4), 3), scipp::value_error);
}
