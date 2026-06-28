// Oracle tests for misc special functions: lambertw, zeta/zetac, struve,
// modstruve, spence.
#include "golden.hpp"
#include "scipp/special/special.hpp"
#include "scipp_test.hpp"

namespace sp = scipp::special;

namespace {
constexpr double R = 1e-10, A = 1e-12;
}  // namespace

TEST_CASE("Lambert W real branches") {
  for (int i = 0; i < golden::lw0_x_n; ++i)
    CHECK_CLOSE(sp::lambertw(golden::lw0_x[i], 0), golden::lw0[i], R, A);
  for (int i = 0; i < golden::lwm1_x_n; ++i)
    CHECK_CLOSE(sp::lambertw(golden::lwm1_x[i], -1), golden::lwm1[i], R, A);
  // Defining identity: W(x) e^{W(x)} = x on both branches.
  for (double x : {0.5, 2.0, 10.0}) {
    double w = sp::lambertw(x, 0);
    CHECK_CLOSE(w * std::exp(w), x, 1e-12, 1e-12);
  }
  // Out of domain returns nan; principal branch passes through the origin.
  CHECK(std::isnan(sp::lambertw(-0.5, 0)));
  CHECK(std::isnan(sp::lambertw(0.0, -1)));
  CHECK(std::isnan(sp::lambertw(-0.5, -1)));
  CHECK_CLOSE(sp::lambertw(0.0, 0), 0.0, R, A);
}

TEST_CASE("Riemann zeta and zetac") {
  for (int i = 0; i < golden::zeta_x_n; ++i) {
    CHECK_CLOSE(sp::zeta(golden::zeta_x[i]), golden::zeta_v[i], R, A);
    CHECK_CLOSE(sp::zetac(golden::zeta_x[i]), golden::zetac_v[i], R, A);
  }
  // Pole at x = 1 and the trivial zero at x = -2.
  CHECK(std::isinf(sp::zeta(1.0)) && sp::zeta(1.0) > 0.0);
  CHECK_CLOSE(sp::zeta(-2.0), 0.0, R, A);
  CHECK_CLOSE(sp::zetac(2.0), sp::zeta(2.0) - 1.0, R, A);
}

TEST_CASE("Struve and modified Struve") {
  const double* hs[] = {golden::struve_h0, golden::struve_h1, golden::struve_h2};
  for (int v = 0; v < 3; ++v)
    for (int i = 0; i < golden::struve_x_n; ++i)
      CHECK_CLOSE(sp::struve(v, golden::struve_x[i]), hs[v][i], R, A);
  for (int i = 0; i < golden::struve_x_n; ++i)
    CHECK_CLOSE(sp::struve(0.5, golden::struve_x[i]), golden::struve_hhalf[i], R, A);

  const double* ls[] = {golden::mstruve_l0, golden::mstruve_l1, golden::mstruve_l2};
  for (int v = 0; v < 3; ++v)
    for (int i = 0; i < golden::mstruve_x_n; ++i)
      CHECK_CLOSE(sp::modstruve(v, golden::mstruve_x[i]), ls[v][i], R, A);
  for (int i = 0; i < golden::mstruve_x_n; ++i)
    CHECK_CLOSE(sp::modstruve(0.5, golden::mstruve_x[i]), golden::mstruve_lhalf[i], R, A);

  // Parity for integer order: H_v(-x) = (-1)^{v+1} H_v(x); same for L_v.
  CHECK_CLOSE(sp::struve(0, -2.0), -sp::struve(0, 2.0), R, A);
  CHECK_CLOSE(sp::struve(1, -2.0), sp::struve(1, 2.0), R, A);
  CHECK_CLOSE(sp::modstruve(0, -2.0), -sp::modstruve(0, 2.0), R, A);
  // x = 0 limit and non-integer order at x < 0 (nan, as SciPy).
  CHECK_CLOSE(sp::struve(0, 0.0), 0.0, R, A);
  CHECK(std::isnan(sp::struve(0.5, -1.0)));
}

TEST_CASE("Spence dilogarithm") {
  CHECK_ARR(spence, [](double x) { return sp::spence(x); }, R, A);
  // Reference points: spence(0) = pi^2/6, spence(1) = 0, spence(2) = -pi^2/12.
  CHECK_CLOSE(sp::spence(1.0), 0.0, R, A);
  CHECK_CLOSE(sp::spence(2.0), -3.14159265358979323846 * 3.14159265358979323846 / 12.0, R, A);
  CHECK(std::isnan(sp::spence(-1.0)));
}
