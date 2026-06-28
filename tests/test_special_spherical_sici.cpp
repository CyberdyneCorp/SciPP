// Oracle tests for spherical Bessel functions and sine/cosine integrals.
#include "golden.hpp"
#include "scypp/special/special.hpp"
#include "scypp_test.hpp"

namespace sp = scypp::special;

namespace {
constexpr double R = 1e-10, A = 1e-12;
}  // namespace

TEST_CASE("spherical Bessel functions") {
  struct Row {
    const double* jn; const double* yn; const double* in; const double* kn;
  };
  const Row rows[] = {
      {golden::sb_jn0, golden::sb_yn0, golden::sb_in0, golden::sb_kn0},
      {golden::sb_jn1, golden::sb_yn1, golden::sb_in1, golden::sb_kn1},
      {golden::sb_jn2, golden::sb_yn2, golden::sb_in2, golden::sb_kn2},
      {golden::sb_jn3, golden::sb_yn3, golden::sb_in3, golden::sb_kn3},
      {golden::sb_jn4, golden::sb_yn4, golden::sb_in4, golden::sb_kn4}};
  for (int n = 0; n < 5; ++n) {
    const Row& r = rows[n];
    for (int i = 0; i < golden::sb_x_n; ++i) {
      const double x = golden::sb_x[i];
      CHECK_CLOSE(sp::spherical_jn(n, x), r.jn[i], R, A);
      CHECK_CLOSE(sp::spherical_yn(n, x), r.yn[i], R, A);
      CHECK_CLOSE(sp::spherical_in(n, x), r.in[i], R, A);
      CHECK_CLOSE(sp::spherical_kn(n, x), r.kn[i], R, A);
    }
  }
  // x = 0 limits (SciPy): jn/in finite, yn = -inf, kn = +inf.
  CHECK_CLOSE(sp::spherical_jn(0, 0.0), 1.0, R, A);
  CHECK_CLOSE(sp::spherical_jn(2, 0.0), 0.0, R, A);
  CHECK_CLOSE(sp::spherical_in(0, 0.0), 1.0, R, A);
  CHECK_CLOSE(sp::spherical_in(3, 0.0), 0.0, R, A);
  CHECK(std::isinf(sp::spherical_yn(0, 0.0)) && sp::spherical_yn(0, 0.0) < 0.0);
  CHECK(std::isinf(sp::spherical_kn(1, 0.0)) && sp::spherical_kn(1, 0.0) > 0.0);
  // Closed forms: j_0(x) = sin x / x, y_0(x) = -cos x / x.
  CHECK_CLOSE(sp::spherical_jn(0, 1.3), std::sin(1.3) / 1.3, R, A);
  CHECK_CLOSE(sp::spherical_yn(0, 1.3), -std::cos(1.3) / 1.3, R, A);
}

TEST_CASE("sine and cosine integrals") {
  // Ci oscillates through zero as it decays, so its meaningful accuracy is
  // absolute; near a zero (e.g. x = 25) the asymptotic floor is ~4e-12.
  constexpr double Aci = 1e-11;
  for (int i = 0; i < golden::sci_x_n; ++i) {
    sp::sici_t s = sp::sici(golden::sci_x[i]);
    CHECK_CLOSE(s.Si, golden::sci_Si[i], R, A);
    CHECK_CLOSE(s.Ci, golden::sci_Ci[i], R, Aci);
  }
  // Si(0) = 0, Ci(0) = -inf; for large x, Si(x) -> pi/2 (correction ~ 1/x).
  sp::sici_t z = sp::sici(0.0);
  CHECK_CLOSE(z.Si, 0.0, R, A);
  CHECK(std::isinf(z.Ci) && z.Ci < 0.0);
  CHECK_CLOSE(sp::sici(1e6).Si, 1.57079632679489661923, 1e-5, 1e-5);
}

TEST_CASE("hyperbolic sine and cosine integrals") {
  for (int i = 0; i < golden::shi_x_n; ++i) {
    sp::shichi_t s = sp::shichi(golden::shi_x[i]);
    CHECK_CLOSE(s.Shi, golden::shi_Shi[i], R, A);
    CHECK_CLOSE(s.Chi, golden::shi_Chi[i], R, A);
  }
  // Shi(0) = 0, Chi(0) = -inf; Shi is odd, Chi is even.
  sp::shichi_t z = sp::shichi(0.0);
  CHECK_CLOSE(z.Shi, 0.0, R, A);
  CHECK(std::isinf(z.Chi) && z.Chi < 0.0);
  CHECK_CLOSE(sp::shichi(-2.5).Shi, -sp::shichi(2.5).Shi, R, A);
  CHECK_CLOSE(sp::shichi(-2.5).Chi, sp::shichi(2.5).Chi, R, A);
}
