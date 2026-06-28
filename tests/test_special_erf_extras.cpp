// Oracle tests for error-function relatives against SciPy golden:
// erfcx, dawsn, wofz, voigt_profile, fresnel.
#include <complex>

#include "golden.hpp"
#include "scipp/special/special.hpp"
#include "scipp_test.hpp"

namespace sp = scipp::special;

namespace {
constexpr double R = 1e-10, A = 1e-12;
}  // namespace

TEST_CASE("erfcx") {
  CHECK_ARR(erfcx, [](double x) { return sp::erfcx(x); }, R, A);
  // Far-negative argument overflows to +inf (matching SciPy).
  CHECK(std::isinf(sp::erfcx(-40.0)));
  // erfcx(0) = 1.
  CHECK_CLOSE(sp::erfcx(0.0), 1.0, R, A);
}

TEST_CASE("dawsn") {
  CHECK_ARR(dawsn, [](double x) { return sp::dawsn(x); }, R, A);
  // Odd function; D(0) = 0.
  CHECK_CLOSE(sp::dawsn(0.0), 0.0, R, A);
  CHECK_CLOSE(sp::dawsn(-2.3), -sp::dawsn(2.3), R, A);
}

TEST_CASE("fresnel") {
  for (int i = 0; i < golden::fr_x_n; ++i) {
    sp::fresnel_t f = sp::fresnel(golden::fr_x[i]);
    CHECK_CLOSE(f.S, golden::fr_S[i], R, A);
    CHECK_CLOSE(f.C, golden::fr_C[i], R, A);
  }
  // S(0) = C(0) = 0; odd functions.
  sp::fresnel_t f0 = sp::fresnel(0.0);
  CHECK_CLOSE(f0.S, 0.0, R, A);
  CHECK_CLOSE(f0.C, 0.0, R, A);
  sp::fresnel_t fp = sp::fresnel(1.7), fm = sp::fresnel(-1.7);
  CHECK_CLOSE(fp.S, -fm.S, R, A);
  CHECK_CLOSE(fp.C, -fm.C, R, A);
}

TEST_CASE("voigt_profile") {
  struct Grid { const double* y; double sigma, gamma; };
  const Grid grids[] = {
      {golden::vt_a, golden::vt_a_sigma, golden::vt_a_gamma},
      {golden::vt_b, golden::vt_b_sigma, golden::vt_b_gamma},
      {golden::vt_c, golden::vt_c_sigma, golden::vt_c_gamma},
      {golden::vt_d, golden::vt_d_sigma, golden::vt_d_gamma},
      {golden::vt_e, golden::vt_e_sigma, golden::vt_e_gamma}};
  for (const auto& g : grids) {
    for (int i = 0; i < golden::vt_x_n; ++i) {
      CHECK_CLOSE(sp::voigt_profile(golden::vt_x[i], g.sigma, g.gamma), g.y[i], R, A);
    }
  }
  // Symmetric in x; sigma <= 0 is out of domain.
  CHECK_CLOSE(sp::voigt_profile(1.3, 1.0, 0.4), sp::voigt_profile(-1.3, 1.0, 0.4), R, A);
  CHECK(std::isnan(sp::voigt_profile(0.0, 0.0, 1.0)));
}

TEST_CASE("wofz") {
  for (int i = 0; i < golden::wz_re_n; ++i) {
    std::complex<double> w = sp::wofz({golden::wz_re[i], golden::wz_im[i]});
    CHECK_CLOSE(w.real(), golden::wz_wre[i], R, A);
    CHECK_CLOSE(w.imag(), golden::wz_wim[i], R, A);
  }
  // w(0) = 1.
  std::complex<double> w0 = sp::wofz({0.0, 0.0});
  CHECK_CLOSE(w0.real(), 1.0, R, A);
  CHECK_CLOSE(w0.imag(), 0.0, R, A);
  // On the imaginary axis w(i y) = erfcx(y).
  std::complex<double> wi = sp::wofz({0.0, 1.5});
  CHECK_CLOSE(wi.real(), sp::erfcx(1.5), R, A);
}
