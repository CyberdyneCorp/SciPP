// Oracle tests for discrete-time LTI: cont2discrete, dstep, dimpulse, dlsim,
// dfreqresp, dbode (validated against scipy.signal).
#include <cmath>
#include <complex>
#include <vector>

#include "golden.hpp"
#include "numpp/core/dtype.hpp"
#include "numpp/core/ndarray.hpp"
#include "scypp/signal/signal.hpp"
#include "scypp_test.hpp"

namespace sg = scypp::signal;
namespace {
using cd = std::complex<double>;
numpp::ndarray vec(const double* d, int n) {
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
void cv(const numpp::ndarray& got, const double* exp, int n, double rtol = 1e-7, double atol = 1e-9) {
  auto g = tov(got);
  CHECK((int)g.size() == n);
  for (int i = 0; i < n && i < (int)g.size(); ++i) CHECK_CLOSE(g[i], exp[i], rtol, atol);
}
}  // namespace
#define V(name) vec(golden::name, golden::name##_n)
#define G(name) golden::name, golden::name##_n

TEST_CASE("discrete LTI cont2discrete") {
  double num[] = {1.0}, den[] = {1.0, 0.5, 1.0};
  sg::TransferFunction tf{vec(num, 1), vec(den, 3)};

  auto z = sg::cont2discrete(tf, 0.1, "zoh");
  cv(z.A, golden::se_d_Ad_d, golden::se_d_Ad_r * golden::se_d_Ad_c, 1e-9, 1e-11);
  cv(z.B, G(se_d_Bd), 1e-9, 1e-11);
  cv(z.C, G(se_d_Cd), 1e-9, 1e-11);
  cv(z.D, G(se_d_Dd), 1e-9, 1e-11);
  CHECK_CLOSE(z.dt, 0.1, 1e-12, 1e-12);

  auto b = sg::cont2discrete(tf, 0.1, "bilinear");
  cv(b.A, golden::se_db_Ad_d, golden::se_db_Ad_r * golden::se_db_Ad_c, 1e-9, 1e-11);
  cv(b.B, G(se_db_Bd), 1e-9, 1e-11);
  cv(b.C, G(se_db_Cd), 1e-9, 1e-11);
  cv(b.D, G(se_db_Dd), 1e-9, 1e-11);

  auto e = sg::cont2discrete(tf, 0.1, "euler");
  cv(e.A, golden::se_de_Ad_d, golden::se_de_Ad_r * golden::se_de_Ad_c, 1e-9, 1e-11);
  cv(e.B, G(se_de_Bd), 1e-9, 1e-11);
}

TEST_CASE("discrete LTI responses") {
  double num[] = {1.0}, den[] = {1.0, 0.5, 1.0};
  sg::TransferFunction tf{vec(num, 1), vec(den, 3)};
  auto sys = sg::cont2discrete(tf, 0.1, "zoh");

  cv(sg::dstep(sys, golden::se_dstep_n).y, G(se_dstep), 1e-8, 1e-10);
  cv(sg::dimpulse(sys, golden::se_dimpulse_n).y, G(se_dimpulse), 1e-8, 1e-10);
  cv(sg::dlsim(sys, V(se_du)).y, G(se_dlsim), 1e-8, 1e-10);
}

TEST_CASE("discrete LTI frequency response") {
  double num[] = {1.0}, den[] = {1.0, 0.5, 1.0};
  sg::TransferFunction tf{vec(num, 1), vec(den, 3)};
  auto sys = sg::cont2discrete(tf, 0.1, "zoh");

  auto fr = sg::dfreqresp(sys, V(se_dw));
  numpp::ndarray Hc = fr.h.astype(numpp::kComplex128).ascontiguousarray();
  const cd* h = Hc.typed_data<cd>();
  for (int i = 0; i < golden::se_dfr_re_n; ++i) {
    CHECK_CLOSE(h[i].real(), golden::se_dfr_re[i], 1e-8, 1e-10);
    CHECK_CLOSE(h[i].imag(), golden::se_dfr_im[i], 1e-8, 1e-10);
  }
  auto bd = sg::dbode(sys, V(se_dw));
  cv(bd.mag, G(se_dbode_mag), 1e-7, 1e-9);
  cv(bd.phase, G(se_dbode_ph), 1e-7, 1e-9);
}
