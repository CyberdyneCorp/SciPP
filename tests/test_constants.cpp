// Oracle tests for scipp::constants against frozen SciPy golden data.
#include <vector>

#include "golden.hpp"
#include "numpp/core/dtype.hpp"
#include "numpp/core/ndarray.hpp"
#include "scipp/constants/constants.hpp"
#include "scipp/error.hpp"
#include "scipp_test.hpp"

namespace cst = scipp::constants;

namespace {
constexpr double R = 1e-12, A = 0.0;
}  // namespace

TEST_CASE("scale constants") {
  CHECK_CLOSE(cst::c, golden::const_c, R, A);
  CHECK_CLOSE(cst::h, golden::const_h, R, A);
  CHECK_CLOSE(cst::hbar, golden::const_hbar, 1e-10, 0.0);
  CHECK_CLOSE(cst::e, golden::const_e, R, A);
  CHECK_CLOSE(cst::k, golden::const_k, R, A);
  CHECK_CLOSE(cst::N_A, golden::const_N_A, R, A);
  CHECK_CLOSE(cst::R, golden::const_R, R, A);
  CHECK_CLOSE(cst::g, golden::const_g, R, A);
  CHECK_CLOSE(cst::pi, golden::const_pi, R, A);
  CHECK_CLOSE(cst::atm, golden::const_atm, R, A);
  CHECK_CLOSE(cst::bar, golden::const_bar, R, A);
  CHECK_CLOSE(cst::inch, golden::const_inch, R, A);
  CHECK_CLOSE(cst::hour, golden::const_hour, R, A);
  CHECK_CLOSE(cst::eV, golden::const_eV, R, A);
}

TEST_CASE("CODATA table lookup") {
  CHECK_CLOSE(cst::value("electron mass"), golden::pc_electron_mass_value, R, A);
  CHECK_CLOSE(cst::precision("electron mass"), golden::pc_electron_mass_prec, 1e-6, 1e-15);
  CHECK_CLOSE(cst::value("proton mass"), golden::pc_proton_mass_value, R, A);
  CHECK(cst::unit("electron mass") == "kg");
  CHECK_CLOSE(cst::value("Boltzmann constant"), golden::pc_boltzmann_value, R, A);
  CHECK_THROWS_AS(cst::value("no such constant"), scipp::value_error);
}

TEST_CASE("unit conversions") {
  CHECK_CLOSE(cst::convert_temperature(100.0, "Celsius", "Fahrenheit"),
              golden::temp_c2f_100, R, 1e-12);
  CHECK_CLOSE(cst::convert_temperature(300.0, "Kelvin", "Celsius"),
              golden::temp_k2c_300, R, 1e-12);
  // round-trip
  CHECK_CLOSE(cst::convert_temperature(
                  cst::convert_temperature(37.0, "C", "F"), "F", "C"),
              37.0, 1e-12, 1e-12);
  CHECK_CLOSE(cst::lambda2nu(500e-9), golden::l2nu_500nm, R, 1.0);
  // wavelength/frequency round-trip
  CHECK_CLOSE(cst::nu2lambda(cst::lambda2nu(632.8e-9)), 632.8e-9, 1e-12, 1e-18);
}
