// Standard waveforms: chirp, sawtooth, square, unit_impulse.
#include "scypp/signal/signal.hpp"

#include <cmath>
#include <vector>

#include "numpp/core/dtype.hpp"
#include "scypp/error.hpp"
#include "scypp/linalg/detail.hpp"

namespace scypp::signal {
namespace {
namespace sd = scypp::linalg::detail;
constexpr double kPi = 3.141592653589793238462643383279502884;
}  // namespace

ndarray chirp(const ndarray& t, double f0, double t1, double f1, const std::string& method,
              double phi) {
  auto tv = sd::to_vec(t);
  std::vector<double> out(tv.size());
  double phi_rad = phi * kPi / 180.0;
  for (size_t i = 0; i < tv.size(); ++i) {
    double x = tv[i], phase;
    if (method == "linear") {
      double beta = (f1 - f0) / t1;
      phase = 2.0 * kPi * (f0 * x + 0.5 * beta * x * x);
    } else if (method == "quadratic") {
      double beta = (f1 - f0) / (t1 * t1);
      phase = 2.0 * kPi * (f0 * x + beta * x * x * x / 3.0);
    } else if (method == "logarithmic") {
      double k = std::pow(f1 / f0, 1.0 / t1);
      phase = 2.0 * kPi * f0 * (std::pow(k, x) - 1.0) / std::log(k);
    } else if (method == "hyperbolic") {
      double c = f0 * t1, df = f1 - f0;
      phase = 2.0 * kPi * (-c / df) * std::log(1.0 - df / (f0 * t1) * x);
    } else {
      throw scypp::value_error("chirp: unknown method " + method);
    }
    out[i] = std::cos(phase + phi_rad);
  }
  return sd::from_vec(out);
}

ndarray sawtooth(const ndarray& t, double width) {
  auto tv = sd::to_vec(t);
  std::vector<double> out(tv.size());
  for (size_t i = 0; i < tv.size(); ++i) {
    double tmod = std::fmod(tv[i], 2.0 * kPi);
    if (tmod < 0) tmod += 2.0 * kPi;
    double frac = tmod / (2.0 * kPi);
    if (frac < width) out[i] = 2.0 * frac / width - 1.0;
    else out[i] = 2.0 * (1.0 - frac) / (1.0 - width) - 1.0;
  }
  return sd::from_vec(out);
}

ndarray square(const ndarray& t, double duty) {
  auto tv = sd::to_vec(t);
  std::vector<double> out(tv.size());
  for (size_t i = 0; i < tv.size(); ++i) {
    double tmod = std::fmod(tv[i], 2.0 * kPi);
    if (tmod < 0) tmod += 2.0 * kPi;
    out[i] = (tmod / (2.0 * kPi) < duty) ? 1.0 : -1.0;
  }
  return sd::from_vec(out);
}

ndarray unit_impulse(int64_t n, int64_t idx) {
  std::vector<double> out(n, 0.0);
  if (idx >= 0 && idx < n) out[idx] = 1.0;
  return sd::from_vec(out);
}

}  // namespace scypp::signal
