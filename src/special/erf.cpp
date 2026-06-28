// Error functions: erf, erfc (libm), and erfinv/erfcinv via a rational initial
// guess (Giles, 2010) refined with one Newton step to double precision.

#include "scipp/special/special.hpp"

#include <cmath>

#include "scipp/detail/elementwise.hpp"

namespace scipp::special {
namespace {

constexpr double kTwoOverSqrtPi = 1.1283791670955126;  // 2/sqrt(pi)

double erfinv_impl(double x) {
  if (x <= -1.0) return x == -1.0 ? -INFINITY : std::nan("");
  if (x >= 1.0) return x == 1.0 ? INFINITY : std::nan("");
  double w = -std::log((1.0 - x) * (1.0 + x));
  double p;
  if (w < 5.0) {
    w -= 2.5;
    p = 2.81022636e-08;
    p = 3.43273939e-07 + p * w;
    p = -3.5233877e-06 + p * w;
    p = -4.39150654e-06 + p * w;
    p = 0.00021858087 + p * w;
    p = -0.00125372503 + p * w;
    p = -0.00417768164 + p * w;
    p = 0.246640727 + p * w;
    p = 1.50140941 + p * w;
  } else {
    w = std::sqrt(w) - 3.0;
    p = -0.000200214257;
    p = 0.000100950558 + p * w;
    p = 0.00134934322 + p * w;
    p = -0.00367342844 + p * w;
    p = 0.00573950773 + p * w;
    p = -0.0076224613 + p * w;
    p = 0.00943887047 + p * w;
    p = 1.00167406 + p * w;
    p = 2.83297682 + p * w;
  }
  double r = p * x;
  // One Newton step on erf to reach double precision.
  r -= (std::erf(r) - x) / (kTwoOverSqrtPi * std::exp(-r * r));
  r -= (std::erf(r) - x) / (kTwoOverSqrtPi * std::exp(-r * r));
  return r;
}

}  // namespace

double erf(double x) { return std::erf(x); }
double erfc(double x) { return std::erfc(x); }
double erfinv(double y) { return erfinv_impl(y); }
double erfcinv(double y) { return erfinv_impl(1.0 - y); }

ndarray erf(const ndarray& x) { return detail::map(x, [](double v) { return std::erf(v); }); }
ndarray erfc(const ndarray& x) { return detail::map(x, [](double v) { return std::erfc(v); }); }
ndarray erfinv(const ndarray& y) { return detail::map(y, [](double v) { return erfinv_impl(v); }); }
ndarray erfcinv(const ndarray& y) {
  return detail::map(y, [](double v) { return erfinv_impl(1.0 - v); });
}

}  // namespace scipp::special
