// Exponential/logarithmic integrals: exp1 (E1), expi (Ei), expn (E_n), exprel.
// Ei is taken from libstdc++ std::expint; E1(x) = -Ei(-x). E_n uses the
// Numerical-Recipes series/continued-fraction split.

#include "scypp/special/special.hpp"

#include <cmath>

#include "scypp/detail/elementwise.hpp"

namespace scypp::special {
namespace {

constexpr double kEuler = 0.5772156649015328606065120900824024;

double expn_impl(int n, double x) {
  constexpr int kMaxIt = 200;
  constexpr double kEps = 1e-15;
  constexpr double kFpMin = 1e-300;
  if (n < 0 || x < 0.0 || (x == 0.0 && (n == 0 || n == 1))) return std::nan("");
  int nm1 = n - 1;
  if (n == 0) return std::exp(-x) / x;
  if (x == 0.0) return 1.0 / nm1;
  if (x > 1.0) {  // Lentz continued fraction
    double b = x + n;
    double c = 1.0 / kFpMin;
    double d = 1.0 / b;
    double h = d;
    for (int i = 1; i <= kMaxIt; ++i) {
      double a = -1.0 * i * (nm1 + i);
      b += 2.0;
      d = 1.0 / (a * d + b);
      c = b + a / c;
      double del = c * d;
      h *= del;
      if (std::fabs(del - 1.0) <= kEps) break;
    }
    return h * std::exp(-x);
  }
  // Power series
  double ans = (nm1 != 0) ? 1.0 / nm1 : -std::log(x) - kEuler;
  double fact = 1.0;
  for (int i = 1; i <= kMaxIt; ++i) {
    fact *= -x / i;
    double del;
    if (i != nm1) {
      del = -fact / (i - nm1);
    } else {
      double psi = -kEuler;
      for (int ii = 1; ii <= nm1; ++ii) psi += 1.0 / ii;
      del = fact * (-std::log(x) + psi);
    }
    ans += del;
    if (std::fabs(del) < std::fabs(ans) * kEps) break;
  }
  return ans;
}

}  // namespace

double expi(double x) { return std::expint(x); }       // Ei(x)
double exp1(double x) { return -std::expint(-x); }      // E1(x), x > 0
double expn(int n, double x) { return expn_impl(n, x); }

double exprel(double x) {
  if (std::fabs(x) < 1e-16) return 1.0;
  return std::expm1(x) / x;
}

ndarray expi(const ndarray& x) { return detail::map(x, [](double v) { return expi(v); }); }
ndarray exp1(const ndarray& x) { return detail::map(x, [](double v) { return exp1(v); }); }
ndarray exprel(const ndarray& x) { return detail::map(x, [](double v) { return exprel(v); }); }

}  // namespace scypp::special
