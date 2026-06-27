// gamma family: gamma, gammaln, loggamma, digamma, polygamma, beta, betaln.
// gamma/gammaln delegate to the C++ standard library (which uses the platform
// libm, matching SciPy/Cephes within tolerance). digamma/polygamma use the
// standard asymptotic series with a recurrence shift; polygamma(n>=1) uses the
// Hurwitz zeta relation psi^(n)(x) = (-1)^(n+1) n! zeta(n+1, x).

#include "scypp/special/special.hpp"

#include <cmath>

#include "scypp/detail/elementwise.hpp"

namespace scypp::special {
namespace {

constexpr double kPi = 3.141592653589793238462643383279502884;

double gamma_sign(double x) {
  if (x > 0.0) return 1.0;
  double fl = std::floor(x);
  if (x == fl) return std::nan("");          // pole
  return (std::fmod(std::fabs(fl), 2.0) == 0.0) ? -1.0 : 1.0;
}

// Hurwitz zeta zeta(s, a) via Euler-Maclaurin (s > 1, a > 0).
double hurwitz_zeta(double s, double a) {
  constexpr int N = 9, M = 5;
  // B_{2j} / (2j)!
  static const double cj[M] = {1.0 / 12.0, -1.0 / 720.0, 1.0 / 30240.0,
                               -1.0 / 1209600.0, 1.0 / 47900160.0};
  double sum = 0.0;
  for (int kk = 0; kk < N; ++kk) sum += std::pow(a + kk, -s);
  double w = a + N;
  sum += std::pow(w, 1.0 - s) / (s - 1.0);
  sum += 0.5 * std::pow(w, -s);
  double poch = s;                 // (s)_{2j-1}, starts at (s)_1 = s
  double wp = std::pow(w, -(s + 1.0));  // w^{-(s + 2j - 1)}, j = 1
  for (int j = 1; j <= M; ++j) {
    sum += cj[j - 1] * poch * wp;
    poch *= (s + 2 * j - 1) * (s + 2 * j);
    wp /= (w * w);
  }
  return sum;
}

}  // namespace

double gamma(double x) { return std::tgamma(x); }
double gammaln(double x) { return std::lgamma(x); }
double loggamma(double x) { return std::lgamma(x); }  // real branch (x > 0)

double betaln(double a, double b) {
  return std::lgamma(a) + std::lgamma(b) - std::lgamma(a + b);
}

double beta(double a, double b) {
  double s = gamma_sign(a) * gamma_sign(b) * gamma_sign(a + b);
  return s * std::exp(betaln(a, b));
}

double digamma(double x) {
  double result = 0.0;
  if (x <= 0.0) {
    if (x == std::floor(x)) return std::nan("");   // poles at 0, -1, -2, ...
    // reflection: psi(x) = psi(1 - x) - pi / tan(pi x)
    return digamma(1.0 - x) - kPi / std::tan(kPi * x);
  }
  while (x < 6.0) {           // shift up into the asymptotic regime
    result -= 1.0 / x;
    x += 1.0;
  }
  double ix = 1.0 / x;
  double f = ix * ix;
  result += std::log(x) - 0.5 * ix;
  result -= f * (1.0 / 12.0 - f * (1.0 / 120.0 - f * (1.0 / 252.0 - f * (1.0 / 240.0))));
  return result;
}

double polygamma(int n, double x) {
  if (n == 0) return digamma(x);
  if (n < 0) return std::nan("");
  double fact = 1.0;
  for (int i = 2; i <= n; ++i) fact *= i;        // n!
  double sign = (n % 2 == 0) ? -1.0 : 1.0;        // (-1)^(n+1)
  return sign * fact * hurwitz_zeta(n + 1.0, x);
}

// ---- ndarray overloads ----
ndarray gamma(const ndarray& x) { return detail::map(x, [](double v) { return gamma(v); }); }
ndarray gammaln(const ndarray& x) { return detail::map(x, [](double v) { return gammaln(v); }); }
ndarray loggamma(const ndarray& x) { return detail::map(x, [](double v) { return loggamma(v); }); }
ndarray digamma(const ndarray& x) { return detail::map(x, [](double v) { return digamma(v); }); }
ndarray polygamma(int n, const ndarray& x) {
  return detail::map(x, [n](double v) { return polygamma(n, v); });
}

}  // namespace scypp::special
