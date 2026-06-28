// Miscellaneous special functions (scipy.special), real argument:
//   lambertw  - Lambert W principal (k=0) and k=-1 real branches via Halley
//   zeta/zetac- Riemann zeta and zeta - 1 via Euler-Maclaurin summation
//   struve    - Struve H_v: power series (small |x|) / Bessel-Y asymptotic
//   modstruve - modified Struve L_v: power series / Bessel-I asymptotic
//   spence    - Spence's dilogarithm Li_2(1 - x) (scipy convention)
//
// Domain edges follow SciPy: out-of-domain returns nan/inf, never throws.

#include "scypp/special/special.hpp"

#include <cmath>

namespace scypp::special {
namespace {

constexpr double kPi = 3.14159265358979323846;
constexpr double kPi2_6 = 1.64493406684822643647;   // pi^2 / 6
constexpr double kInvE = 0.36787944117144232160;    // 1 / e
constexpr double kE = 2.71828182845904523536;

// Halley iteration for w e^w = x, refining an initial guess `w`.
double lambertw_halley(double x, double w) {
  for (int i = 0; i < 60; ++i) {
    const double ew = std::exp(w);
    const double f = w * ew - x;
    if (f == 0.0) break;
    const double wp1 = w + 1.0;
    const double denom = ew * wp1 - (w + 2.0) * f / (2.0 * wp1);
    const double dw = f / denom;
    w -= dw;
    if (std::fabs(dw) <= 1e-17 * (std::fabs(w) + 1e-300)) break;
  }
  return w;
}

// Riemann zeta by Euler-Maclaurin: sum the first N-1 terms explicitly, add the
// integral/midpoint corrections and a Bernoulli tail. Analytically continues to
// the whole real line (s != 1); accurate for s > 1 - 2*kK.
double zeta_euler_maclaurin(double s) {
  constexpr int kN = 12, kK = 7;
  // c_k = B_{2k} / (2k)! for k = 1..7.
  static const double c[kK] = {
      1.0 / 12.0, -1.0 / 720.0, 1.0 / 30240.0, -1.0 / 1209600.0,
      1.0 / 47900160.0, -691.0 / 1307674368000.0, 1.0 / 74724249600.0};
  double tot = 0.0;
  for (int n = 1; n < kN; ++n) tot += std::pow(static_cast<double>(n), -s);
  tot += std::pow(static_cast<double>(kN), 1.0 - s) / (s - 1.0);
  tot += 0.5 * std::pow(static_cast<double>(kN), -s);
  // tail: sum_k c_k * prod_{j=0}^{2k-2}(s+j) * N^{-s-2k+1}
  double prod = s;  // k = 1 factor
  for (int k = 1; k <= kK; ++k) {
    tot += c[k - 1] * prod * std::pow(static_cast<double>(kN), -s - 2.0 * k + 1.0);
    prod *= (s + 2.0 * k - 1.0) * (s + 2.0 * k);  // extend to k+1
  }
  return tot;
}

// Struve H_v / modified Struve L_v power series (modified = true drops the
// (-1)^k sign). Returns the value for x > 0.
double struve_series(double v, double x, bool modified) {
  const double lhalf = std::log(0.5 * x);
  double sum = 0.0;
  for (int k = 0; k < 600; ++k) {
    const double lt = (2.0 * k + v + 1.0) * lhalf - std::lgamma(k + 1.5) -
                      std::lgamma(k + v + 1.5);
    double term = std::exp(lt);
    if (!modified && (k & 1)) term = -term;
    sum += term;
    if (std::fabs(term) <= 1e-19 * std::fabs(sum) && k > 2) break;
  }
  return sum;
}

// Asymptotic expansion for large x:
//   H_v(x) = Y_v(x) + (1/pi) sum_k Gamma(k+1/2) (x/2)^{v-2k-1} / Gamma(v+1/2-k)
//   L_v(x) = I_v(x) - (1/pi) sum_k (-1)^k Gamma(k+1/2) (x/2)^{v-2k-1}/Gamma(...)
// The series is divergent; truncate at its smallest term.
double struve_asymptotic(double v, double x, bool modified) {
  const double base = modified ? iv(v, x) : yv(v, x);
  const double half = 0.5 * x;
  double tail = 0.0, prev = HUGE_VAL;
  for (int k = 0; k < 100; ++k) {
    const double g2 = std::tgamma(v + 0.5 - k);
    if (!std::isfinite(g2) || g2 == 0.0) break;
    double term = std::tgamma(k + 0.5) * std::pow(half, v - 2.0 * k - 1.0) / g2;
    if (modified && (k & 1)) term = -term;
    if (std::fabs(term) > prev) break;
    tail += term;
    prev = std::fabs(term);
  }
  tail /= kPi;
  return modified ? base - tail : base + tail;
}

// Common dispatcher for struve / modstruve over the full real line.
double struve_dispatch(double v, double x, bool modified) {
  if (x == 0.0) return v > -1.0 ? 0.0 : (v == -1.0 ? 2.0 / kPi : NAN);
  double sign = 1.0;
  double ax = x;
  if (x < 0.0) {
    const double vr = std::round(v);
    if (vr != v) return NAN;  // non-integer order: undefined for x < 0
    sign = (static_cast<long long>(vr) % 2 == 0) ? -1.0 : 1.0;  // (-1)^{v+1}
    ax = -x;
  }
  const double val = ax < 16.0 ? struve_series(v, ax, modified)
                               : struve_asymptotic(v, ax, modified);
  return sign * val;
}

// Dilogarithm Li_2(z) for real z <= 1, used by spence.
double dilog_series(double y) {  // sum_{k>=1} y^k / k^2, |y| <= 1/2
  double t = y, sum = y;
  for (int k = 2; k < 500; ++k) {
    t *= y;
    const double add = t / (static_cast<double>(k) * k);
    sum += add;
    if (std::fabs(add) <= 1e-19 * std::fabs(sum) + 1e-300) break;
  }
  return sum;
}

double dilog(double z) {
  if (z == 1.0) return kPi2_6;
  if (z < -1.0)
    return -dilog_series(1.0 / z) - kPi2_6 - 0.5 * std::log(-z) * std::log(-z);
  if (z < 0.0)
    return -dilog_series(z / (z - 1.0)) - 0.5 * std::log(1.0 - z) * std::log(1.0 - z);
  if (z <= 0.5) return dilog_series(z);
  return kPi2_6 - std::log(z) * std::log(1.0 - z) - dilog_series(1.0 - z);
}

}  // namespace

double lambertw(double x, int k) {
  if (k == 0) {
    if (x < -kInvE) return NAN;
    if (x == 0.0) return 0.0;
    double w;
    if (x < -0.3) {  // near the branch point -1/e
      const double p = std::sqrt(2.0 * (kE * x + 1.0));
      w = -1.0 + p - p * p / 3.0 + 11.0 / 72.0 * p * p * p;
    } else if (x <= 1.0) {
      w = x * (1.0 - x + 1.5 * x * x);
    } else {
      const double l1 = std::log(x);
      const double l2 = std::log(l1);
      w = l1 > 1.0 ? l1 - l2 + l2 / l1 : l1;
    }
    return lambertw_halley(x, w);
  }
  if (k == -1) {
    if (x < -kInvE || x >= 0.0) return NAN;
    double w;
    if (x < -0.2) {  // near the branch point -1/e
      const double p = -std::sqrt(2.0 * (kE * x + 1.0));
      w = -1.0 + p - p * p / 3.0 + 11.0 / 72.0 * p * p * p;
    } else {
      const double l1 = std::log(-x);
      const double l2 = std::log(-l1);
      w = l1 - l2 + l2 / l1;
    }
    return lambertw_halley(x, w);
  }
  return NAN;  // other branches are complex-only for real x
}

double zeta(double x) {
  if (x == 1.0) return INFINITY;
  return zeta_euler_maclaurin(x);
}

double zetac(double x) {
  if (x == 1.0) return INFINITY;
  return zeta_euler_maclaurin(x) - 1.0;
}

double struve(double v, double x) { return struve_dispatch(v, x, false); }
double modstruve(double v, double x) { return struve_dispatch(v, x, true); }

double spence(double x) {
  if (x < 0.0) return NAN;
  if (x == 0.0) return kPi2_6;
  return dilog(1.0 - x);
}

}  // namespace scypp::special
