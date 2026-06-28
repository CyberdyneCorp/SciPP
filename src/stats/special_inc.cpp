// Regularized incomplete gamma/beta and their inverses (Numerical-Recipes
// series + continued fraction). Underpin the distribution CDFs/PPFs.
#include "scipp/stats/detail.hpp"

#include <cmath>

namespace scipp::stats::detail {
namespace {
constexpr double kEps = 1e-15;
constexpr double kFpMin = 1e-300;

double gser(double a, double x) {  // series for P(a,x)
  if (x <= 0.0) return 0.0;
  double gln = std::lgamma(a);
  double ap = a, sum = 1.0 / a, del = sum;
  for (int n = 0; n < 1000; ++n) {
    ap += 1.0;
    del *= x / ap;
    sum += del;
    if (std::fabs(del) < std::fabs(sum) * kEps) break;
  }
  return sum * std::exp(-x + a * std::log(x) - gln);
}

double gcf(double a, double x) {  // continued fraction for Q(a,x)
  double gln = std::lgamma(a);
  double b = x + 1.0 - a, c = 1.0 / kFpMin, d = 1.0 / b, h = d;
  for (int i = 1; i < 1000; ++i) {
    double an = -1.0 * i * (i - a);
    b += 2.0;
    d = an * d + b; if (std::fabs(d) < kFpMin) d = kFpMin;
    c = b + an / c; if (std::fabs(c) < kFpMin) c = kFpMin;
    d = 1.0 / d;
    double del = d * c;
    h *= del;
    if (std::fabs(del - 1.0) < kEps) break;
  }
  return std::exp(-x + a * std::log(x) - gln) * h;
}

double betacf(double a, double b, double x) {
  double qab = a + b, qap = a + 1.0, qam = a - 1.0;
  double c = 1.0, d = 1.0 - qab * x / qap;
  if (std::fabs(d) < kFpMin) d = kFpMin;
  d = 1.0 / d;
  double h = d;
  for (int m = 1; m < 1000; ++m) {
    double m2 = 2.0 * m;
    double aa = m * (b - m) * x / ((qam + m2) * (a + m2));
    d = 1.0 + aa * d; if (std::fabs(d) < kFpMin) d = kFpMin;
    c = 1.0 + aa / c; if (std::fabs(c) < kFpMin) c = kFpMin;
    d = 1.0 / d; h *= d * c;
    aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2));
    d = 1.0 + aa * d; if (std::fabs(d) < kFpMin) d = kFpMin;
    c = 1.0 + aa / c; if (std::fabs(c) < kFpMin) c = kFpMin;
    d = 1.0 / d;
    double del = d * c;
    h *= del;
    if (std::fabs(del - 1.0) < kEps) break;
  }
  return h;
}
}  // namespace

double gammainc(double a, double x) {
  if (x < 0.0 || a <= 0.0) return std::nan("");
  if (x == 0.0) return 0.0;
  return (x < a + 1.0) ? gser(a, x) : 1.0 - gcf(a, x);
}
double gammaincc(double a, double x) { return 1.0 - gammainc(a, x); }

double betainc(double a, double b, double x) {
  if (x <= 0.0) return 0.0;
  if (x >= 1.0) return 1.0;
  double bt = std::exp(std::lgamma(a + b) - std::lgamma(a) - std::lgamma(b) +
                       a * std::log(x) + b * std::log(1.0 - x));
  if (x < (a + 1.0) / (a + b + 2.0)) return bt * betacf(a, b, x) / a;
  return 1.0 - bt * betacf(b, a, 1.0 - x) / b;
}

double gammaincinv(double a, double p) {
  if (p <= 0.0) return 0.0;
  if (p >= 1.0) return INFINITY;
  double lo = 0.0, hi = std::max(10.0, a * 4.0);
  while (gammainc(a, hi) < p) hi *= 2.0;
  for (int i = 0; i < 200; ++i) {  // bisection (robust)
    double mid = 0.5 * (lo + hi);
    if (gammainc(a, mid) < p) lo = mid; else hi = mid;
  }
  return 0.5 * (lo + hi);
}

double betaincinv(double a, double b, double p) {
  if (p <= 0.0) return 0.0;
  if (p >= 1.0) return 1.0;
  double lo = 0.0, hi = 1.0;
  for (int i = 0; i < 200; ++i) {
    double mid = 0.5 * (lo + hi);
    if (betainc(a, b, mid) < p) lo = mid; else hi = mid;
  }
  return 0.5 * (lo + hi);
}

}  // namespace scipp::stats::detail
