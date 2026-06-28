// Hypergeometric functions (scipy.special), real arguments:
//   hyp0f1 - confluent limit 0F1(;b;x) via its power series
//   hyp1f1 - Kummer confluent 1F1 = M(a,b,x); Kummer transform stabilises x < 0
//   hyp2f1 - Gauss 2F1; power series plus the Pfaff and 1 - z linear transforms
//   hyperu - Tricomi confluent U from the two 1F1 solutions (non-integer b)
//
// Each routine targets the convergent / numerically stable region documented in
// special.hpp. Out-of-domain or unsupported inputs return nan, never throw.

#include "scipp/special/special.hpp"

#include <cmath>

namespace scipp::special {
namespace {

constexpr double kPi = 3.14159265358979323846;
constexpr int kMaxIt = 2000;
constexpr double kEps = 1e-16;

double nan() { return std::nan(""); }

// True when x is a non-positive integer (a pole of Gamma / vanishing Pochhammer
// denominator), within a tiny tolerance for the supplied (typically exact)
// arguments.
bool is_nonpos_int(double x) {
  return x <= 0.0 && std::fabs(x - std::rint(x)) < 1e-12;
}

// 0F1 series: sum_n x^n / ((b)_n n!).  Term ratio t_{n+1}/t_n = x/((b+n)(n+1)).
double series_0f1(double b, double x) {
  double term = 1.0, sum = 1.0;
  for (int n = 0; n < kMaxIt; ++n) {
    term *= x / ((b + n) * (n + 1));
    sum += term;
    if (std::fabs(term) <= kEps * std::fabs(sum)) break;
  }
  return sum;
}

// 1F1 series: sum_n (a)_n/((b)_n n!) x^n.  Ratio (a+n)/((b+n)(n+1)) * x.
double series_1f1(double a, double b, double x) {
  double term = 1.0, sum = 1.0;
  for (int n = 0; n < kMaxIt; ++n) {
    term *= (a + n) / ((b + n) * (n + 1)) * x;
    sum += term;
    if (std::fabs(term) <= kEps * std::fabs(sum)) break;
  }
  return sum;
}

// 2F1 series: sum_n (a)_n(b)_n/((c)_n n!) z^n.  Converges for |z| < 1.
double series_2f1(double a, double b, double c, double z) {
  double term = 1.0, sum = 1.0;
  for (int n = 0; n < kMaxIt; ++n) {
    term *= (a + n) * (b + n) / ((c + n) * (n + 1)) * z;
    sum += term;
    if (std::fabs(term) <= kEps * std::fabs(sum)) break;
  }
  return sum;
}

}  // namespace

double hyp0f1(double b, double x) {
  if (is_nonpos_int(b)) return nan();
  return series_0f1(b, x);
}

double hyp1f1(double a, double b, double x) {
  if (is_nonpos_int(b)) return nan();
  // Kummer's transformation keeps the summed series convergent and
  // sign-stable for negative argument.
  if (x < 0.0) return std::exp(x) * series_1f1(b - a, b, -x);
  return series_1f1(a, b, x);
}

double hyp2f1(double a, double b, double c, double z) {
  if (is_nonpos_int(c)) return nan();
  if (z == 1.0) {  // Gauss summation theorem (converges only when c-a-b > 0).
    if (c - a - b <= 0.0) return nan();
    return std::tgamma(c) * std::tgamma(c - a - b) /
           (std::tgamma(c - a) * std::tgamma(c - b));
  }
  if (z <= -1.0 - 1e-12 || z > 1.0) return nan();  // outside the delivered region

  // z in (-1, -1/2]: Pfaff transformation maps the argument to [1/3, 1/2).
  if (z <= -0.5) {
    const double w = z / (z - 1.0);
    return std::pow(1.0 - z, -a) * series_2f1(a, c - b, c, w);
  }

  const double s = c - a - b;
  // z in (1/2, 1) with non-integer c - a - b: the 1 - z reflection converges
  // far faster than the direct series near z = 1.
  if (z > 0.5 && std::fabs(s - std::rint(s)) >= 1e-9) {
    const double w = 1.0 - z;
    const double t1 = std::tgamma(c) * std::tgamma(s) /
                      (std::tgamma(c - a) * std::tgamma(c - b)) *
                      series_2f1(a, b, 1.0 - s, w);
    const double t2 = std::pow(w, s) * std::tgamma(c) * std::tgamma(-s) /
                      (std::tgamma(a) * std::tgamma(b)) *
                      series_2f1(c - a, c - b, 1.0 + s, w);
    return t1 + t2;
  }

  // Otherwise the direct series (|z| < 1) is convergent and used directly.
  return series_2f1(a, b, c, z);
}

double hyperu(double a, double b, double x) {
  if (!(x > 0.0)) return nan();
  // Integer b is a removable/limit case handled by a separate formula; deliver
  // the non-integer branch and defer integer b (return nan).
  if (std::fabs(b - std::rint(b)) < 1e-9) return nan();

  const double m1 = series_1f1(a, b, x);
  const double m2 = series_1f1(a - b + 1.0, 2.0 - b, x);
  const double term1 = m1 / (std::tgamma(a - b + 1.0) * std::tgamma(b));
  const double term2 =
      std::pow(x, 1.0 - b) * m2 / (std::tgamma(a) * std::tgamma(2.0 - b));
  return kPi / std::sin(kPi * b) * (term1 - term2);
}

}  // namespace scipp::special
