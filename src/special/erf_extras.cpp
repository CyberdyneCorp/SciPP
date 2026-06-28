// Error-function relatives: erfcx, dawsn, wofz (Faddeeva), voigt_profile and
// fresnel. Everything but erfcx and dawsn is expressed through the Faddeeva
// function w(z); w itself is assembled from three convergent pieces so each
// argument is handled by a method that is accurate where it is used:
//   * a Taylor series in the imaginary part, anchored on the EXACT real-axis
//     value w(x) = exp(-x^2) + i (2/sqrt(pi)) dawson(x), for |Im z| <= 1.3
//     (this is the only stable way to get Re[w] near the real axis, where the
//     usual power/continued-fraction forms suffer catastrophic cancellation);
//   * the Maclaurin series sum_n (i z)^n / Gamma(n/2 + 1) for small |z|;
//   * the Faddeeva continued fraction for large |z|.
// dawson itself uses a power series (small |x|), Rybicki's Gaussian-sum scheme
// (moderate |x|) and the asymptotic series (large |x|).

#include "scypp/special/special.hpp"

#include <cmath>
#include <complex>

#include "scypp/detail/elementwise.hpp"

namespace scypp::special {
namespace {

using cdouble = std::complex<double>;

constexpr double kSqrtPi = 1.7724538509055160273;     // sqrt(pi)
constexpr double kInvSqrtPi = 0.5641895835477562869;  // 1/sqrt(pi)
constexpr double kSqrt2 = 1.4142135623730950488;      // sqrt(2)
constexpr double kSqrt2Pi = 2.5066282746310002416;    // sqrt(2 pi)

// Dawson's integral D(x) = exp(-x^2) integral_0^x exp(t^2) dt.
double dawson_impl(double x) {
  const double s = (x >= 0.0) ? 1.0 : -1.0;
  const double a = std::fabs(x);
  if (a < 0.2) {  // power series D(x) = sum_k (-2)^k x^{2k+1} / (2k+1)!!
    const double x2 = a * a;
    double term = a, total = a;
    for (int k = 1; k < 40; ++k) {
      term *= -2.0 * x2 / (2 * k + 1);
      total += term;
      if (std::fabs(term) < 1e-19 * a) break;
    }
    return s * total;
  }
  if (a < 6.5) {  // Rybicki's exponentially-shifted Gaussian sum
    constexpr double H = 0.25;
    constexpr int kNMax = 12;
    const int n0 = 2 * static_cast<int>(0.5 * a / H + 0.5);
    const double xp = a - n0 * H;
    double e1 = std::exp(2.0 * xp * H);
    const double e2 = e1 * e1;
    double d1 = n0 + 1.0, d2 = d1 - 2.0, sum = 0.0;
    for (int i = 1; i <= kNMax; ++i) {
      const double ci = std::exp(-((2 * i - 1) * H) * ((2 * i - 1) * H));
      sum += ci * (e1 / d1 + 1.0 / (d2 * e1));
      d1 += 2.0;
      d2 -= 2.0;
      e1 *= e2;
    }
    return s * kInvSqrtPi * std::exp(-xp * xp) * sum;
  }
  // asymptotic series D(x) ~ sum_k (2k-1)!! / (2^{k+1} x^{2k+1})
  const double x2 = a * a;
  double term = 0.5 / a, total = term, prev = term;
  for (int k = 1; k < 100; ++k) {
    const double t = term * (2 * k - 1) / (2.0 * x2);
    if (std::fabs(t) > prev) break;  // beyond optimal truncation
    total += t;
    prev = std::fabs(t);
    term = t;
  }
  return s * total;
}

// Exact w on the non-negative real axis.
cdouble w_real(double x) {
  return cdouble(std::exp(-x * x), 2.0 * kInvSqrtPi * dawson_impl(x));
}

// w(x + i y) via Taylor series in (i y) anchored at the exact w(x). Uses the
// recurrence w^{(k+1)} = -2 x w^{(k)} - 2 k w^{(k-1)} from w' = -2 z w + 2i/sqrt(pi).
cdouble w_taylor_y(double x, double y) {
  const cdouble w0 = w_real(x);
  const cdouble iy(0.0, y);
  cdouble wm1 = w0;
  cdouble wk = -2.0 * x * w0 + cdouble(0.0, 2.0 * kInvSqrtPi);  // w'
  cdouble total = w0 + wk * iy;
  cdouble pow = iy;
  for (int k = 1; k < 44; ++k) {
    const cdouble wkp1 = -2.0 * x * wk - 2.0 * static_cast<double>(k) * wm1;
    pow *= iy / static_cast<double>(k + 1);
    total += wkp1 * pow;
    wm1 = wk;
    wk = wkp1;
  }
  return total;
}

// Faddeeva continued fraction w(z) = (i/sqrt(pi)) / (z - (1/2)/(z - (2/2)/...)).
cdouble w_cf(cdouble z) {
  const double az = std::abs(z);
  int n = static_cast<int>(30.0 + 400.0 / az);
  if (n > 256) n = 256;
  cdouble f(0.0, 0.0);
  for (int k = n; k >= 1; --k) f = (0.5 * k) / (z - f);
  return cdouble(0.0, kInvSqrtPi) / (z - f);
}

// Maclaurin series w(z) = sum_n (i z)^n / Gamma(n/2 + 1).
cdouble w_series(cdouble z) {
  const cdouble iz(0.0, 1.0);
  const cdouble q = iz * z;
  cdouble total(0.0, 0.0), p(1.0, 0.0);
  for (int m = 0; m < 90; ++m) {
    total += p / std::tgamma(0.5 * m + 1.0);
    p *= q;
  }
  return total;
}

cdouble wofz_upper(double x, double y) {  // y >= 0
  const double xa = std::fabs(x);
  cdouble w;
  if (y <= 1.3) {
    w = w_taylor_y(xa, y);
  } else {
    const cdouble z(xa, y);
    w = (std::abs(z) >= 2.5) ? w_cf(z) : w_series(z);
  }
  return (x < 0.0) ? std::conj(w) : w;
}

}  // namespace

double erfcx(double x) {
  if (x >= 0.0) {
    if (x < 25.0) return std::exp(x * x) * std::erfc(x);
    // continued fraction erfcx(x) = (1/sqrt(pi)) / (x + (1/2)/(x + (2/2)/...))
    double f = 0.0;
    for (int k = 60; k >= 1; --k) f = (0.5 * k) / (x + f);
    return kInvSqrtPi / (x + f);
  }
  if (x < -26.6) return INFINITY;  // 2 exp(x^2) overflows
  return 2.0 * std::exp(x * x) - erfcx(-x);
}

double dawsn(double x) { return dawson_impl(x); }

std::complex<double> wofz(std::complex<double> z) {
  if (z.imag() >= 0.0) return wofz_upper(z.real(), z.imag());
  // lower half-plane via the reflection w(-z) = 2 exp(-z^2) - w(z)
  return 2.0 * std::exp(-z * z) - wofz_upper(-z.real(), -z.imag());
}

double voigt_profile(double x, double sigma, double gamma) {
  if (sigma <= 0.0) return std::nan("");
  if (gamma == 0.0) {  // pure Gaussian limit
    return std::exp(-x * x / (2.0 * sigma * sigma)) / (sigma * kSqrt2Pi);
  }
  const double d = sigma * kSqrt2;
  const cdouble w = wofz_upper(x / d, gamma / d);
  return w.real() / (sigma * kSqrt2Pi);
}

fresnel_t fresnel(double x) {
  // C(x) + i S(x) = (1+i)/2 [1 - exp(-z^2) w(i z)], z = (sqrt(pi)/2)(1 - i) x.
  const double s = (x >= 0.0) ? 1.0 : -1.0;
  const double a = std::fabs(x);
  const cdouble z = cdouble(kSqrtPi / 2.0, -kSqrtPi / 2.0) * a;  // (sqrt(pi)/2)(1-i)a
  const cdouble iz(-z.imag(), z.real());                        // i z, upper half-plane
  const cdouble erf = 1.0 - std::exp(-z * z) * wofz_upper(iz.real(), iz.imag());
  const cdouble cs = cdouble(0.5, 0.5) * erf;
  return {s * cs.imag(), s * cs.real()};
}

ndarray erfcx(const ndarray& x) { return detail::map(x, [](double v) { return erfcx(v); }); }
ndarray dawsn(const ndarray& x) { return detail::map(x, [](double v) { return dawsn(v); }); }

}  // namespace scypp::special
