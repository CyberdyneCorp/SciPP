// Sine/cosine integrals Si/Ci and their hyperbolic counterparts Shi/Chi
// (scipy.special.sici / shichi), real argument.
//
//   Si(x) = int_0^x sin t / t dt,  Ci(x) = gamma + ln|x| + int_0^x (cos t-1)/t dt
//   Shi(x)= int_0^x sinh t / t dt, Chi(x)= gamma + ln|x| + int_0^x (cosh t-1)/t dt
//
// Si/Ci use the convergent Maclaurin series for small |x| and the divergent
// auxiliary-function asymptotic expansion (DLMF 6.12) with optimal truncation
// for large |x|. Shi/Chi reduce to the exponential integrals already in the
// module: Shi = (Ei + E1)/2, Chi = (Ei - E1)/2 for x > 0; the small-|x| branch
// uses the series to avoid Ei/E1 cancellation. Si and Shi are odd; Ci and Chi
// are even with Ci(0)=Chi(0)=-inf.

#include "scipp/special/special.hpp"

#include <cmath>

namespace scipp::special {
namespace {

constexpr double kPiHalf = 1.57079632679489661923;     // pi/2
constexpr double kEuler = 0.57721566490153286061;      // Euler-Mascheroni
constexpr double kAsympCut = 17.0;  // |x| crossover: series below, asymptotic above
constexpr double kHypSeriesCut = 2.0;  // |x| below: Shi/Chi via series

// Si/Ci by Maclaurin series (valid for all x; used where cancellation is mild).
sici_t sici_series(double a) {
  double p = a, si = a;  // Si term k=0: x/1
  for (int k = 1; k < 600; ++k) {
    p *= -a * a / ((2.0 * k) * (2.0 * k + 1.0));
    const double add = p / (2.0 * k + 1.0);
    si += add;
    if (std::fabs(add) <= 1e-19 * std::fabs(si)) break;
  }
  double m = a * a / 2.0, cs = -m / 2.0, sign = -1.0;  // Ci sum, term k=1
  for (int k = 2; k < 600; ++k) {
    m *= a * a / ((2.0 * k - 1.0) * (2.0 * k));
    sign = -sign;
    cs += sign * m / (2.0 * k);
    if (m / (2.0 * k) <= 1e-19 * (std::fabs(cs) + 1.0)) break;
  }
  return {si, kEuler + std::log(a) + cs};
}

// Si/Ci by asymptotic auxiliary functions f, g with optimal (smallest-term)
// truncation:  Si = pi/2 - f cos x - g sin x,  Ci = f sin x - g cos x.
sici_t sici_asymptotic(double a) {
  const double inv2 = 1.0 / (a * a);
  double term = 1.0 / a, f = term, prev = std::fabs(term);
  for (int m = 1; m < 200; ++m) {
    term *= -(2.0 * m - 1.0) * (2.0 * m) * inv2;
    if (std::fabs(term) > prev) break;
    f += term;
    prev = std::fabs(term);
  }
  term = inv2;
  double g = term;
  prev = std::fabs(term);
  for (int m = 1; m < 200; ++m) {
    term *= -(2.0 * m) * (2.0 * m + 1.0) * inv2;
    if (std::fabs(term) > prev) break;
    g += term;
    prev = std::fabs(term);
  }
  const double c = std::cos(a), s = std::sin(a);
  return {kPiHalf - f * c - g * s, f * s - g * c};
}

// Shi/Chi by Maclaurin series (small |x|, avoids Ei/E1 cancellation).
shichi_t shichi_series(double a) {
  double p = a, shi = a;  // Shi term k=0
  for (int k = 1; k < 600; ++k) {
    p *= a * a / ((2.0 * k) * (2.0 * k + 1.0));
    const double add = p / (2.0 * k + 1.0);
    shi += add;
    if (add <= 1e-19 * shi) break;
  }
  double m = a * a / 2.0, cs = m / 2.0;  // Chi sum term k=1
  for (int k = 2; k < 600; ++k) {
    m *= a * a / ((2.0 * k - 1.0) * (2.0 * k));
    cs += m / (2.0 * k);
    if (m / (2.0 * k) <= 1e-19 * (cs + 1.0)) break;
  }
  return {shi, kEuler + std::log(a) + cs};
}

}  // namespace

sici_t sici(double x) {
  if (x == 0.0) return {0.0, -INFINITY};
  const double a = std::fabs(x);
  sici_t r = a < kAsympCut ? sici_series(a) : sici_asymptotic(a);
  if (x < 0.0) r.Si = -r.Si;  // Si is odd, Ci is even
  return r;
}

shichi_t shichi(double x) {
  if (x == 0.0) return {0.0, -INFINITY};
  const double a = std::fabs(x);
  shichi_t r;
  if (a < kHypSeriesCut) {
    r = shichi_series(a);
  } else {
    const double ei = expi(a), e1 = exp1(a);
    r = {0.5 * (ei + e1), 0.5 * (ei - e1)};
  }
  if (x < 0.0) r.Shi = -r.Shi;  // Shi is odd, Chi is even
  return r;
}

}  // namespace scipp::special
