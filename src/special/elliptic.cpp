// Elliptic integrals, parameterized by m = k^2 (SciPy convention).
//   * complete K(m), E(m): arithmetic-geometric-mean iteration (A&S 17.6).
//   * incomplete F(phi|m), E(phi|m): Carlson symmetric forms R_F, R_D.
//   * ellipj: descending Landen/AGM transformation (Cephes ellpj).
// Out-of-domain inputs return nan/inf (no throw), matching scipy.special.

#include "scypp/special/special.hpp"

#include <cmath>

namespace scypp::special {
namespace {

constexpr double kPi = 3.14159265358979323846;

double max3(double a, double b, double c) {
  return std::fmax(std::fmax(a, b), c);
}

// Carlson R_F(x, y, z): symmetric elliptic integral of the first kind.
double carlson_rf(double x, double y, double z) {
  constexpr double kErrTol = 0.0008;
  double xt = x, yt = y, zt = z, ave, dx, dy, dz;
  do {
    const double sx = std::sqrt(xt), sy = std::sqrt(yt), sz = std::sqrt(zt);
    const double lam = sx * (sy + sz) + sy * sz;
    xt = 0.25 * (xt + lam);
    yt = 0.25 * (yt + lam);
    zt = 0.25 * (zt + lam);
    ave = (xt + yt + zt) / 3.0;
    dx = (ave - xt) / ave;
    dy = (ave - yt) / ave;
    dz = (ave - zt) / ave;
  } while (max3(std::fabs(dx), std::fabs(dy), std::fabs(dz)) > kErrTol);
  const double e2 = dx * dy - dz * dz;
  const double e3 = dx * dy * dz;
  return (1.0 + (e2 / 24.0 - 0.1 - 3.0 * e3 / 44.0) * e2 + e3 / 14.0) /
         std::sqrt(ave);
}

// Carlson R_D(x, y, z): symmetric elliptic integral of the second kind.
double carlson_rd(double x, double y, double z) {
  constexpr double kErrTol = 0.0005;
  double xt = x, yt = y, zt = z, ave, dx, dy, dz;
  double sum = 0.0, fac = 1.0;
  do {
    const double sx = std::sqrt(xt), sy = std::sqrt(yt), sz = std::sqrt(zt);
    const double lam = sx * (sy + sz) + sy * sz;
    sum += fac / (sz * (zt + lam));
    fac *= 0.25;
    xt = 0.25 * (xt + lam);
    yt = 0.25 * (yt + lam);
    zt = 0.25 * (zt + lam);
    ave = 0.2 * (xt + yt + 3.0 * zt);
    dx = (ave - xt) / ave;
    dy = (ave - yt) / ave;
    dz = (ave - zt) / ave;
  } while (max3(std::fabs(dx), std::fabs(dy), std::fabs(dz)) > kErrTol);
  const double ea = dx * dy, eb = dz * dz, ec = ea - eb;
  const double ed = ea - 6.0 * eb, ee = ed + ec + ec;
  return 3.0 * sum +
         fac *
             (1.0 + ed * (-3.0 / 14.0 + (9.0 / 88.0) * ed - (9.0 / 52.0) * dz * ee) +
              dz * ((1.0 / 6.0) * ee +
                    dz * (-(9.0 / 22.0) * ec + dz * (3.0 / 26.0) * ea))) /
             (ave * std::sqrt(ave));
}

// AGM of (1, b); returns the common limit.
double agm1(double b) {
  double a = 1.0;
  for (int n = 0; n < 100; ++n) {
    const double an = 0.5 * (a + b);
    const double bn = std::sqrt(a * b);
    if (std::fabs(a - b) <= 1e-17 * std::fabs(a)) return an;
    a = an;
    b = bn;
  }
  return a;
}

}  // namespace

double ellipk(double m) {
  if (m > 1.0) return std::nan("");
  if (m == 1.0) return INFINITY;
  return kPi / (2.0 * agm1(std::sqrt(1.0 - m)));
}

double ellipkm1(double p) {
  if (p < 0.0) return std::nan("");  // m > 1
  if (p == 0.0) return INFINITY;     // m == 1
  return kPi / (2.0 * agm1(std::sqrt(p)));
}

double ellipe(double m) {
  if (m > 1.0) return std::nan("");
  if (m == 1.0) return 1.0;
  // A&S 17.6: a0=1, b0=sqrt(1-m), c0^2 = m; E = K*(1 - sum 2^{n-1} c_n^2).
  double a = 1.0, b = std::sqrt(1.0 - m);
  double sum = 0.5 * m;   // n = 0 term: 2^{-1} c_0^2
  double coef = 1.0;      // 2^{n-1} for n = 1
  for (int n = 1; n < 100; ++n) {
    const double cn = 0.5 * (a - b);
    const double an = 0.5 * (a + b);
    const double bn = std::sqrt(a * b);
    a = an;
    b = bn;
    sum += coef * cn * cn;
    coef *= 2.0;
    // Stop once c_n is at the rounding floor: it can stagnate at ~1e-16 while
    // the doubling `coef` would otherwise amplify that noise.
    if (std::fabs(cn) <= 1e-15 * a) break;
  }
  const double k = kPi / (2.0 * a);
  return k * (1.0 - sum);
}

double ellipkinc(double phi, double m) {
  const long n = std::lround(phi / kPi);
  const double phir = phi - static_cast<double>(n) * kPi;  // in [-pi/2, pi/2]
  const double s = std::sin(phir), c = std::cos(phir);
  const double y = 1.0 - m * s * s;
  const double fred = s * carlson_rf(c * c, y, 1.0);
  return n != 0 ? fred + 2.0 * n * ellipk(m) : fred;
}

double ellipeinc(double phi, double m) {
  const long n = std::lround(phi / kPi);
  const double phir = phi - static_cast<double>(n) * kPi;
  const double s = std::sin(phir), c = std::cos(phir);
  const double s2 = s * s, y = 1.0 - m * s2;
  const double ered =
      s * carlson_rf(c * c, y, 1.0) - (m / 3.0) * s * s2 * carlson_rd(c * c, y, 1.0);
  return n != 0 ? ered + 2.0 * n * ellipe(m) : ered;
}

ellipj_t ellipj(double u, double m) {
  if (m == 0.0) return {std::sin(u), std::cos(u), 1.0, u};
  if (m == 1.0) {
    const double ch = 1.0 / std::cosh(u);
    return {std::tanh(u), ch, ch, 2.0 * std::atan(std::exp(u)) - 0.5 * kPi};
  }
  // Descending Landen / AGM (Cephes ellpj).
  double a[9], c[9];
  a[0] = 1.0;
  double b = std::sqrt(1.0 - m);
  c[0] = std::sqrt(m);
  double twon = 1.0;
  int i = 0;
  while (std::fabs(c[i] / a[i]) > 1e-16) {
    if (i >= 8) break;
    const double ai = a[i];
    ++i;
    c[i] = 0.5 * (ai - b);
    const double t = std::sqrt(ai * b);
    a[i] = 0.5 * (ai + b);
    b = t;
    twon *= 2.0;
  }
  double phi = twon * a[i] * u;
  double prev = phi;
  do {
    const double t = c[i] * std::sin(phi) / a[i];
    prev = phi;
    phi = 0.5 * (std::asin(t) + phi);
  } while (--i);
  const double cp = std::cos(phi);
  return {std::sin(phi), cp, cp / std::cos(phi - prev), phi};
}

}  // namespace scypp::special
