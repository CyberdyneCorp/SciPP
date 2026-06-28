// Airy functions Ai, Bi and their derivatives, built on the modified/ordinary
// Bessel kernels (DLMF 9.6). For x > 0 the connection is via K_{1/3}/I_{1/3}
// (with negative-order I removed via I_{-v}=I_v+(2/pi)sin(v*pi)K_v); for x < 0
// via J_{1/3}/Y_{1/3} (negative orders removed via the reflection formula).
// x == 0 returns the exact values. Matches scipy.special.airy/airye.

#include "scypp/special/special.hpp"

#include <cmath>

namespace scypp::special {
namespace {

constexpr double kPi = 3.14159265358979323846;
const double kSqrt3 = std::sqrt(3.0);

// Exact values at the origin.
constexpr double kAi0 = 0.355028053887817239260;
constexpr double kAip0 = -0.258819403792806798405;
constexpr double kBi0 = 0.614926627446000735150;
constexpr double kBip0 = 0.448288357353826357692;

airy_t airy_pos(double x) {
  const double zeta = (2.0 / 3.0) * std::pow(x, 1.5);
  const double K13 = kv(1.0 / 3.0, zeta);
  const double K23 = kv(2.0 / 3.0, zeta);
  const double I13 = iv(1.0 / 3.0, zeta);
  const double I23 = iv(2.0 / 3.0, zeta);
  const double s3pi = kSqrt3 / kPi;
  airy_t r;
  r.Ai = (1.0 / kPi) * std::sqrt(x / 3.0) * K13;
  r.Aip = -(1.0 / kPi) * (x / kSqrt3) * K23;
  r.Bi = std::sqrt(x / 3.0) * (2.0 * I13 + s3pi * K13);
  r.Bip = (x / kSqrt3) * (2.0 * I23 + s3pi * K23);
  return r;
}

airy_t airy_neg(double x) {
  const double z = -x;  // |x|
  const double zeta = (2.0 / 3.0) * std::pow(z, 1.5);
  const double J1 = jv(1.0 / 3.0, zeta);
  const double Y1 = yv(1.0 / 3.0, zeta);
  const double J2 = jv(2.0 / 3.0, zeta);
  const double Y2 = yv(2.0 / 3.0, zeta);
  const double h3 = 0.5 * kSqrt3;
  airy_t r;
  r.Ai = (std::sqrt(z) / 3.0) * (1.5 * J1 - h3 * Y1);
  r.Aip = (z / 3.0) * (1.5 * J2 + h3 * Y2);
  r.Bi = std::sqrt(z / 3.0) * (-0.5 * J1 - h3 * Y1);
  r.Bip = (z / kSqrt3) * (0.5 * J2 - h3 * Y2);
  return r;
}

}  // namespace

airy_t airy(double x) {
  if (x == 0.0) return {kAi0, kAip0, kBi0, kBip0};
  return x > 0.0 ? airy_pos(x) : airy_neg(x);
}

airy_t airye(double x) {
  airy_t r = airy(x);
  if (x >= 0.0) {
    const double zeta = (2.0 / 3.0) * std::pow(x, 1.5);
    const double e = std::exp(zeta);
    r.Ai *= e;
    r.Aip *= e;
    r.Bi /= e;
    r.Bip /= e;
  } else {
    // SciPy: the scaled Ai/Aip are nan for negative real argument; Bi/Bip
    // are returned unscaled.
    r.Ai = std::nan("");
    r.Aip = std::nan("");
  }
  return r;
}

}  // namespace scypp::special
