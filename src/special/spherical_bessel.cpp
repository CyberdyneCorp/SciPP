// Spherical Bessel functions of integer order, built on the half-integer
// cylindrical Bessel kernels (DLMF 10.47): for x > 0,
//   j_n(x) = sqrt(pi/(2x)) J_{n+1/2}(x),  y_n(x) = sqrt(pi/(2x)) Y_{n+1/2}(x),
//   i_n(x) = sqrt(pi/(2x)) I_{n+1/2}(x),  k_n(x) = sqrt(pi/(2x)) K_{n+1/2}(x).
// The x = 0 limits and the negative-argument analytic continuations follow
// scipy.special.spherical_jn/yn/in/kn. Ordinary spherical functions are even
// or odd: j_n(-x)=(-1)^n j_n(x), y_n(-x)=(-1)^{n+1} y_n(x), i_n(-x)=(-1)^n i_n(x);
// k_n has no such parity so its negative branch uses the closed-form recurrence
// k_{n+1}=k_{n-1}+(2n+1)/x k_n seeded by k_0=(pi/2x)e^{-x}, k_1=k_0(1+1/x),
// which is the analytic continuation SciPy reports.

#include "scypp/special/special.hpp"

#include <cmath>

namespace scypp::special {
namespace {

constexpr double kPi = 3.14159265358979323846;
constexpr double kPiHalf = 1.57079632679489661923;  // pi/2

inline double prefactor(double a) { return std::sqrt(kPiHalf / a); }

}  // namespace

double spherical_jn(int n, double x) {
  if (x == 0.0) return n == 0 ? 1.0 : 0.0;
  const double a = std::fabs(x);
  double v = prefactor(a) * jv(n + 0.5, a);
  if (x < 0.0 && (n & 1)) v = -v;  // (-1)^n
  return v;
}

double spherical_yn(int n, double x) {
  if (x == 0.0) return -INFINITY;
  const double a = std::fabs(x);
  double v = prefactor(a) * yv(n + 0.5, a);
  if (x < 0.0 && !(n & 1)) v = -v;  // (-1)^(n+1)
  return v;
}

double spherical_in(int n, double x) {
  if (x == 0.0) return n == 0 ? 1.0 : 0.0;
  const double a = std::fabs(x);
  double v = prefactor(a) * iv(n + 0.5, a);
  if (x < 0.0 && (n & 1)) v = -v;  // (-1)^n
  return v;
}

double spherical_kn(int n, double x) {
  if (x == 0.0) return INFINITY;
  if (x > 0.0) return prefactor(x) * kv(n + 0.5, x);
  // x < 0: analytic continuation via the closed-form upward recurrence.
  const double k0 = (kPi / (2.0 * x)) * std::exp(-x);
  if (n == 0) return k0;
  double km1 = k0, km = k0 * (1.0 + 1.0 / x);  // k_0, k_1
  for (int m = 1; m < n; ++m) {
    const double kp = km1 + (2.0 * m + 1.0) / x * km;
    km1 = km;
    km = kp;
  }
  return km;
}

}  // namespace scypp::special
