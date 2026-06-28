// Combinatorial helpers: comb, perm, factorial. Exact mode uses 128-bit integer
// arithmetic and falls back to the gamma-based float path on overflow; inexact
// mode uses lgamma, matching SciPy.

#include "scipp/special/special.hpp"

#include <cmath>
#include <limits>

namespace scipp::special {
namespace {

constexpr unsigned __int128 kU128Max = ~static_cast<unsigned __int128>(0);

// Multiply with overflow detection; returns false on overflow.
bool mul_checked(unsigned __int128& acc, unsigned __int128 m) {
  if (m != 0 && acc > kU128Max / m) return false;
  acc *= m;
  return true;
}

}  // namespace

double factorial(int n, bool exact) {
  if (n < 0) return 0.0;
  if (exact) {
    unsigned __int128 acc = 1;
    for (int i = 2; i <= n; ++i) {
      if (!mul_checked(acc, static_cast<unsigned __int128>(i))) {
        return std::tgamma(static_cast<double>(n) + 1.0);  // overflow -> float path
      }
    }
    return static_cast<double>(acc);
  }
  return std::tgamma(static_cast<double>(n) + 1.0);
}

double perm(int n, int k, bool exact) {
  if (k < 0 || n < 0 || k > n) return 0.0;
  if (exact) {
    unsigned __int128 acc = 1;
    for (int i = 0; i < k; ++i) {
      if (!mul_checked(acc, static_cast<unsigned __int128>(n - i))) {
        return std::exp(std::lgamma(n + 1.0) - std::lgamma(n - k + 1.0));
      }
    }
    return static_cast<double>(acc);
  }
  return std::exp(std::lgamma(n + 1.0) - std::lgamma(n - k + 1.0));
}

double comb(int n, int k, bool exact) {
  if (k < 0 || n < 0 || k > n) return 0.0;
  if (k > n - k) k = n - k;  // symmetry
  if (exact) {
    unsigned __int128 num = 1, den = 1;
    bool ok = true;
    for (int i = 0; i < k; ++i) {
      ok = ok && mul_checked(num, static_cast<unsigned __int128>(n - i));
      ok = ok && mul_checked(den, static_cast<unsigned __int128>(i + 1));
      if (!ok) break;
    }
    if (ok) return static_cast<double>(num / den);
    // overflow -> float path
  }
  return std::exp(std::lgamma(n + 1.0) - std::lgamma(k + 1.0) - std::lgamma(n - k + 1.0));
}

}  // namespace scipp::special
