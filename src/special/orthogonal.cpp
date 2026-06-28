// Orthogonal-polynomial evaluators. Legendre/Hermite/Laguerre come from the
// C++17 special-math family; Chebyshev T and generalized Laguerre use stable
// three-term recurrences matching SciPy's eval_* (physicists' Hermite,
// L_n^{(alpha)} generalized Laguerre).

#include "scipp/special/special.hpp"

#include <cmath>

#include "scipp/detail/elementwise.hpp"

namespace scipp::special {

double eval_legendre(int n, double x) { return std::legendre(static_cast<unsigned>(n), x); }
double eval_hermite(int n, double x) { return std::hermite(static_cast<unsigned>(n), x); }
double eval_laguerre(int n, double x) { return std::laguerre(static_cast<unsigned>(n), x); }

double eval_chebyt(int n, double x) {
  if (n == 0) return 1.0;
  if (n == 1) return x;
  double tkm1 = 1.0, tk = x;
  for (int kk = 2; kk <= n; ++kk) {
    double tkp1 = 2.0 * x * tk - tkm1;
    tkm1 = tk;
    tk = tkp1;
  }
  return tk;
}

double eval_genlaguerre(int n, double alpha, double x) {
  if (n == 0) return 1.0;
  double lkm1 = 1.0;
  double lk = 1.0 + alpha - x;
  for (int kk = 1; kk < n; ++kk) {
    double lkp1 = ((2.0 * kk + 1.0 + alpha - x) * lk - (kk + alpha) * lkm1) / (kk + 1.0);
    lkm1 = lk;
    lk = lkp1;
  }
  return lk;
}

ndarray eval_legendre(int n, const ndarray& x) {
  return detail::map(x, [n](double v) { return eval_legendre(n, v); });
}
ndarray eval_chebyt(int n, const ndarray& x) {
  return detail::map(x, [n](double v) { return eval_chebyt(n, v); });
}
ndarray eval_hermite(int n, const ndarray& x) {
  return detail::map(x, [n](double v) { return eval_hermite(n, v); });
}
ndarray eval_laguerre(int n, const ndarray& x) {
  return detail::map(x, [n](double v) { return eval_laguerre(n, v); });
}
ndarray eval_genlaguerre(int n, double alpha, const ndarray& x) {
  return detail::map(x, [n, alpha](double v) { return eval_genlaguerre(n, alpha, v); });
}

}  // namespace scipp::special
