#pragma once
// scypp::special — port of scipy.special (Phase 1 subset).
// Every element-wise function offers a scalar `double` overload and an
// `ndarray` overload. Domain edges follow SciPy: out-of-domain returns nan/inf
// rather than throwing.

#include <cstdint>
#include <optional>

#include "numpp/core/ndarray.hpp"

namespace scypp::special {

using numpp::ndarray;

// ---- gamma family ---------------------------------------------------------
double  gamma(double x);
double  gammaln(double x);     // log|Gamma(x)|
double  loggamma(double x);    // real branch of log-gamma
double  digamma(double x);     // psi(x)
double  polygamma(int n, double x);
double  beta(double a, double b);
double  betaln(double a, double b);

ndarray gamma(const ndarray& x);
ndarray gammaln(const ndarray& x);
ndarray loggamma(const ndarray& x);
ndarray digamma(const ndarray& x);
ndarray polygamma(int n, const ndarray& x);

// ---- error functions ------------------------------------------------------
double  erf(double x);
double  erfc(double x);
double  erfinv(double y);
double  erfcinv(double y);

ndarray erf(const ndarray& x);
ndarray erfc(const ndarray& x);
ndarray erfinv(const ndarray& y);
ndarray erfcinv(const ndarray& y);

// ---- exponential / logarithmic integrals ----------------------------------
double  exp1(double x);        // E_1(x)
double  expi(double x);        // Ei(x)
double  expn(int n, double x); // E_n(x)
double  exprel(double x);      // (e^x - 1) / x

ndarray exp1(const ndarray& x);
ndarray expi(const ndarray& x);
ndarray exprel(const ndarray& x);

// ---- Bessel functions -----------------------------------------------------
double  jv(double v, double x);
double  yv(double v, double x);
double  iv(double v, double x);
double  kv(double v, double x);
double  jn(int n, double x);
double  yn(int n, double x);
double  i0(double x);
double  i1(double x);
double  k0(double x);
double  k1(double x);

ndarray jv(double v, const ndarray& x);
ndarray yv(double v, const ndarray& x);
ndarray iv(double v, const ndarray& x);
ndarray kv(double v, const ndarray& x);
ndarray i0(const ndarray& x);
ndarray i1(const ndarray& x);
ndarray k0(const ndarray& x);
ndarray k1(const ndarray& x);

// ---- orthogonal polynomial evaluators -------------------------------------
double  eval_legendre(int n, double x);
double  eval_chebyt(int n, double x);
double  eval_hermite(int n, double x);
double  eval_laguerre(int n, double x);
double  eval_genlaguerre(int n, double alpha, double x);

ndarray eval_legendre(int n, const ndarray& x);
ndarray eval_chebyt(int n, const ndarray& x);
ndarray eval_hermite(int n, const ndarray& x);
ndarray eval_laguerre(int n, const ndarray& x);
ndarray eval_genlaguerre(int n, double alpha, const ndarray& x);

// ---- combinatorics --------------------------------------------------------
double comb(int n, int k, bool exact = false);
double perm(int n, int k, bool exact = false);
double factorial(int n, bool exact = false);

// ---- stable reductions ----------------------------------------------------
double  logsumexp(const ndarray& a);
ndarray logsumexp(const ndarray& a, int axis, bool keepdims = false);
ndarray softmax(const ndarray& a, std::optional<int> axis = std::nullopt);
ndarray log_softmax(const ndarray& a, std::optional<int> axis = std::nullopt);

}  // namespace scypp::special
