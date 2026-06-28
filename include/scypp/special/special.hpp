#pragma once
// scypp::special — port of scipy.special (Phase 1 subset).
// Every element-wise function offers a scalar `double` overload and an
// `ndarray` overload. Domain edges follow SciPy: out-of-domain returns nan/inf
// rather than throwing.

#include <complex>
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

// ---- error-function relatives --------------------------------------------
// erfcx(x) = exp(x^2) erfc(x); dawsn(x) = (sqrt(pi)/2) exp(-x^2) erfi(x);
// wofz(z)  = w(z) = exp(-z^2) erfc(-i z) (Faddeeva function, delivered on the
// upper half-plane Im(z) >= 0 that the Voigt profile and Fresnel integrals
// require); voigt_profile(x, sigma, gamma) = Re[w((x + i gamma)/(sigma sqrt2))]
// / (sigma sqrt(2 pi)); fresnel(x) returns the pair (S, C) of Fresnel integrals.
double  erfcx(double x);
double  dawsn(double x);
double  voigt_profile(double x, double sigma, double gamma);
std::complex<double> wofz(std::complex<double> z);
struct fresnel_t { double S, C; };
fresnel_t fresnel(double x);

ndarray erfcx(const ndarray& x);
ndarray dawsn(const ndarray& x);

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

// ---- spherical Bessel functions -------------------------------------------
// Integer order n >= 0 via the half-integer relation to the cylindrical
// kernels: spherical_jn(n,x) = sqrt(pi/(2x)) jv(n+1/2, x) (yv/iv/kv likewise).
// x = 0 limits and negative-argument continuations match SciPy.
double spherical_jn(int n, double x);
double spherical_yn(int n, double x);
double spherical_in(int n, double x);
double spherical_kn(int n, double x);

// ---- sine / cosine integrals ----------------------------------------------
// sici(x)   returns (Si, Ci): Si(x)=int_0^x sin t/t dt, Ci its cosine analogue.
// shichi(x) returns (Shi, Chi): the hyperbolic sine/cosine integrals.
struct sici_t { double Si, Ci; };
struct shichi_t { double Shi, Chi; };
sici_t sici(double x);
shichi_t shichi(double x);

// ---- Airy functions -------------------------------------------------------
// Ai, Ai', Bi, Bi' evaluated together (SciPy returns a 4-tuple). `airye`
// returns the exponentially-scaled variants: for x >= 0,
//   eAi = Ai*exp(zeta), eAip = Aip*exp(zeta), eBi = Bi*exp(-zeta),
//   eBip = Bip*exp(-zeta)  with zeta = (2/3) x^{3/2};
// for x < 0, eAi/eAip are nan (as SciPy) while eBi/eBip equal Bi/Bip.
struct airy_t { double Ai, Aip, Bi, Bip; };
airy_t airy(double x);
airy_t airye(double x);

// ---- elliptic integrals ---------------------------------------------------
// Parameterized by m = k^2 (SciPy convention). Complete integrals use the
// arithmetic-geometric-mean iteration; incomplete integrals use Carlson
// symmetric forms; ellipj uses the descending Landen/AGM transformation.
double ellipk(double m);       // complete K(m)
double ellipkm1(double p);     // K(1 - p), accurate for small p (m near 1)
double ellipe(double m);       // complete E(m)
double ellipkinc(double phi, double m);  // incomplete F(phi | m)
double ellipeinc(double phi, double m);  // incomplete E(phi | m)
struct ellipj_t { double sn, cn, dn, ph; };
ellipj_t ellipj(double u, double m);     // Jacobi elliptic functions

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
