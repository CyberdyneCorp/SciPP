# special Specification

## Purpose
TBD - created by archiving change add-special-constants. Update Purpose after archive.
## Requirements
### Requirement: Gamma and related functions

`scypp::special` SHALL provide `gamma`, `gammaln`, `loggamma`, `digamma`,
`polygamma`, `beta` and `betaln`, evaluated element-wise over `numpp::ndarray`
with a scalar `double` overload, matching SciPy within documented tolerance.
(oracle: scipy/special/_basic.py, scipy/special/cephes)

#### Scenario: Gamma matches SciPy and obeys the recurrence
- GIVEN an array of real arguments excluding non-positive integers
- WHEN `gamma(x)` is evaluated
- THEN the result is `allclose` to SciPy's and satisfies `gamma(x+1) ≈ x*gamma(x)`

#### Scenario: Gamma poles and negative integers
- GIVEN arguments at or below zero
- WHEN `gamma(x)` is evaluated at a non-positive integer
- THEN the result is non-finite (`inf`/`nan`) as SciPy returns, without throwing

#### Scenario: Log-gamma and beta consistency
- GIVEN positive arguments `a`, `b`
- WHEN `gammaln`, `beta` and `betaln` are evaluated
- THEN `gammaln(x)` is `allclose` to `log(abs(gamma(x)))`, and
  `betaln(a,b)` is `allclose` to `gammaln(a)+gammaln(b)-gammaln(a+b)`, both
  matching SciPy

#### Scenario: Digamma and polygamma
- GIVEN positive arguments
- WHEN `digamma(x)` and `polygamma(n,x)` are evaluated
- THEN they are `allclose` to SciPy, with `polygamma(0,x) == digamma(x)`

### Requirement: Error functions

`scypp::special` SHALL provide `erf`, `erfc`, `erfinv` and `erfcinv`,
element-wise over `numpp::ndarray` with scalar overloads, matching SciPy within
documented tolerance. (oracle: scipy/special/cephes/ndtr.c)

#### Scenario: erf/erfc match SciPy and are complementary
- GIVEN an array of real arguments
- WHEN `erf(x)` and `erfc(x)` are evaluated
- THEN both are `allclose` to SciPy and `erf(x) + erfc(x) ≈ 1`

#### Scenario: Inverse error functions round-trip
- GIVEN values `y` in `(-1, 1)`
- WHEN `erfinv(y)` is evaluated
- THEN `erf(erfinv(y)) ≈ y` within tolerance, matching SciPy, and `erfcinv` is the
  inverse of `erfc`

### Requirement: Exponential and logarithmic integrals

`scypp::special` SHALL provide `expn`, `exp1`, `expi` and `exprel`, element-wise
with scalar overloads, matching SciPy within documented tolerance. `exprel` SHALL
be numerically stable near zero. (oracle: scipy/special/cephes/expn.c)

#### Scenario: Exponential integrals match SciPy
- GIVEN positive arguments
- WHEN `exp1(x)`, `expi(x)` and `expn(n,x)` are evaluated
- THEN the results are `allclose` to SciPy

#### Scenario: exprel is stable at the origin
- GIVEN arguments approaching 0
- WHEN `exprel(x)` is evaluated
- THEN it is `allclose` to SciPy and approaches 1 as `x → 0` without catastrophic
  cancellation

### Requirement: Bessel functions

`scypp::special` SHALL provide the Bessel functions `jv`, `yv`, `iv`, `kv`,
integer-order `jn`/`yn`, and the common cases `i0`, `i1`, `k0`, `k1`, element-wise
over the argument with scalar overloads, matching SciPy within documented
tolerance. (oracle: scipy/special/cephes/jv.c, iv.c, kv.c)

#### Scenario: Bessel values match SciPy
- GIVEN an order `v` and an array of arguments
- WHEN `jv(v,x)`, `yv(v,x)`, `iv(v,x)` and `kv(v,x)` are evaluated
- THEN the results are `allclose` to SciPy within the per-function tolerance

#### Scenario: Specialized i0/i1/k0/k1 agree with the general form
- GIVEN an array of positive arguments
- WHEN `i0(x)`/`i1(x)`/`k0(x)`/`k1(x)` are evaluated
- THEN each is `allclose` to the corresponding `iv`/`kv` at integer order and to
  SciPy

### Requirement: Orthogonal polynomial evaluators

`scypp::special` SHALL provide `eval_legendre`, `eval_chebyt`, `eval_hermite`,
`eval_laguerre` and `eval_genlaguerre`, evaluating the degree-`n` polynomial at
the given points via a stable recurrence, matching SciPy within documented
tolerance. (oracle: scipy/special/_orthogonal.py)

#### Scenario: Evaluators match SciPy
- GIVEN a non-negative degree `n` and an array of points `x`
- WHEN an `eval_*` function is called
- THEN the result is `allclose` to SciPy

#### Scenario: Recurrence stability at higher degree
- GIVEN a moderately high degree (e.g. `n = 50`) over `[-1, 1]` or the relevant
  domain
- WHEN the evaluator is called
- THEN the result remains `allclose` to SciPy (no recurrence blow-up)

### Requirement: Combinatorial functions

`scypp::special` SHALL provide `comb`, `perm` and `factorial`, each supporting an
`exact` mode (integer arithmetic) and an inexact mode (gamma-based float),
matching SciPy. Exact mode SHALL fall back to the float path beyond its integer
range, documented in behavior. (oracle: scipy/special/_basic.py)

#### Scenario: Exact combinatorics match SciPy
- GIVEN small non-negative integers `n`, `k`
- WHEN `comb(n,k, exact=true)`, `perm(n,k, exact=true)` and `factorial(n,
  exact=true)` are evaluated
- THEN the results equal SciPy's exact integer values

#### Scenario: Inexact mode uses the gamma path
- GIVEN larger arguments
- WHEN `comb`/`perm`/`factorial` are called with `exact=false`
- THEN the results are `allclose` to SciPy's floating-point values

### Requirement: Stable log-sum-exp and softmax reductions

`scypp::special` SHALL provide `logsumexp`, `softmax` and `log_softmax` over
`numpp::ndarray` with `axis` and `keepdims` support, using a max-shift for
numerical stability, matching SciPy. (oracle: scipy/special/_logsumexp.py)

#### Scenario: logsumexp matches SciPy and is overflow-safe
- GIVEN an array containing large positive values
- WHEN `logsumexp(a, axis)` is evaluated
- THEN the result is `allclose` to SciPy and does not overflow

#### Scenario: softmax normalizes along the axis
- GIVEN an input array and an axis
- WHEN `softmax(a, axis)` is evaluated
- THEN the result is `allclose` to SciPy and sums to 1 along that axis, and
  `log_softmax(a, axis)` is `allclose` to `log(softmax(a, axis))`

### Requirement: Airy functions

`scypp::special` SHALL provide `airy(x)` returning `(Ai, Aip, Bi, Bip)` and the
exponentially-scaled `airye(x)`, evaluated for real `x` and matching SciPy within
documented tolerance (~1e-9). Out-of-domain behavior follows SciPy: `airye`
returns `nan` for the scaled `Ai`/`Aip` at `x < 0` while `Bi`/`Bip` are unscaled.
(oracle: scipy/special/_basic.py, scipy/special/airy.c)

#### Scenario: Airy and derivatives match SciPy
- GIVEN an array of real arguments spanning negative, zero and positive values
- WHEN `airy(x)` is evaluated
- THEN `Ai`, `Aip`, `Bi`, `Bip` are each `allclose` to `scipy.special.airy`

#### Scenario: Airy satisfies the Wronskian
- GIVEN a real argument `x`
- WHEN `airy(x)` is evaluated
- THEN `Ai*Bip - Aip*Bi` is `allclose` to `1/pi`

#### Scenario: Exponentially scaled Airy matches SciPy
- GIVEN an array of real arguments
- WHEN `airye(x)` is evaluated
- THEN for `x >= 0` the four outputs are `allclose` to `scipy.special.airye`, and
  for `x < 0` the scaled `Ai`/`Aip` are `nan` while `Bi`/`Bip` equal the unscaled
  values

### Requirement: Elliptic integrals

`scypp::special` SHALL provide the complete elliptic integrals `ellipk(m)`,
`ellipkm1(p)` (= `ellipk(1-p)`) and `ellipe(m)` via the arithmetic-geometric-mean
iteration, the incomplete integrals `ellipkinc(phi, m)` and `ellipeinc(phi, m)`
via the Carlson symmetric forms, and the Jacobi elliptic functions
`ellipj(u, m)` returning `(sn, cn, dn, ph)`, all parameterized by `m = k^2` and
matching SciPy within documented tolerance. Out-of-domain inputs return `nan`/`inf`
(no throw): `ellipk(1)`/`ellipkm1(0)` are `inf`, `ellipe(1)` is `1`, and `m > 1`
yields `nan`. (oracle: scipy/special/cephes/ellpk.c, ellpe.c, ellie.c, ellpj.c)

#### Scenario: Complete integrals match SciPy
- GIVEN an array of parameters `m <= 1` (including negative `m`)
- WHEN `ellipk(m)` and `ellipe(m)` are evaluated
- THEN both are `allclose` to `scipy.special.ellipk`/`ellipe`, with
  `ellipk(0) = ellipe(0) = pi/2` and `ellipe(1) = 1`

#### Scenario: ellipkm1 is accurate near m = 1
- GIVEN small `p` in `(0, 1]`
- WHEN `ellipkm1(p)` is evaluated
- THEN it is `allclose` to `scipy.special.ellipkm1(p)` (= `ellipk(1-p)`)

#### Scenario: Incomplete integrals match SciPy and reduce to complete
- GIVEN an amplitude `phi` (including `phi > pi/2` and negative `phi`) and a
  parameter `m`
- WHEN `ellipkinc(phi, m)` and `ellipeinc(phi, m)` are evaluated
- THEN both are `allclose` to `scipy.special.ellipkinc`/`ellipeinc`, and at
  `phi = pi/2` they equal `ellipk(m)`/`ellipe(m)`

#### Scenario: Jacobi functions match SciPy and satisfy identities
- GIVEN an argument `u` and a parameter `m`
- WHEN `ellipj(u, m)` is evaluated
- THEN `sn`, `cn`, `dn`, `ph` are `allclose` to `scipy.special.ellipj`, with
  `sn^2 + cn^2 = 1` and `dn^2 + m*sn^2 = 1`

### Requirement: Error-function relatives

`scypp::special` SHALL provide the scaled complementary error function
`erfcx(x)` = `exp(x^2) erfc(x)`, Dawson's integral `dawsn(x)`, the Faddeeva
function `wofz(z)` = `w(z)` = `exp(-z^2) erfc(-i z)` for complex `z` on the upper
half-plane `Im(z) >= 0`, the `voigt_profile(x, sigma, gamma)` =
`Re[w((x + i gamma)/(sigma sqrt2))] / (sigma sqrt(2 pi))`, and the Fresnel
integrals `fresnel(x)` returning `(S, C)`, all matching SciPy within documented
tolerance (~1e-10 rtol / 1e-12 atol). Out-of-domain behavior follows SciPy
without throwing: `erfcx` returns `+inf` when `2 exp(x^2)` overflows for large
negative `x`, and `voigt_profile` returns `nan` for `sigma <= 0`. (oracle:
scipy/special: erfcx, dawsn, wofz, voigt_profile, fresnel)

#### Scenario: Scaled complementary error function matches SciPy
- GIVEN an array of real arguments spanning negative and large positive values
- WHEN `erfcx(x)` is evaluated
- THEN each result is `allclose` to `scipy.special.erfcx`, `erfcx(0) = 1`, and a
  far-negative argument yields `+inf`

#### Scenario: Dawson's integral matches SciPy
- GIVEN an array of real arguments including the small, moderate and asymptotic
  ranges
- WHEN `dawsn(x)` is evaluated
- THEN each result is `allclose` to `scipy.special.dawsn`, with `dawsn(0) = 0`
  and `dawsn(-x) = -dawsn(x)`

#### Scenario: Fresnel integrals match SciPy
- GIVEN an array of real arguments
- WHEN `fresnel(x)` is evaluated
- THEN `S` and `C` are each `allclose` to `scipy.special.fresnel`, with
  `S(0) = C(0) = 0` and both integrals odd in `x`

#### Scenario: Voigt profile matches SciPy
- GIVEN real `x`, `sigma > 0` and `gamma >= 0` (including the `gamma = 0`
  Gaussian limit)
- WHEN `voigt_profile(x, sigma, gamma)` is evaluated
- THEN the result is `allclose` to `scipy.special.voigt_profile`, is symmetric
  in `x`, and `sigma <= 0` returns `nan`

#### Scenario: Faddeeva function matches SciPy on the upper half-plane
- GIVEN complex arguments with `Im(z) >= 0`
- WHEN `wofz(z)` is evaluated
- THEN the real and imaginary parts are each `allclose` to `scipy.special.wofz`,
  `wofz(0) = 1`, and `wofz(i y)` equals `erfcx(y)` on the imaginary axis

### Requirement: Spherical Bessel functions

`scypp::special` SHALL provide the spherical Bessel functions of integer order
`n >= 0` — `spherical_jn(n, x)`, `spherical_yn(n, x)`, `spherical_in(n, x)` and
`spherical_kn(n, x)` — for real `x`, via the half-integer relation
`f_n(x) = sqrt(pi/(2x)) F_{n+1/2}(x)` to the cylindrical kernels, matching SciPy
within documented tolerance (~1e-10). Out-of-domain/limit behavior follows SciPy
(no throw): at `x = 0`, `j_n`/`i_n` are `1` for `n = 0` and `0` otherwise,
`y_n = -inf` and `k_n = +inf`; negative arguments use the analytic continuations
(`j_n`/`i_n` even/odd in `(-1)^n`, `y_n` in `(-1)^{n+1}`, `k_n` via the
closed-form recurrence). (oracle: scipy/special/_spherical_bessel.pyx)

#### Scenario: Spherical Bessel values match SciPy
- GIVEN orders `n = 0..4` and an array of real arguments spanning negative and
  positive values
- WHEN `spherical_jn`, `spherical_yn`, `spherical_in` and `spherical_kn` are
  evaluated
- THEN each is `allclose` to the corresponding
  `scipy.special.spherical_jn/yn/in/kn`

#### Scenario: Origin limits follow SciPy
- GIVEN `x = 0`
- WHEN the spherical functions are evaluated
- THEN `spherical_jn(0,0)=spherical_in(0,0)=1`, `spherical_jn(n,0)=0` and
  `spherical_in(n,0)=0` for `n >= 1`, `spherical_yn(n,0)=-inf` and
  `spherical_kn(n,0)=+inf`

#### Scenario: Closed forms hold at order zero
- GIVEN a positive argument `x`
- WHEN `spherical_jn(0, x)` and `spherical_yn(0, x)` are evaluated
- THEN they equal `sin(x)/x` and `-cos(x)/x` respectively

### Requirement: Sine and cosine integrals

`scypp::special` SHALL provide `sici(x)` returning `(Si, Ci)` and `shichi(x)`
returning `(Shi, Chi)` for real `x`, matching SciPy within documented tolerance
(~1e-10, with `Ci` validated on an absolute tolerance near its zeros).
`Si`/`Shi` are odd and `Ci`/`Chi` are even; `Si(0)=Shi(0)=0` while
`Ci(0)=Chi(0)=-inf` (no throw). (oracle: scipy/special/cephes/sici.c, shichi.c)

#### Scenario: Sine/cosine integrals match SciPy
- GIVEN an array of real arguments spanning the convergent-series and asymptotic
  regimes (including negative values)
- WHEN `sici(x)` is evaluated
- THEN `Si` and `Ci` are `allclose` to `scipy.special.sici`

#### Scenario: Hyperbolic sine/cosine integrals match SciPy
- GIVEN an array of real arguments (including negative values)
- WHEN `shichi(x)` is evaluated
- THEN `Shi` and `Chi` are `allclose` to `scipy.special.shichi`

#### Scenario: Origin and parity behavior
- GIVEN `x = 0` and paired arguments `x`, `-x`
- WHEN `sici` and `shichi` are evaluated
- THEN `Si(0)=Shi(0)=0`, `Ci(0)=Chi(0)=-inf`, with `Si`/`Shi` odd and
  `Ci`/`Chi` even

### Requirement: Lambert W real branches

`scypp::special` SHALL provide `lambertw(x, k)` returning the real Lambert W
branches — the principal branch `k = 0` (defined for `x >= -1/e`) and the
`k = -1` branch (defined on `[-1/e, 0)`) — for real `x`, via Halley iteration,
matching the real part of `scipy.special.lambertw(x, k)` within documented
tolerance (~1e-10). Out-of-domain input returns `nan` (no throw); `lambertw(0, 0)
= 0`.

#### Scenario: Lambert W values match SciPy
- GIVEN arrays of real arguments on the principal branch (`x >= -1/e`) and on the
  `k = -1` branch (`x` in `[-1/e, 0)`)
- WHEN `lambertw(x, 0)` and `lambertw(x, -1)` are evaluated
- THEN each is `allclose` to the real part of `scipy.special.lambertw(x, k)`

#### Scenario: Defining identity holds
- GIVEN a value `w = lambertw(x, k)`
- WHEN `w * exp(w)` is formed
- THEN it equals `x`

#### Scenario: Out-of-domain returns nan
- GIVEN `x < -1/e` for `k = 0`, or `x` outside `[-1/e, 0)` for `k = -1`
- WHEN `lambertw` is evaluated
- THEN the result is `nan`

### Requirement: Riemann zeta

`scypp::special` SHALL provide `zeta(x)` (Riemann zeta, equivalently Hurwitz
zeta with `q = 1`) and `zetac(x) = zeta(x) - 1` for real `x` via Euler-Maclaurin
summation, matching `scipy.special.zeta(x)` / `scipy.special.zetac(x)` within
documented tolerance (~1e-10). `zeta` has a pole at `x = 1` returning `+inf`
(no throw).

#### Scenario: Zeta values match SciPy
- GIVEN an array of real arguments spanning `x < 1` and `x > 1` (excluding the
  pole)
- WHEN `zeta(x)` and `zetac(x)` are evaluated
- THEN they are `allclose` to `scipy.special.zeta` / `scipy.special.zetac`

#### Scenario: Pole and trivial zero
- GIVEN `x = 1` and `x = -2`
- WHEN `zeta` is evaluated
- THEN `zeta(1) = +inf` and `zeta(-2) = 0`

### Requirement: Struve and modified Struve functions

`scypp::special` SHALL provide `struve(v, x)` (Struve `H_v`) and
`modstruve(v, x)` (modified Struve `L_v`) for real order `v` and argument `x`,
using the convergent power series for small `|x|` and the Bessel-based
asymptotic expansion for large `|x|`, matching `scipy.special.struve` /
`scipy.special.modstruve` within documented tolerance (~1e-10). At `x = 0` the
value is `0` for `v > -1`. For `x < 0` only integer order is defined via the
parity `(-1)^{v+1}`; non-integer order at `x < 0` returns `nan` (no throw), as
SciPy.

#### Scenario: Struve values match SciPy
- GIVEN orders `v = 0, 1, 2, 1/2` and an array of positive arguments in the
  convergent-series and asymptotic regimes
- WHEN `struve(v, x)` and `modstruve(v, x)` are evaluated
- THEN they are `allclose` to `scipy.special.struve` / `scipy.special.modstruve`

#### Scenario: Parity and origin behavior
- GIVEN integer order `v` and paired arguments `x`, `-x`, plus `x = 0`
- WHEN `struve`/`modstruve` are evaluated
- THEN `H_v(-x) = (-1)^{v+1} H_v(x)` (likewise `L_v`), `struve(v, 0) = 0` for
  `v > -1`, and non-integer order at `x < 0` is `nan`

### Requirement: Spence's dilogarithm

`scypp::special` SHALL provide `spence(x)` returning Spence's dilogarithm
`Li_2(1 - x)` (SciPy convention) for real `x >= 0`, matching
`scipy.special.spence(x)` within documented tolerance (~1e-10). For `x < 0` the
result is `nan` (no throw).

#### Scenario: Spence values match SciPy
- GIVEN an array of arguments `x >= 0`
- WHEN `spence(x)` is evaluated
- THEN it is `allclose` to `scipy.special.spence(x)`

#### Scenario: Reference points and domain
- GIVEN `x = 0`, `x = 1`, `x = 2` and `x < 0`
- WHEN `spence` is evaluated
- THEN `spence(0) = pi^2/6`, `spence(1) = 0`, `spence(2) = -pi^2/12` and
  `spence(x < 0) = nan`

### Requirement: Confluent hypergeometric limit 0F1

`scypp::special` SHALL provide `hyp0f1(b, x)` returning the confluent limit
`sum_n x^n / ((b)_n n!)` for real `b` and `x` via term-recurrence summation,
matching `scipy.special.hyp0f1(b, x)` within documented tolerance (~1e-9). A
non-positive integer `b` (a pole) returns `nan` (no throw); `hyp0f1(b, 0) = 1`.

#### Scenario: 0F1 values match SciPy
- GIVEN orders `b = 2, 1/2` and an array of real arguments spanning negative and
  positive `x`
- WHEN `hyp0f1(b, x)` is evaluated
- THEN each is `allclose` to `scipy.special.hyp0f1(b, x)`

#### Scenario: Origin and pole
- GIVEN `x = 0` and a non-positive integer `b`
- WHEN `hyp0f1` is evaluated
- THEN `hyp0f1(b, 0) = 1` and `hyp0f1(-1, x) = nan`

### Requirement: Kummer confluent hypergeometric 1F1

`scypp::special` SHALL provide `hyp1f1(a, b, x)` returning Kummer's confluent
`M(a, b, x) = sum_n (a)_n/((b)_n n!) x^n` for real `a, b, x`, using Kummer's
transformation `M(a,b,x) = e^x M(b-a,b,-x)` for `x < 0`, matching
`scipy.special.hyp1f1(a, b, x)` within documented tolerance (~1e-9) over
moderate `|a|, |b|, |x|`. A non-positive integer `b` returns `nan` (no throw);
`hyp1f1(a, b, 0) = 1`.

#### Scenario: 1F1 values match SciPy
- GIVEN parameter sets `(a,b) = (3/2, 5/2), (-2, 3), (1/2, 13/10)` and an array
  of arguments spanning both signs of `x`
- WHEN `hyp1f1(a, b, x)` is evaluated
- THEN each is `allclose` to `scipy.special.hyp1f1(a, b, x)`

#### Scenario: Origin and pole
- GIVEN `x = 0` and a non-positive integer `b`
- WHEN `hyp1f1` is evaluated
- THEN `hyp1f1(a, b, 0) = 1` and `hyp1f1(a, -2, x) = nan`

### Requirement: Gauss hypergeometric 2F1

`scypp::special` SHALL provide `hyp2f1(a, b, c, z)` returning the Gauss
`sum_n (a)_n(b)_n/((c)_n n!) z^n` for real `a, b, c` and real `z` in `(-1, 1]`,
using the direct series, the Pfaff transformation for `z <= -1/2`, the `1 - z`
reflection for `z` in `(1/2, 1)` with non-integer `c - a - b`, and the Gauss
summation theorem at `z = 1` (when `c - a - b > 0`), matching
`scipy.special.hyp2f1(a, b, c, z)` within documented tolerance (~1e-9). A
non-positive integer `c` returns `nan`; `z` outside `(-1, 1]` returns `nan` (no
throw); `hyp2f1(a, b, c, 0) = 1`.

#### Scenario: 2F1 values match SciPy
- GIVEN parameter sets `(a,b,c) = (1/2, 1, 3/2), (1, 2, 7/2), (-3/2, 7/10, 11/5)`
  and an array of `z` in `(-1, 1)` exercising the direct series, the Pfaff
  branch and the `1 - z` reflection
- WHEN `hyp2f1(a, b, c, z)` is evaluated
- THEN each is `allclose` to `scipy.special.hyp2f1(a, b, c, z)`

#### Scenario: Gauss theorem and domain
- GIVEN `z = 0`, the boundary `z = 1` with `c - a - b > 0`, a non-positive
  integer `c`, and `z` outside `(-1, 1]`
- WHEN `hyp2f1` is evaluated
- THEN `hyp2f1(a, b, c, 0) = 1`, `hyp2f1(a, b, c, 1)` equals the Gauss-theorem
  value `Gamma(c)Gamma(c-a-b)/(Gamma(c-a)Gamma(c-b))`, and the out-of-domain
  cases return `nan`

### Requirement: Tricomi confluent hypergeometric U

`scypp::special` SHALL provide `hyperu(a, b, x)` returning Tricomi's confluent
`U(a, b, x)` for real `a`, non-integer real `b`, and `x > 0`, built from the two
`1F1` solutions
`U = pi/sin(pi b) [M(a,b,x)/(Gamma(a-b+1)Gamma(b)) - x^{1-b} M(a-b+1,2-b,x)/(Gamma(a)Gamma(2-b))]`,
matching `scipy.special.hyperu(a, b, x)` within documented tolerance (~1e-8;
a few points lose digits to subtractive cancellation). Integer `b` and
`x <= 0` return `nan` (no throw).

#### Scenario: U values match SciPy
- GIVEN parameter sets `(a,b) = (1/2, 13/10), (2, 2/5), (3/2, 27/10)` and an
  array of positive arguments `x`
- WHEN `hyperu(a, b, x)` is evaluated
- THEN each is `allclose` to `scipy.special.hyperu(a, b, x)`

#### Scenario: Domain
- GIVEN `x <= 0` or integer `b`
- WHEN `hyperu` is evaluated
- THEN the result is `nan`

