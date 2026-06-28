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

