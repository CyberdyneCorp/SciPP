# special Specification

## ADDED Requirements

### Requirement: Gamma and related functions

`scipp::special` SHALL provide `gamma`, `gammaln`, `loggamma`, `digamma`,
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

`scipp::special` SHALL provide `erf`, `erfc`, `erfinv` and `erfcinv`,
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

`scipp::special` SHALL provide `expn`, `exp1`, `expi` and `exprel`, element-wise
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

`scipp::special` SHALL provide the Bessel functions `jv`, `yv`, `iv`, `kv`,
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

`scipp::special` SHALL provide `eval_legendre`, `eval_chebyt`, `eval_hermite`,
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

`scipp::special` SHALL provide `comb`, `perm` and `factorial`, each supporting an
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

`scipp::special` SHALL provide `logsumexp`, `softmax` and `log_softmax` over
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
