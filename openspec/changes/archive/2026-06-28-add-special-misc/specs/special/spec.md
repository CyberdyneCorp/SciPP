# special (delta)

## ADDED Requirements

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
