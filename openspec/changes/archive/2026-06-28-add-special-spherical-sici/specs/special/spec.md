# special (delta)

## ADDED Requirements

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
