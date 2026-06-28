# special (delta)

## ADDED Requirements

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
