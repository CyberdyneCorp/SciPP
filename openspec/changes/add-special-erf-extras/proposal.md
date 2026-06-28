# Add error-function relatives to scypp::special

## Why

`scypp::special` ships `erf`/`erfc`/`erfinv`/`erfcinv` but not the closely
related family that SciPy exposes alongside them: the scaled complementary error
function `erfcx`, Dawson's integral `dawsn`, the Faddeeva function `wofz`, the
`voigt_profile` (which is a scaled `Re[w]`), and the Fresnel integrals
`fresnel`. These were parked on the `add-special-extras` backlog. They share one
analytic core — the Faddeeva function `w(z) = exp(-z^2) erfc(-i z)` — so building
that core once yields `wofz`, `voigt_profile` and `fresnel` together, while
`erfcx` and `dawsn` get dedicated real-line kernels for full accuracy.

## What changes

Extends the **special** capability — `scypp::special` — validated against
`scipy.special`:

- **`erfcx(x)`** = `exp(x^2) erfc(x)`: the direct product for `|x| < 25` (where
  `exp(x^2)` does not overflow) and an asymptotic continued fraction for larger
  `x`; negative `x` uses `erfcx(x) = 2 exp(x^2) - erfcx(-x)`, returning `+inf`
  once `2 exp(x^2)` overflows, as SciPy does.
- **`dawsn(x)`** = `(sqrt(pi)/2) exp(-x^2) erfi(x)`: a power series for small
  `|x|`, Rybicki's exponentially-shifted Gaussian-sum scheme for the moderate
  range and the asymptotic series for large `|x|`.
- **`wofz(z)`** — the Faddeeva function `w(z)` — delivered on the upper
  half-plane `Im(z) >= 0` (and its reflection for `Im(z) < 0`), assembled from a
  Taylor series in the imaginary part anchored on the exact real-axis value
  `w(x) = exp(-x^2) + i (2/sqrt(pi)) dawsn(x)` near the real axis, the Maclaurin
  series for small `|z|`, and the Faddeeva continued fraction for large `|z|`.
- **`voigt_profile(x, sigma, gamma)`** = `Re[w((x + i gamma)/(sigma sqrt2))] /
  (sigma sqrt(2 pi))`, with the `gamma = 0` pure-Gaussian limit handled exactly
  and `sigma <= 0` returning `nan`.
- **`fresnel(x)`** returning `(S, C)` via `C(x) + i S(x) = ((1+i)/2)(1 -
  exp(-z^2) w(i z))` with `z = (sqrt(pi)/2)(1 - i) x`, so the integrals inherit
  the accuracy of the Faddeeva core across the whole real line with no
  small/large-argument seam.

All match SciPy to ~1e-10 (rtol) / 1e-12 (atol) across the tested grids.

## Impact

- Affected specs: **modifies** the `special` capability (adds the
  error-function-relatives requirement).
- Affected code: new `src/special/erf_extras.cpp`; declarations added to
  `include/scypp/special/special.hpp` (`erfcx`, `dawsn`, `voigt_profile`,
  `wofz`, `fresnel`, `fresnel_t`, plus `ndarray` overloads of `erfcx`/`dawsn`);
  `tests/oracle/generate.py` and a new `tests/test_special_erf_extras.cpp`.
- Trims the "Error-fn relatives" line from the `add-special-extras` backlog.

## Non-goals

- `wofz` is delivered for the upper half-plane `Im(z) >= 0` that
  `voigt_profile` and `fresnel` require (with the standard reflection used for
  `Im(z) < 0`). Full-precision evaluation deep in the lower half-plane, where
  `w` grows like `exp(-z^2)` and overflows, is out of scope and matches SciPy's
  overflow behavior rather than its finite values; the complex domain beyond the
  upper half-plane stays on the `add-special-extras` backlog.
- `erfi` is not exposed as a standalone entry point (it is only used internally
  to derive `dawsn`); it remains available implicitly and can graduate later.
