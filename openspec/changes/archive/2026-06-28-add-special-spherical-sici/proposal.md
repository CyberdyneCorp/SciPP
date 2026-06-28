# Add spherical Bessel functions and sine/cosine integrals to scipp::special

## Why

`scipp::special` already ships the cylindrical Bessel kernels (`jv`/`yv`/`iv`/`kv`)
and the exponential integrals (`expi`/`exp1`). The `add-special-extras` backlog
schedules the remaining high-use `scipy.special` surface; the spherical Bessel
family and the sine/cosine integrals are the next two groups to graduate. Both
reduce to machinery already in the module — the spherical functions to the
half-integer cylindrical Bessel kernels, `shichi` to the exponential integrals —
so they need **no NumPP changes**.

## What changes

Adds to the **special** capability, validated against the SciPy oracle:

- **Spherical Bessel functions** (integer order `n >= 0`): `spherical_jn(n, x)`,
  `spherical_yn(n, x)`, `spherical_in(n, x)`, `spherical_kn(n, x)` via the
  half-integer relation `f_n(x) = sqrt(pi/(2x)) F_{n+1/2}(x)` for `x > 0`
  (`F = J, Y, I, K` respectively). The `x = 0` limits follow SciPy: `j_n`/`i_n`
  are `1` at `n = 0` and `0` otherwise, `y_n = -inf`, `k_n = +inf`. Negative
  arguments use the analytic continuations SciPy reports: the ordinary/regular
  functions are even or odd (`j_n(-x)=(-1)^n j_n(x)`, `y_n(-x)=(-1)^{n+1} y_n(x)`,
  `i_n(-x)=(-1)^n i_n(x)`), while `k_n` (which has no such parity) uses the
  closed-form upward recurrence `k_{m+1}=k_{m-1}+(2m+1)/x k_m` seeded by
  `k_0=(pi/2x)e^{-x}`, `k_1=k_0(1+1/x)`.
- **Sine/cosine integrals**: `sici(x) -> (Si, Ci)`. `Si`/`Ci` use the convergent
  Maclaurin series for `|x| < 17` and the divergent auxiliary-function asymptotic
  expansion (DLMF 6.12) with optimal (smallest-term) truncation above. `Si(0)=0`,
  `Ci(0)=-inf`; `Si` is odd and `Ci` is even.
- **Hyperbolic sine/cosine integrals**: `shichi(x) -> (Shi, Chi)`. For `|x| >= 2`
  these reduce to the exponential integrals, `Shi = (Ei + E1)/2`,
  `Chi = (Ei - E1)/2`; for small `|x|` the Maclaurin series is used to avoid the
  `Ei`/`E1` cancellation. `Shi(0)=0`, `Chi(0)=-inf`; `Shi` is odd, `Chi` is even.

## Impact

- Affected specs: **adds** a spherical-Bessel requirement and a sine/cosine
  integral requirement to the `special` capability.
- Affected code: new `src/special/spherical_bessel.cpp` and `src/special/sici.cpp`
  (wired into `src/CMakeLists.txt`); `sici_t`/`shichi_t` structs and declarations
  in `include/scipp/special/special.hpp`; oracle block in
  `tests/oracle/generate.py`; new `tests/test_special_spherical_sici.cpp`.
- Trims the spherical part of the "More Bessel" line and the "Integrals" line
  from the `add-special-extras` backlog.

## Non-goals

- `kelvin` and `hankel1`/`hankel2` (the complex-valued remainder of the
  "More Bessel" backlog line) stay deferred in `add-special-extras`.
- The `derivative=True` flag of `scipy.special.spherical_*` is out of scope; only
  the scalar value form is delivered.
- `Si`/`Ci` in the narrow `16 < |x| < 24` transition band fall between the
  convergent series and the well-converged asymptotic regime and reach ~1e-9
  rather than full double precision; the oracle grid stays outside that band.
  Near a zero of the oscillatory `Ci` (e.g. `x = 25`) the asymptotic floor is
  ~4e-12 absolute, so `Ci` is validated with an absolute tolerance.
