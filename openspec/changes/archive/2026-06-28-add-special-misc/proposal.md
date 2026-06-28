# Add misc special functions (lambertw, zeta, struve, spence) to scipp::special

## Why

`scipp::special` already ships the gamma, error-function, Bessel, spherical
Bessel, sine/cosine integral, Airy and elliptic families. The
`add-special-extras` backlog schedules the remaining high-use `scipy.special`
surface; the "Misc" line — `lambertw`, `zeta`/`zetac`, `struve`/`modstruve`,
`spence` — is the next group to graduate. Each reduces to elementary kernels or
to Bessel kernels already in the module, so they need **no NumPP changes**.

## What changes

Adds to the **special** capability, validated against the SciPy oracle:

- **Lambert W** (`lambertw(x, k=0)`): the principal real branch (`k=0`, defined
  for `x >= -1/e`) and the `k=-1` branch (on `[-1/e, 0)`) via Halley iteration
  from branch-specific initial guesses (a branch-point series near `-1/e`, the
  log/log-log asymptotic for large `x`). SciPy returns complex; for these real
  branches the value equals SciPy's real part. Out-of-domain returns `nan`.
- **Riemann zeta** (`zeta(x)`, `zetac(x) = zeta(x) - 1`): Euler-Maclaurin
  summation (`N = 12` explicit terms plus a 7-term Bernoulli tail) which
  analytically continues to the whole real line. `zeta` has a pole (`+inf`) at
  `x = 1`. `zeta(x)` here is the Hurwitz zeta with `q = 1`, matching SciPy.
- **Struve / modified Struve** (`struve(v, x)`, `modstruve(v, x)`): the Struve
  `H_v` and modified Struve `L_v` via the convergent power series for small
  `|x|` and the divergent Bessel-based asymptotic expansion (`H_v = Y_v + ...`,
  `L_v = I_v - ...`, using the existing `yv`/`iv` kernels) with optimal
  truncation for large `|x|`. For `x < 0` only integer order is defined (parity
  `(-1)^{v+1}`); non-integer order at `x < 0` returns `nan`, as SciPy.
- **Spence's dilogarithm** (`spence(x)`): `Li_2(1 - x)` (SciPy convention) for
  `x >= 0` via the `Li_2` power series with the standard region-reduction
  identities; `x < 0` returns `nan`.

## Impact

- Affected specs: **adds** a misc-special-functions requirement group to the
  `special` capability.
- Affected code: new `src/special/misc.cpp` (wired into `src/CMakeLists.txt`);
  declarations in `include/scipp/special/special.hpp`; oracle block in
  `tests/oracle/generate.py`; new `tests/test_special_misc.cpp`.
- Trims the "Misc" line from the `add-special-extras` backlog.

## Non-goals

- The complex branches of `lambertw` (any `k` other than `0`/`-1`, and the
  complex outputs SciPy produces for `x < -1/e`) are out of scope; only the two
  real branches are delivered.
- `struve`/`modstruve` in the narrow `13 < |x| < 18` transition band fall
  between the cancelling alternating series and the well-converged asymptotic
  regime and reach only ~1e-9 there; the oracle grid stays outside that band
  (series for `|x| <= 10`, asymptotic for `|x| >= 25`). Non-integer order at
  `x < 0` is left as `nan`.
