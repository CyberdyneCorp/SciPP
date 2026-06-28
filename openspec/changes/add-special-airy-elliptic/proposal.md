# Add Airy functions and elliptic integrals to scypp::special

## Why

`scypp::special` already ships the gamma, error-function and Bessel families. The
`add-special-extras` backlog scheduled the remaining high-use `scipy.special`
surface; Airy functions and the elliptic integrals are the first two families to
graduate. Both are pure-CPU scalar kernels that reduce to machinery already in the
module — Airy to the modified/ordinary Bessel kernels (`iv`/`kv`/`jv`/`yv`), and
the elliptic integrals to the arithmetic-geometric mean and Carlson symmetric
forms — so they need **no NumPP changes**.

## What changes

Adds to the **special** capability, validated against the SciPy oracle:

- **Airy**: `airy(x) -> (Ai, Aip, Bi, Bip)` and the exponentially-scaled
  `airye(x)`. For `x > 0` the connection is `Ai = (1/pi) sqrt(x/3) K_{1/3}(zeta)`,
  `Bi = sqrt(x/3)(I_{-1/3}+I_{1/3})(zeta)` with `zeta = (2/3) x^{3/2}`; negative
  orders are removed via `I_{-v} = I_v + (2/pi) sin(v*pi) K_v`. For `x < 0` the
  connection is via `J_{±1/3}`/`J_{±2/3}`, with negative orders removed via the
  `J_{-v}` reflection formula so only positive-order `jv`/`yv` are needed. `x = 0`
  returns the exact constants. `airye` applies `exp(zeta)` / `exp(-zeta)` scaling
  for `x >= 0` and matches SciPy by returning `nan` for the scaled `Ai`/`Aip` at
  `x < 0` (unscaled `Bi`/`Bip`).
- **Complete elliptic integrals**: `ellipk(m)`, `ellipkm1(p) = ellipk(1-p)` and
  `ellipe(m)` via the arithmetic-geometric-mean iteration. `ellipkm1` takes `p`
  directly into the AGM (`b0 = sqrt(p)`) for accuracy as `m -> 1`. `ellipe` uses
  the A&S 17.6 `c_n` sum. Parameterization is `m = k^2` (SciPy convention).
- **Incomplete elliptic integrals**: `ellipkinc(phi, m)` and `ellipeinc(phi, m)`
  via the Carlson symmetric forms `R_F`/`R_D`, with `phi` reduced modulo `pi`
  (quasi-period `2K`/`2E`) so the duplication algorithm runs on the principal
  amplitude.
- **Jacobi elliptic functions**: `ellipj(u, m) -> (sn, cn, dn, ph)` via the
  descending Landen/AGM transformation, with `ph` the accumulated amplitude.

## Impact

- Affected specs: **adds** an Airy requirement and an elliptic-integral
  requirement to the `special` capability.
- Affected code: new `src/special/airy.cpp` and `src/special/elliptic.cpp`
  (wired into `src/CMakeLists.txt`); `airy_t`/`ellipj_t` structs and declarations
  in `include/scypp/special/special.hpp`; oracle block in
  `tests/oracle/generate.py`; new `tests/test_special_airy_elliptic.cpp`.
- Trims the Airy and Elliptic lines from the `add-special-extras` backlog.

## Non-goals

- Complex-argument Airy/elliptic variants and `itairy` (integrals of Airy
  functions) remain deferred.
- The `out`-array / `ndarray` overloads for these multi-output and multi-argument
  functions are out of scope; the scalar struct-returning forms are delivered
  (consistent with SciPy's tuple returns).
- The error-function relatives, more-Bessel, integral, misc and hypergeometric
  families stay in the `add-special-extras` backlog.
