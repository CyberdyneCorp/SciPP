# Add hypergeometric functions (hyp0f1, hyp1f1, hyp2f1, hyperu) to scipp::special

## Why

`scipp::special` already ships the gamma, error-function, Bessel, spherical
Bessel, sine/cosine integral, Airy, elliptic and misc families. The
`add-special-extras` backlog schedules the remaining high-use `scipy.special`
surface; the "Hypergeometric" line — `hyp0f1`, `hyp1f1`, `hyp2f1`, `hyperu` — is
the next group to graduate. Each reduces to a power series plus the existing
gamma kernel, so they need **no NumPP changes**.

## What changes

Adds to the **special** capability, validated against the SciPy oracle:

- **0F1 confluent limit** (`hyp0f1(b, x)`): the power series
  `sum_n x^n / ((b)_n n!)` summed by term recurrence. Non-positive integer `b`
  (a pole) returns `nan`.
- **Kummer 1F1** (`hyp1f1(a, b, x) = M(a, b, x)`): the power series
  `sum_n (a)_n/((b)_n n!) x^n`; for `x < 0` it applies Kummer's transformation
  `M(a,b,x) = e^x M(b-a,b,-x)` so the summed terms stay positive and the result
  is numerically stable. Delivered for moderate `|a|, |b|, |x|`.
- **Gauss 2F1** (`hyp2f1(a, b, c, z)`): the power series
  `sum_n (a)_n(b)_n/((c)_n n!) z^n` converges for `|z| < 1`; it is extended
  toward `|z| -> 1` with the standard linear transformations — the Pfaff
  transformation `(1-z)^{-a} 2F1(a, c-b; c; z/(z-1))` for `z <= -1/2`, and the
  `1 - z` reflection for `z` in `(1/2, 1)` when `c - a - b` is non-integer. At
  `z = 1` the Gauss summation theorem is used when `c - a - b > 0`.
- **Tricomi U** (`hyperu(a, b, x)`): the confluent `U` for `x > 0` built from the
  two `1F1` solutions,
  `U = pi/sin(pi b) * [M(a,b,x)/(Gamma(a-b+1)Gamma(b)) - x^{1-b} M(a-b+1,2-b,x)/(Gamma(a)Gamma(2-b))]`,
  using the existing gamma. Delivered for non-integer `b`.

## Impact

- Affected specs: **adds** a hypergeometric-functions requirement group to the
  `special` capability.
- Affected code: new `src/special/hypergeometric.cpp` (wired into
  `src/CMakeLists.txt`); declarations in `include/scipp/special/special.hpp`;
  oracle block in `tests/oracle/generate.py`; new
  `tests/test_special_hypergeometric.cpp`.
- Trims the "Hypergeometric" line from the `add-special-extras` backlog.

## Non-goals

- `hyp2f1` is delivered on the convergent / linearly-transformable region
  `-1 < z <= 1`. The non-principal sheet (`z > 1`), the slowly-convergent
  approach for which `c - a - b` is a positive integer and `z` is extremely
  close to `1` (logarithmic case), and multi-precision parity at the branch
  point are out of scope and return `nan` outside `(-1, 1]`.
- `hyp1f1` for very large `|x|` (where even the Kummer-transformed series loses
  digits to growth/cancellation) is out of scope; the oracle grid stays at
  moderate `|x|`.
- `hyperu` is delivered for non-integer `b`; integer `b` (the logarithmic limit
  requiring a separate connection formula) returns `nan`, and the connection
  formula's subtractive cancellation limits a few points to ~1e-8 relative
  accuracy. `x <= 0` returns `nan`.
- Only the scalar real-argument overloads are provided (no `ndarray` overloads
  and no complex arguments).
