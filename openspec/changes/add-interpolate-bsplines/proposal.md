# Add B-splines to scypp::interpolate

## Why

The `interpolate` capability ships the piecewise-cubic family (`CubicSpline`,
`PchipInterpolator`, `Akima1DInterpolator`) but not the general B-spline
representation that underpins the rest of scipy.interpolate. B-splines are the
shared substrate for FITPACK (`splrep`/`splev`), `make_interp_spline`, and the
smoothing spline classes. Delivering the `(t, c, k)` representation plus its de
Boor evaluator and the default interpolating builder unblocks that backlog and
gives users an arbitrary-degree interpolant with scipy-compatible knots.

## What changes

Extends the **interpolate** capability — new public types/functions in
`scypp::interpolate`, validated against scipy 1.15:

- **`BSpline`**: the FITPACK `(t, c, k)` representation, evaluated by the de Boor
  recursion with `operator()(const ndarray&)` / `operator()(double)` and
  `t()` / `c()` / `k()` accessors. Default `extrapolate=true` continues the
  boundary polynomial pieces; `false` returns NaN outside `[t[k], t[n]]`,
  matching scipy.
- **`make_interp_spline(x, y, k=3)`**: builds the interpolating B-spline of
  degree `k` using scipy's default not-a-knot averaging knot scheme (degrees 1,
  2, and odd `k`), assembling the banded collocation system `B_j(x_i) c_j = y_i`
  and solving it with `numpp::linalg::solve`.
- **`splev(x, tck)`**: FITPACK-style evaluation of a `(t, c, k)` spline at points
  `x` via the same de Boor evaluator.

## Impact

- Affected specs: **modifies** the `interpolate` capability (adds one
  requirement).
- Affected code: new `src/interpolate/bspline.cpp`, header additions in
  `include/scypp/interpolate/interpolate.hpp`, oracle generator block, and
  `tests/test_interpolate_bspline.cpp`. No changes to existing interpolators.
- Trims the `BSpline`/`make_interp_spline`/`splev` part of the
  `add-interpolate-splines` backlog item.

## Non-goals

- FITPACK smoothing knot **selection** for `splrep` with `s > 0` (the automatic
  knot-placement heuristic). Only `s = 0` interpolation is delivered; an
  externally supplied `tck` can still be evaluated via `BSpline`/`splev`.
- `splint` / `sproot` (definite integral and root finding) and the spline
  antiderivative/derivative spline constructors — separate backlog items.
- `make_interp_spline` with explicit boundary-derivative conditions (`bc_type`)
  and even degrees other than 2; only the default not-a-knot scheme is provided.
