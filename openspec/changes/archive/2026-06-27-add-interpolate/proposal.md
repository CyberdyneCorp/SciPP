# Add interpolate (Phase 6)

## Why

Phase 6 of the ScyPP roadmap. `scipy.interpolate` provides 1-D and N-D
interpolation — used pervasively for resampling, table lookup, and surrogate
modeling. Like `optimize` and `integrate` it is pure algorithm over
`numpp::ndarray`, and its interpolators are **stateful callable objects**
(construct from data, then evaluate at query points), which ScyPP mirrors as
classes with `operator()`.

`scipy.interpolate` is large; this change delivers the deterministic
interpolators (piecewise, spline, regular-grid, and radial-basis) and defers the
FITPACK smoothing splines and the Delaunay-based scattered-data path, which carry
their own large dependencies (smoothing-spline fitting; triangulation, which
overlaps `spatial` in Phase 10).

## What changes

Adds the **interpolate** capability — `scypp::interpolate`, validated against the
SciPy oracle:

- **1-D interpolation**: `Interp1d` (kinds `"linear"`, `"nearest"`, `"previous"`,
  `"next"`) with bounds handling / `fill_value`.
- **Cubic splines**: `CubicSpline` (boundary conditions `"not-a-knot"` (default),
  `"natural"`, `"clamped"`) with derivative evaluation.
- **Monotone / smooth piecewise**: `PchipInterpolator` (monotone cubic Hermite)
  and `Akima1DInterpolator`.
- **N-D regular grid**: `RegularGridInterpolator` (`"linear"`, `"nearest"`) and the
  `interpn` convenience wrapper.
- **Scattered radial basis**: `RBFInterpolator` (kernels `thin_plate_spline`,
  `multiquadric`, `linear`, `cubic`, `gaussian`) with a polynomial tail.

## Impact

- Affected specs: **adds** the `interpolate` capability.
- Affected code: new `include/scypp/interpolate/`, `src/interpolate/`,
  `tests/test_interpolate.cpp`, extended oracle generator. Uses `numpp::linalg::solve`
  for the spline tridiagonal systems and the RBF linear solve.
- Roadmap: checks off Phase 6 in `bootstrap-scypp-foundation/tasks.md`.

## Non-goals (deferred)

- **Smoothing / FITPACK splines**: `splrep`/`splev`, `BSpline`,
  `UnivariateSpline`, `InterpolatedUnivariateSpline`, `make_interp_spline`.
- **Scattered-data triangulation**: `griddata` (`linear`/`cubic`),
  `LinearNDInterpolator`, `NearestNDInterpolator`, `CloughTocher2DInterpolator`
  (Delaunay-based — lands with `spatial` in Phase 10).
- **`BarycentricInterpolator`**, `KroghInterpolator`, and `pchip`/spline
  antiderivative/root helpers beyond derivative evaluation.
