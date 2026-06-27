# Interpolate splines & scattered data (deferred backlog)

## Why

Phase 6 (`add-interpolate`, archived) delivered the deterministic interpolators.
This change is the **tracked backlog** for the FITPACK smoothing splines and the
Delaunay-based scattered-data interpolators. Not implemented yet.

## What changes

Adds (as target requirements) to the **interpolate** capability:

- **B-splines / FITPACK**: `BSpline`, `make_interp_spline`, `splrep`/`splev`,
  `splint`/`sproot`, `UnivariateSpline`, `InterpolatedUnivariateSpline`,
  `LSQUnivariateSpline`, `splprep` parametric splines.
- **Scattered-data (Delaunay)**: `griddata` (`linear`/`cubic`),
  `LinearNDInterpolator`, `NearestNDInterpolator`, `CloughTocher2DInterpolator`
  (depends on `spatial`'s Delaunay triangulation, Phase 10).
- **Other 1-D**: `BarycentricInterpolator`, `KroghInterpolator`,
  `Akima1DInterpolator` makima variant, and spline antiderivative/integral helpers.

## Non-goals
- Implementing anything here; tracking only.
