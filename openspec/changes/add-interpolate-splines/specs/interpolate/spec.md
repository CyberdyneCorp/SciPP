# interpolate Specification

## ADDED Requirements

### Requirement: B-splines and smoothing splines
`scypp::interpolate` SHALL provide `BSpline`, `make_interp_spline`, `splrep`/`splev`
and `UnivariateSpline`/`InterpolatedUnivariateSpline` (FITPACK smoothing splines),
matching SciPy within documented tolerance. (oracle: scipy/interpolate/_bsplines.py, _fitpack_py.py)

#### Scenario: B-spline evaluation and interpolation
- GIVEN knots, coefficients and degree (or samples for `make_interp_spline`)
- WHEN the spline is evaluated at query points
- THEN the values match SciPy and interpolate the samples at the knots

#### Scenario: Smoothing spline fit
- GIVEN noisy samples and a smoothing factor `s`
- WHEN `UnivariateSpline(x, y, s=s)` is fit and evaluated
- THEN the result matches SciPy's spline within tolerance

### Requirement: Scattered-data interpolation
`scypp::interpolate` SHALL provide `griddata` (`linear`/`cubic`),
`LinearNDInterpolator`, `NearestNDInterpolator` and `CloughTocher2DInterpolator`,
built on Delaunay triangulation, matching SciPy within tolerance. (oracle: scipy/interpolate/_ndgriddata.py)

#### Scenario: griddata reproduces samples
- GIVEN scattered points with values and query points
- WHEN `griddata(points, values, xi, method="linear")` is called
- THEN it reproduces the sample values at the sample points and matches SciPy
