# interpolate Specification

## Purpose
TBD - created by archiving change add-interpolate. Update Purpose after archive.
## Requirements
### Requirement: One-dimensional interpolation

`scypp::interpolate` SHALL provide `Interp1d`, a callable built from sorted sample
points supporting kinds `"linear"`, `"nearest"`, `"previous"` and `"next"`, with
out-of-bounds handling via `fill_value`, matching SciPy. (oracle:
scipy/interpolate/_interpolate.py)

#### Scenario: Linear and nearest interpolation match SciPy
- GIVEN samples `(x, y)` and query points within range
- WHEN an `Interp1d` of kind `"linear"` (and `"nearest"`) is evaluated at the
  queries
- THEN the results are `allclose` to SciPy's `interp1d`, and reproduce `y` at the
  sample points

#### Scenario: Out-of-bounds uses fill_value
- GIVEN an `Interp1d` constructed with a `fill_value`
- WHEN it is evaluated outside the sample range
- THEN it returns the fill value (matching SciPy)

### Requirement: Cubic spline interpolation

`scypp::interpolate` SHALL provide `CubicSpline` with boundary conditions
`"not-a-knot"` (default), `"natural"` and `"clamped"`, exposing evaluation of the
spline and its derivatives, matching SciPy. (oracle:
scipy/interpolate/_cubic.py)

#### Scenario: Cubic spline matches SciPy and interpolates the data
- GIVEN samples `(x, y)`
- WHEN a `CubicSpline` (default boundary) is evaluated at query points
- THEN the result is `allclose` to SciPy's `CubicSpline`, passes through the
  samples, and is C²-continuous

#### Scenario: Boundary conditions and derivatives
- GIVEN samples and a boundary condition in {not-a-knot, natural, clamped}
- WHEN the spline and its first derivative (`nu=1`) are evaluated
- THEN both are `allclose` to SciPy for that boundary condition

### Requirement: Monotone and Akima piecewise interpolation

`scypp::interpolate` SHALL provide `PchipInterpolator` (monotone cubic Hermite) and
`Akima1DInterpolator`, matching SciPy. (oracle: scipy/interpolate/_cubic.py)

#### Scenario: PCHIP is shape-preserving and matches SciPy
- GIVEN monotone samples
- WHEN `PchipInterpolator` is evaluated between samples
- THEN the result is `allclose` to SciPy's `PchipInterpolator` and stays monotone
  (no overshoot)

#### Scenario: Akima matches SciPy
- GIVEN samples `(x, y)`
- WHEN `Akima1DInterpolator` is evaluated at query points
- THEN the result is `allclose` to SciPy's `Akima1DInterpolator`

### Requirement: Regular-grid N-D interpolation

`scypp::interpolate` SHALL provide `RegularGridInterpolator` (methods `"linear"`
and `"nearest"`) over an N-D rectilinear grid, and the `interpn` convenience
wrapper, matching SciPy. (oracle: scipy/interpolate/_rgi.py)

#### Scenario: Multilinear interpolation on a grid
- GIVEN a 2-D grid of values and query points inside the grid
- WHEN a `RegularGridInterpolator` of method `"linear"` (and `"nearest"`) is
  evaluated
- THEN the results are `allclose` to SciPy and reproduce grid values at grid nodes

#### Scenario: interpn wrapper
- GIVEN grid points, values and query points
- WHEN `interpn(points, values, xi)` is called
- THEN it returns the same result as the equivalent `RegularGridInterpolator`

### Requirement: Radial-basis-function interpolation

`scypp::interpolate` SHALL provide `RBFInterpolator` for scattered data with
kernels `thin_plate_spline`, `multiquadric`, `linear`, `cubic` and `gaussian` and a
polynomial tail, matching SciPy. (oracle: scipy/interpolate/_rbfinterp.py)

#### Scenario: RBF reproduces samples and matches SciPy
- GIVEN scattered centers `y` with values `d`
- WHEN an `RBFInterpolator` is evaluated at the centers and at intermediate points
- THEN it reproduces `d` at the centers and is `allclose` to SciPy's
  `RBFInterpolator` at the intermediate points

### Requirement: Scattered 2-D interpolation
The system SHALL provide `griddata(points, values, xi, method, fill_value)` that
interpolates scattered 2-D samples onto query points, supporting `method`
`"nearest"` (closest-sample value) and `"linear"` (barycentric over the Delaunay
triangulation), assigning `fill_value` to query points outside the convex hull.

#### Scenario: Linear interpolation of a linear field
- GIVEN scattered points with `values = 2x + 3y + 1`
- WHEN `griddata` is called with `method = "linear"` at interior query points
- THEN the results SHALL equal the field evaluated at the query points
- AND SHALL match `scipy.interpolate.griddata` to `allclose` tolerance.

#### Scenario: Nearest-neighbour interpolation
- GIVEN scattered points and values
- WHEN `griddata` is called with `method = "nearest"`
- THEN each query SHALL receive the value of its closest sample
- AND the results SHALL match SciPy.

#### Scenario: Query outside the convex hull
- GIVEN a query point outside the convex hull of the samples
- WHEN `griddata` is called with `method = "linear"`
- THEN that query SHALL receive `fill_value` (NaN by default).

### Requirement: B-spline representation and interpolation
The system SHALL provide a `scypp::interpolate::BSpline` type holding the FITPACK
`(t, c, k)` representation, evaluated by the de Boor recursion, together with
`make_interp_spline(x, y, k)` for the default interpolating spline and a
`splev(x, tck)` evaluation wrapper. `BSpline` SHALL expose `t()`, `c()`, and
`k()` accessors and SHALL extrapolate off the boundary polynomial pieces by
default.

#### Scenario: de Boor evaluation of a (t, c, k) spline
- GIVEN a knot vector `t`, coefficients `c`, and degree `k` from
  `scipy.interpolate.make_interp_spline` for `k` in {1, 2, 3}
- WHEN `BSpline(t, c, k)` is evaluated at sample points spanning and exceeding
  the knot range
- THEN the results SHALL match `scipy`'s evaluation of the same spline (including
  the extrapolation region) to `allclose` tolerance (~1e-9).

#### Scenario: splev agrees with scipy.interpolate.splev
- GIVEN the same `(t, c, k)` tuple
- WHEN `splev(x, tck)` is evaluated at the sample points
- THEN the results SHALL match `scipy.interpolate.splev(x, (t, c, k))` to
  `allclose` tolerance (~1e-9).

#### Scenario: interpolating B-spline through data
- GIVEN data `(x, y)` and degree `k` in {1, 2, 3}
- WHEN `make_interp_spline(x, y, k)` builds the spline using scipy's default
  not-a-knot knot scheme and solves the collocation system
- THEN evaluating it SHALL reproduce `scipy.interpolate.make_interp_spline(x, y,
  k)` to `allclose` tolerance (~1e-7) and SHALL pass through every data node
  `y_i` at `x_i`.

