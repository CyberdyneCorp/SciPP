# interpolate (delta)

## ADDED Requirements

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
