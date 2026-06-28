# interpolate (delta)

## ADDED Requirements

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
