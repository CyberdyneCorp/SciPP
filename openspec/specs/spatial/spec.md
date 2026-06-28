# spatial Specification

## Purpose
TBD - created by archiving change add-spatial. Update Purpose after archive.
## Requirements
### Requirement: Distance computations

`scipp::spatial` SHALL provide `pdist`, `cdist`, `squareform` and
`distance_matrix` with the metrics `euclidean`, `sqeuclidean`, `cityblock`,
`chebyshev`, `minkowski`, `cosine`, `correlation`, `hamming` and `jaccard`,
matching SciPy. `cdist`/`pdist` SHALL select a backend via NumPP's capability
registry with a portable CPU fallback. (oracle: scipy/spatial/distance.py)

#### Scenario: pdist / cdist / squareform match SciPy
- GIVEN one or two point sets and a metric
- WHEN `pdist`, `cdist` are computed and `squareform` round-trips the condensed
  vector
- THEN the results are `allclose` to SciPy's for each metric

#### Scenario: Distance backend dispatch
- GIVEN a `cdist` run on the CPU path and on the device-reference path
- WHEN the selected backend is queried
- THEN the two results are equal within tolerance and the backend is reported, with
  the CPU kernel always available

### Requirement: KD-tree nearest-neighbor search

`scipp::spatial` SHALL provide `KDTree` with `query` (k nearest neighbors) and
`query_ball_point` (radius search), matching SciPy. (oracle: scipy/spatial/_kdtree.py)

#### Scenario: k-nearest-neighbor query
- GIVEN a point set and query points
- WHEN `KDTree(points).query(x, k)` is called
- THEN the returned distances are `allclose` to SciPy's and the neighbor indices
  match (for distinct distances)

#### Scenario: Radius query
- GIVEN a point set and a radius
- WHEN `query_ball_point(x, r)` is called
- THEN the returned index set matches SciPy's

### Requirement: 2-D convex hull and Delaunay triangulation

`scipp::spatial` SHALL provide `ConvexHull` (vertices, simplices, area, volume) and
`Delaunay` (simplices, `find_simplex`) for 2-D point sets, matching SciPy. (oracle: scipy/spatial/_qhull.pyx)

#### Scenario: Convex hull matches SciPy
- GIVEN a 2-D point set
- WHEN `ConvexHull` is computed
- THEN the hull vertex set matches SciPy and the enclosed area (`volume`) and
  perimeter (`area`) are `allclose` to SciPy's

#### Scenario: Delaunay triangulation matches SciPy
- GIVEN a 2-D point set in general position
- WHEN `Delaunay` is computed
- THEN the set of triangles matches SciPy's (as sorted vertex tuples), and
  `find_simplex` locates the containing triangle for interior query points

### Requirement: 3-D rotations

`scipp::spatial::transform` SHALL provide `Rotation` with conversions to/from
quaternion, matrix, Euler angles and rotation vector, plus `apply`, `inv`,
composition, `magnitude` and `Slerp`, matching SciPy. (oracle: scipy/spatial/transform/_rotation.pyx)

#### Scenario: Rotation conversions round-trip and match SciPy
- GIVEN a rotation specified as a quaternion / matrix / Euler angles / rotvec
- WHEN it is converted between representations and `apply`-ed to vectors
- THEN every representation round-trips, the rotated vectors are `allclose` to
  SciPy's, and `as_euler` matches SciPy for the given sequence

#### Scenario: Composition, inverse and Slerp
- GIVEN two rotations and a set of key rotations with times
- WHEN they are composed, inverted, and interpolated with `Slerp`
- THEN the resulting rotation actions are `allclose` to SciPy's

