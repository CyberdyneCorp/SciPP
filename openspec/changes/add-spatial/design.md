# Design — spatial (Phase 10)

## Distances

`pdist(X, metric)` returns the condensed upper-triangle vector
(`n·(n−1)/2`); `cdist(XA, XB, metric)` returns the `m×n` matrix; `squareform`
converts between condensed and square; `distance_matrix` is Minkowski `cdist`.
Metrics are scalar `dist(u, v)` kernels (`euclidean`, `sqeuclidean`, `cityblock`,
`chebyshev`, `minkowski(p)`, `cosine`, `correlation`, `hamming`, `jaccard`)
matching SciPy's definitions exactly.

`cdist`/`pdist` choose CPU vs device via NumPP's `CapabilityRegistry` (size +
availability), record `last_backend()`, and always keep the portable CPU kernel —
the same dispatch shape as Phase-9 SpMV. The actual GPU kernel is deferred to a
NumPP device backend.

## KD-tree

`KDTree` builds a median-split binary tree over the points (cycling axes).
`query(x, k)` does a branch-and-bound descent with a bounded max-heap of the `k`
best, returning sorted `(distances, indices)`. `query_ball_point(x, r)` collects
all points within radius `r`. Matches SciPy for distinct distances; ties are
returned in index order.

## Computational geometry (2-D)

- **ConvexHull** — Andrew's monotone chain → hull vertex indices in
  counter-clockwise order; `simplices` are the hull edges; `volume` is the enclosed
  area and `area` the perimeter (SciPy's 2-D convention). Validated against SciPy's
  vertex set (rotation-invariant compare) and the area/perimeter.
- **Delaunay** — Bowyer–Watson incremental insertion with a super-triangle →
  triangle simplices (point-index triples). For points in general position the
  triangulation is unique, so the **set** of triangles matches SciPy (compared as
  sorted vertex tuples). `find_simplex(p)` locates the containing triangle via
  barycentric tests.

## Rotations (`transform.Rotation`)

`Rotation` stores a unit quaternion in SciPy's **scalar-last** `(x, y, z, w)`
convention:

- `from_quat`/`as_quat`, `from_matrix`/`as_matrix`, `from_rotvec`/`as_rotvec`
  (axis-angle), `from_euler`/`as_euler` (any 3-char intrinsic/extrinsic sequence,
  degrees or radians).
- `apply(v)` rotates vectors; `inv()` conjugates; `operator*` composes (quaternion
  product); `magnitude()` is the rotation angle.
- `from_euler` composes elementary axis quaternions (intrinsic = upper-case
  sequence, extrinsic = lower-case, per SciPy); `as_euler` uses the general
  Bernardes–Viollet extraction SciPy ships, so all 24 conventions round-trip.
- **Slerp** interpolates a sequence of key rotations at query times via quaternion
  spherical-linear interpolation (shortest path).

Sign conventions follow SciPy (quaternion canonicalization, `as_euler` branch
choices); tests compare rotation *actions* (`apply`, matrices) and round-trips, not
raw quaternion signs.

## Oracle strategy

Compared `allclose` to SciPy: distance vectors/matrices and `squareform`;
KD-tree query distances; `ConvexHull` vertex set + area/volume; `Delaunay` triangle
set + `find_simplex`; and `Rotation` matrices / `apply` outputs / `as_euler` /
`Slerp` results. Tolerances are tight (`~1e-9`) since all are deterministic.

## Open questions

- KD-tree and ConvexHull vertex ordering can differ from SciPy on ties / collinear
  points; tests use order-invariant comparisons. `as_euler` gimbal-lock branches
  follow SciPy's documented choice.
