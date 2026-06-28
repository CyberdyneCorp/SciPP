# Add griddata scattered interpolation

## Why

`scipy.interpolate.griddata` is the standard entry point for interpolating
scattered (unstructured) data onto query points and was deferred from Phase 6.
SciPP already ships the two geometry primitives it needs — a KD-tree and a 2-D
Delaunay triangulation — so `nearest` and `linear` griddata need no NumPP changes.

## What changes

Extends the **interpolate** capability — `scipp::interpolate`, validated against
the SciPy oracle:

- **`griddata(points, values, xi, method, fill_value)`** for 2-D scattered data:
  - `method = "nearest"`: nearest-sample value via the KD-tree.
  - `method = "linear"`: barycentric interpolation over the Delaunay
    triangulation; query points outside the convex hull get `fill_value`
    (NaN by default), matching SciPy.

## Impact

- Affected specs: **modifies** the `interpolate` capability (adds one requirement).
- Affected code: new `src/interpolate/griddata.cpp`, header decl in
  `include/scipp/interpolate/interpolate.hpp`, `tests/test_interpolate_griddata.cpp`,
  extended oracle generator. Reuses `scipp::spatial` KDTree and Delaunay.
- Trims the `griddata` item from the `add-interpolate-splines` backlog.

## Non-goals

- `method = "cubic"` (Clough-Tocher C1 interpolation) — deferred.
- N-D linear interpolation (the Delaunay primitive is 2-D); `nearest` is 2-D here
  for symmetry, extendable when N-D Delaunay lands.
