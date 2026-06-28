# Tasks — griddata

- [x] `griddata` (nearest via KDTree, linear via Delaunay barycentric) in `src/interpolate/griddata.cpp`
- [x] header decl; wire into build
- [x] extend oracle generator (linear field + nearest + outside-hull NaN); regenerate
- [x] `tests/test_interpolate_griddata.cpp` vs SciPy
- [x] full suite green; `openspec validate add-interpolate-griddata --strict`
- [x] trim `griddata` item from `add-interpolate-splines` backlog
