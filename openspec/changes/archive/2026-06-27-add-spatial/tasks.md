# Tasks — spatial (Phase 10)

## 1. Module scaffold
- [x] `include/scypp/spatial/spatial.hpp`; `src/spatial/*.cpp` in `src/CMakeLists.txt`; export from `scypp/scypp.hpp`

## 2. Distances
- [x] metrics (euclidean/sqeuclidean/cityblock/chebyshev/minkowski/cosine/correlation/hamming/jaccard)
- [x] `pdist`, `cdist`, `squareform`, `distance_matrix`; backend dispatch + `last_backend()`

## 3. KD-tree
- [x] `KDTree` build (median split); `query` (k-NN, bounded heap); `query_ball_point`

## 4. Geometry (2-D)
- [x] `ConvexHull` (monotone chain; vertices/simplices/area/volume)
- [x] `Delaunay` (Bowyer–Watson; simplices/find_simplex)

## 5. Rotations
- [x] `Rotation`: from/as quat/matrix/rotvec; `apply`, `inv`, `*`, `magnitude`
- [x] `from_euler`/`as_euler` (general sequences); `Slerp`

## 6. Oracle + validation
- [x] Extend `tests/oracle/generate.py`; regenerate
- [x] `tests/test_spatial.cpp`: distances + dispatch, KD-tree, hull/Delaunay, Rotation vs SciPy
- [x] CPU build green; full suite green; `openspec validate add-spatial --strict`
- [x] Check off Phase 10 in `bootstrap-scypp-foundation/tasks.md`; update README
