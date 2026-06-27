# Tasks — spatial (Phase 10)

## 1. Module scaffold
- [ ] `include/scypp/spatial/spatial.hpp`; `src/spatial/*.cpp` in `src/CMakeLists.txt`; export from `scypp/scypp.hpp`

## 2. Distances
- [ ] metrics (euclidean/sqeuclidean/cityblock/chebyshev/minkowski/cosine/correlation/hamming/jaccard)
- [ ] `pdist`, `cdist`, `squareform`, `distance_matrix`; backend dispatch + `last_backend()`

## 3. KD-tree
- [ ] `KDTree` build (median split); `query` (k-NN, bounded heap); `query_ball_point`

## 4. Geometry (2-D)
- [ ] `ConvexHull` (monotone chain; vertices/simplices/area/volume)
- [ ] `Delaunay` (Bowyer–Watson; simplices/find_simplex)

## 5. Rotations
- [ ] `Rotation`: from/as quat/matrix/rotvec; `apply`, `inv`, `*`, `magnitude`
- [ ] `from_euler`/`as_euler` (general sequences); `Slerp`

## 6. Oracle + validation
- [ ] Extend `tests/oracle/generate.py`; regenerate
- [ ] `tests/test_spatial.cpp`: distances + dispatch, KD-tree, hull/Delaunay, Rotation vs SciPy
- [ ] CPU build green; full suite green; `openspec validate add-spatial --strict`
- [ ] Check off Phase 10 in `bootstrap-scypp-foundation/tasks.md`; update README
