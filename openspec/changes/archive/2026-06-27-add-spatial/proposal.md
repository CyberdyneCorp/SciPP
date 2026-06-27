# Add spatial (Phase 10)

## Why

Phase 10 of the ScyPP roadmap. `scipy.spatial` provides spatial data structures
and computational geometry — nearest-neighbor search, distance computations,
convex hulls / triangulations, and 3-D rotations. Its pairwise-distance kernels
(`cdist`/`pdist`) are a **GPU acceleration target** (dense, embarrassingly
parallel), routed like the Phase-9 SpMV through NumPP's capability registry.

`scipy.spatial` leans on Qhull for N-D geometry; this change delivers the
deterministic core (distances, KD-tree, 2-D convex hull / Delaunay, and the full
`transform.Rotation`) and defers the N-D Qhull geometry and Voronoi diagrams.

## What changes

Adds the **spatial** capability — `scypp::spatial`, validated against the SciPy
oracle:

- **Distances** (`spatial.distance`): `pdist`, `cdist`, `squareform`,
  `distance_matrix`, and the metrics `euclidean`, `sqeuclidean`, `cityblock`,
  `chebyshev`, `minkowski`, `cosine`, `correlation`, `hamming`, `jaccard`; with a
  GPU-accelerable `cdist`/`pdist` dispatch.
- **KD-tree**: `KDTree` with `query` (k nearest neighbors) and `query_ball_point`.
- **Computational geometry (2-D)**: `ConvexHull` (vertices, simplices, area,
  volume) and `Delaunay` (simplices, `find_simplex`).
- **Rotations** (`spatial.transform`): `Rotation` (from/as `quat`/`matrix`/
  `euler`/`rotvec`), `apply`, `inv`, composition, `magnitude`, and `Slerp`.

## Impact

- Affected specs: **adds** the `spatial` capability.
- Affected code: new `include/scypp/spatial/`, `src/spatial/`,
  `tests/test_spatial.cpp`, extended oracle generator.
- Roadmap: checks off Phase 10 in `bootstrap-scypp-foundation/tasks.md`.

## Non-goals (deferred)

- **N-D / 3-D Qhull geometry**: `ConvexHull`/`Delaunay` above 2-D, `Voronoi`,
  `SphericalVoronoi`, `HalfspaceIntersection` — a follow-up (needs an incremental
  N-D hull engine).
- **Actual GPU distance device kernel** — this change ships the dispatch
  architecture + CPU kernel; the CUDA/OpenCL kernel lands with a NumPP device
  backend (tracked).
- **KD-tree extras**: `query_pairs`, `count_neighbors`, `sparse_distance_matrix`,
  and `cKDTree`-specific APIs.
- **Other**: `procrustes`, `directed_hausdorff`, `geometric_slerp`,
  `Rotation.align_vectors`/`reduce`/`mean`, and the boolean-distance long tail.
