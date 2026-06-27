# Add sparse (Phase 9)

## Why

Phase 9 of the ScyPP roadmap. `scipy.sparse` provides sparse matrix formats,
sparse linear algebra, and graph algorithms — essential for large systems where
dense storage is infeasible. It is also the **first subpackage with a GPU
acceleration target**: CSR sparse matrix-vector product (SpMV), the kernel that
dominates iterative solvers and graph propagation.

This change delivers the sparse formats, operations, iterative/direct solvers, and
graph routines on the CPU, plus the **SpMV backend-dispatch architecture** that
routes through NumPP's `CapabilityRegistry` so a future device kernel accelerates
it transparently — validated CPU-side, with cross-backend equivalence the same way
the Phase-2 GEMM path is.

## What changes

Adds the **sparse** capability — `scypp::sparse`, validated against the SciPy
oracle:

- **Formats**: `CooMatrix`, `CsrMatrix`, `CscMatrix` with construction (from
  triplets, from dense), conversion between formats, `toarray`, `nnz`,
  `transpose`, `diagonal`, sparse+sparse addition, and scalar multiply.
- **Constructors**: `eye`, `identity`, `diags`.
- **Operations**: `spmv` (CSR · dense vector), `spmm` (CSR · dense matrix), routed
  through a backend dispatch (`last_backend()` introspection); a portable CPU
  kernel is always present.
- **`sparse.linalg`**: `spsolve` (sparse direct), `cg`, `gmres` (matrix-free
  iterative via `spmv`), `norm`.
- **`sparse.csgraph`**: `dijkstra`, `bellman_ford`, `floyd_warshall`,
  `connected_components`, `minimum_spanning_tree`.

## Impact

- Affected specs: **adds** the `sparse` capability.
- Affected code: new `include/scypp/sparse/`, `src/sparse/`,
  `tests/test_sparse.cpp`, extended oracle generator. Reuses `scypp::linalg` for
  the `spsolve` factorization.
- Roadmap: checks off Phase 9 in `bootstrap-scypp-foundation/tasks.md`.

## Non-goals (deferred)

- **The actual CUDA/OpenCL/Metal CSR-SpMV device kernel** — this change
  establishes the dispatch architecture and the CPU kernel; the device kernel
  lands when a sparse GPU backend is added to NumPP (tracked separately). The
  GPU-equivalence scenario is validated via the CPU-reference path.
- **Other formats** (`DIA`, `LIL`, `BSR`) beyond conversion targets, and
  `sparse.linalg` eigensolvers (`eigsh`/`svds`), `splu`/`spilu`, and the full
  `csgraph` suite (`johnson`, `maximum_flow`, matching) — follow-ups.
