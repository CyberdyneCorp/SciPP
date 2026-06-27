# Tasks — sparse (Phase 9)

## 1. Module scaffold + formats
- [x] `include/scypp/sparse/sparse.hpp`; `src/sparse/*.cpp` in `src/CMakeLists.txt`; export from `scypp/scypp.hpp`
- [x] `CooMatrix`, `CsrMatrix`, `CscMatrix`; `from_coo` (dedup+sort), `from_dense`, `toarray`, `nnz`, conversions, `transpose`, `diagonal`

## 2. Constructors + arithmetic
- [x] `eye`, `identity`, `diags`; sparse+sparse add; scalar multiply

## 3. Products + dispatch
- [x] CPU `spmv`/`spmm`; backend dispatch via NumPP `CapabilityRegistry`; `last_backend()`; CPU-reference device path for equivalence

## 4. sparse.linalg
- [x] `spsolve` (dense factorization via `scypp::linalg::solve`), `cg`, `gmres` (matrix-free), `norm`

## 5. sparse.csgraph
- [x] `dijkstra` (heap), `bellman_ford`, `floyd_warshall`, `connected_components`, `minimum_spanning_tree`

## 6. Oracle + validation
- [x] Extend `tests/oracle/generate.py`; regenerate
- [x] `tests/test_sparse.cpp`: formats/round-trip, spmv/spmm + dispatch equivalence, solvers, csgraph vs SciPy
- [x] CPU build green; full suite green; `openspec validate add-sparse --strict`
- [x] Check off Phase 9 in `bootstrap-scypp-foundation/tasks.md`; update README
