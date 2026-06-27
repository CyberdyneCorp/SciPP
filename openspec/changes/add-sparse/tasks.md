# Tasks — sparse (Phase 9)

## 1. Module scaffold + formats
- [ ] `include/scypp/sparse/sparse.hpp`; `src/sparse/*.cpp` in `src/CMakeLists.txt`; export from `scypp/scypp.hpp`
- [ ] `CooMatrix`, `CsrMatrix`, `CscMatrix`; `from_coo` (dedup+sort), `from_dense`, `toarray`, `nnz`, conversions, `transpose`, `diagonal`

## 2. Constructors + arithmetic
- [ ] `eye`, `identity`, `diags`; sparse+sparse add; scalar multiply

## 3. Products + dispatch
- [ ] CPU `spmv`/`spmm`; backend dispatch via NumPP `CapabilityRegistry`; `last_backend()`; CPU-reference device path for equivalence

## 4. sparse.linalg
- [ ] `spsolve` (dense factorization via `scypp::linalg::solve`), `cg`, `gmres` (matrix-free), `norm`

## 5. sparse.csgraph
- [ ] `dijkstra` (heap), `bellman_ford`, `floyd_warshall`, `connected_components`, `minimum_spanning_tree`

## 6. Oracle + validation
- [ ] Extend `tests/oracle/generate.py`; regenerate
- [ ] `tests/test_sparse.cpp`: formats/round-trip, spmv/spmm + dispatch equivalence, solvers, csgraph vs SciPy
- [ ] CPU build green; full suite green; `openspec validate add-sparse --strict`
- [ ] Check off Phase 9 in `bootstrap-scypp-foundation/tasks.md`; update README
