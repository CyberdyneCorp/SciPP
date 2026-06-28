# Tasks — linprog + nnls

- [x] `linprog` two-phase simplex (Bland's rule) in `src/optimize/linprog.cpp`
- [x] `nnls` Lawson-Hanson active set in `src/optimize/nnls.cpp`
- [x] header decls (`LinprogResult`, `NNLSResult`); wire into build
- [x] extend oracle generator (3 LPs + 2 NNLS cases); regenerate
- [x] `tests/test_optimize_lp.cpp` vs SciPy (optimum, vertex, infeasible/unbounded)
- [x] full suite green; `openspec validate add-optimize-lp --strict`
- [x] trim implemented items from `add-optimize-advanced` backlog
