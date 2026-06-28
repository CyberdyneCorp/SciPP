# Tasks — solve_bvp boundary value solver

- [x] `BvpFn`, `BcFn`, `BvpResult`, and `solve_bvp` signature in `include/scypp/integrate/integrate.hpp`
- [x] 4th-order Simpson/Lobatto collocation residual + global Newton (FD Jacobian, dense Gaussian solve) in `src/integrate/solve_bvp.cpp`; wire into `src/CMakeLists.txt`
- [x] extend oracle generator (harmonic + cubic analytic cases, scipy cross-check); regenerate
- [x] `tests/test_integrate_bvp.cpp` vs analytic solutions and scipy; register in `tests/CMakeLists.txt`
- [x] full suite green; `openspec validate add-integrate-bvp --strict`
- [x] trim `solve_bvp` items from `add-integrate-stiff-bvp` backlog
