# Tasks — minimize methods Powell, CG, L-BFGS-B

- [x] Powell direction-set method (bracket + Brent line search) in `src/optimize/minimize.cpp`
- [x] CG (Polak-Ribiere+, central-difference gradient, strong-Wolfe line search)
- [x] L-BFGS-B two-loop recursion with box bounds via gradient projection
- [x] `Bounds` alias + optional `bounds` parameter on `minimize`; existing callers compile
- [x] extend oracle generator (Powell/CG/L-BFGS-B on Rosenbrock + shifted quadratic, plus a bounded L-BFGS-B case); regenerate
- [x] tests in `tests/test_optimize.cpp` vs SciPy (x ~1e-4, fun ~1e-6)
- [x] full suite green; `openspec validate add-optimize-minimize-methods --strict`
- [x] trim delivered methods from `add-optimize-advanced` backlog
