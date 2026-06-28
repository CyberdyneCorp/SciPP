# Tasks — BDF stiff solver

- [x] BDF Newton stage solve (frozen FD Jacobian, dense Gaussian elimination) in `src/integrate/solve_ivp.cpp`
- [x] adaptive order-1/2 driver (implicit Euler + variable-step BDF2, embedded error estimate, first-step Richardson)
- [x] dispatch `method="BDF"` in `solve_ivp`
- [x] extend oracle generator (analytic stiff scalar + stiff 2-D system); regenerate
- [x] `tests/test_integrate_bdf.cpp` vs analytic solutions
- [x] full suite green; `openspec validate add-integrate-bdf --strict`
- [x] trim BDF item from `add-integrate-stiff-bvp` backlog
