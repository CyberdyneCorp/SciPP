# Tasks — Radau stiff solver

- [x] Radau IIA(3) tableau, dense Gaussian-elimination solve, FD Jacobian in `src/integrate/solve_ivp.cpp`
- [x] simplified-Newton implicit step + Richardson step-doubling adaptive driver
- [x] dispatch `method="Radau"` in `solve_ivp`
- [x] extend oracle generator (stiff scalar + stiff 2-D system); regenerate
- [x] `tests/test_integrate_radau.cpp` vs SciPy
- [x] full suite green; `openspec validate add-integrate-radau --strict`
- [x] trim Radau item from `add-integrate-stiff-bvp` backlog
