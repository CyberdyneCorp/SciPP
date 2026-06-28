# Tasks — integrate completion (backlog, not yet scheduled)

> GitHub issue: [#8](https://github.com/CyberdyneCorp/SciPP/issues/8)

> Tracking artifact. Each item graduates into real implementation work (its own
> change or a focused batch) when picked up; nothing here is implemented yet.

## Stiff initial-value solvers
- [x] FD Jacobian + Newton step infrastructure — delivered via `add-integrate-radau` (dense solve + central-difference Jacobian in `solve_ivp.cpp`)
- [x] `solve_ivp` method `"BDF"` (adaptive-step backward differentiation, order 1-2) — delivered via `add-integrate-bdf`
- [x] `solve_ivp` method `"Radau"` (implicit Runge–Kutta, 3-stage) — delivered via `add-integrate-radau`
- [ ] `solve_ivp` method `"LSODA"` (stiff/non-stiff switching)
- [ ] legacy `odeint` entry point
- [ ] Oracle tests on stiff systems (Van der Pol large μ, Robertson kinetics)

## Boundary-value problems
- [x] `solve_bvp` (collocation + Newton residual solve) — delivered via `add-integrate-bvp` (4th-order collocation, fixed mesh; adaptive refinement deferred)
- [x] Oracle tests against analytic BVP solutions — delivered via `add-integrate-bvp`

## Multidimensional / additional quadrature
- [x] `dblquad`, `tplquad`, `nquad` (nested adaptive `quad`) — delivered via `add-integrate-nested-quad`
- [x] `romberg`, `quad_vec` — delivered via `add-integrate-nested-quad`
- [x] Oracle tests vs SciPy and analytic values — delivered via `add-integrate-nested-quad`
