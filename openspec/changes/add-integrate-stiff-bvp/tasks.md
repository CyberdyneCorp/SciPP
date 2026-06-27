# Tasks — integrate completion (backlog, not yet scheduled)

> Tracking artifact. Each item graduates into real implementation work (its own
> change or a focused batch) when picked up; nothing here is implemented yet.

## Stiff initial-value solvers
- [ ] FD Jacobian + Newton step infrastructure (reuse `scypp::differentiate::jacobian`, `numpp::linalg::solve`)
- [ ] `solve_ivp` method `"BDF"` (variable-order/step backward differentiation)
- [ ] `solve_ivp` method `"Radau"` (implicit Runge–Kutta, 3-stage)
- [ ] `solve_ivp` method `"LSODA"` (stiff/non-stiff switching)
- [ ] legacy `odeint` entry point
- [ ] Oracle tests on stiff systems (Van der Pol large μ, Robertson kinetics)

## Boundary-value problems
- [ ] `solve_bvp` (collocation + Newton residual solve + mesh refinement)
- [ ] Oracle tests against analytic BVP solutions

## Multidimensional / additional quadrature
- [ ] `dblquad`, `tplquad`, `nquad` (nested adaptive `quad`)
- [ ] `romberg`, `quad_vec`
- [ ] Oracle tests vs SciPy and analytic values
