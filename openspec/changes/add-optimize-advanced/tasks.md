# Tasks — advanced optimize (backlog, not yet scheduled)

> Tracking artifact. Each item graduates into real implementation when picked up.

- [x] `linprog` (two-phase simplex) — delivered via `add-optimize-lp`; `milp` (branch-and-bound) still pending
- [ ] `minimize` methods: `Powell`, `CG`, `Newton-CG`, `L-BFGS-B`, `TNC`, trust-region family
- [ ] constrained: `SLSQP`, `trust-constr`, `COBYLA` (+ bound/linear/nonlinear constraints)
- [ ] global: `differential_evolution`, `basinhopping`, `dual_annealing`, `shgo`, `direct`
- [x] `nnls` — delivered via `add-optimize-lp`; `lsq_linear`, `root` (hybr/lm/broyden/krylov) still pending
- [ ] oracle tests for each
