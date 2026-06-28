# Tasks — advanced optimize (backlog, not yet scheduled)

> GitHub issue: [#3](https://github.com/CyberdyneCorp/SciPP/issues/3)

> Tracking artifact. Each item graduates into real implementation when picked up.

- [x] `linprog` (two-phase simplex) — delivered via `add-optimize-lp`; `milp` (branch-and-bound) still pending
- [x] `minimize` methods: `Powell`, `CG`, `L-BFGS-B` — delivered via `add-optimize-minimize-methods`; `Newton-CG`, `TNC`, trust-region family still pending
- [ ] constrained: `SLSQP`, `trust-constr`, `COBYLA` (+ bound/linear/nonlinear constraints)
- [ ] global: `differential_evolution`, `basinhopping`, `dual_annealing`, `shgo`, `direct`
- [x] `nnls` — delivered via `add-optimize-lp`; `lsq_linear`, `root` (hybr/lm/broyden/krylov) still pending
- [ ] oracle tests for each
