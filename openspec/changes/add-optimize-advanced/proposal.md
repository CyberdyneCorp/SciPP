# Advanced optimize (deferred backlog)

## Why

Phase 4 (`add-optimize`, archived) delivered the unconstrained optimization core.
This change is the **tracked backlog** for the deferred solvers. Not implemented yet.

## What changes

Adds (as target requirements) to the **optimize** capability:

- **Linear & mixed-integer programming**: `linprog` (HiGHS-style simplex /
  interior-point) and `milp` (branch-and-bound).
- **Additional `minimize` methods**: `Powell`, `CG`, `Newton-CG`, `L-BFGS-B`,
  `TNC`, `trust-ncg`/`trust-krylov`/`trust-exact`.
- **Constrained minimizers**: `SLSQP`, `trust-constr`, `COBYLA`, with bound and
  (non)linear constraints.
- **Global optimizers**: `differential_evolution`, `basinhopping`,
  `dual_annealing`, `shgo`, `direct`.
- **Misc**: `nnls`, `lsq_linear`, `root` with `hybr`/`lm`/`broyden*`/`krylov`.

## Non-goals
- Implementing anything here; tracking only.
