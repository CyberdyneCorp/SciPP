# Add stiff ODE solvers, BVP, and multidimensional quadrature (integrate completion)

## Why

Phase 5 (`add-integrate`, archived) delivered the **explicit** ODE solvers and
1-D quadrature core. This change is the **tracked backlog** for the remaining
`scipy.integrate` surface deferred there — the implicit/stiff initial-value
solvers, the boundary-value solver, and multidimensional quadrature. It is
recorded now so the work flows through OpenSpec when picked up; it is **not
implemented or archived yet**.

Each area below should graduate into its own focused change (or be implemented
together) when scheduled. They are grouped here because they share the
prerequisite SciPP already has: `numpp::linalg::solve` for the Newton systems and
`scipp::differentiate` for numerical Jacobians.

## What changes

Adds (as target requirements) to the **integrate** capability:

- **Implicit / stiff initial-value solvers**: `solve_ivp` methods `"Radau"`
  (implicit Runge–Kutta), `"BDF"` (variable-order backward differentiation), and
  `"LSODA"` (automatic stiff/non-stiff switching), plus the legacy `odeint`. These
  need a Jacobian (analytic or finite-difference) and a Newton iteration per step.
- **Boundary-value problems**: `solve_bvp` (collocation with a residual Newton
  solve and mesh refinement).
- **Multidimensional quadrature**: `dblquad`, `tplquad`, `nquad` (nested adaptive
  `quad`).
- **Other 1-D quadrature**: `romberg` and `quad_vec`.

## Impact

- Affected specs: extends the `integrate` capability with the stiff/BVP/multidim
  requirements (merged into the baseline only when this change is implemented and
  archived).
- No code in this change — it is the planning/tracking artifact.

## Non-goals

- Implementing anything here; this is the backlog. Each unchecked item in
  `tasks.md` graduates into real work when started.
- QMC integration and `cumulative_simpson` (separate, lower priority).
